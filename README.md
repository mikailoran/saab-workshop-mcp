# saab-workshop-mcp

An MCP server that lets an AI assistant answer Saab maintenance and repair
questions by searching the official Saab WIS (Workshop Information System)
documentation.

## Introduction

[SaabWisOnline](https://saabwisonline.com) hosts the workshop manuals Saab
technicians used — torque specs, step-by-step procedures, diagnostic trouble
codes. It's an excellent resource, but it's a folder tree you have to navigate
by hand, and you need to already know where a procedure lives to find it.

This project makes that corpus searchable in natural language from inside a
conversation with an AI assistant. Ask "how do I bleed the brakes on a 2007
9-3?" and the assistant retrieves the relevant passages from the manual — with
links back to the source page — instead of guessing from memory.

It speaks the [Model Context Protocol](https://modelcontextprotocol.io), an open
standard for exposing tools to language models, so it works with any
MCP-compatible client — desktop assistants, coding agents, editor integrations,
and agent frameworks alike — regardless of which model is behind it. Nothing in
the server is tied to a particular vendor.

It works in three stages:

| Stage | Component | What it does |
| --- | --- | --- |
| 1. Crawl | [scraper/crawl.py](scraper/crawl.py) | Walks a model/year subtree of saabwisonline.com and saves each procedure as JSON |
| 2. Ingest | [server/ingest.py](server/ingest.py) | Chunks the documents, embeds them locally, and builds a Chroma vector store |
| 3. Serve | [server/mcp_server.py](server/mcp_server.py) | Exposes `search_saab_manual(query, k)` as an MCP tool |

Everything runs locally. Embeddings are computed on-device with
[`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) via
`sentence-transformers` — no API keys, no per-query cost, no data leaving your
machine.

The reference corpus is the **9-3 (9440), model year 2007**: ~10,450 documents
chunked into ~13,000 indexed passages. Any other model/year on the site can be
crawled and indexed the same way.

## Installation

Requires Python 3.11+ (developed on 3.14). `scraper/` and `server/` are
independent subprojects with their own dependencies and virtualenvs — the
scraper only needs `requests`/`beautifulsoup4`, while the server pulls in
`torch` via `sentence-transformers`, so there's no reason to install both if
you only need one.

```bash
git clone https://github.com/mikailoran/saab-workshop-mcp.git
cd saab-workshop-mcp

# Scraper (stage 1)
python -m venv scraper/.venv
source scraper/.venv/bin/activate
pip install -r scraper/requirements.txt      # or requirements-dev.txt for pytest + ruff
deactivate

# Server (stages 2 and 3)
python -m venv server/.venv
source server/.venv/bin/activate
pip install -r server/requirements.txt       # or requirements-dev.txt
```

The embedding model (~130 MB) downloads from the HuggingFace Hub on first use
and is cached locally thereafter.

### 1. Crawl the manual

```bash
cd scraper
python crawl.py --model 9-3-9440 --year 2007
```

Output goes to a timestamped `data_<date>_<time>/` directory, with a
`data_last_run` symlink pointing at the most recent run. A full crawl takes
several hours at the default 1-second delay — see
[crawling etiquette](#crawling-etiquette) below.

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--delay` | Seconds between requests (default `1.0`) |
| `--max-docs` / `--max-pages` | Stop early — good for a quick test run |
| `--skip-dtcs` | Skip the Diagnostic Trouble Codes section (thousands of short, near-identical pages) |
| `--dfs` | Crawl depth-first instead of breadth-first |

The crawl writes its manifest even if interrupted, so `Ctrl+C` is safe.

### 2. Build the index

```bash
cd server
python ingest.py --model 9-3-9440 --year 2007
```

Reads from `../scraper/data_last_run` by default and persists a Chroma
collection to `server/index/`. Re-running upserts rather than duplicating, so
it's safe to run again after a fresh crawl. Expect ~215 MB on disk for the
reference corpus.

Sanity-check the result without involving MCP at all:

```bash
python retrieval.py "how do I replace the oil sump" --k 5
```

### 3. Connect it to an MCP client

The server supports both MCP transports. Which one you need depends on whether
your client runs on the same machine as the index:

| Transport | How it works | Use it when |
| --- | --- | --- |
| **stdio** (default) | The client spawns the server as a local subprocess and talks over stdin/stdout | The client runs on the same machine — desktop assistants, coding agents, editor integrations |
| **Streamable HTTP** | The client connects to a URL over the network | The client is hosted/browser-based, or you're sharing one server with several people |

#### Local (stdio)

Most clients read the same `mcpServers` configuration shape. Add an entry
pointing at the virtualenv's interpreter and the server script:

```json
{
  "mcpServers": {
    "saab-manual": {
      "command": "/absolute/path/to/server/.venv/bin/python3",
      "args": ["/absolute/path/to/server/mcp_server.py"]
    }
  }
}
```

Where that config lives varies by client — a JSON file in the client's config
directory, a project-level file, or a settings UI. Some clients also offer a CLI
subcommand that writes the same entry for you. Check your client's MCP
documentation for the location; the fields above are the portable part.

**Use absolute paths**, both for the interpreter and the script. The client
launches the server from an arbitrary working directory, and the venv
interpreter is what has the dependencies installed.

Most clients need a restart or an explicit reconnect before a newly added server
shows up. Once connected, ask a maintenance question and the assistant should
call `search_saab_manual` on its own.

#### Remote (Streamable HTTP)

Hosted and browser-based clients can't spawn a local process, so they need the
server reachable over the network. Set `MCP_TRANSPORT=streamable-http` (plus
optional `MCP_HOST`/`MCP_PORT`) and deploy it somewhere always-on, then add the
resulting URL in your client.

[server/Dockerfile](server/Dockerfile) builds a self-contained CPU-only image
with the model weights and the prebuilt index baked in:

```bash
cd server
docker build -t saab-rag .
docker run -p 8000:8000 saab-rag
```

Note the image includes the index, so it must be built *after* stage 2.

> **The server has no authentication.** Anyone who can reach the URL can query
> it. Put it behind a reverse proxy, network restriction, or auth layer before
> exposing it publicly — and see [Legal](#legal) on redistributing manual
> content.

## How to get your model name from SaabWisOnline

The `--model` and `--year` arguments come straight out of the site's URLs.
Browse to your car's manual on [saabwisonline.com](https://saabwisonline.com)
and read the two path segments after the domain:

```
https://saabwisonline.com/9-3-9440/2007/
                         └─model─┘ └year┘
```

So a 2007 9-3 (chassis 9440) is `--model 9-3-9440 --year 2007`. The model slug
combines the model name and its chassis/platform code, since the same name
covers different platforms across generations.

## Crawling etiquette

SaabWisOnline is a free community resource run on someone else's budget. This
crawler is built to be a good citizen, and you should keep it that way:

- **It respects `robots.txt`** — every URL is checked with `RobotFileParser`
  before being fetched, and disallowed paths are skipped.
- **It identifies itself** with a descriptive User-Agent that says what it is
  and how to characterize the traffic.
- **It rate-limits** to one request per second by default. Please don't lower
  `--delay` to hammer the site; a full crawl is a few hours of background work.
- **It stays in scope** — links are followed only within the requested
  `model/year` subtree, and external links are ignored.
- **It backs off on errors** with exponential retry rather than retrying
  immediately.
- **Crawl once, index many times.** The scraped JSON is saved to disk so you can
  re-chunk and re-embed as much as you like without touching the site again.

If you're testing changes to the crawler, use `--max-docs 20` and `--dfs` rather than
repeatedly pulling the full corpus.

## Future proofing

The crawler depends on the site's current HTML structure — leaf documents are
detected by a nested `<html>` block inside `#content`, a quirk of how the
original WIS export tool dumped each procedure.

Planned hardening:

- An additional engine filtering feature to only scrape engine specific content.
- A nightly job that checks crawling is still possible and that the site
  structure hasn't changed, so breakage surfaces before someone needs the tool.
- A dedicated fast-path tool for the `dtcs/` section — structured lookup by
  fault code (`P0300`, `B3001-05`) rather than semantic search over prose, which
  is a poor fit for short, near-identical code descriptions.
- Result re-ranking and a retrieval-quality evaluation harness.

## Contributing

- Both subprojects use `pytest` (tests under `tests/`) and `ruff` for linting
  and import sorting. Config lives in [pyproject.toml](pyproject.toml):
  120-column lines, rules `E`/`F`/`I`/`UP`/`B`/`SIM`.
- Install dev dependencies with `pip install -r <subproject>/requirements-dev.txt`.
- Run `ruff check .` and `pytest` before opening a PR.
- Scraped data and built indexes are gitignored deliberately — please don't
  commit corpus data or `server/index/` (see [Legal](#legal)).

## Legal

This project ships **no manual content** — only the code to fetch and index it.
The workshop documentation on SaabWisOnline is the intellectual property of its
respective rights holders, and running this crawler produces a local copy for
your own use.

- Use it for personal research, maintenance, and repair on cars you own or work
  on. Don't redistribute the scraped corpus or a built index publicly.
- If you host the MCP server for others, you are republishing that content —
  understand what you're doing before pointing a public URL at it, and consider
  keeping access limited to people who would otherwise use the site directly.
- The crawler respects `robots.txt`, but robots.txt is not a license. Be
  respectful of the source.
- This project is unaffiliated with Saab, Orio AB, General Motors, NEVS, or
  SaabWisOnline.

Nothing here is legal advice; you are responsible for how you use it.

## Donate

If this project is useful to you, the people who actually deserve your money are
the ones hosting the manuals:

**[Donate to SaabWisOnline](https://saabwisonline.com)** — they host the
documentation this entire project depends on, for free.