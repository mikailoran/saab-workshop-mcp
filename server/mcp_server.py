#!/usr/bin/env python3
"""MCP server exposing Saab manual retrieval as a tool for MCP clients.

Wraps retrieval.search() as a FastMCP tool. Transport defaults to stdio --
the standard local transport for Claude Desktop / Claude Code -- but can be
switched to streamable-http (for a hosted deployment) via the MCP_TRANSPORT
env var. The corpus is single-scope (one model/year: 9-3-9440/2007), so the
tool only takes query/k; model/year use retrieval.search()'s own matching
defaults.

Usage:
    python mcp_server.py                            # stdio (default)
    MCP_TRANSPORT=streamable-http python mcp_server.py   # HTTP, for deployment
    mcp dev mcp_server.py   # local dev/inspection, via the mcp[cli] extra

Env vars (only used for streamable-http):
    MCP_TRANSPORT: "stdio" (default) or "streamable-http".
    MCP_HOST: bind host, default "0.0.0.0".
    MCP_PORT: bind port, default 8000.
"""

import os
from pathlib import Path
from typing import Literal, get_args

from mcp.server.fastmcp import FastMCP
from retrieval import SearchResult, search

# Resolved relative to this file, not the launching process's cwd -- an MCP
# client may start this server from an arbitrary working directory, so
# retrieval.search()'s cwd-relative Path("index") default can't be used here.
INDEX_DIR = Path(__file__).resolve().parent / "index"

Transport = Literal["stdio", "streamable-http"]
_VALID_TRANSPORTS = get_args(Transport)


def _get_transport() -> Transport:
    """Read and validate the MCP_TRANSPORT env var, defaulting to stdio."""
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport not in _VALID_TRANSPORTS:
        raise ValueError(f"MCP_TRANSPORT must be one of {_VALID_TRANSPORTS}, got {transport!r}")
    return transport  # type: ignore[return-value]


mcp = FastMCP(
    "saab-manual",
    host=os.environ.get("MCP_HOST", "0.0.0.0"),
    port=int(os.environ.get("MCP_PORT", "8000")),
)


@mcp.tool()
def search_saab_manual(query: str, k: int = 5) -> list[SearchResult]:
    """Search the Saab 9-3 (9440) 2007 workshop manual for relevant passages.

    Args:
        query: natural-language question, e.g. "how do I replace the oil sump".
        k: number of passages to return.

    Returns:
        Up to k passages, ranked by relevance (best first), each carrying
        its source title/breadcrumb/url so the caller can cite it.
    """
    return search(query, k, index_dir=INDEX_DIR)


if __name__ == "__main__":
    mcp.run(transport=_get_transport())
