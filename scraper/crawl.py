#!/usr/bin/env python3
"""Crawler for saabwisonline.com workshop documentation.

Walks the folder hierarchy under https://saabwisonline.com/{model}/{year}/,
follows only links that stay within that subtree, and saves each leaf
document (an actual procedure/article, as opposed to a folder listing) as
a JSON file mirroring the site's URL path under --output-dir. A manifest.json
listing every saved document is written alongside them for the later RAG
ingestion step.

Usage:
    python crawl.py                              # defaults: 9-3-9440 / 2007
    python crawl.py --model 9-5-9600 --year 2010
    python crawl.py --max-pages 20 --delay 1.5    # quick/polite test run
    python crawl.py --dtcs                        # also crawl the (large) DTCs section
    python crawl.py --dfs                          # crawl depth-first instead of breadth-first
"""

import argparse
import json
import sys
import time
import urllib.robotparser as robotparser
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://saabwisonline.com"
USER_AGENT = "SaabWISOnlineRAGCrawler/1.0 (personal research/RAG project; low-volume, respects robots.txt)"


def load_robots(session: requests.Session) -> robotparser.RobotFileParser:
    """Fetch and parse robots.txt so callers can check can_fetch() before requesting a URL."""
    rp = robotparser.RobotFileParser()
    resp = session.get(f"{BASE_URL}/robots.txt", timeout=15)
    rp.parse(resp.text.splitlines())
    return rp


def fetch(session: requests.Session, url: str, retries: int = 3) -> str | None:
    """GET url, retrying with backoff on errors/non-2xx. Returns HTML, or None on 404 or after exhausting retries."""
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 404:
                print(f"  [404] {url}")
                return None
            print(f"  [{resp.status_code}] {url} (attempt {attempt}/{retries})")
        except requests.RequestException as exc:
            print(f"  [error] {url}: {exc} (attempt {attempt}/{retries})")
        time.sleep(2**attempt)
    return None


def fetch_content_div(session: requests.Session, url: str):
    """Fetch url and return its parsed #content div, or None if the page can't be used.

    Args:
        session: requests session to fetch with.
        url: absolute URL to fetch.

    Returns:
        The page's #content div (a bs4 Tag) or None if the fetch failed or the
        page has no #content div.
    """
    html = fetch(session, url)
    if html is None:
        return None

    soup = BeautifulSoup(html, "html.parser")
    return soup.find("div", id="content")


def is_leaf(content_div) -> bool:
    # Leaf documents embed a full (invalid, nested) <html> block inside
    # #content -- a leftover from how the original WIS export tool dumped
    # each procedure. Folder-listing pages never have this, only <a> links.
    return content_div.find("html") is not None


def clean_text(nested_html) -> str:
    raw = nested_html.get_text(separator="\n")
    lines = [line.strip() for line in raw.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def extract_links(content_div, current_url: str, scope_prefix: str) -> list[str]:
    breadcrumbs = content_div.find("div", id="breadcrumbs")
    links = []
    for a in content_div.find_all("a", href=True):
        if breadcrumbs and a in breadcrumbs.find_all("a"):
            continue
        href = a["href"]
        absolute = urljoin(current_url, href)
        parsed = urlparse(absolute)
        if parsed.netloc and parsed.netloc != urlparse(BASE_URL).netloc:
            continue  # external link (SCNA, SaabNet, PayPal, ...)
        path = parsed.path
        if not path.endswith("/"):
            path += "/"
        if not path.startswith(scope_prefix):
            continue  # stay inside the requested model/year subtree
        links.append(f"{BASE_URL}{path}")
    return links


def should_explore(link: str, dtcs_prefix: str, include_dtcs: bool) -> bool:
    """Whether a discovered link should be queued, based on the script's crawl filters."""
    return include_dtcs or not link.startswith(dtcs_prefix)


def save_leaf(content_div, url: str, url_path: str, output_dir: Path) -> dict:
    """Write a leaf document's title/breadcrumb/text/html as JSON under output_dir, mirroring its URL path."""
    title_tag = content_div.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else url_path

    breadcrumbs_div = content_div.find("div", id="breadcrumbs")
    if breadcrumbs_div:
        crumbs = [s.strip() for s in breadcrumbs_div.stripped_strings if s.strip() != "/"]
        breadcrumb = " / ".join(crumbs)
    else:
        breadcrumb = ""

    nested_html = content_div.find("html")
    text = clean_text(nested_html)

    record = {
        "url": url,
        "title": title,
        "breadcrumb": breadcrumb,
        "text": text,
        "html": str(nested_html),
    }

    rel_path = url_path.strip("/") + ".json"
    out_file = output_dir / rel_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"url": url, "title": title, "breadcrumb": breadcrumb, "file": str(out_file.relative_to(output_dir))}


def crawl(
    model: str,
    year: str,
    output_dir: Path,
    delay: float,
    max_pages: int | None,
    include_dtcs: bool = False,
    dfs: bool = False,
) -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    rp = load_robots(session)

    scope_prefix = f"/{model}/{year}/"
    start_url = f"{BASE_URL}{scope_prefix}"
    dtcs_prefix = f"{BASE_URL}{scope_prefix}dtcs/"

    queue: deque[str] = deque([start_url])
    visited: set[str] = set()
    manifest: list[dict] = []

    # try/finally ensures the manifest is written even if the crawl is interrupted
    try:
        while queue:
            if max_pages is not None and len(visited) >= max_pages:
                print(f"Reached --max-pages limit ({max_pages}), stopping.")
                break

            # deque as FIFO (popleft) gives BFS; as LIFO (pop) gives DFS
            url = queue.pop() if dfs else queue.popleft()

            # Add current URL to visited set to avoid reprocessing it
            if url in visited:
                continue
            visited.add(url)

            # Check if URL is allowed by robots.txt before fetching
            if not rp.can_fetch(USER_AGENT, url):
                print(f"  [robots.txt disallow] {url}")
                continue

            # Fetch and parse enqueued URL 
            print(f"[{len(visited)}] {url}")
            content_div = fetch_content_div(session, url)
            # Sleep between requests
            time.sleep(delay)
            if content_div is None:
                continue
            path = urlparse(url).path

            if is_leaf(content_div):
                # Save leaf document as JSON and add to manifest
                entry = save_leaf(content_div, url, path, output_dir)
                manifest.append(entry)
            else:
                # Enqueue links found on non-leaf pages for further crawling
                links = extract_links(content_div, url, scope_prefix)
                for link in links:
                    if should_explore(link, dtcs_prefix, include_dtcs) and link not in visited:
                        queue.append(link)
    finally:
        manifest_file = output_dir / model / year / "manifest.json"
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"\nSaved {len(manifest)} documents ({len(visited)} pages visited).")
        print(f"Manifest: {manifest_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="9-3-9440", help="model/chassis slug, e.g. 9-3-9440")
    parser.add_argument("--year", default="2007", help="model year, e.g. 2007")
    parser.add_argument("--output-dir", default="data", type=Path, help="where to write scraped JSON documents")
    parser.add_argument("--delay", default=1.0, type=float, help="seconds to wait between requests")
    parser.add_argument("--max-pages", default=None, type=int, help="stop after visiting this many pages (for testing)")
    parser.add_argument("--dtcs", action="store_true", help="also crawl the (large) Diagnostic Trouble Codes section")
    parser.add_argument("--dfs", action="store_true", help="crawl depth-first instead of the default breadth-first")
    args = parser.parse_args()

    try:
        crawl(args.model, args.year, args.output_dir, args.delay, args.max_pages, args.dtcs, args.dfs)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
