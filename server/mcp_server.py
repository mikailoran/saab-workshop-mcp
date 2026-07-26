#!/usr/bin/env python3
"""MCP server exposing Saab manual retrieval as a tool for MCP clients.

Wraps retrieval.search() as a FastMCP tool, served over stdio -- the
standard local transport for Claude Desktop / Claude Code. The corpus is
single-scope (one model/year: 9-3-9440/2007), so the tool only takes
query/k; model/year use retrieval.search()'s own matching defaults.

Usage:
    python mcp_server.py
    mcp dev mcp_server.py   # local dev/inspection, via the mcp[cli] extra
"""

from pathlib import Path

from mcp.server.fastmcp import FastMCP
from retrieval import SearchResult, search

# Resolved relative to this file, not the launching process's cwd -- an MCP
# client may start this server from an arbitrary working directory, so
# retrieval.search()'s cwd-relative Path("index") default can't be used here.
INDEX_DIR = Path(__file__).resolve().parent / "index"

mcp = FastMCP("saab-manual")


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
    mcp.run()
