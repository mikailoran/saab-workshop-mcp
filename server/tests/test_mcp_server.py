"""Tests for mcp_server.py's own wiring, not retrieval.search()'s results."""

from pathlib import Path

import mcp_server


def test_index_dir_resolves_next_to_module_regardless_of_cwd(tmp_path, monkeypatch):
    """INDEX_DIR must not depend on the launching process's cwd."""
    monkeypatch.chdir(tmp_path)
    assert Path(mcp_server.__file__).resolve().parent / "index" == mcp_server.INDEX_DIR
