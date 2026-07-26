#!/usr/bin/env python3
"""Ingests scraped Saab manual JSON documents into a local Chroma vector store.

Reads scraper/data/{model}/{year}/manifest.json, loads each document's full
JSON record, cleans and chunks its text, embeds each chunk locally, and
upserts into a persistent Chroma collection under --index-dir.

Usage:
    python ingest.py --data-dir ../scraper/data --model 9-3-9440 --year 2007
"""

import argparse
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import chromadb
import ftfy
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MAX_CHUNK_CHARS = 1800  # ~512 tokens at ~4 chars/token, with headroom
CHUNK_OVERLAP_LINES = 2  # small overlap so context isn't lost at chunk boundaries
UPSERT_BATCH_SIZE = 500  # stay under Chroma's max add-batch size


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap_lines: int = CHUNK_OVERLAP_LINES) -> list[str]:
    """Greedily pack text's lines into chunks of at most max_chars each.

    Docs that fit under max_chars come back as a single chunk. Longer docs
    split on line boundaries -- since clean_text() in crawl.py already puts
    one semantic unit (a numbered step, a table row, a paragraph) per line,
    this never cuts one mid-way, whether the doc is a numbered procedure,
    prose, or a flat reference table.

    Args:
        text: the document's cleaned text.
        max_chars: soft cap on each chunk's length.
        overlap_lines: number of trailing lines carried into the next chunk,
            so context isn't lost right at a chunk boundary.

    Returns:
        A list of chunks (always at least one, even for empty text).
    """
    lines = text.splitlines()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        # Flush the current chunk and start a new one, optionally carrying over the last few lines for context.
        if current and current_len + len(line) + 1 > max_chars:
            chunks.append("\n".join(current))
            current = current[-overlap_lines:] if overlap_lines else []
            current_len = sum(len(existing) + 1 for existing in current)
        # Append the line to the (possibly just-reset) current chunk.
        current.append(line)
        current_len += len(line) + 1

    # Flush whatever's left as the final chunk.
    if current:
        chunks.append("\n".join(current))
    return chunks or [""]


# TODO: refactor dataclasses duplicated in server/ingest.py
@dataclass(frozen=True)
class LeafRecord:
    """A leaf document's scraped content, as saved to its JSON file."""

    url: str
    title: str
    breadcrumb: str
    text: str
    html: str


@dataclass(frozen=True)
class ChunkMetadata:
    """Citation metadata for one chunk."""

    url: str
    title: str
    breadcrumb: str
    file: str
    chunk_index: int


@dataclass(frozen=True)
class Chunk:
    """One chunk of a leaf document's text, ready to embed and index."""

    text: str
    metadata: ChunkMetadata

    @property
    def id(self) -> str:
        """Deterministic chunk id, unique within a document and stable across re-ingests."""
        return f"{self.metadata.file}::{self.metadata.chunk_index}"


def iter_chunks(data_dir: Path, manifest_paths: list[str]) -> Iterator[Chunk]:
    """Yield a Chunk for every chunk across every manifest entry.

    Args:
        data_dir: data root of scraped content.
        manifest_paths: list of leaf document file paths found in manifest.json.

    Yields:
        Chunks with a deterministic id, the chunk's text, and metadata for later citation.
    """
    for rel_leaf_path in manifest_paths:
        abs_leaf_path = data_dir / rel_leaf_path
        record = LeafRecord(**json.loads(abs_leaf_path.read_text(encoding="utf-8")))
        # Fix mojibake and other unicode issues from record
        record_title = ftfy.fix_text(record.title)
        record_breadcrumb = ftfy.fix_text(record.breadcrumb)
        record_text = ftfy.fix_text(record.text)
        for i, piece in enumerate(chunk_text(record_text)):
            metadata = ChunkMetadata(
                url=record.url,
                title=record_title,
                breadcrumb=record_breadcrumb,
                file=rel_leaf_path,
                chunk_index=i,
            )
            yield Chunk(text=piece, metadata=metadata)


def ingest(data_dir: Path, model: str, year: str, index_dir: Path) -> None:
    """Embed every chunk of every document in a manifest and upsert them into a persistent Chroma collection.

    Args:
        data_dir: data root of scraped content.
        model: model/chassis slug, e.g. 9-3-9440.
        year: model year, e.g. 2007.
        index_dir: directory to persist the Chroma store to.
    """
    # Manifest lists every scraped leaf document's file path, relative to data_dir
    manifest_path = data_dir / model / year / "manifest.json"
    manifest_entries: list[str] = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"Loaded manifest @ {manifest_path}.")

    # Chunk every document up front so embedding can run as one batch below
    chunks = list(iter_chunks(data_dir, manifest_entries))
    print(f"{len(manifest_entries)} documents chunked into {len(chunks)} chunks")

    print(f"Loading embedding model {EMBEDDING_MODEL}...")
    try:
        embedder = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load embedding model {EMBEDDING_MODEL!r}. Check your internet connection -- "
            "the model weights are downloaded from the Hugging Face Hub on first use."
        ) from exc

    # Batch-encode all chunks at once (rather than one at a time) for throughput
    print("Embedding chunks...")
    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = embedder.encode(chunk_texts, show_progress_bar=True, batch_size=64, normalize_embeddings=True).tolist()

    # Create or update the persistent collection for this model/year
    client = chromadb.PersistentClient(path=str(index_dir))
    collection_name = f"{model}-{year}"
    collection = client.get_or_create_collection(collection_name)

    # Upsert in batches to stay under Chroma's max add-batch size; upsert (not
    # add) so re-running this function overwrites rather than duplicates
    for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
        end = start + UPSERT_BATCH_SIZE
        batch = chunks[start:end]
        collection.upsert(
            ids=[chunk.id for chunk in batch],
            embeddings=embeddings[start:end],
            documents=[c.text for c in batch],
            metadatas=[asdict(c.metadata) for c in batch],
        )

    print(f"Indexed {len(chunks)} chunks into {index_dir} (collection '{collection_name}')")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir",
        default=Path("../scraper/data_last_run"),
        type=Path,
        help="scraper data root (defaults to the `data_last_run` symlink)",
    )
    parser.add_argument("--model", default="9-3-9440", help="model/chassis slug, e.g. 9-3-9440")
    parser.add_argument("--year", default="2007", help="model year, e.g. 2007")
    parser.add_argument("--index-dir", default=Path("index"), type=Path, help="where to persist the Chroma store")
    args = parser.parse_args()

    ingest(args.data_dir, args.model, args.year, args.index_dir)


if __name__ == "__main__":
    main()
