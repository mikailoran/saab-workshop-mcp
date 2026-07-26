#!/usr/bin/env python3
"""Retrieval core for the Saab manual vector store built by ingest.py.

Embeds a query with the same local model used at ingest time and searches
the persistent Chroma collection, returning ranked passages with metadata
for citation.

Usage:
    python retrieval.py "how do I replace the oil sump" --k 5
"""

import argparse
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# BGE models' recommended instruction prefix for the query side of asymmetric
# (short query -> long passage) retrieval; not applied to indexed passages.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class SearchResult:
    """One retrieved passage, ranked by relevance."""

    text: str
    title: str
    breadcrumb: str
    url: str
    distance: float


@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    """Load the embedding model once and reuse it across calls."""
    return SentenceTransformer(EMBEDDING_MODEL)


@cache
def _get_collection(index_dir: Path, model: str, year: str) -> Collection:
    """Open (and cache) a persistent Chroma client/collection for a given index dir + model/year."""
    client = chromadb.PersistentClient(path=str(index_dir))
    return client.get_collection(f"{model}-{year}")


def search(
    query: str,
    k: int = 5,
    index_dir: Path = Path("index"),
    model: str = "9-3-9440",
    year: str = "2007",
) -> list[SearchResult]:
    """Embed query and return the k most relevant passages from the manual.

    Args:
        query: natural-language question, e.g. "how do I replace the oil sump".
        k: number of passages to return.
        index_dir: directory the Chroma store was persisted to by ingest.py.
        model: model/chassis slug the store was built for, e.g. 9-3-9440.
        year: model year the store was built for, e.g. 2007.

    Returns:
        Up to k results, ranked by relevance (best first, lower distance = more relevant).
    """
    embedder = _get_embedder()
    collection = _get_collection(index_dir, model, year)

    query_embedding = embedder.encode(QUERY_INSTRUCTION + query, normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    return [
        SearchResult(
            text=document,
            title=metadata["title"],
            breadcrumb=metadata["breadcrumb"],
            url=metadata["url"],
            distance=distance,
        )
        for document, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0], strict=True
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", help="natural-language question")
    parser.add_argument("--k", default=5, type=int, help="number of results to return")
    parser.add_argument("--index-dir", default=Path("index"), type=Path, help="Chroma store directory")
    parser.add_argument("--model", default="9-3-9440", help="model/chassis slug, e.g. 9-3-9440")
    parser.add_argument("--year", default="2007", help="model year, e.g. 2007")
    args = parser.parse_args()

    results = search(args.query, args.k, args.index_dir, args.model, args.year)
    for i, result in enumerate(results, start=1):
        print(f"[{i}] {result.title}  (distance={result.distance:.4f})")
        print(f"    {result.breadcrumb}")
        print(f"    {result.url}")
        preview = result.text[:200].replace("\n", " ")
        print(f"    {preview}...")
        print()


if __name__ == "__main__":
    main()
