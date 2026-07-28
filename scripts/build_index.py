#!/usr/bin/env python3
"""Build FAISS index from data/articles.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings
from src.rag.index import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS RAG index")
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--index-dir", type=str, default=str(settings.index_dir))
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    store = build_index(
        index_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    n_vecs = store.index.ntotal
    print(f"Index built: {index_dir}")
    print(f"Vectors: {n_vecs}")
    print(f"Embedding model: {settings.embedding_model}")


if __name__ == "__main__":
    main()
