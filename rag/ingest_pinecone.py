#!/usr/bin/env python3
"""Ingest synthetic corpus into Pinecone."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import DATA_DIR, get_env
from stores.pinecone import PineconeStore


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ingest corpus into Pinecone")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Skip namespace wipe before upsert",
    )
    args = parser.parse_args()

    store = PineconeStore()
    stats = store.ingest(data_dir=args.data_dir, reset=not args.no_reset)
    index = get_env("PINECONE_INDEX", store.index_name)
    print(
        f"Ingested {stats.chunks} chunks from {stats.documents} documents → "
        f"Pinecone index {index!r} (namespace={store.namespace})"
    )


if __name__ == "__main__":
    main()
