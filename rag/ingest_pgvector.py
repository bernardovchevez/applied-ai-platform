#!/usr/bin/env python3
"""Ingest synthetic corpus into Postgres + pgvector."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import DATA_DIR
from stores.pgvector import PgVectorStore


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Ingest corpus into pgvector")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Keep existing tables; upsert documents/chunks",
    )
    parser.add_argument(
        "--lora-tag",
        action="store_true",
        help="Tag each chunk with LoRA rights-tier classifier at ingest",
    )
    parser.add_argument(
        "--lora-adapter",
        type=Path,
        default=None,
        help="Path to LoRA adapter dir (default: finetune checkpoint)",
    )
    args = parser.parse_args()

    store = PgVectorStore()
    stats = store.ingest(
        data_dir=args.data_dir,
        reset=not args.no_reset,
        lora_tag=args.lora_tag,
        lora_adapter_dir=args.lora_adapter,
    )
    print(
        f"Ingested {stats.chunks} chunks from {stats.documents} documents → pgvector "
        f"({store.database_url.split('@')[-1]})"
    )
    if stats.lora_tagged_chunks:
        print(f"LoRA tier tags written on {stats.lora_tagged_chunks} chunks")


if __name__ == "__main__":
    main()
