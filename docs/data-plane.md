# Applied AI Data Plane Architecture

Reference architecture connecting enterprise document and warehouse sources to RAG, MCP servers, and multi-agent workflows.

```
SharePoint / Email          S3 (documents)           Warehouse
       │                          │                      │
       └──────────┬───────────────┴──────────────────────┘
                  ▼
           Ingest & normalize (batch + event)
                  ▼
           Chunk + embed + metadata enrich
                  ▼
     ┌────────────┴────────────┐
     ▼                         ▼
 Vector index              Feature tables
 (pgvector / Pinecone /     (Snowflake / Databricks /
  Databricks VS)             dbt marts)
     │                         │
     └────────────┬────────────┘
                  ▼
        API / MCP / Agent layer
                  ▼
     Apps (Next.js) + Enterprise systems
```

## Design principles

1. **Agents never read raw sources directly:** Agents query indexed vectors and structured warehouse marts only.
2. **Dual outputs:** Vector indexes handle semantic recall; feature tables provide authoritative structured rows.
3. **SQL before vectors for clearance:** `lookup_rights` queries structured SQL first ([finetune-vs-rag.md](finetune-vs-rag.md)).
4. **Consistent embedding contract:** Same embedding model enforced across batch ingest, event workers, and query time.
5. **Batch + event-driven freshness:** Nightly reconciliation paired with incremental event-driven updates.

## Pipeline commands

| Step | Command / path |
|------|----------------|
| Author & publish | [applied-ai-docs](https://github.com/bernardovchevez/applied-ai-docs) → lake raw + events |
| Land in lake (batch manifest) | `python platform/pipelines/ingest_batch.py` (optional `--seed-from-synthetic` for dev) |
| Parquet manifest | `data/lake/parquet/documents.parquet` + `ingest_manifest.json` |
| Emit ingest events | `ingest_batch.py --emit-events` -> `data/lake/events/document_ingested/` |
| Incremental embed worker | `python platform/pipelines/embed_worker.py` |
| Full-batch embed + index | `python rag/ingest_pgvector.py --data-dir data/lake/raw` |
| LoRA tier tags (optional) | `ingest_pgvector.py --lora-tag --data-dir data/lake/raw` |
| Query RAG | `python rag/query_pgvector.py` |
| Agent MCP server | `agents/mcp_server/server.py` |

## Target SLAs

| Stage | Batch (nightly) | Event-driven |
|-------|-----------------|--------------|
| Source -> lake | < 24 h | < 5 min |
| Lake -> vector index queryable | nightly window | < 15 min p95 |
| Feature mart fresh | after nightly job | < 15 min p95 |
| Access control (`confidentiality`) | enforced on ingest | `metadata_only` events |

## Related docs

- [mcp-enterprise.md](mcp-enterprise.md)
- [finetune-vs-rag.md](finetune-vs-rag.md)
- [responsible-ai-media.md](responsible-ai-media.md)
