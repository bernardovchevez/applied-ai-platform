# Platform data directories

| Path | Purpose |
|------|---------|
| `lake/raw/` | **Published** document bronze (written by [applied-ai-docs](https://github.com/bernardovchevez/applied-ai-docs) publish) |
| `lake/parquet/` | Batch ingest manifest |
| `lake/events/` | `document_ingested` events for `embed_worker.py` |

**Seed content** (Harbor Line synthetic library) lives in **applied-ai-docs** → `seed/harbor-line/`. Import via the docs app, then publish to `lake/raw/`.
