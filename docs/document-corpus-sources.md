# Document corpus sources of truth

## Ownership

| Layer | Repository | Role |
|-------|------------|------|
| **Authoring & catalog** | [applied-ai-docs](https://github.com/bernardovchevez/applied-ai-docs) | Libraries, metadata, markdown bodies, publish workflow |
| **Bronze landing** | `applied-ai-platform/data/lake/raw/` | Files + `sharepoint-meta.json` written on **Publish** from docs |
| **Vector index & AI** | `applied-ai-platform` | pgvector, BFF RAG/agents, evals, MCP tools |

Slack and web clients talk only to the **platform BFF**. They do not read the docs catalog directly.

## Localhost workflow

1. Run **applied-ai-docs** (`./run.sh` on port 8010).
2. Create or edit documents in the UI (or seed once from synthetic).
3. **Publish** (per item or all pending) → lake raw + `document_ingested` events.
4. Run **`python platform/pipelines/embed_worker.py`** (or enable `RUN_EMBED_WORKER` in docs).
5. Query via BFF, Slack, or web.

## `data/synthetic/`

Seed pack for demos and first-time bootstrap. **Not** the live edit path. Use docs **Seed from synthetic** or:

```bash
python platform/pipelines/ingest_batch.py --seed-from-synthetic --emit-events
python platform/pipelines/embed_worker.py
```

## Defaults

- `shared.config.DATA_DIR` → `data/lake/raw` (published corpus).
- `ingest_batch.py` does **not** copy synthetic unless `--seed-from-synthetic` is passed.

See also [data-plane.md](data-plane.md).
