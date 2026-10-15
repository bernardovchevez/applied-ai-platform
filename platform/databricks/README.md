# Databricks Vector Search: Delta Sync Pipeline

Export local lake embeddings to a **Unity Catalog gold Delta table** and sync with **Databricks Vector Search** (Delta Sync) for filtered ANN retrieval.

## Local export

Requires Ollama with `nomic-embed-text` (same embedding model as `pgvector` ingest):

```bash
cd applied-ai-platform
pip install -r requirements-pipelines.txt

# Ensure lake raw exists
python platform/pipelines/ingest_batch.py --skip-sync

# Export gold parquet + pre-embedded demo query vectors
python platform/pipelines/export_databricks_gold.py --write-query-vectors
```

Outputs (gitignored under `data/lake/databricks/`):

| File | Purpose |
|------|---------|
| `gold_document_embeddings.parquet` | ~81 chunk rows with 768-d embeddings + metadata |
| `query_vectors.json` | Pre-embedded demo questions (model parity with `pgvector`) |

## Upload to Databricks

1. Sign in to Databricks with **Unity Catalog** and **Serverless** enabled.
2. Upload both files to DBFS or a Unity Catalog Volume, e.g. `/FileStore/applied_ai/`:
   - `databricks fs cp gold_document_embeddings.parquet dbfs:/FileStore/applied_ai/`

## Import notebook

1. **Workspace** -> **Import** -> `platform/databricks/gold_vector_search.py`
2. Attach to a serverless notebook cluster and run all cells.

The notebook:
1. Creates `main.applied_ai.gold_document_embeddings` Delta table (Change Data Feed enabled)
2. Creates Vector Search endpoint `applied-ai-search`
3. Creates a **Delta Sync** index on the gold table
4. Runs filtered ANN similarity queries (`doc_type = 'rights_clearance'`)

## Latency benchmark comparison

```bash
python platform/pipelines/benchmark_pgvector.py \
  "Is the theme song cleared for EMEA streaming?" --doc-type rights_clearance
```

## Related docs

- [data-plane.md](../../docs/data-plane.md)
