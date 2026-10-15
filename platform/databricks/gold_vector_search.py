# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Delta + Databricks Vector Search (Delta Sync)
# MAGIC
# MAGIC **Pipeline:** Embed job -> **Unity Catalog gold Delta** -> **Vector Search Delta Sync** -> filtered ANN query.
# MAGIC
# MAGIC **Prerequisites**
# MAGIC 1. Databricks **Free Edition** or trial workspace with **Unity Catalog** + **Serverless**
# MAGIC 2. Local export (same Ollama vectors as pgvector):
# MAGIC    ```bash
# MAGIC    python platform/pipelines/export_databricks_gold.py --write-query-vectors
# MAGIC    ```
# MAGIC 3. Upload to DBFS / Volume:
# MAGIC    - `gold_document_embeddings.parquet`
# MAGIC    - `query_vectors.json`
# MAGIC
# MAGIC **Compare latency locally:**
# MAGIC ```bash
# MAGIC python platform/pipelines/benchmark_pgvector.py \
# MAGIC   "Is the theme song cleared for EMEA streaming?" --doc-type rights_clearance
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

# Upload paths — adjust after copying from data/lake/databricks/
PARQUET_PATH = "/FileStore/applied_ai/gold_document_embeddings.parquet"
QUERY_VECTORS_PATH = "/FileStore/applied_ai/query_vectors.json"

# Unity Catalog target (Free Edition often uses catalog `main`)
CATALOG = "main"
SCHEMA = "applied_ai"
TABLE = "gold_document_embeddings"
FULL_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE}"

# AI Search (formerly Vector Search)
ENDPOINT_NAME = "applied-ai-search"
INDEX_NAME = f"{FULL_TABLE}_index"
EMBEDDING_DIM = 768  # nomic-embed-text (default)

# Demo query (pre-embedded locally for model parity)
DEMO_QUESTION = "Is the theme song cleared for EMEA streaming?"
DOC_TYPE_FILTER = "rights_clearance"
TOP_K = 4

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load exported gold parquet → Delta (enable CDF for Delta Sync)

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

gold_df = spark.read.parquet(PARQUET_PATH)
gold_df = gold_df.withColumn("updated_at", gold_df.updated_at.cast("timestamp"))

(
    gold_df.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(FULL_TABLE)
)

spark.sql(
    f"""
    ALTER TABLE {FULL_TABLE}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """
)

display(spark.table(FULL_TABLE).groupBy("doc_type").count().orderBy("count", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create AI Search endpoint + Delta Sync index
# MAGIC
# MAGIC Free Edition: **one** AI Search endpoint, Delta Sync only (no Direct Vector Access).
# MAGIC Requires Unity Catalog + Serverless. First sync may take a few minutes.

# COMMAND ----------

# MAGIC %pip install -q databricks-vectorsearch

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient(disable_notice=True)

# Create endpoint if missing (skip if applied-ai-search already exists)
try:
    vsc.create_endpoint(name=ENDPOINT_NAME, endpoint_type="STANDARD")
    print(f"Created endpoint {ENDPOINT_NAME}")
except Exception as exc:
    if "already exists" in str(exc).lower() or "RESOURCE_ALREADY_EXISTS" in str(exc):
        print(f"Endpoint {ENDPOINT_NAME} already exists")
    else:
        raise

# Delta Sync index — source of truth is the gold Delta table
try:
    vsc.create_delta_sync_index(
        endpoint_name=ENDPOINT_NAME,
        index_name=INDEX_NAME,
        source_table_name=FULL_TABLE,
        primary_key="chunk_id",
        embedding_vector_column="embedding",
        embedding_dimension=EMBEDDING_DIM,
        pipeline_type="TRIGGERED",
    )
    print(f"Created index {INDEX_NAME}")
except Exception as exc:
    if "already exists" in str(exc).lower() or "RESOURCE_ALREADY_EXISTS" in str(exc):
        print(f"Index {INDEX_NAME} already exists — triggering sync")
        index = vsc.get_index(endpoint_name=ENDPOINT_NAME, index_name=INDEX_NAME)
        index.sync()
    else:
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Wait for index sync, then query with metadata filter
# MAGIC
# MAGIC Filter: `doc_type = 'rights_clearance'` (same as local `query_pgvector.py --doc-type rights_clearance`).

# COMMAND ----------

import json
import time

index = vsc.get_index(endpoint_name=ENDPOINT_NAME, index_name=INDEX_NAME)

# Poll until ONLINE (adjust timeout for first run)
for attempt in range(30):
    desc = index.describe()
    status = desc.get("status", {}).get("indexed_row_count")
    state = desc.get("status", {}).get("message", desc.get("status"))
    print(f"Sync check {attempt + 1}: {state}")
    if desc.get("status", {}).get("ready") or (status and status > 0):
        break
    time.sleep(10)
else:
    print("Index may still be syncing — try re-running this cell.")

query_payload = json.loads(
    dbutils.fs.head(QUERY_VECTORS_PATH)  # noqa: F821
)
query_vector = query_payload["queries"][DEMO_QUESTION]
print(f"Query model: {query_payload.get('embed_model_version')}")

start = time.perf_counter()
results = index.similarity_search(
    query_vector=query_vector,
    columns=["chunk_text", "file_name", "doc_type", "rights_tier", "series"],
    filters={"doc_type": DOC_TYPE_FILTER},
    num_results=TOP_K,
)
elapsed_ms = (time.perf_counter() - start) * 1000

print(f"\nQuestion: {DEMO_QUESTION}")
print(f"Filter: doc_type = {DOC_TYPE_FILTER}")
print(f"AI Search latency: {elapsed_ms:.1f} ms")
print(f"Hits: {len(results.get('result', {}).get('data_array', []))}")

display(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Compare with local pgvector
# MAGIC
# MAGIC Run on your laptop (same question + filter):
# MAGIC ```bash
# MAGIC python platform/pipelines/benchmark_pgvector.py \
# MAGIC   "Is the theme song cleared for EMEA streaming?" --doc-type rights_clearance
# MAGIC ```
# MAGIC
# MAGIC | Backend | Vectors in | Incremental sync | Governance |
# MAGIC |---------|------------|------------------|------------|
# MAGIC | **pgvector** | Postgres | `embed_worker.py` per doc | App-managed |
# MAGIC | **Databricks Vector Search** | Delta gold table | Delta Sync | Unity Catalog |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Troubleshooting
# MAGIC
# MAGIC | Issue | Fix |
# MAGIC |-------|-----|
# MAGIC | No AI Search UI | Enable **Serverless** + **Unity Catalog** on workspace |
# MAGIC | CDF error on index create | Re-run cell 2 (`delta.enableChangeDataFeed = true`) |
# MAGIC | Free Edition endpoint limit | Reuse `applied-ai-search`; delete old endpoint if needed |
# MAGIC | Query dimension mismatch | Re-export with `export_databricks_gold.py` (768d nomic) |
# MAGIC | Empty filter results | Confirm `doc_type` values: `display(spark.table(FULL_TABLE).select('doc_type').distinct())` |
