# AI Platform, Data Plane & Operations (`platform/`)

## Overview

Production platform layer combining batch and event-driven data lake ingestion, a PEFT/LoRA rights-tier classifier, a FastAPI OpenAPI presentation BFF, and `Langfuse` + `MLflow` observability and evaluation pipelines.

## Structure

```
platform/
├── finetune/           # PEFT / LoRA rights-tier sequence classifier (flan-t5-base)
├── pipelines/          # Batch parquet + event-driven incremental ingest workers
├── databricks/         # Databricks Unity Catalog / Delta Sync Vector Search export
├── bff/                # FastAPI presentation BFF (openapi.yaml)
└── ops/                # Langfuse tracing, MLflow evaluation logging, rollback runbook
```

## PEFT / LoRA Classifier (`finetune/`)

See [docs/finetune-vs-rag.md](../docs/finetune-vs-rag.md) for architecture tradeoffs between SQL lookup, RAG, and small classifiers at ingest.

```bash
pip install -r requirements-finetune.txt
python platform/finetune/generate_dataset.py
python platform/finetune/train_classifier.py
python platform/finetune/eval_classifier.py
python platform/finetune/compare_baselines.py
```

## Data Plane & Incremental Ingest (`pipelines/`)

See [docs/data-plane.md](../docs/data-plane.md).

```bash
pip install -r requirements-pipelines.txt
python platform/pipelines/ingest_batch.py
python platform/pipelines/ingest_batch.py --skip-sync --emit-events
python platform/pipelines/embed_worker.py               # consume events -> pgvector
python platform/pipelines/export_databricks_gold.py --write-query-vectors
python rag/ingest_pgvector.py --data-dir data/lake/raw --lora-tag
```

## FastAPI Presentation BFF (`bff/`)

OpenAPI BFF (`bff/openapi.yaml`) serving web, mobile, CTV, and internal agent clients. Consumed by the companion Next.js application in **`applied-ai-web/`**.

- [bff/README.md](bff/README.md): API contract, SSE streaming, human approval gate, multi-client headers (`web` / `ios` / `ctv`)

```bash
# Terminal 1 - BFF
pip install -r platform/bff/requirements-bff.txt
cd platform/bff && uvicorn main:app --reload --port 8000

# Terminal 2 - Next.js client (applied-ai-web)
cd ../applied-ai-web && cp .env.local.example .env.local && npm install && npm run dev
```

## Observability, Evaluations & Governance (`ops/`)

- [ops/README.md](ops/README.md): `Langfuse` request/tool tracing and `MLflow` evaluation tracking
- [ops/rollback-runbook.md](ops/rollback-runbook.md): Prompt/model revert procedure and smoke eval gates
- [docs/responsible-ai-media.md](../docs/responsible-ai-media.md): Responsible AI implementation map

```bash
# Run 6-backend RAG keyword baseline and log to MLflow
./platform/ops/run_rag_keyword_baseline.sh

# Run 7-scenario agent evaluation and log to MLflow
python agents/evals/run_agent_eval.py --mlflow

# Inspect MLflow runs
./platform/ops/mlflow_ui.sh
python platform/ops/mlflow_list_runs.py
```
