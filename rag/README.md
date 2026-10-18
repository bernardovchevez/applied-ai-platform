# Enterprise Doc Q&A & Evaluation Pipeline (`rag/`)

## Overview

Answers questions over synthetic media operations and rights clearance documents with grounded citations, metadata filtering, and automated evaluation harnesses across 6 retrieval backends.

## Commands

```bash
# Default corpus: data/lake/raw/ (publish from applied-ai-docs)
python ingest.py
python ingest_pgvector.py

# Ingest from data lake (after ingest_batch.py)
python ingest_pgvector.py --data-dir ../data/lake/raw
python ingest_pgvector.py --data-dir ../data/lake/raw --lora-tag   # LoRA chunk tags

# CLI query (Chroma)
python query.py "Is the theme song cleared for EMEA streaming?"

# LangChain RAG
python query_langchain.py --reindex "Is the theme song cleared for EMEA streaming?"

# LlamaIndex RAG
python query_llamaindex.py --reindex "Is the theme song cleared for EMEA streaming?"

# Multi-backend keyword eval (6 backends)
python evals/run_eval.py
python evals/run_eval.py --backend hybrid

# RAGAS semantic eval
python evals/run_ragas.py --backend chroma --limit 3
python evals/run_ragas.py --backend all --write-baseline

# promptfoo regression (compares 3 backends x 23 gold questions)
cd evals
PROMPTFOO_PYTHON=../../.venv/bin/python npx promptfoo@latest eval -c promptfooconfig.yaml
PROMPTFOO_PYTHON=../../.venv/bin/python npx promptfoo@latest view
```

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for pipeline comparison (Chroma vs LangChain vs LlamaIndex).

## Pipeline

```
applied-ai-docs publish -> data/lake/raw/ -> chunk -> embed -> Vector Store -> retrieve -> LLM
```

## Key files

| File | Purpose |
|------|---------|
| `ingest.py` | ChromaDB chunking and embedding ingestion |
| `ingest_pgvector.py` | Postgres `pgvector` + SQL clearance table ingestion |
| `query.py` | Direct ChromaDB RAG CLI |
| `query_langchain.py` | LangChain retrieval pipeline |
| `query_llamaindex.py` | LlamaIndex retrieval pipeline |
| `ARCHITECTURE.md` | Framework and retrieval comparison notes |
| `evals/gold_qa.json` | 23 gold Q&A benchmark pairs |
| `evals/run_eval.py` | Multi-backend keyword evaluation harness |
| `evals/run_ragas.py` | RAGAS semantic evaluation harness |
| `evals/promptfooconfig.yaml` | `promptfoo` regression configuration |
