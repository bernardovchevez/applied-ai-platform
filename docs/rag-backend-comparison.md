# RAG backend comparison (prototype evals)

This document describes how to **test, measure, and compare** retrieval backends in `applied-ai-platform`, and records a **localhost benchmark** run used to inform the BFF default (`pgvector`).

**Intent:** Prototyping to determine approach, not running six vector stores in production. The FastAPI BFF uses **pgvector only** for `/v1/retrieve` and `/v1/chat/stream`; other backends are comparison prototypes exercised via CLI eval harnesses.

## Gold set and harness

| Item | Location |
|------|----------|
| Questions (23) | `rag/evals/gold_qa.json` |
| Keyword/source evaluator | `rag/evals/run_eval.py` |
| RAGAS (subset of backends) | `rag/evals/run_ragas.py` |
| MLflow logging | `platform/ops/mlflow_eval.py`, experiment `media-rag-evals` |
| Web dashboard | `applied-ai-web` → `/admin/evals` (via BFF `GET /v1/evals/summary`) |

Keyword checks include required terms, optional `expected_contains_any`, forbidden phrases, and optional `expected_sources`.

## Prerequisites (localhost)

```bash
cd applied-ai-platform
source .venv/bin/activate
python scripts/check_env.py --env localhost   # Ollama or OpenAI for LLM + embeddings

# pgvector (BFF default)
docker compose up -d
python rag/ingest_pgvector.py

# Chroma + hybrid + langchain + llamaindex
python rag/ingest.py

# Pinecone (optional comparison)
# Set PINECONE_API_KEY and PINECONE_INDEX in .env, then:
python rag/ingest_pinecone.py
```

## Run keyword eval

**All backends (long; may timeout on LlamaIndex with local Ollama):**

```bash
./platform/ops/run_rag_keyword_baseline.sh
```

**Individual backends (recommended for local iteration):**

```bash
python rag/evals/run_eval.py --backend pgvector
python rag/evals/run_eval.py --backend hybrid
python rag/evals/run_eval.py --backend chroma
python rag/evals/run_eval.py --backend langchain
python rag/evals/run_eval.py --backend pinecone
python rag/evals/run_eval.py --backend llamaindex
```

**Write baseline + MLflow:**

```bash
python rag/evals/run_eval.py --backend all --write-baseline --mlflow
```

Artifacts:

- `rag/evals/results.json` (last run; prefer `--backend all` for full dashboard row set)
- `rag/evals/baseline.json` (merged pass rates)
- `platform/ops/mlflow.db` (compare runs in MLflow UI)

```bash
./platform/ops/mlflow_ui.sh
```

## RAGAS (LLM-as-judge)

RAGAS is wired for **chroma, langchain, llamaindex** (`run_ragas.py --backend`). Smoke test:

```bash
python rag/evals/run_ragas.py --backend chroma --limit 3 --judge ollama
```

Use RAGAS on the **shortlist** after keyword eval. Local Ollama judges can timeout or return NaN; prefer OpenAI judge (`--judge openai`) for stable RAGAS when `OPENAI_API_KEY` is set.

## Localhost benchmark (2026-10-16)

Environment: **Ollama** `llama3.2` + `nomic-embed-text`, pgvector **81 chunks / 27 docs**, Pinecone index re-ingested before run.

### Keyword pass rate (23 questions)

| Backend | Pass | Pass rate | Wall time (s) | Role |
|---------|------|-----------|---------------|------|
| chroma | 22/23 | 95.7% | 50.3 | Fast local prototype |
| langchain | 20/23 | 87.0% | 36.0 | Framework comparison |
| **pgvector** | **22/23** | **95.7%** | **83.3** | **BFF / production-shaped default** |
| hybrid | 22/23 | 95.7% | 83.8 | BM25 + vector (Chroma dense leg) |
| pinecone | 22/23 | 95.7% | 56.6 | Managed SaaS comparison |
| llamaindex | (incomplete) | n/a | n/a | Run aborted (Ollama timeout); prior baseline 21/23 |

Snapshot JSON: [`rag/evals/results-localhost-2026-10-16.json`](../rag/evals/results-localhost-2026-10-16.json).

### Interpretation

1. **Keyword scores cluster at 22/23** for pgvector, chroma, hybrid, and pinecone on this run. **LangChain** lagged (20/23), mainly on wording/format mismatches vs gold keywords, not necessarily wrong retrieval.
2. **pgvector** matches top keyword accuracy and aligns with **SQL/metadata filters**, **clearance_rows**, Docker Postgres, and the **BFF code path** clients use (web, Slack).
3. **hybrid** tied on pass rate but with similar latency to pgvector here; worth re-running on keyword-heavy gold items (for example Ad Ops / sponsorship phrasing) before promoting over pgvector.
4. **LlamaIndex** is costly on local Ollama (multi-minute per question); treat as framework lab, not default ops.
5. **RAGAS smoke** (chroma, 3 questions, Ollama judge): mean **faithfulness 0.75** with judge timeouts on 2/3 rows. Do not use as a deploy gate until judge is stable (OpenAI) or limits are raised.

### Recommended decision (prototype → hardening)

| Phase | Action |
|-------|--------|
| Now | Standardize demos and clients on **pgvector** via BFF (already implemented). |
| Next | Shortlist **hybrid vs pgvector** on failing gold items only; add regression in CI when pass rate drops. |
| Prod hardening | Single ingest pipeline, eval gate on `gold_qa.json`, retire duplicate query paths or mark as archived. |

## Validate the chosen path (BFF)

After backend choice, confirm the **same questions** through the API clients use:

```bash
cd platform/bff && ./run.sh --env localhost
curl -s http://localhost:8000/health | jq .
```

Use `applied-ai-web` or `applied-ai-slack` sample prompts: [applied-ai-slack docs](https://github.com/bernardovchevez/applied-ai-slack/blob/main/docs/local-slack-testing.md#sample-prompts-for-localhost-demos).

## Related docs

- [README.md](../README.md) (multi-backend overview)
- [platform/ops/README.md](../platform/ops/README.md) (Langfuse + MLflow)
- [docs/finetune-vs-rag.md](finetune-vs-rag.md) (when RAG vs classifier vs SQL)
