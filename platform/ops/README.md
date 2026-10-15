# Ops & Observability

Tracing and eval-run logging for the Applied AI platform layer.

## Langfuse (BFF)

Optional traces when Langfuse keys are set in `.env`. Module: [`langfuse_tracing.py`](langfuse_tracing.py).

### Chat: `POST /v1/chat/stream`

| Span | Type | What it captures |
|------|------|------------------|
| `v1/chat/stream` | root | Question, `X-Session-Id`, `prompt_version`, `X-Client-Platform`, final answer, `latency_ms` |
| `pgvector.retrieve` | retriever | `top_k`, source filenames |
| `llm.stream` | generation | Model name, truncated system/user messages, streamed answer |

### Agent: `POST /v1/agent/run`

| Span | Type | What it captures |
|------|------|------------------|
| `v1/agent/run` | root | Prompt, `auto_approve`, resume decision, `latency_ms`, status |
| `agents.workflow` | agent | Workflow execution (skipped on approve/reject-only resume) |
| `lookup_rights`, `search_docs`, … | tool | One span per entry in `tool_trace` |
| `human_approval` | tool | Approve/reject on resume |

**Session correlation:** human-gate runs use `thread_id` as Langfuse session id (run + approve/reject share one session). Optional `X-Session-Id` header overrides.

### Enable

1. Create a project at [cloud.langfuse.com](https://cloud.langfuse.com) (free tier).
2. Copy keys into `.env`:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

3. Restart BFF, send a chat from `applied-ai-web` (header `X-Session-Id` is auto-set).
4. Open Langfuse → Traces → filter by session id.

When keys are **unset**, tracing is a no-op (no import errors, no network calls).

### Smoke test

```bash
cd applied-ai-platform/platform/bff && ./run.sh

# Another terminal: stream one question (session id for correlation)
curl -N -X POST http://127.0.0.1:8000/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -H 'X-Session-Id: smoke-obs-1' \
  -H 'X-Client-Platform: internal' \
  -d '{"question":"What is the clearance index?","top_k":4}'
```

Check Langfuse for trace `v1/chat/stream` with session `smoke-obs-1`.

```bash
# Agent with human gate (session = thread_id in Langfuse)
curl -X POST http://127.0.0.1:8000/v1/agent/run \
  -H 'Content-Type: application/json' \
  -H 'X-Client-Platform: internal' \
  -d '{"prompt":"Clearance for Harbor Line S01E04 in EMEA?","auto_approve":false}'
# Note thread_id from 202 response, then approve:
curl -X POST http://127.0.0.1:8000/v1/agent/run \
  -H 'Content-Type: application/json' \
  -d '{"thread_id":"<uuid>","resume":"approve"}'
```

## MLflow (eval runs)

Log offline eval results to compare prompt/backend changes over time. Module: [`mlflow_eval.py`](mlflow_eval.py).

| Experiment | Script | Metrics |
|------------|--------|---------|
| `media-rag-evals` | `rag/evals/run_eval.py --mlflow` | `pass_rate`, `passed`, `total`, `total_time_s` |
| `media-rag-evals` | `rag/evals/run_ragas.py --mlflow` | RAGAS means + `total_time_s` |
| `media-agent-evals` | `agents/evals/run_agent_eval.py --mlflow` | `pass_rate`, `passed`, `total` |

Default tracking URI: `sqlite:///platform/ops/mlflow.db` (override with `MLFLOW_TRACKING_URI` in `.env`).

### Run + view

```bash
cd applied-ai-platform
pip install mlflow   # or: pip install -r requirements.txt

# Quick agent smoke (7 scenarios, logs one MLflow run)
python agents/evals/run_agent_eval.py --mlflow

# RAG keyword eval across all backends
python rag/evals/run_eval.py --backend all --write-baseline --mlflow
# or:
./platform/ops/run_rag_keyword_baseline.sh

# RAGAS smoke (3 questions)
python rag/evals/run_ragas.py --backend chroma --limit 3 --mlflow

# RAGAS -> MLflow (run after OPENAI_API_KEY is set)
# ./platform/ops/run_rag_ragas_mlflow.sh

# UI: browse runs locally (use helper script to avoid wrong/empty store)
./platform/ops/mlflow_ui.sh
# Open http://127.0.0.1:5001

**MLflow 3 UI:** Do **not** use `#/experiments/1` links; those are MLflow 2 hash routes and appear empty in v3.

1. Open http://127.0.0.1:5001/ (root, no hash)
2. Sidebar -> **Model development** -> **Experiments**
3. Click **media-agent-evals** (agent runs) or **media-rag-evals** (RAG runs)

**Terminal fallback** (if the UI still looks empty in MLflow 3):

```bash
python platform/ops/mlflow_list_runs.py
```

Manual UI start:

```bash
mlflow ui --backend-store-uri "sqlite:////$(pwd)/platform/ops/mlflow.db" --port 5001
```

Each run logs params (backend, judge model, etc.), metrics, and the results JSON as an artifact.

## RAG keyword baseline

One-shot script to refresh the eval dashboard comparison table and MLflow `media-rag-evals` experiment:

```bash
./platform/ops/run_rag_keyword_baseline.sh
```

Runs all **6 backends** (chroma, langchain, llamaindex, pgvector, pinecone, hybrid) against `gold_qa.json`, writes:

| Output | Purpose |
|--------|---------|
| `rag/evals/results.json` | `/admin/evals` keyword table |
| `rag/evals/baseline.json` | `keyword` pass rates (preserves RAGAS section) |
| `ops/mlflow.db` | one MLflow run per backend |

Flags on `run_eval.py`: `--backend all --write-baseline --mlflow`. Avoid single-backend runs without `--backend all`; they overwrite `results.json` with one row.

## RAGAS → MLflow

`--mlflow` on `run_ragas.py` + [`run_rag_ragas_mlflow.sh`](run_rag_ragas_mlflow.sh) are wired. Run the full baseline once `OPENAI_API_KEY` is set.

```bash
# After OPENAI_API_KEY in .env
./platform/ops/run_rag_ragas_mlflow.sh
```

Runs **chroma, langchain, llamaindex** with default metric **faithfulness** when executed. Logs per backend:

| MLflow field | Source |
|--------------|--------|
| `params.eval_type` | `rag_ragas` |
| `params.judge_model` | Ollama or OpenAI (auto) |
| `params.metric_names` | e.g. `faithfulness` |
| `metrics.faithfulness` | mean RAGAS score |
| artifact | `ragas_results.json` |

**Judge note:** local Ollama often returns `n/a` on multi-metric runs; use `--judge openai` for `answer_relevancy` / `context_precision`.

`--write-baseline` merges into `baseline.json` under `ragas` (preserves `keyword` section from keyword baseline).

## LangSmith (agent evals)

Optional LangGraph/ReAct traces when running the agent eval harness. Module: [`langsmith_tracing.py`](langsmith_tracing.py).

| Run config field | Purpose |
|------------------|---------|
| `run_name` | `agent-eval-{scenario_id}` (one trace per gold scenario) |
| `metadata` | `scenario_id`, `backend`, eval harness path |
| `tags` | `agent-eval`, backend key, scenario id |

### Enable

1. Create an API key at [smith.langchain.com/settings](https://smith.langchain.com/settings).
2. Add to `.env`:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=applied-ai-platform-agents
```

3. Run evals with `--langsmith` (sets tracing env for the process even if `.env` has tracing off):

```bash
cd applied-ai-platform
python agents/evals/run_agent_eval.py --langsmith --limit 1
python agents/evals/run_agent_eval.py --langsmith --mlflow
```

4. Open [smith.langchain.com](https://smith.langchain.com) → project **applied-ai-platform-agents** → filter tags `agent-eval`.

**Note:** Per-scenario `run_name`/tags apply to the default **`agent_v1.py`** backend. With `--graph` or `--mcp`, tracing is enabled at project level only.

When the API key is **unset**, `--langsmith` exits with a clear error (no silent no-op).

## Rollback runbook

Prompt/model/agent revert procedure + smoke eval gate: **[rollback-runbook.md](rollback-runbook.md)**

Quick flow:

1. Capture `prompt_version` + MLflow runs  
2. `git checkout <SHA> -- guardrails/prompts/system_rag.txt` (or `workflow.py`)  
3. Restart BFF -> smoke evals -> optional MLflow log  
4. Record incident summary in team runbook log
