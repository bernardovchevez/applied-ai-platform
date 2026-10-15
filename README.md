# Applied AI Platform & MCP Workflow Engine

[![CI](https://github.com/bernardovchevez/applied-ai-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/bernardovchevez/applied-ai-platform/actions/workflows/ci.yml)

End-to-end applied AI reference platform for media operations, rights clearance, and production workflows. Combines multi-backend RAG (`pgvector`, `Chroma`, `Pinecone`, hybrid BM25 + vector), a multi-agent LangGraph + **Model Context Protocol (MCP)** workflow service with deterministic guardrails and human-in-the-loop approval gates, and a FastAPI presentation BFF with `Langfuse` tracing and `MLflow` evaluation tracking.

| Module | Directory | Focus |
|-------|-----------|-------|
| **1. Enterprise Doc Q&A & Evals** | `rag/` | Multi-backend RAG (`pgvector`, `Chroma`, `Pinecone`, hybrid BM25 + vector) + 23-question gold eval & RAGAS harnesses |
| **2. Multi-Agent & MCP Workflows** | `agents/` | LangGraph supervisor + ReAct sub-agents, custom MCP server (`stdio` / `SSE`), deterministic grounding & escalation guards |
| **3. AI Platform, Data Plane & Ops** | `platform/` | FastAPI OpenAPI BFF, batch + event-driven lake ingest, PEFT/LoRA classifier, `Langfuse` tracing, `MLflow` eval tracking |

Companion clients (consume the FastAPI BFF over HTTP/SSE):
- **`applied-ai-web/`** (`X-Client-Platform: web`): Next.js browser client
- **`applied-ai-slack/`** (`X-Client-Platform: slack`): Slack Bolt (`@slack/bolt`) slash commands, `@media-ops` thread assistant, and interactive Block Kit human approval cards

## Quick start

```bash
cd applied-ai-platform
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add API keys and/or use Ollama
python guardrails/provider_compare.py
```

### Local Ollama (zero API billing)

```bash
brew install ollama
ollama serve                    # or open the Ollama app
ollama pull llama3.2            # chat
ollama pull nomic-embed-text    # RAG embeddings
```

Add to `.env` to prefer Ollama over OpenAI:

```env
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
```

```bash
python guardrails/provider_compare.py --preset summarize --providers ollama
python rag/ingest.py
python rag/query.py "Is the theme song cleared for EMEA streaming?"
```

## Postgres + pgvector

```bash
docker compose up -d
python rag/ingest_pgvector.py
python rag/query_pgvector.py "Is the theme song cleared for EMEA streaming?"
python rag/query_pgvector.py "EMEA music clearance" --sql-demo
python rag/query_pgvector.py "..." --doc-type rights_clearance --rights-tier cleared
```

## Optional: Pinecone

Set `PINECONE_API_KEY` in `.env`, then:

```bash
python rag/ingest_pinecone.py
python rag/query_pinecone.py "Is the theme song cleared for EMEA streaming?"
python rag/query_pinecone.py "EMEA music clearance" --filter-demo
python rag/query_pinecone.py "..." --doc-type rights_clearance --rights-tier cleared
```

## Hybrid BM25 + vector

Requires Chroma ingest (`python rag/ingest.py`). BM25 index is built in-memory from corpus.

```bash
python rag/query_hybrid.py "Riverdale Docs Q3 sponsorship" --show-context
python rag/evals/run_eval.py --backend hybrid
```

## Data plane & incremental ingest

Lake ingest + pgvector from `data/lake/raw/` (see [docs/data-plane.md](docs/data-plane.md)):

```bash
pip install -r requirements-pipelines.txt
python platform/pipelines/ingest_batch.py
python rag/ingest_pgvector.py --data-dir data/lake/raw
python rag/ingest_pgvector.py --data-dir data/lake/raw --lora-tag
```

Event-driven incremental path: `ingest_batch.py --emit-events` -> `embed_worker.py`.

## Web & Slack clients + FastAPI BFF (Localhost, Stage & Production)

Platform BFF in `platform/bff/`; Next.js client in sibling **`applied-ai-web/`** and Slack Bolt client in sibling **`applied-ai-slack/`**. All three repositories support three runtime environments via `APP_ENV` (`localhost`, `stage`, `production`):

| Environment | Env template | LLM / Embeddings default | BFF State Store (`agent_sessions`, `feedback`) | BFF Auth (`BFF_API_KEY`) |
|-------------|--------------|--------------------------|------------------------------------------------|--------------------------|
| `localhost` | `.env.localhost.example` | Ollama (`llama3.2` + `nomic-embed-text` 768d) or OpenAI | Local SQLite (`platform/bff/data/*.db`) | Optional (open by default) |
| `stage` | `.env.stage.example` | OpenAI (`gpt-4o-mini` + `text-embedding-3-small` 1536d) | Shared PostgreSQL (`DATABASE_URL`) | Bearer / `X-API-Key` |
| `production` | `.env.production.example` | OpenAI (`gpt-4o-mini` + `text-embedding-3-small` 1536d) | Shared PostgreSQL (`DATABASE_URL`) | Bearer / `X-API-Key` (`/docs` off) |

```bash
# Localhost (Terminal 1 - BFF, Terminal 2 - web client)
cd platform/bff && ./run.sh --env localhost
cd ../applied-ai-web && cp .env.local.example .env.local && npm run dev

# Stage or Production (multi-worker Uvicorn or Docker)
cp .env.stage.example .env.stage
./platform/bff/run.sh --env stage

# Or via Docker Compose (stage / production profile)
APP_ENV=stage docker compose --profile stage up --build -d
```

See [platform/README.md](platform/README.md) and [platform/bff/README.md](platform/bff/README.md).

## Testing & Continuous Integration

Unit and BFF contract tests run offline without external LLM/database calls (`tests/test_platform_and_bff.py`), and execute automatically on every push and pull request via GitHub Actions (`.github/workflows/ci.yml`):

```bash
# Syntax verification across all platform packages
python -m compileall -q shared stores guardrails rag agents platform scripts tests

# Fast unit & BFF contract test suite (env resolution, RAG/agent grounding guards, SQLite state, API key auth)
pytest -v

# Environment & dependency readiness check
python scripts/check_env.py --env localhost
```

## Observability, evaluations & governance

```bash
# Langfuse tracing - set LANGFUSE_* in .env, then run BFF + web client
cd platform/bff && ./run.sh

# MLflow + evaluation baselines
./platform/ops/run_rag_keyword_baseline.sh
python agents/evals/run_agent_eval.py --mlflow
./platform/ops/mlflow_ui.sh
```

Architecture & operational docs:
- [docs/data-plane.md](docs/data-plane.md)
- [docs/mcp-enterprise.md](docs/mcp-enterprise.md)
- [docs/finetune-vs-rag.md](docs/finetune-vs-rag.md)
- [docs/responsible-ai-media.md](docs/responsible-ai-media.md)
- [platform/ops/rollback-runbook.md](platform/ops/rollback-runbook.md)

## Repository structure

```
applied-ai-platform/
├── guardrails/                # LLM provider benchmark & layered prompt guardrails
├── rag/                       # Multi-backend RAG pipeline & evaluation harnesses
├── agents/                    # LangGraph multi-agent workflows & MCP server (stdio/SSE)
├── platform/                  # FastAPI BFF, data plane pipelines, LoRA classifier, MLOps
├── shared/                    # Shared LLM/embedding configuration & utilities
├── stores/                    # Vector & SQL store adapters (pgvector, Chroma, Pinecone, hybrid)
├── tests/                     # Unit & FastAPI BFF contract test suite (pytest)
├── data/synthetic/            # Synthetic media operations & rights clearance corpus
├── docs/                      # Architecture & governance documentation
└── .github/workflows/ci.yml   # GitHub Actions CI (Python 3.12 compileall + pytest)
```
