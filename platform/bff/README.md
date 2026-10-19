# Media Ops AI Platform BFF

HTTP presentation layer for a decoupled platform architecture: one platform API, many clients.

```
applied-ai-platform/                    ← platform (RAG, agents, MCP, data plane)
└── platform/bff/  ← this BFF (FastAPI)
    ├── main.py           ← app wiring, CORS, routers
    ├── routes/           ← HTTP handlers by surface
    ├── models.py, deps.py, rag_helpers.py, bootstrap.py
    ├── agent_runner.py, health_checks.py, …

applied-ai-web/                         ← sibling Next.js client (X-Client-Platform: web)
applied-ai-slack/                       ← sibling Slack Bolt client (X-Client-Platform: slack)
applied-ai-mobile/                      ← future
ctv-partner-sdk/                        ← future (same contract)
```

**Contract:** [`openapi.yaml`](openapi.yaml): source of truth for all clients (web, Slack, mobile, CTV, and MCP hosts).

---

## Endpoints (v0.1)

| Method | Path | Status | Purpose |
|--------|------|--------|---------|
| GET | `/health` | [x] | pgvector + LLM dependency check (`environment`, `state_store`) |
| GET | `/v1/version` | [x] | Active `prompt_version`, `embed_model`, `environment`, and `state_store` |
| GET | `/openapi.yaml` | [x] | Raw spec |
| GET | `/docs` | [x] | Swagger UI (enabled in `localhost`/`stage`; disabled by default in `production`) |
| POST | `/v1/retrieve` | [x] | Chunks only (CTV / on-device LLM) |
| POST | `/v1/chat/stream` | [x] | SSE streaming RAG (unbuffered headers) |
| POST | `/v1/feedback` | [x] | Thumbs to SQLite (`localhost`) or PostgreSQL (`stage`/`production`) |
| GET | `/v1/evals/summary` | [x] | Admin eval dashboard |
| POST | `/v1/agent/run` | [x] | Multi-agent workflow + optional human approval |

---

## Client headers

| Header | Values | Use |
|--------|--------|-----|
| `X-Client-Platform` | `web`, `ios`, `android`, `ctv`, `slack`, `mcp`, `internal` | Analytics + `format=compact` routing |
| `X-Session-Id` | UUID | Correlate chat, feedback, traces |
| `Authorization` / `X-API-Key` | `Bearer <BFF_API_KEY>` | Required on `/v1/*` routes when `BFF_API_KEY` is configured (`stage` / `production`) |

---

## Run server

**One-time** (if BFF deps are missing; already in root `requirements.txt`):

```bash
cd applied-ai-platform
source .venv/bin/activate
pip install -r requirements.txt
# or: pip install -r platform/bff/requirements-bff.txt
```

**Start / restart** (supports `localhost`, `stage`, and `production`):

```bash
cd applied-ai-platform/platform/bff
./run.sh                    # defaults to APP_ENV=localhost (--reload)
./run.sh --env stage        # multi-worker stage mode (.env.stage)
./run.sh --env production   # multi-worker production mode (.env.production, /docs disabled)
```

Or via Docker from the repository root:

```bash
cd applied-ai-platform
APP_ENV=stage docker compose --profile stage up --build -d
```

Or manually:

```bash
cd applied-ai-platform
source .venv/bin/activate
cd platform/bff
uvicorn main:app --reload --port 8000
```

- Swagger: http://localhost:8000/docs (enabled in `localhost` and `stage`; disabled in `production` unless `BFF_ENABLE_DOCS=true`)  
- OpenAPI: http://localhost:8000/openapi.yaml  
- Health: http://localhost:8000/health (reports `environment` and `state_store`)  

---

## CORS (Next.js / browser clients)

The BFF allows cross-origin requests from local Next.js dev by default:

| Origin | Purpose |
|--------|---------|
| `http://localhost:3000` | Next.js (`applied-ai-web`) |
| `http://127.0.0.1:3000` | Same, when using loopback IP |

Override with comma-separated origins in `.env`:

```bash
BFF_CORS_ORIGINS=http://localhost:3000,https://your-app.vercel.app
```

Allowed client headers: `Content-Type`, `X-Client-Platform`, `X-Session-Id`, `Authorization`.

Preflight check:

```bash
curl -i -X OPTIONS http://localhost:8000/v1/retrieve \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type,X-Client-Platform"
```

---

## Implementation map

| BFF route | Platform module |
|-----------|-----------------|
| `/v1/retrieve` | `query_pgvector.retrieve()` |
| `/v1/chat/stream` | retrieve + `shared.llm` stream |
| `/v1/feedback` | `bff/feedback.py` -> SQLite (`localhost`) or PostgreSQL (`stage`/`production`) |
| `/v1/evals/summary` | read `rag/evals/results.json` |
| `/v1/agent/run` | `agents/agent_v1.py` + `bff/agent_sessions.py` (SQLite in `localhost`, PostgreSQL in `stage`/`production`) |

Clients **never** import Python from the platform, only HTTP/SSE.

---

## Sibling web client (`applied-ai-web/`)

The canonical UI lives in the sibling repository:

```bash
# ../applied-ai-web/.env.local
NEXT_PUBLIC_BFF_URL=http://localhost:8000
```

```bash
cd ../applied-ai-web
cp .env.local.example .env.local
npm install && npm run dev   # http://localhost:3000
```

See [`applied-ai-web/README.md`](../../../applied-ai-web/README.md).

---

## Sibling Slack client (`applied-ai-slack/`)

Slack Bolt app (Socket Mode on localhost) calling the same BFF routes with `X-Client-Platform: slack` and `format=compact` on chat streams.

```bash
# ../applied-ai-slack/.env.localhost
BFF_URL=http://localhost:8000
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...
SLACK_APP_TOKEN=xapp-...
```

```bash
cd ../applied-ai-slack
cp .env.localhost.example .env.localhost
npm install && npm run build && npm run start:localhost
```

Slash commands: `/media-ask`, `/media-clearance`, `/media-status`. Sample prompts: [applied-ai-slack/docs/local-slack-testing.md](../../../applied-ai-slack/docs/local-slack-testing.md#sample-prompts-for-localhost-demos).

GitHub: [bernardovchevez/applied-ai-slack](https://github.com/bernardovchevez/applied-ai-slack).

---

## Agent workflow (`POST /v1/agent/run`)

Runs the clearance workflow agent (`agent_v1.run_agent`) via `agent_runner.py`.

**Auto-approve (200):**

```bash
curl -X POST http://localhost:8000/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Clearance for Harbor Line S01E04 in EMEA?","auto_approve":true}'
```

**Human approval gate (202 → approve/reject):**

```bash
# 1. Run without auto_approve → 202 + thread_id
curl -X POST http://localhost:8000/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Clearance for Harbor Line S01E04 in EMEA?","auto_approve":false}'

# 2. Approve or reject
curl -X POST http://localhost:8000/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"<uuid>","resume":"approve"}'
```

Pending sessions are stored in `bff/data/agent_sessions.db` in `localhost` (gitignored) and in PostgreSQL (`DATABASE_URL`) in `stage` and `production`.

---

## Observability

Optional **Langfuse** traces on `POST /v1/chat/stream` and `POST /v1/agent/run` when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set in `.env`. Chat uses `X-Session-Id`; agent human-gate runs correlate via `thread_id`.

Implementation: [`../ops/langfuse_tracing.py`](../ops/langfuse_tracing.py) · setup: [`../ops/README.md`](../ops/README.md).

When keys are unset, tracing is a no-op with no behavior change for local dev.

---

## Related docs

- [data-plane.md](../../docs/data-plane.md)
- [mcp-enterprise.md](../../docs/mcp-enterprise.md): MCP vs BFF division
- [responsible-ai-media.md](../../docs/responsible-ai-media.md): Responsible AI & governance
