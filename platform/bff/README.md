# Media Ops AI Platform BFF

HTTP presentation layer for a decoupled platform architecture: one platform API, many clients.

```
applied-ai-platform/                    ← platform (RAG, agents, MCP, data plane)
└── platform/bff/  ← this BFF (FastAPI)

applied-ai-web/                         ← sibling Next.js client (future repo)
applied-ai-mobile/                      ← future
ctv-partner-sdk/                   ← future (same contract)
```

**Contract:** [`openapi.yaml`](openapi.yaml) — source of truth for all clients (web, mobile, CTV, and MCP hosts).

---

## Endpoints (v0.1 stub)

| Method | Path | Status | Purpose |
|--------|------|--------|---------|
| GET | `/health` | [x] | pgvector + LLM dependency check |
| GET | `/v1/version` | [x] stub | Prompt / embed version pins |
| GET | `/openapi.yaml` | [x] | Raw spec |
| GET | `/docs` | [x] | Swagger UI (FastAPI) |
| POST | `/v1/retrieve` | [x] | Chunks only (CTV / on-device LLM) |
| POST | `/v1/chat/stream` | [x] | SSE streaming RAG |
| POST | `/v1/feedback` | [x] | Thumbs → SQLite |
| GET | `/v1/evals/summary` | [x] | Admin eval dashboard |
| POST | `/v1/agent/run` | [x] | Multi-agent workflow + optional human approval |

---

## Client headers

| Header | Values | Use |
|--------|--------|-----|
| `X-Client-Platform` | `web`, `ios`, `android`, `ctv`, `mcp`, `internal` | Analytics + `format=compact` routing |
| `X-Session-Id` | UUID | Correlate chat, feedback, traces |

---

## Run server

**One-time** (if BFF deps missing — already in root `requirements.txt`):

```bash
cd applied-ai-platform
source .venv/bin/activate
pip install -r requirements.txt
# or: pip install -r platform/bff/requirements-bff.txt
```

**Start / restart** (no pip needed):

```bash
cd applied-ai-platform/platform/bff
./run.sh
```

Or manually:

```bash
cd applied-ai-platform
source .venv/bin/activate
cd platform/bff
uvicorn main:app --reload --port 8000
```

- Swagger: http://localhost:8000/docs  
- OpenAPI: http://localhost:8000/openapi.yaml  
- Health: http://localhost:8000/health  

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
| `/v1/feedback` | `bff/feedback.py` → SQLite |
| `/v1/evals/summary` | read `rag/evals/results.json` |
| `/v1/agent/run` | `agents/agent_v1.py` or MCP |

Clients **never** import Python from the platform — only HTTP/SSE.

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

Pending sessions are stored in `bff/data/agent_sessions.db` (gitignored).

---

## Observability

Optional **Langfuse** traces on `POST /v1/chat/stream` and `POST /v1/agent/run` when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set in `.env`. Chat uses `X-Session-Id`; agent human-gate runs correlate via `thread_id`.

Implementation: [`../ops/langfuse_tracing.py`](../ops/langfuse_tracing.py) · setup: [`../ops/README.md`](../ops/README.md).

When keys are unset, tracing is a no-op — no behavior change for local dev.

---

## Related docs

- [data-plane.md](../../docs/data-plane.md)
- [mcp-enterprise.md](../../docs/mcp-enterprise.md) — MCP vs BFF division
- [responsible-ai-media.md](../../docs/responsible-ai-media.md) — Responsible AI & governance
