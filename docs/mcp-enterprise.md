# Embedding MCP in Enterprise Application Teams

Architecture and rollout pattern for Model Context Protocol (MCP) servers so application teams can consume governed AI tools without duplicating RAG, rights lookup, and guardrails across clients.

**Implementation:** `agents/mcp_server/server.py`

---

## Why MCP for media & enterprise workflows

| Without MCP | With MCP |
|-------------|----------|
| Each team wires RAG + APIs differently | Single **tool contract** (`search_docs`, `lookup_rights`, `draft_brief`, `escalate_legal`) |
| Prompt logic scattered across clients | **Server owns** tool schemas, prompts, and backend routing |
| Hard to audit what the model can call | Central **allowlist** per team and environment |
| IDE agents, internal workflows, and web apps duplicate glue | **Clients** differ; **capabilities** stay consistent |

MCP acts as the **agent-facing adapter** on top of Postgres, `pgvector`, and internal domain services.

---

## Architecture

```
SharePoint / S3 docs
        ↓
  Ingest + embed + metadata
        ↓
 pgvector / warehouse (clearance_territories)
        ↓
 ┌──────┴──────┐
 │  REST API   │  <- Web, mobile, CTV clients (FastAPI BFF)
 └──────┬──────┘
        │
 ┌──────┴──────┐
 │ MCP server  │  <- IDE assistants, Claude Desktop, LangGraph agents
 │ (stdio/SSE) │
 └──────┬──────┘
        ↓
   LLM clients (with layered prompt guardrails + deterministic post-processing)
```

See also: [data-plane.md](data-plane.md)

---

## Rollout pattern

### 1. Register the server

- **Name:** `media-workflow`
- **Transport:** `stdio` for local IDE clients; `SSE` / streamable HTTP for shared environments
- **Ownership:** Platform engineering maintains MCP schemas and transport; domain teams own tool backends

### 2. Tool boundaries

| Tool | Backend owner | Client use |
|------|---------------|------------|
| `search_docs` | Data / search platform | Corpus research & grounded Q&A |
| `lookup_rights` | Rights / legal data | Authoritative territory clearance checks |
| `draft_brief` | Production workflow | Grounded production brief synthesis |
| `escalate_legal` | Compliance | Triggered automatically when confidence < 0.7 |
| `check_ip_flags` | Legal / ad sales | Sponsorship and placement validation |

Business logic stays in shared Python modules (`tools.py`, `stores/`), keeping MCP handlers as thin JSON-RPC adapters.

### 3. Guardrails

MCP exposes capabilities, while deterministic guards stay enforced server-side or in workflow post-processing (`workflow.py`):

- Grounded RAG refusal enforcement (`apply_grounding_guard`)
- Territory clearance alignment (`apply_clearance_grounding_guard`)
- Automatic compliance escalation below confidence thresholds (`apply_auto_escalation`)
- Human-in-the-loop approval before external output (`graph.py` checkpoint interrupt)

### 4. Developer experience

**MCP Host (`stdio` configuration):**

```json
{
  "mcpServers": {
    "media-workflow": {
      "command": "python",
      "args": ["agents/mcp_server/server.py"],
      "cwd": "/path/to/applied-ai-platform"
    }
  }
}
```

**CLI verification:**

```bash
python agents/mcp_server/server.py --smoke-test
python agents/mcp_server/client_demo.py
```
