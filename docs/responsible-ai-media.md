# Responsible AI & Governance Architecture

Governance controls implemented across the platform's RAG, MCP, and multi-agent workflows.

---

## Summary map

| Governance control | Implementation |
|--------------------|----------------|
| **Source citations & grounding** | RAG source attribution (`shared/rag_text.py`) + deterministic grounding guards (`workflow.py`) |
| **Human-in-the-loop review gate** | FastAPI BFF `POST /v1/agent/run` (`202 Accepted` + `thread_id` -> explicit approve/reject) |
| **Audit trail & observability** | Session-scoped `Langfuse` traces (`X-Session-Id` / `thread_id`) + SQLite audit tables |
| **Regression gates** | 23-question RAG benchmark + 7-scenario agent evaluation logged to `MLflow` |

---

## Controls by layer

### 1. Data & metadata controls
- **Owned synthetic corpus:** LoRA fine-tuning (`platform/finetune/`) and vector stores operate exclusively on structured synthetic media operations documents (`data/synthetic/`).
- **Document-level metadata filtering:** Ingestion attaches `confidentiality`, `department`, `doc_type`, and `rights_tier` metadata (`rag/ingest_pgvector.py`), enabling filtered retrieval via `stores/pgvector.py` and `POST /v1/retrieve`.
- **Authoritative territory status:** Clearance status (`cleared`, `pending`, `excluded`, `unknown`) is stored in structured SQL (`clearance_territories`) and cross-checked by `lookup_rights` and deterministic post-processing guards.

### 2. Output & guardrail controls
- **Mandatory source attribution:** RAG responses append verified source filenames; agent runs return structured `sources[]` alongside tool traces.
- **Deterministic grounding & contradiction guards:** `apply_grounding_guard`, `apply_clearance_grounding_guard`, and `answer_contradicts_lookup` in `agents/workflow.py` prevent the LLM from overriding SQL clearance statuses or answering out-of-scope queries.
- **Human review gate:** External-facing or compliance-sensitive production briefs pause at a human approval checkpoint (`202 Accepted` in `bff/agent_sessions.py` or SQLite checkpointer in `graph.py`) until explicitly approved or rejected.

### 3. Operational & release controls
- **Prompt & model version pinning:** The BFF computes and logs a `prompt_version` hash of `system_rag.txt` across SSE streams, feedback records, and `Langfuse` trace metadata.
- **Rollback runbook:** Documented revert and smoke-evaluation procedure in [`platform/ops/rollback-runbook.md`](../platform/ops/rollback-runbook.md).
- **End-to-end audit correlation:** Every chat stream and agent workflow correlates retrieval spans, tool invocations, and user feedback via shared `session_id` / `thread_id` keys across `Langfuse` and local audit storage.
