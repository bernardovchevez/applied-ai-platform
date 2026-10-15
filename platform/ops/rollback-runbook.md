# Rollback Runbook: Applied AI Platform

**Purpose:** Undo a prompt, model, agent, or retrieval regression and verify system health against evaluation gates.

**Scope:** Platform BFF + RAG + MCP/Agent workflows (`applied-ai-platform` + `applied-ai-web`).

**Related:** [responsible-ai-media.md](../../docs/responsible-ai-media.md) · [ops/README.md](README.md)

---

## When to rollback

| Signal | Where to look | Typical cause |
|--------|---------------|---------------|
| Wrong clearance / hallucination on gold Q | `/admin/evals`, `run_eval.py` | Prompt or grounding regression |
| Agent scenario fail (e.g. Netflix CEO) | `agent_results.json`, Langfuse | `workflow.py` guard weakened |
| User thumbs-down spike | `bff/data/feedback.db`, Langfuse session | Prompt or model drift |
| MLflow pass_rate drop vs last good run | `mlflow_list_runs.py`, MLflow UI | Promoted change without eval gate |
| Legal/compliance escalation | Manual report | Out-of-scope answer slipped through |

**Rule:** If keyword eval or agent eval fails a **known gate scenario**, treat as rollback-worthy before continuing to demo or ship.

---

## Step 0: Capture current state (2 min)

Before changing anything, record what is live:

```bash
cd applied-ai-platform

# Prompt fingerprint (BFF chat + feedback + Langfuse metadata)
python -c "
from pathlib import Path
import hashlib
p = Path('guardrails/prompts/system_rag.txt')
print('prompt_version:', hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.exists() else 'missing')
"

# Last eval artifacts
python -c "
import json
from pathlib import Path
for f in [
    'rag/evals/results.json',
    'agents/evals/agent_results.json',
]:
    p = Path(f)
    if p.exists():
        d = json.loads(p.read_text())
        print(f, d.get('timestamp'), d.get('passed'), '/', d.get('total'))
"

# MLflow snapshot
python platform/ops/mlflow_list_runs.py --experiment all
```

Note the **git commit** and **Langfuse trace** (if keys set) for the failing session.

---

## Step 1: Identify last known good (LKG)

| Layer | LKG source | What to record |
|-------|------------|----------------|
| **Prompt** | `git log guardrails/prompts/system_rag.txt` | Commit SHA + `prompt_version` hash |
| **Agent guards** | `git log agents/workflow.py` | Commit SHA |
| **LLM model** | `.env` → `OLLAMA_CHAT_MODEL` / `CHAT_MODEL` | Model id string |
| **Offline eval** | MLflow `media-rag-evals` / `media-agent-evals` | Run id with acceptable `pass_rate` |
| **Keyword baseline** | `rag/evals/baseline.json` | `keyword` + `ragas` sections |

**Promotion rule:** Do not treat a change as “shipped” until:

1. `run_eval.py --backend hybrid` ≥ **21/23** (or your team threshold)
2. `run_agent_eval.py` = **7/7**
3. MLflow run logged with `--mlflow` (optional but recommended)

---

## Step 2: Rollback by layer

### A. RAG system prompt (most common)

**Files:** `guardrails/prompts/system_rag.txt`  
**Affects:** BFF chat (`POST /v1/chat/stream`), workflow agent (appended to agent rules)

```bash
cd applied-ai-platform

# See history
git log --oneline -5 -- guardrails/prompts/system_rag.txt

# Revert to previous commit (replace <SHA> with LKG)
git checkout <SHA> -- guardrails/prompts/system_rag.txt

# Confirm new fingerprint
python -c "
import hashlib
from pathlib import Path
p = Path('guardrails/prompts/system_rag.txt')
print(hashlib.sha256(p.read_bytes()).hexdigest()[:12])
"
```

**Restart BFF** (uvicorn `--reload` may pick up file reads on next request; restart is safer):

```bash
# Stop existing BFF (Ctrl+C or kill port 8000), then:
cd platform/bff && ./run.sh
```

**Prod analog:** Redeploy pinned prompt artifact from MLflow Prompt Registry / config service; update `prompt_version` tag in Langfuse.

---

### B. Agent workflow / grounding guards

**Files:** `agents/workflow.py` (primary), optionally `agent_v1.py`  
**Affects:** BFF `POST /v1/agent/run`, human approval gate

```bash
git log --oneline -5 -- agents/workflow.py
git checkout <SHA> -- agents/workflow.py
```

Restart BFF (agent path loads workflow at request time; restart clears any cached imports).

---

### C. LLM / embedding model

**Files:** `.env` (never commit secrets)

```bash
# Example: revert Ollama chat model
# OLLAMA_CHAT_MODEL=llama3.2   ← set to LKG value
```

Restart BFF and any standalone eval processes. Re-run evals; model swaps can change pass rates without code changes.

**Prod analog:** Pin model id in deployment manifest; rollback = previous model version in registry.

---

### D. LoRA rights-tier classifier

**Files:** `platform/finetune/checkpoints/flan-t5-lora-rights-tier/adapter/`  
**Affects:** Ingest tagging via `--lora-tag`, optional tier metadata on chunks

```bash
# If you saved an earlier adapter, copy it back over adapter/
# Or re-run from checkpoint-N:
ls platform/finetune/checkpoints/flan-t5-lora-rights-tier/

# Re-ingest only if tier tags changed retrieval metadata
python rag/ingest_pgvector.py
```

**Prod analog:** Model registry version rollback + optional re-index job.

---

### E. Vector index / corpus (heavy rollback)

**When:** Bad re-ingest, chunking change, corrupted pgvector data  
**Avoid unless necessary** (slow path).

```bash
# Re-ingest from synthetic corpus (pgvector)
python rag/ingest_pgvector.py

# Pinecone (if used)
python rag/ingest_pinecone.py
```

Compare retrieval on one gold question before full eval suite.

---

## Step 3: Smoke gate (required after any rollback)

Run in order; all must pass before calling rollback complete.

```bash
cd applied-ai-platform

# 1. RAG keyword: fast hybrid slice (production BFF backend)
python rag/evals/run_eval.py --backend hybrid --limit 5 --no-write

# 2. Agent: known guardrail scenarios
python agents/evals/run_agent_eval.py \
  --id out_of_scope_netflix --no-write
python agents/evals/run_agent_eval.py \
  --id emea_cleared --no-write

# 3. BFF live (BFF must be running)
curl -sN -X POST http://127.0.0.1:8000/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -H 'X-Session-Id: rollback-smoke-1' \
  -d '{"question":"Is the theme song cleared for EMEA streaming?","top_k":4}' \
  | tail -3

# 4. Optional: log clean MLflow point for audit trail
python agents/evals/run_agent_eval.py --mlflow --no-write
```

**Full regression** (before demo or “promotion”):

```bash
./platform/ops/run_rag_keyword_baseline.sh
python agents/evals/run_agent_eval.py --mlflow
```

---

## Step 4: Verify observability alignment

| Check | Action |
|-------|--------|
| **Langfuse** | New traces show updated `prompt_version` in metadata |
| **Feedback DB** | New rows in `bff/data/feedback.db` use new `prompt_version` |
| **MLflow** | Post-rollback run logged; compare `pass_rate` to pre-incident run |
| **Eval dashboard** | `applied-ai-web` → `/admin/evals` reflects latest JSON artifacts |

```bash
python platform/ops/mlflow_list_runs.py --experiment rag
python platform/ops/mlflow_list_runs.py --experiment agent
```

---

## Step 5: Document the incident (1 paragraph)

Record in the team incident log:

- **What changed** (prompt SHA, workflow commit, model id)
- **Symptom** (which gold Q / agent scenario / user report)
- **LKG restored to** (commit, prompt hash, MLflow run id)
- **Smoke gate results** (pass/fail counts)
- **Follow-up** (CI eval on prompt PR, stricter gate, etc.)

---

## Quick reference: files that affect answers

| Component | Path |
|-----------|------|
| RAG system prompt | `guardrails/prompts/system_rag.txt` |
| Rejection / scrubber | `guardrails/prompts/rejection.txt`, `scrubber.txt` |
| Agent grounding | `agents/workflow.py` |
| Agent ReAct prompt | `agents/agent_v1.py` (`load_system_prompt`) |
| BFF RAG route | `platform/bff/main.py` |
| Gold eval set | `rag/evals/gold_qa.json` |
| Agent scenarios | `agents/evals/agent_scenarios.json` |
| Env / models | `.env` |
| Eval baselines | `rag/evals/baseline.json`, `results.json`, `agent_results.json` |

---

## Enterprise rollout extensions

For multi-environment production deployments, extend this git-based runbook with:

- Prompt registry with version aliases (`production`, `candidate`)
- CI blocking merge on eval regression (`.github/workflows/eval.yml`)
- Automated rollback on SLO breach (feature flags / blue-green deploy)
- Centralized SIEM audit log alongside Langfuse traces
