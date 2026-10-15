# RAG vs. Small Classifier Architecture

**Modules:** RAG (`rag/`) + MCP/Agents (`agents/`) + PEFT/LoRA Classifier (`platform/finetune/`)  
**Base Model:** `google/flan-t5-base` · **Labels:** `cleared | pending | denied | unknown`

---

## When to use which

| Use **RAG + tools** (`search_docs`, `lookup_rights`, agents) | Use **small classifier** (fine-tuned LoRA) |
|--------------------------------------------------------------|-------------------------------------------|
| Answer requires **multiple documents** and synthesis | **Single snippet** -> fixed label |
| User needs **citations** (filename, memo, audit trail) | High **volume**, same classification task repeated |
| Question is **open-ended** natural language | Output is **one of N tiers** |
| Knowledge **changes** with new memos / deals | Label taxonomy is **stable** |
| Workflow: research -> compliance -> brief | Workflow: ingest tagging, pre-filtering, batch scoring |

**Design rule:** When compliance requires source citations and territory specifics, use SQL lookup (`lookup_rights`) + RAG (`search_docs`). When the ingestion pipeline needs fast tier tags across thousands of chunks, use the fine-tuned LoRA classifier.

---

## Tradeoffs

| Dimension | RAG + agent (`search_docs`, `lookup_rights`) | Small classifier (LoRA) |
|-----------|-----------------------------------------------|-------------------------|
| **Latency** | Seconds (embed + retrieve + LLM + tools) | Milliseconds per snippet |
| **Cost at scale** | Higher (LLM + embedding per query) | Low after training (CPU/GPU inference only) |
| **Determinism** | Variable (requires deterministic post-processing guards) | High (fixed weights -> consistent label) |
| **Explainability** | Strong (chunk citations + structured SQL rows) | Supporting signal only (class + confidence) |

---

## Combined stack in this repository

```text
Prompts + guards     -> policy enforcement (refuse, cite, escalate)
lookup_rights (SQL)  -> authoritative territory clearance row
search_docs (RAG)    -> narrative + multi-document context
Classifier (LoRA)    -> fast tier tagging on raw chunks at ingest
Agent + MCP          -> multi-step orchestration for production briefs
```

### Data plane integration

| Path | Command | Target column |
|------|---------|---------------|
| Batch lake ingest | `ingest_pgvector.py --data-dir data/lake/raw --lora-tag` | `document_chunks.rights_tier` |
| Event-driven worker | `embed_worker.py --lora-tag` | `document_chunks.rights_tier` |
