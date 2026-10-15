#!/usr/bin/env bash
# Full RAG keyword baseline across all 6 backends + MLflow + baseline.json
set -euo pipefail

OPS_DIR="$(cd "$(dirname "$0")" && pwd)"
LAB_ROOT="$(cd "$OPS_DIR/../.." && pwd)"
EVAL="$LAB_ROOT/rag/evals/run_eval.py"

cd "$LAB_ROOT"

if [[ -f "$LAB_ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$LAB_ROOT/.venv/bin/activate"
fi

echo "Full RAG keyword baseline: 6 backends × gold_qa.json"
echo "Requires: Ollama (LLM), pgvector (Postgres), Pinecone API key in .env"
echo ""

python "$EVAL" --backend all --write-baseline --mlflow "$@"

echo ""
echo "Artifacts:"
echo "  rag/evals/results.json    : dashboard keyword table"
echo "  rag/evals/baseline.json   : keyword pass rates (+ RAGAS if present)"
echo "  platform/ops/mlflow.db     : media-rag-evals runs"
echo ""
echo "View:"
echo "  ./platform/ops/mlflow_ui.sh"
echo "  python platform/ops/mlflow_list_runs.py"
echo "  applied-ai-web → http://localhost:3000/admin/evals"
