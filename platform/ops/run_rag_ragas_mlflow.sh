#!/usr/bin/env bash
# RAGAS semantic eval → MLflow (media-rag-evals)
# Run once OPENAI_API_KEY is configured for multi-metric judge scoring.
set -euo pipefail

OPS_DIR="$(cd "$(dirname "$0")" && pwd)"
LAB_ROOT="$(cd "$OPS_DIR/../.." && pwd)"
EVAL="$LAB_ROOT/rag/evals/run_ragas.py"

cd "$LAB_ROOT"

if [[ -f "$LAB_ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$LAB_ROOT/.venv/bin/activate"
fi

echo "RAGAS -> MLflow: chroma + langchain + llamaindex x gold_qa.json"
echo "Cloud judge: set OPENAI_API_KEY for multi-metric faithfulness scores (~20 to 35 min full run)."
echo "Default: faithfulness only. Ollama judge often returns null; keyword baseline is the local gate."
echo ""

python "$EVAL" \
  --backend all \
  --metrics faithfulness \
  --write-baseline \
  --mlflow \
  "$@"

echo ""
echo "Artifacts:"
echo "  rag/evals/ragas_results.json  : dashboard RAGAS table"
echo "  rag/evals/baseline.json       : ragas means (+ keyword if present)"
echo "  platform/ops/mlflow.db        : media-rag-evals runs (eval_type=rag_ragas)"
echo ""
echo "View:"
echo "  ./platform/ops/mlflow_ui.sh"
echo "  python platform/ops/mlflow_list_runs.py --experiment rag"
echo "  applied-ai-web → http://localhost:3000/admin/evals"
