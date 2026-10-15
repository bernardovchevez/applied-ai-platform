#!/usr/bin/env bash
# Start MLflow UI pointed at the platform sqlite store (same URI as --mlflow eval scripts).
set -euo pipefail

OPS_DIR="$(cd "$(dirname "$0")" && pwd)"
LAB_ROOT="$(cd "$OPS_DIR/../.." && pwd)"
DB="$OPS_DIR/mlflow.db"
PORT="${MLFLOW_UI_PORT:-5001}"
HOST="${MLFLOW_UI_HOST:-127.0.0.1}"

print_links() {
  echo "MLflow UI → http://${HOST}:${PORT}/"
  echo "Store: $DB"
  echo ""
  echo "MLflow 3 UI (no hash URLs; #/experiments/1 will look empty):"
  echo "  1. Open http://${HOST}:${PORT}/"
  echo "  2. Sidebar → Model development → Experiments"
  echo "  3. Click media-agent-evals or media-rag-evals"
  echo ""
  echo "Terminal fallback (always works):"
  echo "  python platform/ops/mlflow_list_runs.py"
}

port_in_use() {
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1
}

mlflow_responding() {
  curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1
}

if [[ ! -f "$DB" ]]; then
  echo "No MLflow database at $DB"
  echo "Log a run first, e.g.:"
  echo "  cd $LAB_ROOT"
  echo "  python agents/evals/run_agent_eval.py --mlflow"
  exit 1
fi

if port_in_use; then
  print_links
  echo ""
  if mlflow_responding; then
    echo "Port ${PORT} is already in use; MLflow UI is running. Use navigation above."
  else
    echo "Port ${PORT} is already in use by another process."
    echo "Stop it: lsof -ti :${PORT} | xargs kill"
    echo "Or use another port: MLFLOW_UI_PORT=5002 $0"
  fi
  exit 0
fi

print_links
echo ""

if [[ -f "$LAB_ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$LAB_ROOT/.venv/bin/activate"
fi

exec mlflow ui --backend-store-uri "sqlite:////${DB}" --port "$PORT" --host "$HOST"
