#!/usr/bin/env bash
# Start BFF for localhost, stage, or production.
# Usage:
#   ./run.sh                    # defaults to APP_ENV=localhost
#   ./run.sh --env stage        # runs stage config
#   ./run.sh --env production   # runs production multi-worker server
set -euo pipefail

BFF_DIR="$(cd "$(dirname "$0")" && pwd)"
LAB_ROOT="$(cd "$BFF_DIR/../.." && pwd)"
VENV="$LAB_ROOT/.venv/bin/activate"

TARGET_ENV="${APP_ENV:-localhost}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
WORKERS="${WEB_CONCURRENCY:-2}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env|-e)
      TARGET_ENV="$2"
      shift 2
      ;;
    --port|-p)
      PORT="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

export APP_ENV="$TARGET_ENV"

if [[ -f "$VENV" ]]; then
  # shellcheck source=/dev/null
  source "$VENV"
elif ! command -v uvicorn >/dev/null 2>&1; then
  echo "Missing venv at $LAB_ROOT/.venv and uvicorn not on PATH."
  echo "Run: cd $LAB_ROOT && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

cd "$BFF_DIR"

if [[ "$APP_ENV" == "localhost" || "$APP_ENV" == "local" || "$APP_ENV" == "dev" ]]; then
  exec uvicorn main:app --reload --host "$HOST" --port "$PORT"
else
  exec uvicorn main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --timeout-keep-alive 65
fi
