#!/usr/bin/env bash
# Start BFF from anywhere — no pip install on restart.
set -euo pipefail

BFF_DIR="$(cd "$(dirname "$0")" && pwd)"
LAB_ROOT="$(cd "$BFF_DIR/../.." && pwd)"
VENV="$LAB_ROOT/.venv/bin/activate"

if [[ ! -f "$VENV" ]]; then
  echo "Missing venv at $LAB_ROOT/.venv — run: cd $LAB_ROOT && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

# shellcheck source=/dev/null
source "$VENV"
cd "$BFF_DIR"
exec uvicorn main:app --reload --port 8000
