#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3.12}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON=python3
fi

echo "→ Using $PYTHON ($($PYTHON --version))"
echo "→ Creating venv..."
rm -rf .venv
"$PYTHON" -m venv .venv
source .venv/bin/activate

echo "→ Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "-> Created .env from .env.example; add your API keys"
else
  echo "→ .env already exists"
fi

echo ""
echo "Setup complete. Next steps:"
echo "  source .venv/bin/activate"
echo "  python guardrails/provider_compare.py --list-prompts"
echo "  python guardrails/provider_compare.py --preset summarize"
echo ""
echo "RAG pipeline:"
echo "  python rag/ingest.py"
echo "  python rag/query.py \"Is the theme song cleared for EMEA streaming?\""
