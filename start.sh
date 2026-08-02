#!/usr/bin/env bash
# Start local (WSL/desktop): mesmo fluxo do bootstrap, sem clone.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null; then
    uv venv .venv --seed && uv pip install --python .venv/bin/python -r requirements.txt
  else
    python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt
  fi
fi
exec .venv/bin/python server.py
