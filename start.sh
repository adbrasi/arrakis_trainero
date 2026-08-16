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
PORT="${WEB_PORT:-8090}"
# libera a porta se uma instância anterior ficou pendurada
pkill -f "[s]erver\.py" >/dev/null 2>&1 || true
command -v fuser >/dev/null 2>&1 && fuser -k "$PORT/tcp" >/dev/null 2>&1 || true
exec .venv/bin/python server.py
