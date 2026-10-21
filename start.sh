#!/usr/bin/env bash
# Start local (WSL/desktop): mesmo fluxo do bootstrap, sem clone.
set -euo pipefail
cd "$(dirname "$0")"
APP_DIR="$(pwd)"
PORT="${WEB_PORT:-8090}"
log() { printf '\033[1;33m[trainero]\033[0m %s\n' "$*"; }

stop_previous_server() {
  # The server cancels its running job on SIGTERM and waits for the trainer to
  # exit (up to ~30 s) — killing it harder would orphan a training on the GPU.
  local pidfile="$1/data/server.pid" pid
  if [ -f "$pidfile" ]; then
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -n "$pid" ] && grep -q "server.py" "/proc/$pid/cmdline" 2>/dev/null; then
      log "parando a instância anterior (pid $pid)…"
      kill -TERM "$pid" 2>/dev/null || true
      # an exited process can linger as a zombie with an empty cmdline
      for _ in $(seq 1 45); do grep -q "server.py" "/proc/$pid/cmdline" 2>/dev/null || break; sleep 1; done
    fi
  fi
  # a server from before the pidfile existed: it was started as
  # "<app>/.venv/bin/python server.py", which names this app and no other
  if pkill -TERM -f "^$1/.venv/bin/python server.py" >/dev/null 2>&1; then
    for _ in $(seq 1 45); do pgrep -f "^$1/.venv/bin/python server.py" >/dev/null || break; sleep 1; done
  fi
  command -v fuser >/dev/null 2>&1 && fuser -k "$PORT/tcp" >/dev/null 2>&1 || true
}

sync_requirements() {
  # a `git pull` that adds a dependency must reach the venv, or the new
  # server dies on an import
  local stamp="$1/.venv/.requirements.sha" want
  want="$(sha256sum "$1/requirements.txt" | cut -d' ' -f1)"
  [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$want" ] && return
  log "instalando dependências do servidor…"
  if command -v uv >/dev/null; then
    uv pip install -q --python "$1/.venv/bin/python" -r "$1/requirements.txt"
  else
    "$1/.venv/bin/pip" install -q -r "$1/requirements.txt"
  fi
  echo "$want" > "$stamp"
}

if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null; then uv venv .venv --seed; else python3 -m venv .venv; fi
fi
sync_requirements "$APP_DIR"
stop_previous_server "$APP_DIR"
exec .venv/bin/python server.py
