#!/usr/bin/env bash
# Arrakis Trainero — bootstrap para vast.ai / RunPod.
#
#   export HF_TOKEN="hf_..."            # upload/download HuggingFace
#   export OPENROUTER_API_KEY="sk-..."  # captions via LLM (opcional)
#   curl -L https://raw.githubusercontent.com/adbrasi/arrakis_trainero/main/bootstrap.sh | bash
#
# Tudo dentro de funções: bash executa o stream conforme chega, então um
# download truncado não pode rodar meio deploy (padrão do arrakis_start).

set -euo pipefail

REPO_URL="${TRAINERO_REPO:-https://github.com/adbrasi/arrakis_trainero.git}"
BRANCH="${TRAINERO_BRANCH:-}"   # vazio = branch padrão do repo
WORKSPACE="${WORKSPACE_DIR:-/workspace}"
APP_DIR="$WORKSPACE/arrakis_trainero"
PORT="${WEB_PORT:-8090}"

log() { printf '\033[1;33m[trainero]\033[0m %s\n' "$*"; }

install_system_deps() {
  log "dependências de sistema (git, aria2, unzip, megatools, ffmpeg)…"
  local APT="apt-get"
  if command -v sudo >/dev/null && [ "$(id -u)" != "0" ]; then APT="sudo apt-get"; fi
  $APT update -qq || true
  $APT install -y -qq git aria2 unzip megatools ffmpeg curl python3-venv >/dev/null 2>&1 || true
}

install_uv() {
  if command -v uv >/dev/null || [ -x "$HOME/.local/bin/uv" ]; then return; fi
  log "instalando uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
}

clone_repo() {
  mkdir -p "$WORKSPACE"
  if [ -d "$APP_DIR/.git" ]; then
    log "repo já existe — atualizando…"
    if [ -n "$BRANCH" ]; then
      git -C "$APP_DIR" fetch origin "$BRANCH" || true
      git -C "$APP_DIR" checkout -B "$BRANCH" "origin/$BRANCH" || true
    else
      git -C "$APP_DIR" pull --ff-only || true
    fi
  elif [ -f "$(pwd)/server.py" ] && [ -d "$(pwd)/trainero" ]; then
    APP_DIR="$(pwd)"
    log "rodando do checkout local: $APP_DIR"
  else
    log "clonando $REPO_URL${BRANCH:+ ($BRANCH)}…"
    git clone --depth 1 ${BRANCH:+--branch "$BRANCH"} "$REPO_URL" "$APP_DIR"
  fi
}

make_venv() {
  export PATH="$HOME/.local/bin:$PATH"
  if [ ! -x "$APP_DIR/.venv/bin/python" ]; then
    log "criando venv leve do servidor…"
    if command -v uv >/dev/null; then
      uv venv "$APP_DIR/.venv" --seed
    else
      python3 -m venv "$APP_DIR/.venv"
    fi
  fi
  sync_requirements "$APP_DIR"
}

stop_previous_server() {
  # The server cancels its running job on SIGTERM and waits for the trainer to
  # exit (up to ~30 s) — killing it harder would orphan a training on the GPU.
  local pidfile="$1/data/server.pid" pid
  if [ -f "$pidfile" ]; then
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -n "$pid" ] && grep -q "server.py" "/proc/$pid/cmdline" 2>/dev/null; then
      log "parando a instância anterior (pid $pid)…"
      kill -TERM "$pid" 2>/dev/null || true
      for _ in $(seq 1 45); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    fi
  fi
  # a server from before the pidfile existed: free the port the old way
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

main() {
  install_system_deps
  install_uv
  clone_repo
  make_venv
  stop_previous_server "$APP_DIR"
  log "subindo em http://0.0.0.0:$PORT  (abra a porta $PORT no painel do pod)"
  cd "$APP_DIR"
  mkdir -p "$APP_DIR/data"
  # detached from this terminal: closing the pod's web terminal must not take
  # the server — and a running training — down with it
  setsid nohup "$APP_DIR/.venv/bin/python" server.py >> "$APP_DIR/data/server.log" 2>&1 < /dev/null &
  log "servidor em segundo plano (log: $APP_DIR/data/server.log). Ctrl+C só para de mostrar o log."
  sleep 2
  exec tail -n 20 -f "$APP_DIR/data/server.log"
}

main "$@"
