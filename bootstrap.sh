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
    git -C "$APP_DIR" pull --ff-only || true
  elif [ -f "$(pwd)/server.py" ] && [ -d "$(pwd)/trainero" ]; then
    APP_DIR="$(pwd)"
    log "rodando do checkout local: $APP_DIR"
  else
    log "clonando $REPO_URL…"
    git clone --depth 1 "$REPO_URL" "$APP_DIR"
  fi
}

make_venv() {
  export PATH="$HOME/.local/bin:$PATH"
  if [ ! -x "$APP_DIR/.venv/bin/python" ]; then
    log "criando venv leve do servidor…"
    if command -v uv >/dev/null; then
      uv venv "$APP_DIR/.venv" --seed
      uv pip install --python "$APP_DIR/.venv/bin/python" -r "$APP_DIR/requirements.txt"
    else
      python3 -m venv "$APP_DIR/.venv"
      "$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"
    fi
  fi
}

free_port() {
  # uma instância anterior segurando a porta é o motivo mais comum de o bootstrap
  # morrer com "Address already in use" na segunda vez que se roda o comando.
  # pkill cobre o caso normal; fuser é o reforço e pode não existir no container.
  pkill -f "[s]erver\.py" >/dev/null 2>&1 || true
  command -v fuser >/dev/null 2>&1 && fuser -k "$PORT/tcp" >/dev/null 2>&1 || true
  sleep 1
}

main() {
  install_system_deps
  install_uv
  clone_repo
  make_venv
  free_port
  log "subindo em http://0.0.0.0:$PORT  (abra a porta $PORT no painel do pod)"
  cd "$APP_DIR"
  exec "$APP_DIR/.venv/bin/python" server.py
}

main "$@"
