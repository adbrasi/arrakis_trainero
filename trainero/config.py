"""Paths, environment and persistent state for Arrakis Trainero.

Everything lives under WORKSPACE (/workspace on pods, cwd fallback for
local/WSL), mirroring arrakis_start/character_animatrem conventions so the
same bootstrap works on vast.ai and RunPod.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent

_ws = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
WORKSPACE = _ws if _ws.exists() else REPO_DIR.parent

ENGINES_DIR = WORKSPACE / "engines"
MODELS_DIR = WORKSPACE / "models"
PROJECTS_DIR = WORKSPACE / "trainero_projects"
DATA_DIR = REPO_DIR / "data"
STATE_FILE = DATA_DIR / "state.json"

WEB_PORT = int(os.environ.get("WEB_PORT", "8090"))

HF_TOKEN_ENVS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif", ".m4v"}


def hf_token() -> str | None:
    for env in HF_TOKEN_ENVS:
        if os.environ.get(env):
            return os.environ[env]
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def ensure_dirs() -> None:
    for d in (ENGINES_DIR, MODELS_DIR, PROJECTS_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)


_state_lock = threading.RLock()


def load_state() -> dict:
    with _state_lock:
        try:
            return json.loads(STATE_FILE.read_text())
        except FileNotFoundError:
            return {}
        except Exception:
            # Preserve the corrupt file instead of silently replacing it
            # (an abrupt pod death mid-write would otherwise erase history).
            try:
                STATE_FILE.rename(STATE_FILE.with_suffix(".json.corrupt"))
            except OSError:
                pass
            return {}


def save_state(state: dict) -> None:
    with _state_lock:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(DATA_DIR), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, STATE_FILE)
            dir_fd = os.open(str(DATA_DIR), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def update_state(**kwargs) -> dict:
    with _state_lock:
        state = load_state()
        state.update(kwargs)
        save_state(state)
        return state


def gpu_info() -> dict:
    """Name and VRAM (MiB) of GPU 0 via nvidia-smi; {} when unavailable."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        line = out.stdout.strip().splitlines()[0]
        name, mem = [p.strip() for p in line.split(",")[:2]]
        return {"name": name, "vram_mb": int(float(mem))}
    except Exception:
        return {}
