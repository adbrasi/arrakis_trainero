"""Engine installation: git clone + isolated venv per trainer.

Each engine gets its own venv (torch versions and package names collide —
both musubi checkouts install as `musubi_tuner`). `uv` builds them fast and
shares the wheel cache; plain venv+pip is the fallback.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .config import ENGINES_DIR
from .jobs import Job, JobFailed
from .presets import ENGINES

TORCH_SPEC = ["torch", "torchvision"]


def engine_dir(name: str) -> Path:
    return ENGINES_DIR / ENGINES[name]["dir"]


def venv_python(name: str) -> Path:
    return engine_dir(name) / ".venv" / "bin" / "python"


def venv_bin(name: str, tool: str) -> Path:
    return engine_dir(name) / ".venv" / "bin" / tool


def is_installed(name: str) -> bool:
    return venv_python(name).exists()


def _uv() -> str | None:
    return shutil.which("uv") or (
        str(Path.home() / ".local/bin/uv") if (Path.home() / ".local/bin/uv").exists() else None
    )


def ensure_engine(name: str, job: Job) -> None:
    spec = ENGINES[name]
    dest = engine_dir(name)
    ENGINES_DIR.mkdir(parents=True, exist_ok=True)

    if not (dest / ".git").exists():
        job.log(f"Clonando {spec['repo']} ({spec['branch']})...")
        job.run(["git", "clone", "--depth", "1", "--single-branch",
                 "--branch", spec["branch"], spec["repo"], str(dest)])
    else:
        job.log(f"Engine {name} já clonado.")

    if is_installed(name):
        job.log(f"venv de {name} já existe.")
        return

    uv = _uv()
    venv = dest / ".venv"
    if uv:
        job.run([uv, "venv", str(venv), "--python", "3.11", "--seed"])
        pip = [uv, "pip", "install", "--python", str(venv / "bin/python")]
    else:
        job.run(["python3", "-m", "venv", str(venv)])
        pip = [str(venv / "bin/python"), "-m", "pip", "install"]

    job.log(f"Instalando torch no venv de {name} (pode demorar)...")
    job.run(pip + TORCH_SPEC)

    if spec["install"] == "package":
        job.run(pip + ["-e", "."], cwd=dest)
    else:
        req = dest / "requirements.txt"
        if not req.exists():
            raise JobFailed(f"{name}: requirements.txt não encontrado em {dest}")
        # sd-scripts/data_araknideo requirements contain '.' — install from cwd
        job.run(pip + ["-r", "requirements.txt"], cwd=dest)

    job.run(pip + ["accelerate>=1.6.0", "huggingface_hub[hf_xet]>=0.31.4", "protobuf", "six"])
    _write_accelerate_config(job)
    job.log(f"✔ Engine {name} pronto.")


def _write_accelerate_config(job: Job) -> None:
    cfg = Path.home() / ".cache/huggingface/accelerate/default_config.yaml"
    if cfg.exists():
        return
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        "compute_environment: LOCAL_MACHINE\n"
        "distributed_type: 'NO'\n"
        "mixed_precision: bf16\n"
        "num_machines: 1\n"
        "num_processes: 1\n"
        "use_cpu: false\n"
    )
    job.log("accelerate configurado (single GPU, bf16).")


def ensure_system_tool(tool: str, apt_pkg: str, job: Job) -> bool:
    """Best-effort apt install for helpers (unzip, aria2c, megatools)."""
    if shutil.which(tool):
        return True
    apt = ["apt-get"]
    if shutil.which("sudo") and os.geteuid() != 0:
        apt = ["sudo", "apt-get"]
    job.log(f"Instalando {apt_pkg}...")
    try:
        subprocess.run(apt + ["update", "-qq"], check=False, capture_output=True, timeout=300)
        subprocess.run(apt + ["install", "-y", "-qq", apt_pkg], check=False, capture_output=True, timeout=600)
    except Exception:
        pass
    return shutil.which(tool) is not None
