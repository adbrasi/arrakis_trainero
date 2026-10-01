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


#: written as the very last install step. The venv's python appears the moment
#: `uv venv` runs, long before torch and the requirements land, so "python
#: exists" called a half-installed engine ready forever after one failed pip —
#: and the training then crashed on an import, after the base-model download.
INSTALL_MARKER = ".trainero_installed"


def _spec_stamp(name: str) -> str:
    import hashlib
    import json

    return hashlib.sha256(json.dumps(ENGINES[name], sort_keys=True).encode()).hexdigest()[:16]


def is_installed(name: str) -> bool:
    marker = engine_dir(name) / ".venv" / INSTALL_MARKER
    try:
        return venv_python(name).exists() and marker.read_text().strip() == _spec_stamp(name)
    except OSError:
        return False


def _uv() -> str | None:
    return shutil.which("uv") or (
        str(Path.home() / ".local/bin/uv") if (Path.home() / ".local/bin/uv").exists() else None
    )


def _pip(name: str, job: Job) -> list[str]:
    venv = engine_dir(name) / ".venv"
    uv = _uv()
    if not (venv / "bin" / "python").exists():
        if uv:
            job.run([uv, "venv", str(venv), "--python", "3.11", "--seed"])
        else:
            job.run(["python3", "-m", "venv", str(venv)])
    if uv:
        return [uv, "pip", "install", "--python", str(venv / "bin/python")]
    return [str(venv / "bin/python"), "-m", "pip", "install"]


def _install_target(spec: dict) -> list[str]:
    extras = spec.get("extras")
    return ["-e", f".[{extras}]" if extras else "."]


def ensure_engine(name: str, job: Job) -> None:
    spec = ENGINES[name]
    dest = engine_dir(name)
    ENGINES_DIR.mkdir(parents=True, exist_ok=True)

    pulled = False
    if not (dest / ".git").exists():
        # a leftover directory from an interrupted clone would make git refuse
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        job.log(f"Clonando {spec['repo']} ({spec['branch']})...")
        job.run(["git", "clone", "--depth", "1", "--single-branch",
                 "--branch", spec["branch"], spec["repo"], str(dest)])
    elif spec.get("pull"):
        # A stale engine is bad; a phase that refuses to run is worse. Never let
        # a network hiccup or a local commit in the clone stop the job.
        try:
            job.run(["git", "pull", "--ff-only"], cwd=dest)
            pulled = True
        except Exception as exc:
            job.log(f"⚠ git pull falhou em {name} ({exc}) — seguindo com o clone atual.")
    else:
        job.log(f"Engine {name} já clonado.")

    if is_installed(name):
        if pulled and spec["install"] == "package":
            # new code may bring a new dependency; a satisfied install is a no-op
            job.run(_pip(name, job) + _install_target(spec), cwd=dest)
        job.log(f"venv de {name} pronto.")
        return

    pip = _pip(name, job)
    if spec.get("torch", True):
        job.log(f"Instalando torch no venv de {name} (pode demorar)...")
        job.run(pip + TORCH_SPEC)

    if spec["install"] == "package":
        job.run(pip + _install_target(spec), cwd=dest)
    else:
        req = dest / "requirements.txt"
        if not req.exists():
            raise JobFailed(f"{name}: requirements.txt não encontrado em {dest}")
        # sd-scripts requirements contain '.' — install from cwd
        job.run(pip + ["-r", "requirements.txt"], cwd=dest)

    if spec.get("torch", True):
        job.run(pip + ["accelerate>=1.6.0", "huggingface_hub[hf_xet]>=0.31.4", "protobuf", "six"])
        _write_accelerate_config(job)
        _check_cuda(name, job)
    (dest / ".venv" / INSTALL_MARKER).write_text(_spec_stamp(name))
    job.log(f"✔ Engine {name} pronto.")


def _check_cuda(name: str, job: Job) -> None:
    """Fail at install, not after a 40 GB download: an unpinned torch can be
    built for a CUDA newer than the pod's driver, and then it sees no GPU."""
    if not shutil.which("nvidia-smi"):
        job.log("⚠ nvidia-smi não encontrado — sem GPU visível; o treino não vai rodar aqui.")
        return
    probe = subprocess.run(
        [str(venv_python(name)), "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)"],
        capture_output=True, text=True, timeout=300)
    if probe.returncode != 0:
        raise JobFailed(
            f"o torch instalado em {name} não enxerga a GPU (driver antigo para esta build "
            f"de CUDA?). Use um pod com driver mais novo ou instale um torch compatível em "
            f"{engine_dir(name) / '.venv'}. Detalhe: {(probe.stderr or probe.stdout)[-300:]}")


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


def supports_sampling(engine: str) -> bool:
    """Whether this engine's trainer accepts --sample_prompts.

    Confirmed present in musubi upstream. The LTX fork and sd-scripts are
    checked by reading their sources — cheaper and safer than running the
    trainer with a flag it may reject.
    """
    if engine == "musubi":
        return True
    root = engine_dir(engine)
    if not root.exists():
        return False
    for py in root.rglob("*.py"):
        # the engine's own .venv holds tens of thousands of files and none of
        # them is the trainer — scanning it would cost seconds for nothing
        if ".venv" in py.parts or "site-packages" in py.parts:
            continue
        try:
            if "--sample_prompts" in py.read_text(encoding="utf-8", errors="ignore"):
                return True
        except OSError:
            continue
    return False
