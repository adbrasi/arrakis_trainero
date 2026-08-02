"""Base-model downloads: aria2c when available, hf_hub_download fallback.

Files land in MODELS_DIR/<model_key>/<local_subpath>. A download entry whose
repo file ends with "/" (or is "/") means snapshot of that directory/repo.
Token is never put on argv — aria2c gets URL+header via a 0600 input file
(same rationale as arrakis_start: /proc/<pid>/cmdline is world-readable).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import MODELS_DIR, hf_token
from .jobs import Job, JobFailed
from .presets import MODELS


def model_root(model_key: str) -> Path:
    return MODELS_DIR / model_key


def resolve(model_key: str, rel: str) -> Path:
    return model_root(model_key) / rel


def _present(path: Path) -> bool:
    if path.is_dir():
        return any(path.rglob("*.safetensors")) or any(path.rglob("*.pth"))
    return path.is_file() and path.stat().st_size > 1024


def ensure_models(model_key: str, job: Job) -> None:
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    root = model_root(model_key)
    root.mkdir(parents=True, exist_ok=True)
    token = hf_token()
    for repo, remote, local in MODELS[model_key]["downloads"]:
        dest = root / local
        if _present(dest):
            job.log(f"✔ {local} já existe.")
            continue
        job.check_cancel()
        if remote.endswith("/"):
            _snapshot(repo, remote.strip("/"), dest, token, job)
        else:
            _single_file(repo, remote, dest, token, job)
        if not _present(dest):
            raise JobFailed(f"download de {repo}/{remote} não produziu {dest}")


def _snapshot(repo: str, subdir: str, dest: Path, token: str | None, job: Job) -> None:
    from huggingface_hub import snapshot_download

    job.log(f"Baixando snapshot {repo}" + (f" ({subdir}/)" if subdir else "") + "...")
    kwargs = {"local_dir": str(dest), "token": token}
    if subdir:
        kwargs["allow_patterns"] = [f"{subdir}/*"]
    snapshot_download(repo, **kwargs)
    # flatten "<dest>/<subdir>/*" -> "<dest>/*" so presets reference dest directly
    inner = dest / subdir if subdir else dest
    if subdir and inner.exists():
        for item in inner.iterdir():
            target = dest / item.name
            if not target.exists():
                shutil.move(str(item), str(target))
        shutil.rmtree(inner, ignore_errors=True)


def _single_file(repo: str, remote: str, dest: Path, token: str | None, job: Job) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    size_hint = ""
    job.log(f"Baixando {repo}/{remote}{size_hint}...")
    if shutil.which("aria2c"):
        url = f"https://huggingface.co/{repo}/resolve/main/{remote}"
        if _aria2(url, dest, token, job):
            return
        job.log("aria2c falhou, tentando huggingface_hub...")
    from huggingface_hub import hf_hub_download

    tmp_dir = dest.parent / ".hf_dl"
    tmp_dir.mkdir(exist_ok=True)
    path = hf_hub_download(repo, filename=remote, token=token, local_dir=str(tmp_dir))
    shutil.move(path, dest)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _aria2(url: str, dest: Path, token: str | None, job: Job) -> bool:
    part = dest.with_suffix(dest.suffix + ".part")
    argfile = None
    try:
        fd, argfile = tempfile.mkstemp(prefix=".aria2-", dir=str(dest.parent))
        lines = [url, f"  out={part.name}", f"  dir={dest.parent}"]
        if token:
            lines.append(f"  header=Authorization: Bearer {token}")
        os.write(fd, ("\n".join(lines) + "\n").encode())
        os.close(fd)
        os.chmod(argfile, 0o600)
        job.run([
            "aria2c", "-x", "8", "-s", "8", "--continue=true",
            "--console-log-level=warn", "--summary-interval=15",
            "--auto-file-renaming=false", "--allow-overwrite=true",
            "-i", argfile,
        ])
        os.replace(part, dest)
        return True
    except Exception:
        return False
    finally:
        if argfile and os.path.exists(argfile):
            os.unlink(argfile)
