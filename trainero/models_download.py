"""Base-model downloads. Xet first, aria2c as fallback.

Xet (hf_xet) reconstructs a file from deduplicated chunks over many parallel
connections and skips whatever the pod already has, so it is the fastest path
by a wide margin — it is the default here whenever the file is Xet-backed. A
file that is not Xet-backed would go over huggingface_hub's single HTTP stream,
which aria2c beats with 8 connections, so that case flips the order.

Files land in MODELS_DIR/<model_key>/<local_subpath>. A download entry whose
repo file ends with "/" (or is "/") means snapshot of that directory/repo.

Nothing partial is ever visible at the destination: every transfer lands in a
sibling ".part"/".hfpart" and is moved into place only once it completed, so
"the path exists" is the whole readiness test.

Token is never put on argv — aria2c gets URL+header via a 0600 input file and
the huggingface_hub child reads HF_TOKEN from its environment (same rationale
as arrakis_start: /proc/<pid>/cmdline is world-readable).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import MODELS_DIR, REPO_DIR, hf_token
from .jobs import Cancelled, Job, JobFailed
from .presets import MODELS


def model_root(model_key: str) -> Path:
    return MODELS_DIR / model_key


def resolve(model_key: str, rel: str) -> Path:
    return model_root(model_key) / rel


def _present(path: Path) -> bool:
    """A transfer is moved into place only when it finished, so existence is
    the test. Anything half-downloaded is sitting in a .part sibling."""
    if path.is_dir():
        return True
    return path.is_file() and path.stat().st_size > 1024


# ---------------------------------------------------------------------------
# Transport choice
# ---------------------------------------------------------------------------


def xet_ready() -> bool:
    """hf_xet installed and not disabled by the environment."""
    try:
        import hf_xet  # noqa: F401
    except ImportError:
        return False
    return os.environ.get("HF_HUB_DISABLE_XET", "").strip().lower() not in ("1", "true", "yes")


def prefers_xet(repo: str, remote: str, token: str | None) -> bool:
    """Should this file go over Xet?

    Only a definite "this file is not stored on Xet" sends it to aria2c. A probe
    that fails (rate limit, gated repo, no network yet) is not evidence of
    anything — Xet is the path we want whenever it can work, and the child
    process carries the token the probe may have lacked.
    """
    if not xet_ready():
        return False
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    try:
        meta = get_hf_file_metadata(hf_hub_url(repo, remote), token=token)
    except Exception:  # noqa: BLE001
        return True
    return getattr(meta, "xet_file_data", None) is not None


# ---------------------------------------------------------------------------


def ensure_models(model_key: str, job: Job) -> None:
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    root = model_root(model_key)
    root.mkdir(parents=True, exist_ok=True)
    token = hf_token()
    job.log("Transferência: Xet ativo." if xet_ready() else
            "Transferência: hf_xet indisponível — caindo para aria2c/HTTP.")
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
    job.log(f"Baixando snapshot {repo}" + (f" ({subdir}/)" if subdir else "") + " via Xet/hub…")
    _hf_fetch(repo, "", subdir, dest, token, job)


def _single_file(repo: str, remote: str, dest: Path, token: str | None, job: Job) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/main/{remote}"
    have_aria2 = bool(shutil.which("aria2c"))
    use_xet = prefers_xet(repo, remote, token)
    job.log(f"Baixando {repo}/{remote} via " +
            ("Xet." if use_xet else "aria2c." if have_aria2 else "huggingface_hub (HTTP)."))

    def by_xet() -> bool:
        try:
            _hf_fetch(repo, remote, "", dest, token, job)
            return True
        except JobFailed as exc:
            job.log(f"⚠ Xet/hub falhou: {exc}")
            return False

    def by_aria2() -> bool:
        if not have_aria2:
            return False
        if _aria2(url, dest, token, job):
            return True
        job.log("⚠ aria2c falhou.")
        return False

    order = (by_xet, by_aria2) if use_xet else (by_aria2, by_xet)
    for attempt in order:
        if attempt():
            return
    raise JobFailed(f"não consegui baixar {repo}/{remote} por nenhum transporte")


def _hf_fetch(repo: str, remote: str, subdir: str, dest: Path,
              token: str | None, job: Job) -> None:
    """Hand one transfer to trainero.hf_fetch in a child process (see its
    docstring for why it is a child and not an in-process call)."""
    stage = dest.parent / (dest.name + ".hfpart")
    spec = {"repo": repo, "remote": remote, "subdir": subdir,
            "dest": str(dest), "stage": str(stage)}
    env = {"HF_XET_HIGH_PERFORMANCE": "1"}
    if token:
        env["HF_TOKEN"] = token  # env, never argv
    job.run([sys.executable, "-m", "trainero.hf_fetch", json.dumps(spec)],
            cwd=REPO_DIR, env=env)


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
    except Cancelled:
        raise  # cancelling must stop the job, not silently try the next transport
    except (JobFailed, OSError, subprocess.SubprocessError):
        return False
    finally:
        if argfile and os.path.exists(argfile):
            os.unlink(argfile)
