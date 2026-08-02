"""Single-slot background job runner with live log + progress for the UI.

One job at a time (import, caption, train) — the UI polls /api/status.
Subprocesses run in their own process group so cancel kills the whole tree.
Progress is parsed from the child's stdout (steps/epochs/loss) without ever
blocking it: a reader thread owns the pipe and appends to the job log file.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

_lock = threading.RLock()
_current: "Job | None" = None

# steps:  4%|▌  | 120/3000 [01:00<23:00, 2.0it/s, avr_loss=0.052]
_STEP_RE = re.compile(r"steps?:\s*\d+%.*?\|\s*(\d+)/(\d+)")
_LOSS_RE = re.compile(r"(?:avr_loss|loss)[=:]\s*([0-9.]+(?:e-?\d+)?)")
_EPOCH_RE = re.compile(r"epoch (\d+)/(\d+)", re.IGNORECASE)


class Cancelled(Exception):
    pass


class JobFailed(Exception):
    pass


class Job:
    def __init__(self, kind: str, title: str, log_path: Path):
        self.kind = kind
        self.title = title
        self.log_path = log_path
        self.phase = ""
        self.phases: list[dict] = []  # [{name, status: pending|running|done|failed}]
        self.status = "running"       # running | done | failed | cancelled
        self.error = ""
        self.progress: dict = {}      # {step, total_steps, epoch, total_epochs, loss}
        self.extra: dict = {}         # job-specific payload (hf links, counts…)
        self.started_at = time.time()
        self.finished_at: float | None = None
        self._cancel = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._log_lock = threading.Lock()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("")

    # -- logging -----------------------------------------------------------
    def log(self, msg: str) -> None:
        line = msg if msg.endswith("\n") else msg + "\n"
        with self._log_lock:
            with open(self.log_path, "a", encoding="utf-8", errors="replace") as f:
                f.write(line)

    def log_tail(self, max_bytes: int = 16384) -> str:
        try:
            size = self.log_path.stat().st_size
            with open(self.log_path, "rb") as f:
                if size > max_bytes:
                    f.seek(size - max_bytes)
                data = f.read()
            return data.decode("utf-8", errors="replace")
        except OSError:
            return ""

    # -- phases ------------------------------------------------------------
    def set_phases(self, names: list[str]) -> None:
        self.phases = [{"name": n, "status": "pending"} for n in names]

    def start_phase(self, name: str) -> None:
        self.phase = name
        for p in self.phases:
            if p["name"] == name:
                p["status"] = "running"
        self.log(f"\n===== {name} =====")

    def end_phase(self, name: str, ok: bool = True) -> None:
        for p in self.phases:
            if p["name"] == name:
                p["status"] = "done" if ok else "failed"

    # -- cancellation ------------------------------------------------------
    def cancel(self) -> None:
        self._cancel.set()
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                return

            def _escalate():
                time.sleep(15)
                if proc.poll() is None:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass

            threading.Thread(target=_escalate, daemon=True).start()

    def check_cancel(self) -> None:
        if self._cancel.is_set():
            raise Cancelled()

    # -- subprocess with streaming ----------------------------------------
    def run(self, cmd: list[str], cwd: Path | None = None, env: dict | None = None,
            parse_progress: bool = False) -> int:
        """Run a child, stream every output line to the log, optionally parse
        training progress. Raises Cancelled/JobFailed."""
        self.check_cancel()
        self.log("$ " + " ".join(str(c) for c in cmd))
        full_env = dict(os.environ)
        if env:
            full_env.update(env)
        full_env.setdefault("PYTHONUNBUFFERED", "1")
        proc = subprocess.Popen(
            [str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=full_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._proc = proc
        assert proc.stdout is not None
        try:
            for raw in iter(proc.stdout.readline, b""):
                line = raw.decode("utf-8", errors="replace")
                # tqdm redraws with \r: keep only the last frame of the line
                line = line.replace("\x1b[A", "").split("\r")[-1].rstrip("\n")
                if line.strip():
                    self.log(line)
                if parse_progress:
                    self._parse_progress(line)
            proc.wait()
        finally:
            self._proc = None
            try:
                proc.stdout.close()
            except OSError:
                pass
        if self._cancel.is_set():
            raise Cancelled()
        if proc.returncode != 0:
            raise JobFailed(f"comando saiu com código {proc.returncode}")
        return proc.returncode

    def _parse_progress(self, line: str) -> None:
        m = _STEP_RE.search(line)
        if m:
            self.progress["step"] = int(m.group(1))
            self.progress["total_steps"] = int(m.group(2))
        m = _LOSS_RE.search(line)
        if m:
            try:
                self.progress["loss"] = float(m.group(1))
            except ValueError:
                pass
        m = _EPOCH_RE.search(line)
        if m:
            self.progress["epoch"] = int(m.group(1))
            self.progress["total_epochs"] = int(m.group(2))

    # -- status snapshot ---------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "kind": self.kind,
            "title": self.title,
            "status": self.status,
            "phase": self.phase,
            "phases": self.phases,
            "progress": dict(self.progress),
            "extra": dict(self.extra),
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def current() -> Job | None:
    return _current


def start(kind: str, title: str, log_path: Path, target) -> Job:
    """Reserve the slot and run `target(job)` in a daemon thread.
    Raises RuntimeError if a job is already running (server-side guard —
    the client-side one is lost on page reload)."""
    global _current
    with _lock:
        if _current is not None and _current.status == "running":
            raise RuntimeError("Já existe um job em andamento.")
        job = Job(kind, title, log_path)
        _current = job

    def _runner():
        try:
            target(job)
            job.status = "done"
        except Cancelled:
            job.status = "cancelled"
            job.log("⚠ Cancelado pelo usuário.")
        except JobFailed as exc:
            job.status = "failed"
            job.error = str(exc)
            job.log(f"✘ FALHA: {exc}")
        except Exception as exc:  # noqa: BLE001 — surface anything to the UI
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            import traceback

            job.log("✘ ERRO INESPERADO:\n" + traceback.format_exc())
        finally:
            job.finished_at = time.time()
            if job.phase:
                job.end_phase(job.phase, ok=job.status == "done")

    threading.Thread(target=_runner, daemon=True).start()
    return job
