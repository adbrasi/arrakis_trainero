"""One huggingface_hub transfer, as a child process.

Run as a child for three reasons: cancelling a job kills the process group and
that has to stop the download; the Xet progress then streams into the job log
like every other phase; and the command echoed into that log stays one readable
line you can paste into a shell.

    python -m trainero.hf_fetch '<json spec>'

spec: {repo, remote, subdir, dest, stage}
  remote set  -> single file
  remote ""   -> snapshot of `subdir` (or the whole repo when subdir is "")

Resume: `stage` is derived from the destination and never randomised, and
huggingface_hub keeps its partial payload in `<stage>/.cache/huggingface/
download`. A killed transfer therefore picks up where it stopped on the next
run instead of starting the 17 GB again. The destination only appears once the
transfer finished, so an interrupted download can never look complete.

The token comes from HF_TOKEN in the environment — never argv, which is
world-readable via /proc.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path


class Progress:
    """tqdm-compatible sink that prints one live line per update.

    huggingface_hub's own bar writes through tqdm, which disables itself when
    the stream is not a terminal — and here it never is, so a 17 GB transfer
    logged absolutely nothing between "Baixando…" and "ok". Injecting a
    tqdm_class is the supported way in (arrakis_start's hf_xet_worker does the
    same). Frames end in \\r so the job log keeps one updating line instead of
    thousands of near-identical ones.
    """

    interval = 1.0  # seconds between frames; hf_xet updates far faster than that

    def __init__(self, *_args, **kwargs):
        self.desc = str(kwargs.get("desc") or "download")
        self.total = int(kwargs.get("total") or 0)
        self.n = int(kwargs.get("initial") or 0)
        self.resumed_at = self.n
        self._lock = threading.RLock()
        self._sampled_at = time.monotonic()
        self._sampled_n = self.n
        self._emitted_at = 0.0
        self._rate = 0.0
        self.format_dict = {"rate": None}  # some hub versions read this back

    # -- tqdm surface ------------------------------------------------------
    def update(self, amount=1) -> None:
        with self._lock:
            self.n += int(amount or 0)
        self._emit()

    def set_description(self, desc: str, refresh: bool = True) -> None:
        with self._lock:
            self.desc = str(desc)
        if refresh:
            self._emit()

    def set_postfix_str(self, *_a, **_kw) -> None:
        pass

    def refresh(self, *_a, **_kw) -> None:
        self._emit()

    def close(self) -> None:
        self._emit(force=True, final=True)

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        self.close()

    # -- output ------------------------------------------------------------
    def _emit(self, force: bool = False, final: bool = False) -> None:
        with self._lock:
            now = time.monotonic()
            if not force and now - self._emitted_at < self.interval:
                return
            elapsed, moved = now - self._sampled_at, self.n - self._sampled_n
            if elapsed > 0 and moved > 0:
                self._rate = moved / elapsed
                self.format_dict["rate"] = self._rate
            self._sampled_at, self._sampled_n, self._emitted_at = now, self.n, now
            line = self._render()
        # print outside the lock: the job reader is on the other end of a pipe
        print(line, end="\n" if final else "\r", flush=True)

    def _render(self) -> str:
        parts = [self.desc, _size(self.n) + (f"/{_size(self.total)}" if self.total else "")]
        if self.total:
            parts.append(f"{self.n / self.total * 100:.0f}%")
        if self._rate:
            parts.append(f"{_size(self._rate)}/s")
        if self.resumed_at:
            parts.append(f"(retomado em {_size(self.resumed_at)})")
        return "  ".join(parts)


def _size(n: float) -> str:
    for unit, step in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= step:
            return f"{n / step:.1f}{unit}"
    return f"{n:.0f}B"


def with_progress(fn, **kwargs):
    """Call a hub download with our reporter, and without it if a future hub
    release changes that argument — losing the bar beats losing Xet."""
    try:
        return fn(**kwargs, tqdm_class=Progress)
    except (TypeError, AttributeError) as exc:
        print(f"[hf] progresso indisponível nesta versão do hub ({type(exc).__name__}: {exc})",
              flush=True)
        return fn(**kwargs)


def main(argv: list[str]) -> int:
    spec = json.loads(argv[1])
    dest, stage = Path(spec["dest"]), Path(spec["stage"])

    from huggingface_hub import hf_hub_download, snapshot_download

    try:
        import hf_xet  # noqa: F401
    except ImportError:
        print("[hf] hf_xet ausente — transferência HTTP simples", flush=True)

    stage.mkdir(parents=True, exist_ok=True)
    if spec["remote"]:
        got = with_progress(hf_hub_download, repo_id=spec["repo"],
                            filename=spec["remote"], local_dir=str(stage))
        os.replace(got, dest)
    else:
        sub = spec["subdir"]
        with_progress(snapshot_download, repo_id=spec["repo"], local_dir=str(stage),
                      **({"allow_patterns": [sub + "/*"]} if sub else {}))
        shutil.rmtree(stage / ".cache", ignore_errors=True)
        inner = stage / sub if sub else stage
        if not inner.is_dir() or not any(inner.iterdir()):
            print(f"[hf] {spec['repo']} não tem nada em {sub or '/'}", flush=True)
            return 1
        os.replace(inner, dest)
    shutil.rmtree(stage, ignore_errors=True)
    print("[hf] ok", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
