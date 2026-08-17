"""HuggingFace: repo creation + continuous checkpoint upload during training.

Repo is created before training starts (animatrem pattern: the repo is born
documented even if training dies) and carries the run as data — trainero_config.json
plus captions.json, never a generated model card. A daemon watcher uploads every *.safetensors
that appears in output_dir once the file is complete (the safetensors header
says exactly how long the file must be); dedupe via .hf_uploaded.log. A
quota/auth failure disables further attempts instead of retrying every
checkpoint.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from .config import hf_token
from .jobs import Job


def hf_username() -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi(token=hf_token()).whoami()["name"]
    except Exception:
        return None


def create_repo(repo_id: str, private: bool, job: Job) -> str | None:
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token())
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        job.log(f"✔ Repo HF pronto: https://huggingface.co/{repo_id}")
        return repo_id
    except Exception as exc:
        job.log(f"⚠ Não consegui criar o repo HF ({exc}). Upload desativado.")
        return None


def upload_text(repo_id: str, path_in_repo: str, content: str, job: Job) -> None:
    try:
        from huggingface_hub import HfApi

        HfApi(token=hf_token()).upload_file(
            path_or_fileobj=content.encode(), path_in_repo=path_in_repo,
            repo_id=repo_id, repo_type="model",
        )
    except Exception as exc:
        job.log(f"⚠ upload de {path_in_repo} falhou: {exc}")


def model_card(label: str, base_model: str) -> str:
    """The one thing a LoRA page has to say: which model it was trained on.

    Not a generated description — prose nobody wrote goes stale the moment
    anything is edited by hand, and the rest of the run is already in the repo
    as data. `base_model` is the field HuggingFace indexes and renders as
    "Finetuned from", so the name is stated once, where it is machine-readable.
    """
    front = ["---"]
    if base_model:
        front.append(f"base_model: {base_model}")
    front += ["tags:", "  - lora", "---", "", f"# {label}", ""]
    return "\n".join(front)


def upload_run_files(repo_id: str, job: Job, *, info: dict, captions: dict) -> None:
    """Put what the run actually was into the repo, as data.

    The README carries the base model and nothing else; the facts belong in
    files — the resolved config, and the captions the LoRA was trained on.
    """
    label = info.get("model_label") or info.get("model") or ""
    if label:
        upload_text(repo_id, "README.md",
                    model_card(label, info.get("base_model", "")), job)
    upload_text(repo_id, "trainero_config.json",
                json.dumps(info, indent=2, ensure_ascii=False), job)
    if captions:
        upload_text(repo_id, "captions.json",
                    json.dumps(captions, indent=2, ensure_ascii=False), job)


class UploadWatcher:
    """Uploads new .safetensors from output_dir while training runs."""

    def __init__(self, repo_id: str, output_dir: Path, job: Job,
                 convert=None):
        self.repo_id = repo_id
        self.output_dir = output_dir
        self.job = job
        self.convert = convert  # optional fn(Path) -> Path|None (e.g. ComfyUI format)
        self._stop = threading.Event()
        self._disabled = False
        self._log = output_dir / ".hf_uploaded.log"
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        # The log dedupes within one run. It must not survive into the next one:
        # a retrain of the same project writes checkpoints with the same names,
        # and a leftover log made the watcher skip every one of them — the run
        # finished "ok" with nothing but the config file on HuggingFace.
        try:
            self._log.unlink()
        except OSError:
            pass
        self._thread.start()

    def stop_and_sweep(self):
        self._stop.set()
        self._thread.join(timeout=1800)

    # -- internals ---------------------------------------------------------
    def _uploaded(self) -> set[str]:
        try:
            return set(self._log.read_text().split())
        except OSError:
            return set()

    def _mark(self, name: str) -> None:
        with open(self._log, "a") as f:
            f.write(name + "\n")

    def _loop(self):
        while True:
            stopping = self._stop.wait(30)
            self._sweep(final=stopping)
            # The flag is read AFTER the sweep on purpose. Checking it in the
            # while condition lost the final sweep whenever stop landed while a
            # sweep was already in flight — a 2 GB upload takes minutes, and the
            # last checkpoint, the one that matters, never went up.
            if stopping:
                return

    def _sweep(self, final: bool = False):
        if self._disabled:
            return
        done = self._uploaded()
        for ckpt in sorted(self.output_dir.glob("*.safetensors")):
            if ckpt.name in done or self._disabled:
                continue
            if not _complete(ckpt):
                # mid-write on a periodic sweep — the next one picks it up.
                # On the final sweep it means the writer died first (a
                # cancelled run SIGTERMs the trainer mid-checkpoint), and a
                # truncated file must not reach the repo looking like weights.
                if final:
                    self.job.log(f"⚠ {ckpt.name} incompleto no disco "
                                 f"(escrita interrompida) — não enviado.")
                continue
            files = [ckpt]
            if self.convert:
                try:
                    extra = self.convert(ckpt)
                    if extra:
                        files.append(extra)
                except Exception as exc:  # conversion is best-effort
                    self.job.log(f"⚠ conversão de {ckpt.name} falhou: {exc}")
            for f in files:
                if self._upload(f):
                    self._mark(f.name)
                    done.add(f.name)

    def _upload(self, path: Path) -> bool:
        try:
            from huggingface_hub import HfApi

            self.job.log(f"↑ HF: enviando {path.name}...")
            HfApi(token=hf_token()).upload_file(
                path_or_fileobj=str(path), path_in_repo=path.name,
                repo_id=self.repo_id, repo_type="model",
            )
            self.job.log(f"✔ HF: {path.name} enviado.")
            links = self.job.extra.setdefault("hf_files", [])
            if path.name not in links:
                links.append(path.name)
            return True
        except Exception as exc:
            msg = str(exc)
            self.job.log(f"⚠ upload de {path.name} falhou: {msg[:300]}")
            if "403" in msg or "401" in msg or "quota" in msg.lower():
                self.job.log("⚠ Erro de auth/quota — uploads desativados para este treino.")
                self._disabled = True
            return False


def _complete(path: Path) -> bool:
    """Whether a .safetensors file is entirely on disk.

    Size stability (the previous test) cannot tell a finished checkpoint from
    one whose writer was killed mid-write — a dead writer is perfectly stable.
    The format itself can: the first 8 bytes carry the header length, the
    header carries every tensor's end offset, and the file is complete iff it
    is exactly as long as the header promises. This also replaces the 15s
    stability wait per checkpoint — a file still being written simply fails
    the size equation and is retried on the next sweep.
    """
    import struct

    try:
        size = path.stat().st_size
        if size < 8:
            return False
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            if header_len <= 0 or 8 + header_len > size:
                return False
            header = json.loads(f.read(header_len))
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    if not isinstance(header, dict):
        return False
    payload_end = max((entry["data_offsets"][1] for entry in header.values()
                       if isinstance(entry, dict) and "data_offsets" in entry),
                      default=0)
    return size == 8 + header_len + payload_end
