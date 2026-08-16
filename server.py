#!/usr/bin/env python3
"""Arrakis Trainero — web server (stdlib only, arrakis_start pattern).

Single-port HTTP: static UI + JSON API, polling instead of websockets
(cloud pods proxy exactly one port). Run: python server.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trainero import dataset as ds
from trainero import jobs
from trainero.jobs import JobFailed
from trainero.captioner import generate_captions
from trainero.config import (IMAGE_EXTS, VIDEO_EXTS, WEB_PORT, ensure_dirs,
                             gpu_info, hf_token, load_state, update_state)
from trainero.engines import is_installed
from trainero.presets import (CAPTION_PROFILES, MODEL_ORDER, MODELS,
                              STYLE_RUSH_SCHEDULE, public_presets,
                              style_rush_models, suggest_schedule)
from trainero.training import project_dir, run_training, slugify

WEB_DIR = Path(__file__).resolve().parent / "web"

# A job reads the dataset from the first phase to the last; anything that writes
# to it meanwhile changes the run under its feet. jobs.start() is the mutex for
# job-vs-job, and these two cover the writes that are plain HTTP requests:
# uploads/clears wait for the job slot, and a train waits for the uploads.
_uploads = 0
_uploads_lock = threading.Lock()


class _UploadInFlight:
    def __enter__(self):
        global _uploads
        with _uploads_lock:
            _uploads += 1

    def __exit__(self, *_exc):
        global _uploads
        with _uploads_lock:
            _uploads -= 1
        return False


def uploads_in_flight() -> int:
    with _uploads_lock:
        return _uploads

# musubi writes samples as <output_name>_e{epoch:06d}_{idx:02d}_{ts}[_{seed}].png
# and falls back to a bare step counter when the run is step-based.
_SAMPLE_RE = re.compile(r"_(e?)(\d{6})_(\d{2})_")


def parse_sample_name(name: str) -> tuple[int, int]:
    """(epoch, index) from a sample filename; (-1, -1) when it does not match."""
    # the LAST match, not the first: a project named "makima_202601_01_v2" puts a
    # digit run in the slug that would otherwise be read as the epoch
    matches = list(_SAMPLE_RE.finditer(name))
    if not matches:
        return (-1, -1)
    m = matches[-1]
    epoch = int(m.group(2)) if m.group(1) == "e" else -1
    return (epoch, int(m.group(3)))


def list_samples(sample_dir: Path) -> list[dict]:
    """Samples newest first — the UI shows the freshest epoch at the top."""
    try:
        files = [p for p in sample_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    except OSError:
        return []
    # epoch leads: it is the real ordering when the trainer knows it. mtime only
    # breaks ties (step-based runs report epoch -1 for every file).
    files.sort(key=lambda p: (parse_sample_name(p.name)[0], p.stat().st_mtime, p.name),
               reverse=True)
    out = []
    for p in files:
        epoch, idx = parse_sample_name(p.name)
        out.append({"name": p.name, "epoch": epoch, "idx": idx})
    return out


#: how many thumbnails the contact sheet shows before collapsing into "+N"
THUMB_LIMIT = 24
THUMB_EDGE = 220


def dataset_thumbs(directory: Path, limit: int = THUMB_LIMIT) -> tuple[list[str], int]:
    """The first `limit` image names in a dataset, plus how many there are."""
    try:
        names = sorted(p.name for p in directory.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    except OSError:
        return [], 0
    return names[:limit], len(names)


def render_thumb(path: Path, edge: int = THUMB_EDGE) -> bytes:
    """A small square-ish JPEG. The browser would otherwise pull full-size
    originals — a 40-image dataset of 4K art is hundreds of MB per poll."""
    import io

    from PIL import Image

    with Image.open(path) as im:
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        im.thumbnail((edge, edge), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=78)
    return buf.getvalue()


def safe_sample_name(name: str) -> bool:
    """A plain .png basename — no separators, no traversal, no NUL.

    A NUL byte survives Path().name and endswith() but makes open() raise
    ValueError, which is not an OSError — it would escape the handler.
    """
    return bool(name) and "\x00" not in name and name == Path(name).name \
        and name.lower().endswith(".png") and ".." not in name


def current_project() -> str:
    return load_state().get("project", "")


def dataset_dir(side: str = "pos") -> Path:
    pdir = project_dir(current_project() or "projeto")
    return pdir / {"neg": "dataset_neg", "convert": "dataset_convert",
                   "restore": "dataset_restore"}.get(side, "dataset")


def sample_dir() -> Path:
    return project_dir(current_project() or "projeto") / "output" / "sample"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, fmt, *args):  # quiet access log
        pass

    # -- helpers -----------------------------------------------------------
    def _json(self, payload: dict, code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, msg: str, code: int = 400) -> None:
        self._json({"error": msg}, code)

    def _body_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    # -- GET ---------------------------------------------------------------
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/status":
            return self._status()
        if url.path == "/api/presets":
            return self._presets()
        if url.path == "/api/logs":
            job = jobs.current()
            return self._json({"log": job.log_tail() if job else ""})
        if url.path == "/api/samples":
            return self._json({"samples": list_samples(sample_dir())})
        if url.path == "/api/sample":
            q = parse_qs(url.query)
            return self._serve_sample(q.get("name", [""])[0], q.get("thumb", [""])[0] == "1")
        if url.path == "/api/dataset/thumbs":
            side = parse_qs(url.query).get("side", ["pos"])[0]
            names, total = dataset_thumbs(dataset_dir(side))
            return self._json({"names": names, "total": total})
        if url.path == "/api/dataset/thumb":
            q = parse_qs(url.query)
            return self._serve_thumb(q.get("side", ["pos"])[0], q.get("name", [""])[0])
        return super().do_GET()

    def _serve_thumb(self, side: str, name: str):
        if not name or "\x00" in name or name != Path(name).name or ".." in name:
            return self._error("nome inválido")
        path = dataset_dir(side) / name
        if path.suffix.lower() not in IMAGE_EXTS:
            return self._error("não é imagem")
        try:
            data = render_thumb(path)
        except (OSError, ValueError):
            return self._error("miniatura indisponível", 404)
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=60")
        self.end_headers()
        self.wfile.write(data)

    def _serve_sample(self, name: str, thumb: bool = False):
        """The gallery asks for thumb=1: a 1024px sample PNG is 1-2 MB and the
        grid cell is 150px, so a run with 40 epochs would push ~60 MB per view."""
        if not safe_sample_name(name):
            return self._error("nome de sample inválido")
        path = sample_dir() / name
        try:
            data = render_thumb(path, edge=340) if thumb else path.read_bytes()
        except (OSError, ValueError):
            return self._error("sample não encontrado", 404)
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg" if thumb else "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    def _status(self):
        job = jobs.current()
        project = current_project()
        stats = ds.inspect(dataset_dir("pos")) if project else {}
        stats_neg = ds.inspect(dataset_dir("neg")) if project else {}
        payload = {
            "project": project,
            "dataset": stats,
            "dataset_neg": stats_neg,
            "job": job.snapshot() if job else None,
            "gpu": gpu_info(),
            "hf_token": bool(hf_token()),
            "openrouter": bool(os.environ.get("OPENROUTER_API_KEY")),
            "trigger": load_state().get("trigger", ""),
            "dataset_convert": ds.inspect(dataset_dir("convert")) if project else {},
            "dataset_restore": ds.inspect(dataset_dir("restore")) if project else {},
            "style_rush_models": style_rush_models(),
            "engines": {name: is_installed(name) for name in ("musubi", "musubi-ltx", "sd-scripts", "captioner")},
        }
        if project and stats.get("items"):
            payload["schedules"] = {key: suggest_schedule(key, stats["items"]) for key in MODEL_ORDER}
        self._json(payload)

    def _presets(self):
        self._json({
            "models": public_presets(),
            "order": MODEL_ORDER,
            "caption_profiles": CAPTION_PROFILES,
            # Style Rush ignores suggest_schedule: showing the suggested numbers
            # in the panel would promise a run that is not the one that happens
            "style_rush_schedule": STYLE_RUSH_SCHEDULE,
        })

    # -- POST --------------------------------------------------------------
    def do_POST(self):
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/project":
                return self._set_project()
            if url.path == "/api/dataset/import":
                return self._import(query)
            if url.path == "/api/dataset/file":
                return self._upload_file(query)
            if url.path == "/api/dataset/clear":
                return self._clear_dataset(query)
            if url.path == "/api/captions":
                return self._captions()
            if url.path == "/api/train":
                return self._train()
            if url.path == "/api/cancel":
                job = jobs.current()
                if job and job.status == "running":
                    job.cancel()
                return self._json({"ok": True})
            if url.path == "/api/shutdown":
                return self._shutdown()
            return self._error("rota desconhecida", 404)
        except RuntimeError as exc:  # job slot busy
            return self._error(str(exc), 409)

    def _set_project(self):
        body = self._body_json()
        name = (body.get("name") or "").strip()
        if not name:
            return self._error("nome vazio")
        # only touch the trigger when the client actually sent one: the UI omits
        # it until it has read the stored value, so a fast typist cannot wipe it
        fields = {"project": name}
        if "trigger" in body:
            fields["trigger"] = (body.get("trigger") or "").strip()
        update_state(**fields)
        pdir = project_dir(name)
        (pdir / "dataset").mkdir(parents=True, exist_ok=True)
        self._json({"ok": True, "slug": slugify(name)})

    def _shutdown(self):
        """Stop the server from the UI. Without this the only way out is finding
        the pid on the pod, and a leftover process makes the next bootstrap fail
        with 'Address already in use'."""
        job = jobs.current()
        if job and job.status == "running" and not self._body_json().get("force"):
            return self._error("há um job rodando — cancele antes ou confirme para encerrar", 409)
        if job and job.status == "running":
            job.cancel()
        self._json({"ok": True})
        # shutdown() blocks until serve_forever returns, so it can never run on
        # the thread that is currently serving this request
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def _require_project(self) -> str | None:
        project = current_project()
        if not project:
            self._error("defina o nome do projeto primeiro")
            return None
        return project

    def _require_idle(self) -> bool:
        """Refuse to touch the dataset while a job is reading it."""
        job = jobs.current()
        if job and job.status == "running":
            self._error(f"{job.title} está rodando — cancele antes de mexer no dataset", 409)
            return False
        return True

    def _import(self, query):
        project = self._require_project()
        if not project:
            return
        source = (self._body_json().get("source") or "").strip()
        side = query.get("side", "pos")
        target = dataset_dir(side)
        pdir = project_dir(project)
        job = jobs.start("import", f"Importando dataset ({side})", pdir / "logs" / "import.log",
                         lambda j: ds.import_source(source, target, j))
        self._json({"ok": True, "job": job.kind}, 202)

    def _upload_file(self, query):
        project = self._require_project()
        if not project:
            return
        if not self._require_idle():
            return
        name = Path(query.get("name", "")).name  # basename only — no traversal
        if not name:
            return self._error("nome do arquivo ausente")
        side = query.get("side", "pos")
        target_root = dataset_dir(side)
        sub = query.get("dir", "")
        if sub == "control":
            target_root = target_root / "control"
        target_root.mkdir(parents=True, exist_ok=True)
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self._error("corpo vazio")
        dest = target_root / name
        with _UploadInFlight():
            with open(dest, "wb") as f:
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            ext = ds.archive_ext(name)
            if ext:
                import subprocess
                import uuid

                class _SyncJob:  # tiny shim: extraction logging goes nowhere useful here
                    def log(self, *_): pass

                    def run(self, cmd, **kw):
                        subprocess.run([str(c) for c in cmd], check=True,
                                       cwd=kw.get("cwd"), capture_output=True)

                # unique per request: parallel uploads of two zips must not share staging
                staging = target_root.parent / f"_upload_staging_{uuid.uuid4().hex}"
                try:
                    ds.extract_archive(dest, staging / "src", _SyncJob())
                    ds.normalize_into(staging, target_root, _SyncJob())
                except (JobFailed, subprocess.CalledProcessError, OSError) as exc:
                    return self._error(f"extração de {name} falhou: {exc}")
                finally:
                    dest.unlink(missing_ok=True)
                    shutil.rmtree(staging, ignore_errors=True)
        self._json({"ok": True, "stats": ds.inspect(dataset_dir(side))})

    def _clear_dataset(self, query):
        project = self._require_project()
        if not project:
            return
        if not self._require_idle():
            return
        ds.clear_dataset(project_dir(project), query.get("side", "pos"))
        self._json({"ok": True})

    def _captions(self):
        project = self._require_project()
        if not project:
            return
        body = self._body_json()
        profile = body.get("profile") or "default"
        side = body.get("side", "pos")
        media = "video" if ds.inspect(dataset_dir(side)).get("videos") else "image"
        prompt_vars = {}
        var = body.get("var_name")
        if var and body.get("trigger"):
            prompt_vars[var] = body["trigger"]
        pdir = project_dir(project)
        target = dataset_dir(side)
        job = jobs.start("captions", "Gerando captions com LLM", pdir / "logs" / "captions.log",
                         lambda j: generate_captions(target, media, profile, prompt_vars, j))
        self._json({"ok": True, "job": job.kind}, 202)

    def _train(self):
        project = self._require_project()
        if not project:
            return
        body = self._body_json()
        model_key = body.get("model")
        if model_key not in MODELS:
            return self._error(f"modelo inválido: {model_key}")
        pending = uploads_in_flight()
        if pending:
            return self._error(f"{pending} upload(s) ainda em andamento — espere terminar", 409)
        params = {
            "project": project,
            "model": model_key,
            "mode": body.get("mode", "lora"),
            "trigger": (body.get("trigger") or load_state().get("trigger") or "").strip(),
            "overrides": body.get("overrides") or {},
            "slider_targets": body.get("slider_targets") or [],
        }
        pdir = project_dir(project)
        title = f"Treinando {MODELS[model_key]['label']}"
        job = jobs.start("train", title, pdir / "logs" / "train.log",
                         lambda j: run_training(j, params))
        self._json({"ok": True, "job": job.kind}, 202)


def main():
    ensure_dirs()
    port = WEB_PORT
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    # without this a handler thread can keep the process alive after shutdown,
    # and the port stays taken until someone hunts down the pid
    server.daemon_threads = True
    print(f"⚔ Arrakis Trainero em http://0.0.0.0:{port}")
    try:
        server.serve_forever()
        print("Desligado pela interface. Até logo.")
    except KeyboardInterrupt:
        job = jobs.current()
        if job and job.status == "running":
            job.cancel()
        print("\nAté logo.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
