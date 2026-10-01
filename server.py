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
from trainero import style_rush as sr
from trainero import project as pj
from trainero.jobs import JobFailed
from trainero.captioner import caption_datasets
from trainero.config import (DATA_DIR, IMAGE_EXTS, PROJECTS_DIR, VIDEO_EXTS, WEB_PORT, ensure_dirs,
                             gpu_info, hf_token, load_state, save_state,
                             update_state)
from trainero.engines import is_installed
from trainero.presets import (CAPTION_PROFILES, LORA_KINDS, MODEL_ORDER, MODELS,
                              caption_profile, default_caption_profile,
                              STYLE_RUSH_SCHEDULE, public_presets,
                              style_rush_models, suggest_schedule)
from trainero.training import project_dir, run_training, slugify

WEB_DIR = Path(__file__).resolve().parent / "web"

# A job reads the dataset from the first phase to the last; anything that writes
# to it meanwhile changes the run under its feet. jobs.start() is the mutex for
# job-vs-job, and these two cover the writes that are plain HTTP requests:
# uploads/clears wait for the job slot, and a train or a clear waits for the
# uploads (a clear racing an in-flight upload would delete files mid-write).
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
                       if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTS)
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


def current_meta() -> dict:
    project = current_project()
    return pj.load_meta(project_dir(project)) if project else {}


def migrate_global_trigger() -> None:
    """Move the one trigger the global state may still hold into its project.

    Ran once at startup. Leaving it in state.json is what made the trigger
    follow the session instead of the dataset; leaving it in *both* places
    would be a second channel, so it is removed from state as it moves.
    """
    state = load_state()
    trigger, project = state.get("trigger"), state.get("project")
    if trigger and project:
        pdir = project_dir(project)
        if pdir.is_dir() and not pj.load_meta(pdir).get("trigger"):
            pj.save_meta(pdir, name=project, trigger=trigger)
    if "trigger" in state:
        state.pop("trigger")
        save_state(state)


def dataset_dir(side: str = "pos") -> Path:
    pdir = project_dir(current_project() or "projeto")
    return pdir / {"neg": "dataset_neg", "convert": "dataset_convert",
                   "restore": "dataset_restore"}.get(side, "dataset")


def disk_free_gb() -> float:
    try:
        return round(shutil.disk_usage(PROJECTS_DIR.parent).free / 1024**3, 1)
    except OSError:
        return -1.0


def shown_project() -> str:
    """The project whose job output is on screen.

    The progress card shows the current job; its samples and log belong to the
    job's own project. Reading them from the selected project instead showed
    another project's gallery (or none) under a running training as soon as
    the owner typed a new name.
    """
    job = jobs.current()
    return (job.project if job and job.project else "") or current_project() or "projeto"


def sample_dir() -> Path:
    return project_dir(shown_project()) / "output" / "sample"


def unique_upload_name(directory: Path, name: str) -> str:
    """`name`, or `stem_2.ext`, `stem_3.ext`… — the first free media name.

    A phone names every photo image.jpg, and a second batch reuses 1.png..N.png:
    replacing the file kept the old image's .txt beside the new image, so the
    run trained on captions describing pictures no longer there.
    """
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate, n = name, 2
    while (directory / candidate).exists() or (directory / f"{Path(candidate).stem}.txt").exists():
        candidate = f"{stem}_{n}{suffix}"
        n += 1
    return candidate


def same_bytes(a: Path, b: Path) -> bool:
    import filecmp

    try:
        return a.stat().st_size == b.stat().st_size and filecmp.cmp(a, b, shallow=False)
    except OSError:
        return False


def valid_upload_name(name: str) -> bool:
    return bool(name) and "\x00" not in name and name == Path(name).name \
        and not name.startswith(".") and "/" not in name and "\\" not in name


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

    def _content_length(self) -> int:
        try:
            return max(0, int(self.headers.get("Content-Length") or 0))
        except ValueError:
            return 0

    def _body_json(self) -> dict:
        """The request body as a dict; anything else reads as empty.

        Cached: a handler may call it twice, and the stream can only be read once.
        """
        if hasattr(self, "_body_cache"):
            return self._body_cache
        length = self._content_length()
        data = {}
        if 0 < length <= 1 << 20:
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                data = {}
        self._body_cache = data if isinstance(data, dict) else {}
        return self._body_cache

    def _guarded(self, handler) -> None:
        """Any error becomes a JSON answer instead of a dropped connection,
        which the browser can only report as "Failed to fetch"."""
        try:
            handler()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            try:
                self._error(f"erro interno: {type(exc).__name__}: {exc}", 500)
            except Exception:
                pass

    # -- GET ---------------------------------------------------------------
    def do_GET(self):
        self._guarded(self._do_get)

    def do_POST(self):
        self._guarded(self._do_post)

    def _do_get(self):
        url = urlparse(self.path)
        if url.path == "/api/status":
            return self._status()
        if url.path == "/api/presets":
            return self._presets()
        if url.path == "/api/projects":
            current = current_project()
            return self._json({"projects": pj.list_projects(),
                               "current": slugify(current) if current else ""})
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
        except Exception:  # unreadable, truncated, decompression bomb…
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
        except Exception:
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
        meta = current_meta()
        payload = {
            "project": project,
            "dataset": stats,
            "dataset_neg": stats_neg,
            "job": job.snapshot() if job else None,
            "gpu": gpu_info(),
            "hf_token": bool(hf_token()),
            "openrouter": bool(os.environ.get("OPENROUTER_API_KEY")),
            "trigger": meta.get("trigger", ""),
            "trigger_locked": bool(meta.get("origin")),
            "origin": (meta.get("origin") or {}).get("project", ""),
            "dataset_convert": ds.inspect(dataset_dir("convert")) if project else {},
            "dataset_restore": ds.inspect(dataset_dir("restore")) if project else {},
            "style_rush_models": style_rush_models(),
            "engines": {name: is_installed(name) for name in ("musubi", "musubi-ltx", "sd-scripts", "captioner")},
            "kind": meta.get("kind", ""),
            "caption_profile": meta.get("caption_profile", ""),
            "disk_free_gb": disk_free_gb(),
        }
        if project:
            media = "video" if stats.get("videos") and not stats.get("images") else "image"
            # which profile the job would write missing captions with, per model:
            # the UI says it before the click instead of the log saying it after
            chosen = caption_profile(meta.get("caption_profile") or "")
            payload["caption_plan"] = {
                key: (chosen["key"] if chosen and chosen["media"] == media
                      else default_caption_profile(key, meta.get("kind"), media))
                for key in MODEL_ORDER}
        if project and stats.get("items"):
            payload["schedules"] = {key: suggest_schedule(key, stats["items"]) for key in MODEL_ORDER}
        self._json(payload)

    def _presets(self):
        self._json({
            "models": public_presets(),
            "order": MODEL_ORDER,
            "caption_profiles": CAPTION_PROFILES,
            "lora_kinds": LORA_KINDS,
            # Style Rush ignores suggest_schedule: showing the suggested numbers
            # in the panel would promise a run that is not the one that happens
            "style_rush_schedule": STYLE_RUSH_SCHEDULE,
            "default_convert_target": sr.DEFAULT_CONVERT_TARGET,
        })

    # -- POST --------------------------------------------------------------
    def _do_post(self):
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/project":
                return self._set_project()
            if url.path == "/api/project/fork":
                return self._fork_project()
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
        switching = slugify(name) != slugify(current_project() or "")
        update_state(project=name)
        pdir = project_dir(name)
        (pdir / "dataset").mkdir(parents=True, exist_ok=True)
        fields = {"name": name}
        # The trigger only travels with a save of the SAME project. On a switch
        # the field still shows the previous project's word, and writing it
        # would give project B the trigger of project A — the server ignores it
        # and answers with B's stored value for the UI to show.
        if "trigger" in body and not switching and not pj.trigger_locked(pdir):
            fields["trigger"] = (body.get("trigger") or "").strip()
        if body.get("kind") in LORA_KINDS:
            fields["kind"] = body["kind"]
        if "caption_profile" in body:
            prof = body.get("caption_profile") or ""
            if prof and caption_profile(prof) is None:
                return self._error(f"perfil de caption desconhecido: {prof}")
            fields["caption_profile"] = prof
        meta = pj.save_meta(pdir, **fields)
        self._json({"ok": True, "slug": slugify(name), "switched": switching,
                    "trigger": meta.get("trigger", ""), "kind": meta.get("kind", ""),
                    "caption_profile": meta.get("caption_profile", "")})

    def _fork_project(self):
        """Fill the current (empty) project with another project's dataset."""
        project = self._require_project()
        if not project:
            return
        if not self._require_idle():
            return
        source = (self._body_json().get("source") or "").strip()
        if not source:
            return self._error("escolha o projeto de origem")
        try:
            copied = pj.fork_dataset(project_dir(source), project_dir(project))
        except pj.ForkError as exc:
            return self._error(str(exc), 409)
        except OSError as exc:
            return self._error(f"cópia falhou: {exc}", 500)
        self._json({"ok": True, "copied": copied,
                    "stats": ds.inspect(dataset_dir("pos"))})

    def _shutdown(self):
        """Stop the server from the UI. Without this the only way out is finding
        the pid on the pod, and a leftover process makes the next bootstrap fail
        with 'Address already in use'."""
        job = jobs.current()
        if job and job.status == "running" and not self._body_json().get("force"):
            return self._error("há um job rodando — cancele antes ou confirme para encerrar", 409)
        self._json({"ok": True})
        # shutdown() blocks until serve_forever returns, so it can never run on
        # the thread that is currently serving this request
        def stop():
            stop_running_job()
            self.server.shutdown()

        threading.Thread(target=stop, daemon=True).start()

    def _require_project(self) -> str | None:
        project = current_project()
        if not project:
            self._error("defina o nome do projeto primeiro")
            return None
        return project

    def _require_no_uploads(self) -> bool:
        pending = uploads_in_flight()
        if pending:
            self._error(f"{pending} upload(s) ainda em andamento — espere terminar", 409)
            return False
        return True

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
        if not self._require_no_uploads():
            return
        source = (self._body_json().get("source") or "").strip()
        side = query.get("side", "pos")
        target = dataset_dir(side)
        pdir = project_dir(project)
        job = jobs.start("import", f"Importando dataset ({side})", pdir / "logs" / "import.log",
                         lambda j: ds.import_source(source, target, j), project=project)
        self._json({"ok": True, "job": job.kind}, 202)

    def _upload_file(self, query):
        project = self._require_project()
        if not project:
            return
        if not self._require_idle():
            return
        name = query.get("name", "")
        if not valid_upload_name(name):
            return self._error(f"nome de arquivo inválido: {name!r}")
        side = query.get("side", "pos")
        target_root = dataset_dir(side)
        sub = query.get("dir", "")
        if sub == "control":
            target_root = target_root / "control"
        target_root.mkdir(parents=True, exist_ok=True)
        length = self._content_length()
        if length <= 0:
            return self._error("corpo vazio")
        dest = target_root / name
        # staged next to the destination: the trainer and inspect() glob the
        # dataset dir by media extension, so a half-received file must never
        # carry a media name — a dropped connection would leave a truncated
        # image that trains without a single error
        tmp = target_root / f".{name}.uploading"
        with _UploadInFlight():
            try:
                with open(tmp, "wb") as f:
                    remaining = length
                    while remaining > 0:
                        chunk = self.rfile.read(min(1 << 20, remaining))
                        if not chunk:
                            break
                        f.write(chunk)
                        remaining -= len(chunk)
                if remaining > 0:
                    return self._error(f"upload de {name} truncado "
                                       f"(faltaram {remaining} bytes) — envie de novo")
                is_media = Path(name).suffix.lower() in (IMAGE_EXTS | VIDEO_EXTS)
                if is_media and dest.exists():
                    if same_bytes(tmp, dest):
                        # the same file sent twice: nothing new, and the
                        # caption beside it still describes it
                        return self._json({"ok": True, "name": name, "duplicate": True,
                                           "stats": ds.inspect(dataset_dir(side))})
                    dest = target_root / unique_upload_name(target_root, name)
                os.replace(tmp, dest)
                if dest.suffix.lower() in IMAGE_EXTS:
                    ds.bake_orientation(dest)
            except OSError as exc:
                return self._error(f"não consegui gravar {name}: {exc}", 507)
            finally:
                tmp.unlink(missing_ok=True)
            ext = ds.archive_ext(name)
            if ext:
                import uuid

                # unique per request: parallel uploads of two zips must not share staging
                staging = target_root.parent / f"_upload_staging_{uuid.uuid4().hex}"
                try:
                    sync = jobs.SyncJob()
                    ds.extract_archive(dest, staging / "src", sync)
                    ds.normalize_into(staging, target_root, sync)
                except (JobFailed, OSError) as exc:
                    return self._error(f"extração de {name} falhou: {exc}")
                finally:
                    dest.unlink(missing_ok=True)
                    shutil.rmtree(staging, ignore_errors=True)
        self._json({"ok": True, "name": dest.name, "stats": ds.inspect(dataset_dir(side))})

    def _clear_dataset(self, query):
        project = self._require_project()
        if not project:
            return
        if not self._require_idle():
            return
        pending = uploads_in_flight()
        if pending:
            return self._error(f"{pending} upload(s) ainda em andamento — "
                               f"espere terminar antes de limpar", 409)
        side = query.get("side", "pos")
        pdir = project_dir(project)
        ds.clear_dataset(pdir, side)
        if side == "pos":
            pj.clear_origin(pdir)
        self._json({"ok": True})

    def _captions(self):
        project = self._require_project()
        if not project:
            return
        if not self._require_no_uploads():
            return
        body = self._body_json()
        meta = current_meta()
        redo = bool(body.get("redo"))
        trigger = (body.get("trigger") or meta.get("trigger") or "").strip()
        profile = body.get("profile") or None
        if profile and caption_profile(profile) is None:
            return self._error(f"perfil de caption desconhecido: {profile}")
        sides = [sd for sd in (body.get("sides") or ["pos", "neg"]) if sd in ("pos", "neg")]
        # every side that needs it, in one job: the card counts both, so one
        # click has to fix both (it used to caption only the first)
        dirs = [dataset_dir(sd) for sd in sides
                if (ds.inspect(dataset_dir(sd))["items"] if redo
                    else ds.inspect(dataset_dir(sd))["missing_captions"])]
        if not dirs:
            return self._error("nada para legendar")
        pdir = project_dir(project)
        model = body.get("model") if body.get("model") in MODELS else None
        kind = meta.get("kind")

        def run(j):
            caption_datasets(j, dirs, model_key=model, kind=kind, trigger=trigger,
                             profile=profile, redo=redo)
            if redo:
                # every caption was rewritten with the trigger from step 01: an
                # inherited trigger no longer describes anything on disk
                pj.clear_origin(pdir)

        job = jobs.start("captions", "Gerando captions com LLM",
                         pdir / "logs" / "captions.log", run, project=project)
        self._json({"ok": True, "job": job.kind}, 202)

    def _train(self):
        project = self._require_project()
        if not project:
            return
        body = self._body_json()
        meta = current_meta()
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
            "trigger": (body.get("trigger") or meta.get("trigger") or "").strip(),
            "overrides": body.get("overrides") if isinstance(body.get("overrides"), dict) else {},
            "slider_targets": body.get("slider_targets") or [],
            # what the captions are written with, if the job has to write them
            "kind": meta.get("kind") or "",
            "caption_profile": meta.get("caption_profile") or "",
        }
        if params["mode"] not in ("lora", "slider", "style-rush"):
            return self._error(f"modo inválido: {params['mode']}")
        pdir = project_dir(project)
        title = f"Treinando {MODELS[model_key]['label']}"
        job = jobs.start("train", title, pdir / "logs" / "train.log",
                         lambda j: run_training(j, params), project=project)
        self._json({"ok": True, "job": job.kind}, 202)


def stop_running_job(timeout: float = 30.0) -> None:
    """Cancel the job and wait for its process group to go.

    Without the wait the server process exits first, the trainer keeps the GPU
    with a dead stdout pipe, and the next server starts a second training into
    an out-of-memory.
    """
    import time

    job = jobs.current()
    if not job or job.status != "running":
        return
    job.cancel()
    deadline = time.monotonic() + timeout
    while job.status == "running" and time.monotonic() < deadline:
        time.sleep(0.5)


def main():
    import signal

    ensure_dirs()
    migrate_global_trigger()
    port = WEB_PORT
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    # without this a handler thread can keep the process alive after shutdown,
    # and the port stays taken until someone hunts down the pid
    server.daemon_threads = True

    def on_signal(signum, _frame):
        # a bootstrap re-run or `kill` arrives as SIGTERM
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_signal)
    # a closed web terminal must not cancel a two-hour training
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    # bootstrap.sh stops exactly this process through the pidfile instead of a
    # `pkill -f server.py` that would also hit any other server.py on the pod
    pidfile = DATA_DIR / "server.pid"
    pidfile.write_text(str(os.getpid()))
    print(f"⚔ Arrakis Trainero em http://0.0.0.0:{port}")
    try:
        server.serve_forever()
        print("Desligado pela interface. Até logo.")
    except KeyboardInterrupt:
        print("\nEncerrando — cancelando o job em andamento, se houver...")
        stop_running_job()
        print("Até logo.")
    finally:
        server.server_close()
        try:
            if pidfile.read_text().strip() == str(os.getpid()):
                pidfile.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
