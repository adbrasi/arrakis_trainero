#!/usr/bin/env python3
"""Arrakis Trainero — web server (stdlib only, arrakis_start pattern).

Single-port HTTP: static UI + JSON API, polling instead of websockets
(cloud pods proxy exactly one port). Run: python server.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
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
                              public_presets, suggest_schedule)
from trainero.training import project_dir, run_training, slugify

WEB_DIR = Path(__file__).resolve().parent / "web"


def current_project() -> str:
    return load_state().get("project", "")


def dataset_dir(side: str = "pos") -> Path:
    pdir = project_dir(current_project() or "projeto")
    return pdir / ("dataset" if side != "neg" else "dataset_neg")


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
        return super().do_GET()

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
            return self._error("rota desconhecida", 404)
        except RuntimeError as exc:  # job slot busy
            return self._error(str(exc), 409)

    def _set_project(self):
        name = (self._body_json().get("name") or "").strip()
        if not name:
            return self._error("nome vazio")
        update_state(project=name)
        pdir = project_dir(name)
        (pdir / "dataset").mkdir(parents=True, exist_ok=True)
        self._json({"ok": True, "slug": slugify(name)})

    def _require_project(self) -> str | None:
        project = current_project()
        if not project:
            self._error("defina o nome do projeto primeiro")
            return None
        return project

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
        side = query.get("side", "pos")
        shutil.rmtree(dataset_dir(side), ignore_errors=True)
        dataset_dir(side).mkdir(parents=True, exist_ok=True)
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
        params = {
            "project": project,
            "model": model_key,
            "mode": body.get("mode", "lora"),
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
    print(f"⚔ Arrakis Trainero em http://0.0.0.0:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        job = jobs.current()
        if job and job.status == "running":
            job.cancel()
        print("\nAté logo.")


if __name__ == "__main__":
    main()
