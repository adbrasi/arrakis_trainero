"""Per-project metadata and dataset forking.

A project is a directory under PROJECTS_DIR plus a `project.json` holding what
belongs to the project rather than to the browser session: its display name,
the trigger word its captions were written with, and — when the dataset was
copied from another project — where it came from.

The trigger used to live in the global state file, right next to the name of
the currently selected project. That is correct only while there is exactly one
project: selecting a second one kept the first one's trigger, so a run could
train captions that say `arkstyle` while sampling `novastyle`, with nothing
failing. The trigger describes what is written inside the .txt files, so it
lives with them.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from . import config
from .config import IMAGE_EXTS, VIDEO_EXTS

META_NAME = "project.json"

#: everything a second model can reuse as-is. `cache/` and `output/` are absent
#: on purpose: they are functions of the model, which is the whole reason the
#: copy is being made.
FORKABLE = ("dataset", "dataset_neg", "dataset_convert", "dataset_restore")


class ForkError(Exception):
    """A copy that would destroy or silently mix datasets."""


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------

def load_meta(pdir: Path) -> dict:
    try:
        data = json.loads((pdir / META_NAME).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_meta(pdir: Path, meta: dict) -> dict:
    pdir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(pdir), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, pdir / META_NAME)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise
    return meta


def save_meta(pdir: Path, **fields) -> dict:
    """Merge `fields` into the project metadata, atomically."""
    meta = load_meta(pdir)
    meta.update(fields)
    return _write_meta(pdir, meta)


def clear_origin(pdir: Path) -> None:
    """Forget that the dataset was a copy, releasing the trigger.

    Called when the dataset itself is cleared: with the inherited images and
    captions gone there is nothing left for the inherited trigger to describe.
    """
    meta = load_meta(pdir)
    if meta.pop("origin", None) is not None:
        _write_meta(pdir, meta)


def trigger_locked(pdir: Path) -> bool:
    """A copied dataset arrives with the trigger already written into every
    caption on disk. Accepting a new word there would train one thing and
    sample another — the honest way to change it is to caption again, which
    rewrites those files and is what releases the lock."""
    return bool(load_meta(pdir).get("origin"))


# ---------------------------------------------------------------------------
# listing
# ---------------------------------------------------------------------------

def _count_media(directory: Path) -> int:
    try:
        return sum(1 for f in directory.iterdir()
                   if f.is_file() and f.suffix.lower() in (IMAGE_EXTS | VIDEO_EXTS))
    except OSError:
        return 0


def describe(pdir: Path) -> dict:
    """What the picker needs to show one project as a dataset source."""
    meta = load_meta(pdir)
    return {
        "slug": pdir.name,
        "name": meta.get("name") or pdir.name,
        "trigger": meta.get("trigger", ""),
        "items": _count_media(pdir / "dataset"),
        "convert": _count_media(pdir / "dataset_convert"),
        "restore": _count_media(pdir / "dataset_restore"),
        "origin": (meta.get("origin") or {}).get("project", ""),
    }


def list_projects() -> list[dict]:
    """Every project on disk, biggest dataset first.

    Projects with no images are still listed — the UI needs to show that the
    name is taken — but they are not offerable as a copy source.
    """
    # read through the module: the tests (and WORKSPACE_DIR) rebind it
    try:
        dirs = [d for d in config.PROJECTS_DIR.iterdir() if d.is_dir()]
    except OSError:
        return []
    out = [describe(d) for d in dirs]
    out.sort(key=lambda p: (-p["items"], p["slug"]))
    return out


# ---------------------------------------------------------------------------
# forking
# ---------------------------------------------------------------------------

def _has_content(pdir: Path) -> str:
    """Name of the first forkable directory that is not empty, or ""."""
    for name in FORKABLE:
        if _count_media(pdir / name):
            return name
    return ""


def fork_dataset(src: Path, dst: Path) -> dict:
    """Copy every model-independent directory of `src` into the empty `dst`.

    A copy, not a symlink or a hardlink. A reference makes two projects share
    mutable files, and the captions are plain .txt sitting next to the images —
    there is no way to share the images without sharing those too. Then
    "Limpar dataset" on the copy empties the original, and editing one caption
    edits both. A few hundred MB and a few seconds buys each project ownership
    of what it holds.
    """
    if not src.is_dir():
        raise ForkError(f"projeto de origem não existe: {src.name}")
    if src.resolve() == dst.resolve():
        raise ForkError("origem e destino são o mesmo projeto")
    if not _count_media(src / "dataset"):
        raise ForkError(f"{src.name} não tem imagens para copiar")
    occupied = _has_content(dst)
    if occupied:
        raise ForkError(f"{dst.name} já tem {occupied} — limpe antes de copiar")

    copied = {}
    for name in FORKABLE:
        source = src / name
        if not source.is_dir() or not any(source.iterdir()):
            continue
        target = dst / name
        # stage under a sibling name and rename: a copy interrupted halfway
        # would otherwise leave a dataset that looks complete and trains short
        staging = dst / f".{name}.incoming"
        shutil.rmtree(staging, ignore_errors=True)
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, staging)
        shutil.rmtree(target, ignore_errors=True)
        os.replace(staging, target)
        copied[name] = _count_media(target)

    meta = load_meta(src)
    save_meta(dst, trigger=meta.get("trigger", ""),
              origin={"project": src.name, "trigger": meta.get("trigger", "")})
    return copied
