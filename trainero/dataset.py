"""Dataset import + inspection.

One input field accepts: local folder, local archive, HTTP(S) URL, Mega link,
HuggingFace repo (user/repo) or HF file URL. Browser uploads land in the same
staging dir. The source is NEVER mutated — everything is copied/extracted into
projects/<name>/dataset/. A `control/` subfolder survives the flatten as
<dataset>/control/ (edit-model control images, paired by stem).

Patterns ported from character_animatrem (mega dir-diff, wrapper descent,
archive-in-snapshot extraction, media+.txt pairing).
"""

from __future__ import annotations

import os
import random
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

from .config import IMAGE_EXTS, VIDEO_EXTS, hf_token
from .engines import ensure_system_tool
from .jobs import Job, JobFailed

ARCHIVE_EXTS = (".zip", ".rar", ".7z", ".tar", ".tar.gz", ".tgz", ".tar.xz", ".tar.bz2", ".tbz2")
_HF_FILE_RE = re.compile(r"^https?://huggingface\.co/(datasets/)?([\w\-.]+/[\w\-.]+)/(?:resolve|blob)/[^/]+/(.+?)(?:\?.*)?$")


# ---------------------------------------------------------------------------
# Source detection
# ---------------------------------------------------------------------------

def detect_source(value: str) -> str:
    v = value.strip()
    if not v:
        return "empty"
    if v.startswith(("https://mega.nz/", "https://mega.io/", "mega://")):
        return "mega"
    if _HF_FILE_RE.match(v):
        return "hf-file"
    if re.match(r"^https?://", v):
        return "url"
    p = Path(v).expanduser()
    if p.is_dir():
        return "local-dir"
    if p.is_file() and archive_ext(p.name):
        return "local-archive"
    if "/" in v and not v.startswith(("/", ".")) and len(v.split("/")) == 2 \
            and all(re.match(r"^[\w\-.]+$", part) for part in v.split("/")):
        return "hf-repo"
    return "unknown"


def archive_ext(name: str) -> str | None:
    low = name.lower()
    for ext in ARCHIVE_EXTS:
        if low.endswith(ext):
            return ext
    return None


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def import_source(source: str, dataset_dir: Path, job: Job) -> None:
    kind = detect_source(source)
    job.log(f"Fonte detectada: {kind}")
    staging = dataset_dir.parent / "_staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    # a failed or cancelled download must not leave GBs behind in the project
    try:
        _import_into(kind, source, staging, dataset_dir, job)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _import_into(kind: str, source: str, staging: Path, dataset_dir: Path, job: Job) -> None:
    if kind == "empty":
        raise JobFailed("fonte vazia")
    elif kind == "unknown":
        raise JobFailed(f"não reconheci a fonte: {source!r} (pasta, zip, URL, Mega ou user/repo do HF)")
    elif kind == "local-dir":
        src = Path(source).expanduser().resolve()
        project = dataset_dir.parent.resolve()
        if src == project or project in src.parents or src in project.parents:
            raise JobFailed("essa pasta contém (ou está dentro de) o próprio projeto — "
                            "aponte para a pasta das imagens")
        _copy_media_tree(src, staging / "src")
    elif kind == "local-archive":
        extract_archive(Path(source).expanduser(), staging / "src", job)
    elif kind == "hf-repo":
        _hf_snapshot(source, staging / "src", job)
    elif kind == "hf-file":
        m = _HF_FILE_RE.match(source.strip())
        is_ds, repo, fname = bool(m.group(1)), m.group(2), m.group(3)
        _hf_file(repo, fname, is_ds, staging, job)
    elif kind == "mega":
        _mega(source.strip(), staging, job)
    elif kind == "url":
        _http(source.strip(), staging, job)

    _resolve_archives(staging, job)
    normalize_into(staging, dataset_dir, job)


def _copytree(src: Path, dest: Path) -> None:
    shutil.copytree(src, dest, dirs_exist_ok=True)


def _copy_media_tree(src: Path, dest: Path) -> None:
    """Copy only what an import keeps (media, captions, archives).

    A whole-tree copy of a local folder also copied checkpoints, caches and
    anything else living there — tens of GB — only to throw them away.
    """
    keep = IMAGE_EXTS | VIDEO_EXTS | {".txt"}
    for f in src.rglob("*"):
        if not f.is_file() or is_junk(f.relative_to(src)):
            continue
        if f.suffix.lower() not in keep and not archive_ext(f.name):
            continue
        target = dest / f.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)


def is_junk(rel: Path) -> bool:
    """macOS resource forks (`__MACOSX/`, `._name`) and hidden files.

    Finder puts a `._001.png` next to every image of a zip; it has an image
    extension, no caption and no pixels, and it used to count as a dataset item.
    """
    return any(part == "__MACOSX" or part.startswith(".") for part in rel.parts)


def _hf_snapshot(repo: str, dest: Path, job: Job) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError

    from huggingface_hub.errors import GatedRepoError

    job.log(f"Baixando repo HF {repo}...")
    token = hf_token()
    for repo_type in ("dataset", "model"):
        try:
            snapshot_download(repo, repo_type=repo_type, local_dir=str(dest), token=token)
            return
        except GatedRepoError:  # a subclass of RepositoryNotFoundError
            raise JobFailed(f"{repo} é gated — aceite os termos no HuggingFace com a conta "
                            f"do HF_TOKEN e tente de novo")
        except RepositoryNotFoundError:
            continue
    raise JobFailed(f"repo HF não encontrado (ou privado sem acesso pelo HF_TOKEN): {repo}")


def _hf_file(repo: str, fname: str, is_dataset: bool, staging: Path, job: Job) -> None:
    from huggingface_hub import hf_hub_download

    job.log(f"Baixando arquivo HF {repo}/{fname}...")
    path = hf_hub_download(repo, filename=fname, token=hf_token(),
                           repo_type="dataset" if is_dataset else "model",
                           local_dir=str(staging / "_dl"))
    _place_downloaded(Path(path), staging, job)


def _mega(url: str, staging: Path, job: Job) -> None:
    tool = None
    if shutil.which("mega-get"):
        tool = "mega-get"
    elif shutil.which("megadl"):
        tool = "megadl"
    elif ensure_system_tool("megadl", "megatools", job):
        tool = "megadl"
    if not tool:
        raise JobFailed("instale 'megatools' ou 'megacmd' para baixar do Mega")
    dl = staging / "_dl"
    dl.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in dl.iterdir()}
    job.log(f"Baixando do Mega via {tool}...")
    if tool == "mega-get":
        job.run([tool, url, str(dl)])
    else:
        job.run([tool, "--path", str(dl), url])
    new = [dl / n for n in ({p.name for p in dl.iterdir()} - before)]
    if not new:
        raise JobFailed("o download do Mega não produziu nada")
    for item in new:
        _place_downloaded(item, staging, job)


def _http(url: str, staging: Path, job: Job) -> None:
    dl = staging / "_dl"
    dl.mkdir(parents=True, exist_ok=True)
    m = re.search(r"/([^/?#]+)(?:[?#]|$)", url)
    fname = m.group(1) if m else "dataset.zip"
    if not archive_ext(fname):
        fname += ".zip"
    dest = dl / fname
    job.log(f"Baixando {fname}...")
    if shutil.which("aria2c"):
        job.run(["aria2c", "-x", "8", "--console-log-level=warn", "--summary-interval=15",
                 "-o", fname, "-d", str(dl), url])
    elif shutil.which("wget"):
        job.run(["wget", "-q", "--show-progress", "-O", str(dest), url])
    else:
        job.run(["curl", "-L", "--progress-bar", "-o", str(dest), url])
    _place_downloaded(dest, staging, job)


def _place_downloaded(item: Path, staging: Path, job: Job) -> None:
    if item.is_dir():
        _copytree(item, staging / "src")
    elif archive_ext(item.name):
        extract_archive(item, staging / "src", job)
    else:
        (staging / "src").mkdir(parents=True, exist_ok=True)
        shutil.move(str(item), staging / "src" / item.name)


def extract_archive(archive: Path, dest: Path, job: Job) -> None:
    ext = archive_ext(archive.name)
    if ext is None:
        raise JobFailed(f"extensão não reconhecida: {archive.name}")
    dest.mkdir(parents=True, exist_ok=True)
    job.log(f"Extraindo {archive.name}...")
    if ext == ".zip":
        if not ensure_system_tool("unzip", "unzip", job):
            raise JobFailed("unzip indisponível")
        job.run(["unzip", "-q", "-o", str(archive), "-d", str(dest)])
    elif ext == ".rar":
        if not ensure_system_tool("unrar", "unrar", job):
            raise JobFailed("unrar indisponível")
        job.run(["unrar", "x", "-o+", "-idq", str(archive), str(dest) + os.sep])
    elif ext == ".7z":
        if not ensure_system_tool("7z", "p7zip-full", job):
            raise JobFailed("p7zip indisponível")
        job.run(["7z", "x", f"-o{dest}", "-y", str(archive)])
    else:
        # archives also come from URLs/Mega/HF — never trust member paths
        with tarfile.open(archive, "r:*") as tf:
            try:
                tf.extractall(dest, filter="data")
            except TypeError:  # Python < 3.12: validate members manually
                base = dest.resolve()
                for member in tf.getmembers():
                    target = (dest / member.name).resolve()
                    if base not in target.parents and target != base:
                        raise JobFailed(f"caminho suspeito no tar: {member.name}")
                tf.extractall(dest)


def _resolve_archives(root: Path, job: Job) -> None:
    """HF snapshots sometimes ship .zip inside the repo — extract in place."""
    for arc in list(root.rglob("*")):
        if arc.is_file() and archive_ext(arc.name):
            marker = arc.with_suffix(arc.suffix + ".extracted")
            if marker.exists():
                continue
            extract_archive(arc, arc.parent, job)
            marker.touch()
            arc.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Normalization: staging -> flat dataset dir (+ control/)
# ---------------------------------------------------------------------------

def _descend_wrapper(root: Path) -> Path:
    """my_dataset.zip/my_dataset/... — walk down single-folder wrappers."""
    cur = root
    for _ in range(8):
        entries = [p for p in cur.iterdir() if not p.name.startswith(".")]
        dirs = [p for p in entries if p.is_dir()]
        files = [p for p in entries if p.is_file() and archive_ext(p.name) is None
                 and not p.name.endswith(".extracted")]
        if len(dirs) == 1 and not files:
            cur = dirs[0]
        else:
            break
    return cur


def normalize_into(staging: Path, dataset_dir: Path, job: Job) -> None:
    src = _descend_wrapper(staging)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    control_dir = dataset_dir / "control"
    moved = 0
    media_exts = IMAGE_EXTS | VIDEO_EXTS
    for f in sorted(src.rglob("*")):
        if not f.is_file() or is_junk(f.relative_to(src)):
            continue
        ext = f.suffix.lower()
        in_control = "control" in [p.name.lower() for p in f.relative_to(src).parents if p.name]
        if ext in media_exts:
            target_root = control_dir if in_control else dataset_dir
            target_root.mkdir(parents=True, exist_ok=True)
            dest = target_root / f.name
            if dest.exists():
                dest = target_root / f"{f.parent.name}_{f.name}"
            if dest.exists():
                continue
            shutil.move(str(f), str(dest))
            moved += 1
            txt = f.with_suffix(".txt")
            if txt.exists() and not in_control:
                txt_dest = dest.with_suffix(".txt")
                if not txt_dest.exists():
                    shutil.move(str(txt), str(txt_dest))
        elif ext == ".txt" and not in_control:
            # orphan caption whose media may arrive later in the walk — copy
            # only if a same-stem media file already landed
            for m_ext in media_exts:
                if (dataset_dir / (f.stem + m_ext)).exists():
                    dest = dataset_dir / f.name
                    if not dest.exists():
                        shutil.move(str(f), str(dest))
                    break
    job.log(f"✔ {moved} arquivos de mídia importados.")
    if moved == 0:
        raise JobFailed("nenhuma imagem/vídeo encontrado na fonte")


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def captions_map(dataset_dir: Path) -> dict[str, str]:
    """{media filename: caption} for everything in the dataset that has one."""
    out: dict[str, str] = {}
    if not dataset_dir.exists():
        return out
    for f in sorted(dataset_dir.iterdir()):
        if not is_media_item(f):
            continue
        txt = f.with_suffix(".txt")
        try:
            caption = txt.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if caption:
            out[f.name] = caption
    return out


# Fixed so the sample prompt survives a resume. The sample is only worth looking
# at because it is the same prompt at every epoch.
SAMPLE_CAPTION_SEED = 8821


def sample_caption(dataset_dir: Path) -> str:
    """One caption from the dataset, to stand in as the sample prompt.

    A caption from the dataset is the only prompt guaranteed to sit inside the
    distribution being trained: it carries, by construction, whatever trigger
    and vocabulary the dataset uses. Empty string when nothing has a caption.
    """
    captions = sorted(set(captions_map(Path(dataset_dir)).values()))
    if not captions:
        return ""
    return random.Random(SAMPLE_CAPTION_SEED).choice(captions)


#: built FROM the positive dataset, so they cannot outlive it
DERIVED_DIRS = ("dataset_convert", "dataset_restore", "cache")


def clear_dataset(project_dir: Path, side: str) -> None:
    """Empty one side of the dataset, and everything derived from it.

    The synthetic datasets and the latent/TE caches are functions of the images
    in `dataset/`. Their manifests only ever verify that their own files exist,
    never where they came from — so replacing the dataset and leaving them
    behind makes the next Style Rush announce "já completo (50 pares)" and train
    150 pairs built from images that are no longer on disk. Nothing fails, and
    trainero_config.json records the wrong thing as fact.
    """
    import shutil

    target = project_dir / {"neg": "dataset_neg", "convert": "dataset_convert",
                            "restore": "dataset_restore"}.get(side, "dataset")
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)
    if side == "pos":
        for derived in DERIVED_DIRS:
            shutil.rmtree(project_dir / derived, ignore_errors=True)


def captions_digest(dataset_dir: Path) -> str:
    """Fingerprint of every caption in the directory, name included.

    Cheap enough to take on each run and the only way to notice that a cached
    text-encoder output no longer matches the text it was built from — the
    cache file name carries nothing about the caption.
    """
    import hashlib

    h = hashlib.sha256()
    for name, caption in sorted(captions_map(dataset_dir).items()):
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(caption.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def is_media_item(f: Path) -> bool:
    """A training item: a visible media file. Dotfiles are never items — that
    covers macOS `._x.png` forks and the `.x.png.uploading` staging names."""
    return f.is_file() and not f.name.startswith(".") \
        and f.suffix.lower() in (IMAGE_EXTS | VIDEO_EXTS)


def has_caption(item: Path) -> bool:
    """One rule for "this item is captioned", used by inspect and by the
    captioner's retry loop — two copies of it would drift."""
    txt = item.with_suffix(".txt")
    try:
        return txt.stat().st_size > 0
    except OSError:
        return False


def uncaptioned(dataset_dir: Path) -> list[Path]:
    """Every media item still without a caption, as paths."""
    if not dataset_dir.exists():
        return []
    return [f for f in sorted(dataset_dir.iterdir())
            if is_media_item(f) and not has_caption(f)]


def inspect(dataset_dir: Path) -> dict:
    images, videos, captions, missing = 0, 0, 0, []
    control = 0
    if dataset_dir.exists():
        for f in sorted(dataset_dir.iterdir()):
            if not is_media_item(f):
                continue
            ext = f.suffix.lower()
            if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
                if ext in IMAGE_EXTS:
                    images += 1
                else:
                    videos += 1
                if has_caption(f):
                    captions += 1
                else:
                    missing.append(f.name)
        cdir = dataset_dir / "control"
        if cdir.exists():
            control = sum(1 for f in cdir.iterdir()
                          if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
    return {
        "images": images,
        "videos": videos,
        "items": images + videos,
        "captions": captions,
        "missing_captions": len(missing),
        "missing_sample": missing[:8],
        "control_images": control,
    }
