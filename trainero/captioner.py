"""LLM captioning via captionkreator (`captioner run`), with this repo's profiles.

captionkreator is the engine: discovery, WD14 booru tags, the OpenRouter model
cascade, JSON extraction, validation, resume state and cost accounting all live
there. What lives here is what is specific to training a LoRA:

- the prompt profiles (caption_profiles/<media>/<name>/), passed with
  --prompts-dir, so a caption is written for the model being trained;
- the decision of what to do with an item no model would caption: it blocks
  the whole training, so it leaves the dataset — but only after proving the
  OpenRouter account is alive, because a dead key looks exactly like every
  model refusing;
- the list of items the primary model refused, which Style Rush reads to avoid
  paying gpt-image-2 for slots it would refuse too.

The cascade is the profile's `model` + `fallback`. A refusal is not retried on
the same model forever (profiles set retries: 1); it moves down the chain.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from . import dataset as ds
from .config import IMAGE_EXTS, REPO_DIR, VIDEO_EXTS
from .engines import engine_dir, ensure_engine, venv_bin
from .jobs import Job, JobFailed
from .presets import caption_profile, default_caption_profile

PROFILES_DIR = REPO_DIR / "caption_profiles"
FLAGGED_FILE = ".caption_refused.json"
QUARANTINE_DIR = "descartadas"
#: written by captionkreator inside the dataset; all of it describes the captions
STATE_FILES = (".captioner.sqlite", ".captioner.sqlite-wal", ".captioner.sqlite-shm",
               "captions.json")
#: bookkeeping of the previous engine (data_araknideo); cleared with the captions
LEGACY_FILES = (".tagger_log.json",)

# One knob for the whole cascade; captionkreator paces itself down on 429s.
CAPTION_CONCURRENCY = int(os.environ.get("CAPTION_CONCURRENCY", "64"))

MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS

_PROGRESS_RE = re.compile(r"^progress: (\d+)/(\d+)")
_COST_RE = re.compile(r"cost \$([0-9.]+)")


def clear_captions(dataset_dir, job: Job | None = None) -> int:
    """Delete every caption in the dataset so the next run writes them all again.

    The .txt alone is not enough: captionkreator restores a missing sidecar from
    its state without asking the model again — the property that makes an
    interrupted run free to resume. Its state, the refusal list and captions.json
    are conclusions drawn about the old captions, so they go together.

    QUARANTINE_DIR is not touched: it sits beside the dataset and is the owner's
    only copy of what was moved out.
    """
    dataset_dir = Path(dataset_dir)
    removed = 0
    if not dataset_dir.is_dir():
        return 0
    for item in sorted(dataset_dir.iterdir()):
        if not item.is_file() or item.suffix.lower() not in MEDIA_EXTS:
            continue
        try:
            item.with_suffix(".txt").unlink()
        except OSError:
            continue
        removed += 1
    for name in (*STATE_FILES, *LEGACY_FILES, FLAGGED_FILE):
        (dataset_dir / name).unlink(missing_ok=True)
    if job:
        job.log(f"{removed} captions apagadas — todas serão reescritas.")
    return removed


def captioner_cmd(dataset_dir: Path, profile: str, trigger: str = "") -> list[str]:
    cmd = [
        str(venv_bin("captioner", "captioner")), "run", str(dataset_dir),
        "--profile", profile,
        "--prompts-dir", str(PROFILES_DIR),
        # control/ holds edit-model references, not training items
        "--no-recursive",
        # fill gaps only: an imported or hand-edited caption is the owner's
        "--skip-captioned",
        "--concurrency", str(CAPTION_CONCURRENCY),
    ]
    if trigger:
        cmd += ["--var", f"trigger={trigger}"]
    return cmd


def openrouter_problem() -> str:
    """Empty string when the account can actually serve requests, else why not.

    Guards the discard: a dead key or an empty balance leaves every item
    uncaptioned, which looks exactly like every model refusing them. Moving the
    owner's images out because his card expired is not a recovery.
    """
    import urllib.error
    import urllib.request

    key = os.environ.get("OPENROUTER_API_KEY", "")
    base = os.environ.get("OPENROUTER_BASE_URL", "").strip().rstrip("/") \
        or "https://openrouter.ai/api/v1"
    req = urllib.request.Request(f"{base}/key", headers={"Authorization": f"Bearer {key}"})
    try:
        data = json.load(urllib.request.urlopen(req, timeout=30)).get("data", {})
    except urllib.error.HTTPError as exc:
        return f"OpenRouter recusou a chave (HTTP {exc.code})"
    except Exception as exc:  # offline, DNS, timeout — anything but a refusal
        return f"não consegui falar com o OpenRouter ({type(exc).__name__})"
    remaining = data.get("limit_remaining")
    if remaining is not None and remaining <= 0:
        return "OpenRouter sem crédito"
    return ""


def fallback_names(dataset_dir: Path) -> set[str]:
    """Items whose caption came from a fallback route, per captions.json."""
    try:
        rows = json.loads((Path(dataset_dir) / "captions.json").read_text()).get("items", [])
    except (OSError, json.JSONDecodeError, AttributeError):
        return set()
    return {r.get("filename") for r in rows
            if isinstance(r, dict) and r.get("fallback") is True and r.get("filename")}


def record_flagged(dataset_dir: Path, names: set[str], profile: str) -> None:
    """Remember which items the primary model would not caption.

    They are the ones a content filter objects to, which is the same objection
    gpt-image-2 raises in the Style Rush conversion phase — and there a refusal
    costs a paid slot. Kept as a union across runs: a resumed run only sees the
    items it captioned itself.
    """
    path = dataset_dir / FLAGGED_FILE
    try:
        known = set(json.loads(path.read_text()).get("refused_by_primary", []))
    except (OSError, json.JSONDecodeError, AttributeError):
        known = set()
    known |= names
    path.write_text(json.dumps(
        {"profile": profile, "refused_by_primary": sorted(known)},
        indent=2, ensure_ascii=False))


def flagged_names(dataset_dir: Path) -> set[str]:
    """File names the primary caption model refused, as recorded by record_flagged."""
    try:
        return set(json.loads((Path(dataset_dir) / FLAGGED_FILE).read_text())
                   .get("refused_by_primary", []))
    except (OSError, json.JSONDecodeError, AttributeError):
        return set()


def quarantine_uncaptionable(dataset_dir: Path, items: list[Path], job: Job) -> None:
    """Move items no model would caption out of the dataset, so the run goes on.

    Moved, not deleted: the only evidence that a model *refused* is that it
    returned nothing, which is also what a retired model id returns. The account
    check upstream catches the common case and cannot catch all of them.
    """
    dest = dataset_dir.parent / QUARANTINE_DIR
    dest.mkdir(parents=True, exist_ok=True)
    moved = 0
    for item in items:
        try:
            item.replace(dest / item.name)
        except OSError as exc:
            job.log(f"⚠ não consegui mover {item.name}: {exc}")
            continue
        txt = item.with_suffix(".txt")
        if txt.exists():
            txt.replace(dest / txt.name)
        moved += 1
        job.log(f"✗ fora do dataset (nenhum modelo aceitou): {item.name}")
    if moved:
        job.log(f"⚠ {moved} itens movidos para {dest}")


def _uncaptioned(dataset_dir: Path, media: str) -> list[Path]:
    exts = IMAGE_EXTS if media == "image" else VIDEO_EXTS
    return [p for p in ds.uncaptioned(dataset_dir) if p.suffix.lower() in exts]


def _progress_hook(job: Job, errors: list[str]):
    def on_line(line: str) -> None:
        if line.strip().startswith("error:"):
            errors.append(line.strip()[len("error:"):].strip())
        m = _PROGRESS_RE.match(line.strip())
        if not m:
            return
        job.progress.update({"step": int(m.group(1)), "total_steps": int(m.group(2)),
                             "unit": "captions"})
        cost = _COST_RE.search(line)
        if cost:
            job.extra["caption_cost"] = float(cost.group(1))
    return on_line


def generate_captions(dataset_dir, profile: str, job: Job, trigger: str = "",
                      redo: bool = False) -> None:
    """Caption every item of `dataset_dir` that has no caption yet.

    Ends with every remaining item captioned, or raises. What no model would
    caption is moved out (see quarantine_uncaptionable); an infrastructure
    failure raises instead and leaves the dataset as it was.
    """
    spec = caption_profile(profile)
    if spec is None:
        raise JobFailed(f"perfil de caption desconhecido: {profile}")
    trigger = (trigger or "").strip()
    if spec["trigger"] and not trigger:
        raise JobFailed(f"o perfil '{spec['label']}' precisa da trigger word — preencha no passo 01")
    dataset_dir = Path(dataset_dir)
    media = spec["media"]
    if not redo and not _uncaptioned(dataset_dir, media):
        job.log("Todos os itens já têm caption.")
        return
    # checked before the clear: a redo without a key would delete every
    # caption and then fail to write a single one
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise JobFailed("defina OPENROUTER_API_KEY para gerar captions com LLM")
    if redo:
        clear_captions(dataset_dir, job)

    ensure_engine("captioner", job)
    job.log(f"Perfil de caption: {spec['label']} ({profile})")
    errors: list[str] = []
    code = job.run(captioner_cmd(dataset_dir, profile, trigger if spec["trigger"] else ""),
                   cwd=engine_dir("captioner"), check=False, on_line=_progress_hook(job, errors))
    job.progress.clear()
    # 0 = all done, 1 = some items failed (handled below). Anything else is the
    # run refusing to start — bad profile, missing variable — before any spend.
    if code not in (0, 1):
        why = errors[-1] if errors else "veja o log"
        raise JobFailed(f"o captioner não rodou (código {code}): {why}")

    rescued = fallback_names(dataset_dir)
    if rescued:
        record_flagged(dataset_dir, rescued, profile)

    # only this profile's media: a video left uncaptioned by an image profile was
    # never asked about, and moving it out would be a decision nobody made
    missing = _uncaptioned(dataset_dir, media)
    if missing:
        # Never destroy on an infrastructure failure: out of credit looks
        # exactly like every model refusing, and one of the two is recoverable.
        problem = openrouter_problem()
        if problem:
            raise JobFailed(
                f"{problem} — {len(missing)} itens ficaram sem caption. "
                f"Resolva e rode de novo; nada saiu do dataset e o que já tem "
                f"caption não é refeito.")
        quarantine_uncaptionable(dataset_dir, missing, job)

    job.log("✔ Captions geradas.")


def caption_datasets(job: Job, dirs: list[Path], *, model_key: str | None, kind: str | None,
                     trigger: str = "", profile: str | None = None, redo: bool = False) -> None:
    """Caption whatever is missing in each dataset directory.

    A dataset can hold images and videos (LTX trains on both), and one
    profile only speaks one of them, so each media present gets its own: the
    chosen profile where its media matches, the default for (model, kind)
    elsewhere. Redo clears each directory once, before its first pass.
    """
    chosen = caption_profile(profile) if profile else None
    for d in dirs:
        stats = ds.inspect(d)
        first = True
        for media in ("image", "video"):
            count = stats["images"] if media == "image" else stats["videos"]
            if not count:
                continue
            if not redo and not _uncaptioned(d, media):
                continue
            key = chosen["key"] if chosen and chosen["media"] == media \
                else default_caption_profile(model_key, kind, media)
            generate_captions(d, key, job, trigger=trigger, redo=redo and first)
            first = False
