"""LLM captioning via data_araknideo (PixAI booru tags -> Grok/OpenRouter).

Same invocation character_animatrem uses; the prompt profiles live in the
captioner clone (prompts/<image|video>/<profile>/). Needs OPENROUTER_API_KEY.

Three passes, because a caption is not optional: a single item without one
blocks the whole training. The cheap model runs first; whatever it refuses on
content grounds goes to a second model with different policies; whatever both
refuse leaves the dataset. Refusal is a property of the image, not a transient
error, so retrying the same model would only cost time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import dataset as ds
from .config import IMAGE_EXTS, VIDEO_EXTS
from .engines import engine_dir, ensure_engine, venv_python
from .jobs import Job, JobFailed

TAGGER_LOG = ".tagger_log.json"
FLAGGED_FILE = ".caption_refused.json"
QUARANTINE_DIR = "descartadas"

# The captioner's CLI names this stage "grok" for historical reasons; the flag
# takes any OpenRouter model id with vision.
MUSE_SPARK = "meta/muse-spark-1.2-contributor"
GEMINI_FLASH = "google/gemini-3.7-flash"
GROK = "x-ai/grok-4.20"

# The order depends on what the mode does with the primary's refusal. The only
# consumer of .caption_refused.json is Style Rush's content_flagged, which uses
# it to avoid paying for a slot gpt-image-2 would refuse — so there the primary
# has to be the model whose filter most resembles OpenAI's, which is Gemini. A
# plain LoRA has no paid phase and nothing reads the list, so its order is pure
# cost and quality: Muse Spark captions the same explicit material for a tenth
# of the price.
DEFAULT_CAPTION_MODELS = {
    "lora": [MUSE_SPARK, GEMINI_FLASH, GROK],
    "style-rush": [GEMINI_FLASH, MUSE_SPARK, GROK],
}

# What record_flagged means by "the strict model". Kept as a name because the
# flag file records it and the Style Rush half reasons about it.
DEFAULT_CAPTION_MODEL = DEFAULT_CAPTION_MODELS["style-rush"][0]

# The tagger's own default is 32, which is what a thousand-image dataset waits
# on. One knob, not one per model.
CAPTION_CONCURRENCY = int(os.environ.get("CAPTION_CONCURRENCY", "64"))

MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS


def caption_models(mode: str = "lora") -> list[str]:
    """The cascade for this mode, cheapest-useful first.

    CAPTION_MODELS overrides every mode at once: a comma-separated list is one
    knob for a cascade of any length, where numbered env vars would silently
    pin it to the length they were written for.
    """
    override = os.environ.get("CAPTION_MODELS", "").strip()
    if override:
        return [m.strip() for m in override.split(",") if m.strip()]
    return list(DEFAULT_CAPTION_MODELS.get(mode) or DEFAULT_CAPTION_MODELS["lora"])


def prune_stale_log(dataset_dir: Path, job: Job | None = None) -> int:
    """Drop processing-log entries whose caption file is gone.

    The tagger skips anything already in its log, which is what makes an
    interrupted run resume for free. The log is only trustworthy while it
    agrees with the disk: a .txt deleted by hand (or never written, because the
    LLM returned nothing) would otherwise be skipped forever. Reconciling here
    keeps "delete the caption to redo it" working without a reprocess-all flag.
    """
    path = Path(dataset_dir) / TAGGER_LOG
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return 0
    processed = data.get("processed")
    if not isinstance(processed, dict):
        return 0
    stale = [f for f in processed if not Path(f).with_suffix(".txt").exists()]
    if not stale:
        return 0
    for f in stale:
        del processed[f]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    if job:
        job.log(f"{len(stale)} itens sem caption no disco voltaram para a fila.")
    return len(stale)


def _tagger_cmd(dataset_dir: Path, media: str, profile: str,
                prompt_vars: dict[str, str], model: str) -> list[str]:
    cap_dir = engine_dir("captioner")
    cmd = [
        str(venv_python("captioner")),
        str(cap_dir / "tag_images_by_wd14_tagger.py"),
        str(dataset_dir),
        "--taggers", "pixai,grok",
        "--grok_provider", "openrouter",
        "--grok_model", model,
        "--grok_concurrency", str(CAPTION_CONCURRENCY),
        "--prompt_profile", profile,
        "--remove_underscore",
        "--thresh", "0.30",
        # No --force. That flag reprocesses every file already in the tagger's
        # log, so an interrupted run (OpenRouter running out of credits at image
        # 190 of 282) charged for the whole dataset again on the retry. Both
        # callers want the same thing — fill what is missing — and the UI only
        # ever offers the button while something is missing.
        "--grok_context_from_existing",  # manual .txt vira contexto do LLM
    ]
    if media == "video":
        cmd.append("--video")
    for key, value in prompt_vars.items():
        if value:
            cmd += ["--prompt_var", f"{key}={value}"]
    return cmd


def _pass(dataset_dir: Path, media: str, profile: str, prompt_vars: dict[str, str],
          job: Job, model: str) -> list[Path]:
    """One tagger run with one model. Returns what is still uncaptioned.

    The stale-log prune has to happen before every run, not once: a refusal can
    leave the item in the tagger's log without a caption beside it, and the next
    run would then skip the very file it is here to rescue.
    """
    prune_stale_log(dataset_dir, job)
    job.run(_tagger_cmd(dataset_dir, media, profile, prompt_vars, model),
            cwd=engine_dir("captioner"))
    return ds.uncaptioned(dataset_dir)


def openrouter_problem() -> str:
    """Empty string when the account can actually serve requests, else why not.

    Guards the discard: a dead key or an empty balance leaves every item
    uncaptioned, which looks exactly like every model refusing them. Deleting
    the owner's images because his card expired is not a recovery.
    """
    import urllib.error
    import urllib.request

    key = os.environ.get("OPENROUTER_API_KEY", "")
    req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                 headers={"Authorization": f"Bearer {key}"})
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


def record_flagged(dataset_dir: Path, items: list[Path], model: str) -> None:
    """Remember which items the primary model would not caption.

    They are the ones a content filter objects to, which is the same objection
    gpt-image-2 raises in the Style Rush conversion phase — and there a refusal
    costs a paid slot. Written next to the dataset so that phase can read it
    without re-running any model. `model` is recorded because the primary is
    not the same in every mode, and a list is only as meaningful as the filter
    that produced it.
    """
    path = dataset_dir / FLAGGED_FILE
    try:
        known = set(json.loads(path.read_text()).get("refused_by_primary", []))
    except (OSError, json.JSONDecodeError, AttributeError):
        known = set()
    known |= {i.name for i in items}
    path.write_text(json.dumps(
        {"model": model, "refused_by_primary": sorted(known)},
        indent=2, ensure_ascii=False))


def flagged_names(dataset_dir: Path) -> set[str]:
    """File names the strict caption model refused, as recorded by record_flagged."""
    try:
        return set(json.loads((Path(dataset_dir) / FLAGGED_FILE).read_text())
                   .get("refused_by_primary", []))
    except (OSError, json.JSONDecodeError, AttributeError):
        return set()


def quarantine_uncaptionable(dataset_dir: Path, items: list[Path], job: Job) -> None:
    """Move items no model would caption out of the dataset, so the run goes on.

    Moved, not deleted. An uncaptioned item blocks the training outright, so it
    has to leave the dataset — but the only evidence that a model *refused* is
    that it returned nothing, which is also what a dead endpoint, a retired
    model id and a rate limit return. The account check upstream catches the
    common case and cannot catch all of them, and unlink on the owner's only
    copy of an imported image is not a recoverable mistake. QUARANTINE_DIR sits
    beside the dataset, so nothing scans it and he can look at what left.
    """
    dest = dataset_dir.parent / QUARANTINE_DIR
    dest.mkdir(parents=True, exist_ok=True)
    moved = []
    for item in items:
        try:
            item.replace(dest / item.name)
        except OSError as exc:
            job.log(f"⚠ não consegui mover {item.name}: {exc}")
            continue
        txt = item.with_suffix(".txt")
        if txt.exists():
            txt.replace(dest / txt.name)
        moved.append(item)
        job.log(f"✗ fora do dataset (nenhum modelo aceitou): {item.name}")
    try:
        log = dataset_dir / TAGGER_LOG
        data = json.loads(log.read_text())
        gone = {str(i) for i in moved}
        data["processed"] = {k: v for k, v in data.get("processed", {}).items()
                             if k not in gone}
        log.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    if moved:
        job.log(f"⚠ {len(moved)} itens movidos para {dest}")


def generate_captions(dataset_dir, media: str, profile: str, prompt_vars: dict[str, str],
                      job: Job, mode: str = "lora") -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise JobFailed("defina OPENROUTER_API_KEY para gerar captions com LLM")
    dataset_dir = Path(dataset_dir)
    ensure_engine("captioner", job)

    models = caption_models(mode)
    job.log(f"Cascata de caption: {' → '.join(models)}")

    missing = _pass(dataset_dir, media, profile, prompt_vars, job, models[0])
    refused_by_primary = missing
    for model in models[1:]:
        if not missing:
            break
        job.log(f"{len(missing)} sem caption — tentando {model}: "
                f"{', '.join(p.name for p in missing[:5])}"
                f"{'…' if len(missing) > 5 else ''}")
        missing = _pass(dataset_dir, media, profile, prompt_vars, job, model)

    # Flag only what a later model rescued. "Still uncaptioned after pass 1" has
    # two causes that look identical on disk — the model refused it, or the API
    # was down — and only one of them justifies excluding the image from the
    # paid conversion phase forever. A second model producing a caption is the
    # proof that the first one refused on content, not that the account died.
    rescued = [p for p in refused_by_primary if p not in set(missing)]
    if rescued:
        record_flagged(dataset_dir, rescued, models[0])

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
