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
from .engines import engine_dir, ensure_engine, venv_python
from .jobs import Job, JobFailed

TAGGER_LOG = ".tagger_log.json"

# The captioner's CLI names this stage "grok" for historical reasons; the flag
# takes any OpenRouter model id with vision. Override with CAPTION_MODEL.
DEFAULT_CAPTION_MODEL = os.environ.get("CAPTION_MODEL", "google/gemini-3.7-flash")
# Gemini blocks with PROHIBITED_CONTENT on material Grok captions without
# complaint, which is the entire reason there is a second pass.
FALLBACK_CAPTION_MODEL = os.environ.get("CAPTION_FALLBACK_MODEL", "x-ai/grok-4.20")


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


def discard_uncaptionable(dataset_dir: Path, items: list[Path], job: Job) -> None:
    """Remove items no model would caption, so the run can go on.

    An uncaptioned item stops the training outright, and these have already been
    refused by two models with different content policies. Every name is logged
    because this deletes the owner's data.
    """
    log = dataset_dir / TAGGER_LOG
    for item in items:
        job.log(f"✗ removida (nenhum modelo aceitou): {item.name}")
        item.with_suffix(".txt").unlink(missing_ok=True)
        item.unlink(missing_ok=True)
    try:
        data = json.loads(log.read_text())
        gone = {str(i) for i in items}
        data["processed"] = {k: v for k, v in data.get("processed", {}).items()
                             if k not in gone}
        log.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    job.log(f"⚠ {len(items)} itens descartados do dataset.")


def generate_captions(dataset_dir, media: str, profile: str, prompt_vars: dict[str, str],
                      job: Job) -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise JobFailed("defina OPENROUTER_API_KEY para gerar captions com LLM")
    dataset_dir = Path(dataset_dir)
    ensure_engine("captioner", job)

    missing = _pass(dataset_dir, media, profile, prompt_vars, job, DEFAULT_CAPTION_MODEL)

    if missing and FALLBACK_CAPTION_MODEL != DEFAULT_CAPTION_MODEL:
        job.log(f"{len(missing)} recusadas por {DEFAULT_CAPTION_MODEL} — "
                f"tentando {FALLBACK_CAPTION_MODEL}: "
                f"{', '.join(p.name for p in missing[:5])}"
                f"{'…' if len(missing) > 5 else ''}")
        missing = _pass(dataset_dir, media, profile, prompt_vars, job,
                        FALLBACK_CAPTION_MODEL)

    if missing:
        discard_uncaptionable(dataset_dir, missing, job)

    job.log("✔ Captions geradas.")
