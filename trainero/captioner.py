"""LLM captioning via data_araknideo (PixAI booru tags -> Grok/OpenRouter).

Same invocation character_animatrem uses; the prompt profiles live in the
captioner clone (prompts/<image|video>/<profile>/). Needs OPENROUTER_API_KEY.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .config import IMAGE_EXTS, VIDEO_EXTS
from .engines import engine_dir, ensure_engine, venv_python
from .jobs import Job, JobFailed

TAGGER_LOG = ".tagger_log.json"

# The captioner's CLI names this stage "grok" for historical reasons; the flag
# takes any OpenRouter model id with vision. Override with CAPTION_MODEL.
DEFAULT_CAPTION_MODEL = os.environ.get("CAPTION_MODEL", "google/gemini-3.7-flash")


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


def generate_captions(dataset_dir, media: str, profile: str, prompt_vars: dict[str, str],
                      job: Job) -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise JobFailed("defina OPENROUTER_API_KEY para gerar captions com LLM")
    prune_stale_log(Path(dataset_dir), job)
    ensure_engine("captioner", job)
    cap_dir = engine_dir("captioner")
    cmd = [
        str(venv_python("captioner")),
        str(cap_dir / "tag_images_by_wd14_tagger.py"),
        str(dataset_dir),
        "--taggers", "pixai,grok",
        "--grok_provider", "openrouter",
        "--grok_model", DEFAULT_CAPTION_MODEL,
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
    job.run(cmd, cwd=cap_dir)
    job.log("✔ Captions geradas.")
