"""LLM captioning via data_araknideo (PixAI booru tags -> Grok/OpenRouter).

Same invocation character_animatrem uses; the prompt profiles live in the
captioner clone (prompts/<image|video>/<profile>/). Needs OPENROUTER_API_KEY.
"""

from __future__ import annotations

import os

from .engines import engine_dir, ensure_engine, venv_python
from .jobs import Job, JobFailed

# The captioner's CLI names this stage "grok" for historical reasons; the flag
# takes any OpenRouter model id with vision. Override with CAPTION_MODEL.
DEFAULT_CAPTION_MODEL = os.environ.get("CAPTION_MODEL", "google/gemini-3.7-flash")


def generate_captions(dataset_dir, media: str, profile: str, prompt_vars: dict[str, str],
                      job: Job) -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise JobFailed("defina OPENROUTER_API_KEY para gerar captions com LLM")
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
        "--force",
        "--grok_context_from_existing",  # manual .txt vira contexto do LLM
    ]
    if media == "video":
        cmd.append("--video")
    for key, value in prompt_vars.items():
        if value:
            cmd += ["--prompt_var", f"{key}={value}"]
    job.run(cmd, cwd=cap_dir)
    job.log("✔ Captions geradas.")
