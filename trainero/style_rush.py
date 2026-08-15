"""Style Rush: build the synthetic style-conversion dataset.

The owner ships one dataset of images. This module turns it into a second,
paired dataset: for each of the SLOT_COUNT slots, GPT Image restyles one of
those images into a *different* style. That restyled image becomes the control
image and the untouched original becomes the target, so the trained LoRA learns
"any style -> the owner's style".

The slot count is fixed. A dataset smaller than SLOT_COUNT simply reuses its
images, always under a different style prompt.
"""

from __future__ import annotations

import random
from pathlib import Path

from .config import REPO_DIR

SLOT_COUNT = 50
CAPTION_TEMPLATE = "convert the style of this image to the {trigger} style"
PROMPTS_FILE = REPO_DIR / "data" / "style_prompts.txt"

# Fixed so a cancelled run resumes onto the same plan instead of paying the
# API again for a different selection.
PLAN_SEED = 1707


def load_style_prompts(path: Path | None = None) -> list[str]:
    """The style prompts, one per line. Blank lines and '#' comments ignored."""
    src = path or PROMPTS_FILE
    try:
        lines = src.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FileNotFoundError(f"style prompts file missing: {src}") from exc
    prompts = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
    if len(prompts) < SLOT_COUNT:
        raise ValueError(f"{src} has {len(prompts)} prompts, need {SLOT_COUNT}")
    return prompts[:SLOT_COUNT]


def plan_slots(images: list[Path], prompts: list[str]) -> list[dict]:
    """One slot per prompt, each pointing at a primary image and a fallback.

    The fallback is what the slot retries with when the primary is refused by
    moderation; it is always a different image when the dataset has more than
    one. Selection is deterministic so a resumed run rebuilds the same plan.
    """
    if not images:
        raise ValueError("dataset base vazio — nada para converter")
    pool = sorted(str(p) for p in images)
    rng = random.Random(PLAN_SEED)
    order = list(pool)
    rng.shuffle(order)

    slots = []
    for i in range(SLOT_COUNT):
        primary = order[i % len(order)]
        sources = [primary]
        if len(order) > 1:
            sources.append(order[(i + 1) % len(order)])
        slots.append({"slot": f"slot_{i:02d}", "prompt": prompts[i], "sources": sources})
    return slots
