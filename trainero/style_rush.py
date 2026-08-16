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

import json
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import imagegen
from .config import IMAGE_EXTS, REPO_DIR
from .jobs import Cancelled, JobFailed

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


MANIFEST_NAME = ".style_rush.json"
RETRIABLE_ATTEMPTS = 3  # per image, for 429/5xx — moderation is not retried here


def base_images(base_dir: Path) -> list[Path]:
    """Images at the top level of the base dataset (control/ is not one of ours)."""
    if not base_dir.exists():
        return []
    return sorted(p for p in base_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _load_manifest(convert_dir: Path) -> dict:
    try:
        return json.loads((convert_dir / MANIFEST_NAME).read_text())
    except (OSError, json.JSONDecodeError):
        return {"slots": {}}


def _save_manifest(convert_dir: Path, manifest: dict) -> None:
    (convert_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))


def _generate_with_retries(generate, prompt: str, source: Path):
    """Retry the same image on transient errors; let RefusedError through so the
    caller can move to a different image instead."""
    last: Exception | None = None
    for attempt in range(RETRIABLE_ATTEMPTS):
        try:
            return generate(prompt, source)
        except imagegen.RetriableError as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise last


def fit_control_to_target(png: bytes, target: Path) -> bytes:
    """Center-crop the generated image to the target's exact aspect ratio.

    This is what makes a pair a pair. musubi buckets the control and the target
    independently — BucketSelector.calculate_bucket_resolution derives the shape
    from each image's OWN aspect ratio, and resize_image_to_bucket then scales to
    cover and center-crops. Two different ratios therefore become two different
    bucket shapes cropped along different axes, so the model would be shown a
    control framed differently from the target it must reproduce.

    The generated image rarely matches: gpt-image-2 only renders the handful of
    ratios in SUPPORTED_RATIOS, so a 1000x800 source (1.25) comes back as 4:3.
    Cropping here, before anything is written, makes both land in one bucket.
    """
    import io

    from PIL import Image

    with Image.open(target) as t:
        target_ratio = t.size[0] / t.size[1]
    with Image.open(io.BytesIO(png)) as control:
        width, height = control.size
        if abs(width / height - target_ratio) < 1e-3:
            return png
        if width / height > target_ratio:  # too wide — trim the sides
            new_w, new_h = round(height * target_ratio), height
        else:                              # too tall — trim top and bottom
            new_w, new_h = width, round(width / target_ratio)
        left, top = (width - new_w) // 2, (height - new_h) // 2
        buf = io.BytesIO()
        control.crop((left, top, left + new_w, top + new_h)).save(buf, format="PNG")
    return buf.getvalue()


def build_convert_dataset(base_dir: Path, convert_dir: Path, trigger: str, job,
                          generate=imagegen.generate, workers: int = 4) -> dict:
    """Produce SLOT_COUNT (control, target, caption) triples under convert_dir.

    control/<slot>.png  the restyled image, cropped to the target's aspect ratio
    <slot>.png          an untouched copy of the source image (the target)
    <slot>.txt          the same caption for every slot

    Slots already recorded as ok in the manifest are skipped, so a cancelled run
    resumes without paying for them again.
    """
    images = base_images(base_dir)
    if not images:
        raise JobFailed("dataset base vazio — importe as imagens antes de treinar")

    slots = plan_slots(images, load_style_prompts())
    control_dir = convert_dir / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(convert_dir)
    caption = CAPTION_TEMPLATE.format(trigger=trigger)

    lock = threading.Lock()
    totals = {"pairs": 0, "refused": 0, "failed": 0, "cost": 0.0}

    def done(name: str) -> bool:
        entry = manifest["slots"].get(name)
        return bool(entry and entry.get("status") == "ok"
                    and (convert_dir / f"{name}.png").exists()
                    and (control_dir / f"{name}.png").exists())

    pending = [s for s in slots if not done(s["slot"])]
    totals["pairs"] = len(slots) - len(pending)

    # the caption carries the trigger word, so a run with a changed trigger has
    # to rewrite it even for slots it will not regenerate
    for slot in slots:
        if (convert_dir / f"{slot['slot']}.png").exists():
            (convert_dir / f"{slot['slot']}.txt").write_text(caption, encoding="utf-8")

    if pending:
        job.log(f"Gerando {len(pending)} imagens de conversão com {imagegen.MODEL} "
                f"(quality low) — ~${imagegen.COST_PER_IMAGE * len(pending):.2f}")
    else:
        job.log(f"Dataset de conversão já completo ({len(slots)} pares).")

    def record(name: str, entry: dict, **deltas) -> None:
        """Persist one slot's outcome. The manifest is written on every slot, not
        once at the end: a cancelled run (job.check_cancel raises) or a dead pod
        would otherwise lose the record of images already paid for."""
        with lock:
            for k, v in deltas.items():
                totals[k] += v
            manifest["slots"][name] = entry
            _save_manifest(convert_dir, manifest)

    def work(slot: dict) -> None:
        job.check_cancel()
        name = slot["slot"]
        refusals = 0
        for source in slot["sources"]:  # primary, then the fallback image
            try:
                png, cost = _generate_with_retries(generate, slot["prompt"], Path(source))
                png = fit_control_to_target(png, Path(source))
                (control_dir / f"{name}.png").write_bytes(png)
                shutil.copyfile(source, convert_dir / f"{name}.png")
                (convert_dir / f"{name}.txt").write_text(caption, encoding="utf-8")
            except imagegen.RefusedError as exc:
                refusals += 1
                record(name, {"status": "refused", "prompt": slot["prompt"],
                              "source": source, "error": str(exc)})
                continue
            except Cancelled:
                raise
            except Exception as exc:
                # one unreadable file (an .avif this Pillow cannot open) or one
                # malformed response must cost its own slot, never the phase
                record(name, {"status": "failed", "prompt": slot["prompt"], "source": source,
                              "error": f"{type(exc).__name__}: {exc}"},
                       failed=1, refused=refusals)
                job.log(f"⚠ {name}: {type(exc).__name__}: {exc}")
                return

            record(name, {"status": "ok", "prompt": slot["prompt"], "source": source,
                          "cost": cost},
                   pairs=1, refused=refusals, cost=cost)
            return

        with lock:
            totals["refused"] += refusals
        job.log(f"⚠ {name}: recusado nas duas tentativas — slot descartado.")

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(work, pending))

    if totals["pairs"] == 0:
        raise JobFailed(
            "o dataset de conversão ficou vazio — todas as imagens foram recusadas ou "
            "falharam. Sem ele o LoRA não aprende a converter estilo.")

    job.log(f"✔ Dataset de conversão: {totals['pairs']} pares, "
            f"{totals['refused']} recusas, custo ~${totals['cost']:.2f}")
    return totals
