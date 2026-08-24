"""Style Rush: build the two synthetic paired datasets.

The owner ships one dataset of images. This module turns it into two more,
both of them (control -> target) pairs where the target is always an untouched
original, so every pair teaches "bad input -> the owner's image":

  conversion   GPT Image restyles the original into a *different* style. That
               restyled image is the control, so the LoRA learns
               "any style -> the owner's style".
  restoration  the original is run through the tiled-grit degradation, and the
               damaged copy is the control, so the LoRA learns to undo the
               artefacts an upscaler or generator leaves behind. Reverse
               engineering: teach the repair by manufacturing the damage.

Conversion costs API money per image; restoration is local CPU and free. The
conversion half stops when it has DEFAULT_CONVERT_TARGET *successes*, not when
it has tried that many times: a refusal costs an attempt, never a pair. A
dataset smaller than the target simply reuses its images — conversion always
under a different style prompt, restoration always under a different grit seed.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import captioner, degrade, imagegen
from .config import IMAGE_EXTS, REPO_DIR
from .jobs import Cancelled, JobFailed

# The conversion half stops when it has this many *successes*. A refusal used
# to cost a pair; now it costs an attempt.
DEFAULT_CONVERT_TARGET = 100
# Enough queue to absorb refusals, and a hard end for a dataset where every
# image is refused — otherwise the loop runs until the account is empty.
ATTEMPT_MULTIPLIER = 3
# gpt-image-2 calls are slow and independent. The budget reservation in
# build_convert_dataset is what makes raising this safe.
CONVERT_WORKERS = int(os.environ.get("CONVERT_WORKERS", "8"))
CAPTION_TEMPLATE = "convert the style of this image to the {trigger} style"
PROMPTS_FILE = REPO_DIR / "data" / "style_prompts.txt"

# Fixed so a cancelled run resumes onto the same plan instead of paying the
# API again for a different selection.
PLAN_SEED = 1707

# -- restoration half --------------------------------------------------------
RESTORE_COUNT = 100
RESTORE_CAPTION = "fix the noise in the image, enhance the quality"
RESTORE_MANIFEST_NAME = ".style_rush_restore.json"
# Grit settings. The seed varies per slot (RESTORE_SEED + index) so the 100
# damaged images do not all carry the same tile pattern — one fixed pattern is
# a watermark the LoRA would learn to subtract instead of learning to restore.
RESTORE_SEED = 4703
RESTORE_SEVERITY = 0.6
RESTORE_MULTIPLIER = 1.0


def load_style_prompts(path: Path | None = None) -> list[str]:
    """The style prompts, one per line. Blank lines and '#' comments ignored.

    No minimum count and no truncation: the attempt queue cycles the list, so
    any number of prompts serves any target.
    """
    src = path or PROMPTS_FILE
    try:
        lines = src.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FileNotFoundError(f"style prompts file missing: {src}") from exc
    prompts = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
    if not prompts:
        raise ValueError(f"{src} não tem nenhum prompt de estilo")
    return prompts


def plan_attempts(images: list[Path], prompts: list[str], target: int,
                  avoid: set[str] | None = None) -> list[dict]:
    """An ordered queue of (prompt, source) attempts, longer than the target.

    A queue the length of the target dies on its first refusal, which is the
    bug this replaces. ATTEMPT_MULTIPLIER gives the run room to route around
    refused images and, just as importantly, a place to stop: a dataset where
    every image is refused ends here instead of spending the whole balance.

    The image index carries an extra `i // len(prompts)` so that when the prompt
    list wraps, the pairing shifts by one image. Without it the queue would
    repeat the exact (prompt, source) pair it already spent an attempt on.

    Selection is deterministic so a resumed run rebuilds the same queue.

    `avoid` holds file names a content filter has already objected to — the
    primary caption model refused them, or gpt-image-2 itself did on an earlier
    run. Feeding those back in buys another refusal at full price.
    """
    if not images:
        raise ValueError("dataset base vazio — nada para converter")
    avoid = avoid or set()
    usable = [p for p in images if p.name not in avoid]
    if not usable:
        raise ValueError(
            f"todas as {len(images)} imagens do dataset já foram recusadas por "
            f"filtro de conteúdo — não há o que mandar para o gpt-image-2")

    order = sorted(str(p) for p in usable)
    rng = random.Random(PLAN_SEED)
    rng.shuffle(order)

    return [{"attempt": f"slot_{i:03d}",
             "prompt": prompts[i % len(prompts)],
             "source": order[(i + i // len(prompts)) % len(order)]}
            for i in range(target * ATTEMPT_MULTIPLIER)]


MANIFEST_NAME = ".style_rush.json"
RETRIABLE_ATTEMPTS = 3  # per image, for 429/5xx — moderation is not retried here


def base_images(base_dir: Path) -> list[Path]:
    """Images at the top level of the base dataset (control/ is not one of ours)."""
    if not base_dir.exists():
        return []
    return sorted(p for p in base_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def content_flagged(base_dir: Path, convert_dir: Path) -> set[str]:
    """File names a content filter has already objected to, from both sources.

    The caption phase records what the strict model refused; the conversion
    manifest records what gpt-image-2 refused on an earlier run. Both are the
    same fact about the image, and both cost money to rediscover.
    """
    manifest = _load_manifest(convert_dir)
    flagged = captioner.flagged_names(base_dir)
    flagged |= set(manifest.get("refused_images", []))
    for entry in manifest.get("slots", {}).values():
        if entry.get("status") == "refused" and entry.get("source"):
            flagged.add(Path(entry["source"]).name)
    return flagged


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
                          generate=imagegen.generate, workers: int = CONVERT_WORKERS,
                          target: int = DEFAULT_CONVERT_TARGET) -> dict:
    """Produce `target` (control, target, caption) triples under convert_dir.

    control/<slot>.png  the restyled image, cropped to the target's aspect ratio
    <slot>.png          an untouched copy of the source image (the target)
    <slot>.txt          the same caption for every pair

    The pair is named after the attempt that produced it, so the numbering has
    gaps wherever an attempt was refused. Nothing downstream cares: musubi walks
    the directory. Numbering by success count instead would mean two numbering
    schemes living side by side, one for work and one for output.

    Pairs already recorded as ok in the manifest count toward the target, so a
    cancelled run resumes without paying for them again.
    """
    images = base_images(base_dir)
    if not images:
        raise JobFailed("dataset base vazio — importe as imagens antes de treinar")

    avoid = content_flagged(base_dir, convert_dir)
    if avoid:
        job.log(f"{len(avoid)} imagens fora da conversão por recusa de filtro de "
                f"conteúdo (o gpt-image-2 recusaria de novo, cobrando a tentativa).")
    try:
        attempts = plan_attempts(images, load_style_prompts(), target, avoid=avoid)
    except (ValueError, FileNotFoundError) as exc:
        raise JobFailed(str(exc))

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

    pending = [a for a in attempts if not done(a["attempt"])]
    totals["pairs"] = len(attempts) - len(pending)
    # what is still to buy. Guarded by `lock` from here on.
    budget = target - totals["pairs"]

    # the caption carries the trigger word, so a run with a changed trigger has
    # to rewrite it even for pairs it will not regenerate
    for existing in convert_dir.glob("slot_*.png"):
        existing.with_suffix(".txt").write_text(caption, encoding="utf-8")

    if budget > 0:
        job.log(f"Gerando {budget} imagens de conversão com {imagegen.MODEL} "
                f"(quality low, {workers} em paralelo) — "
                f"~${imagegen.COST_PER_IMAGE * budget:.2f}")
    else:
        job.log(f"Dataset de conversão já completo ({totals['pairs']} pares).")

    # Refusals live per image, not per attempt: with a target far above the
    # dataset size the same picture is the source many times over, so without
    # this every worker pays to rediscover the same no.
    refused_now: set[str] = set(manifest.get("refused_images", []))

    def record(name: str, entry: dict, **deltas) -> None:
        """Persist one attempt's outcome. The manifest is written on every
        attempt, not once at the end: a cancelled run (job.check_cancel raises)
        or a dead pod would otherwise lose the record of images already paid."""
        with lock:
            for k, v in deltas.items():
                totals[k] += v
            manifest["slots"][name] = entry
            _save_manifest(convert_dir, manifest)

    def note_refused(source: str) -> None:
        with lock:
            refused_now.add(Path(source).name)
            manifest["refused_images"] = sorted(refused_now)
            _save_manifest(convert_dir, manifest)

    def claim() -> bool:
        """Take one slot of budget before spending money on it.

        This is what makes the target exact. Without it, the N workers still in
        flight when the last pair lands each go on to buy one more image."""
        nonlocal budget
        with lock:
            if budget <= 0:
                return False
            budget -= 1
            return True

    def release() -> None:
        """Give the budget back — this attempt produced no pair."""
        nonlocal budget
        with lock:
            budget += 1

    # ThreadPoolExecutor.map keeps dispatching after a task raises — the error
    # only surfaces when the results are consumed. Without this the whole queue
    # still runs, and a dead account is asked hundreds more times.
    account_dead: list[Exception] = []

    def work(attempt: dict) -> None:
        if account_dead:
            return
        job.check_cancel()
        name, source = attempt["attempt"], attempt["source"]
        with lock:
            if Path(source).name in refused_now:
                return
        if not claim():
            return                      # the target is already met
        try:
            png, cost = _generate_with_retries(generate, attempt["prompt"], Path(source))
        except imagegen.AccountError as exc:
            release()
            with lock:            # not this attempt's problem — it ends the phase
                account_dead.append(exc)
            return
        except imagegen.RefusedError as exc:
            release()
            note_refused(source)
            record(name, {"status": "refused", "prompt": attempt["prompt"],
                          "source": source, "error": str(exc)}, refused=1)
            return
        except Cancelled:
            release()
            raise
        except Exception as exc:
            # failed before the call was billed: one unreadable file (an .avif
            # this Pillow cannot open) costs its own attempt, never the phase
            release()
            record(name, {"status": "failed", "prompt": attempt["prompt"], "source": source,
                          "error": f"{type(exc).__name__}: {exc}"}, failed=1)
            job.log(f"⚠ {name}: {type(exc).__name__}: {exc}")
            return

        # billed from here on, so the cost is counted whatever happens next
        try:
            png = fit_control_to_target(png, Path(source))
            (control_dir / f"{name}.png").write_bytes(png)
            shutil.copyfile(source, convert_dir / f"{name}.png")
            (convert_dir / f"{name}.txt").write_text(caption, encoding="utf-8")
        except Cancelled:
            release()
            raise
        except Exception as exc:
            release()               # paid, but there is no pair to show for it
            record(name, {"status": "failed", "prompt": attempt["prompt"], "source": source,
                          "paid": True, "cost": cost,
                          "error": f"{type(exc).__name__}: {exc}"},
                   failed=1, cost=cost)
            job.log(f"⚠ {name}: imagem paga mas não gravada "
                    f"({type(exc).__name__}: {exc}) — ${cost:.4f} perdidos")
            return

        record(name, {"status": "ok", "prompt": attempt["prompt"], "source": source,
                      "cost": cost}, pairs=1, cost=cost)

    if budget > 0:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(work, pending))
        if account_dead:
            _save_manifest(convert_dir, manifest)
            raise JobFailed(
                f"{account_dead[0]} — {totals['pairs']} de {target} pares prontos. "
                f"Resolva e rode de novo: o que já foi pago não é refeito.")

    if totals["pairs"] == 0:
        raise JobFailed(
            "o dataset de conversão ficou vazio — todas as imagens foram recusadas ou "
            "falharam. Sem ele o LoRA não aprende a converter estilo.")

    if totals["pairs"] < target:
        job.log(f"⚠ meta de {target} pares não atingida: {totals['pairs']} prontos depois "
                f"de {len(pending)} tentativas. {len(refused_now)} imagens do dataset "
                f"foram recusadas por moderação.")

    job.log(f"✔ Dataset de conversão: {totals['pairs']} pares, "
            f"{totals['refused']} recusas, custo ~${totals['cost']:.2f}")
    return totals


# ---------------------------------------------------------------------------
# Restoration dataset
# ---------------------------------------------------------------------------


def convert_sources(convert_dir: Path) -> set[str]:
    """The base images the conversion dataset actually consumed.

    Read from the manifest rather than re-derived from plan_attempts, because
    the queue is three times longer than the target — it says which images
    *might* have been used, and only the manifest says which ones were.
    """
    manifest = _load_manifest(convert_dir)
    return {entry["source"] for entry in manifest.get("slots", {}).values()
            if entry.get("status") == "ok" and entry.get("source")}


def plan_restore(images: list[Path], used: set[str],
                 count: int = RESTORE_COUNT) -> list[dict]:
    """One slot per damaged copy, preferring images the conversion half did not
    already take.

    Overlap is avoided, not forbidden: a dataset with fewer unused images than
    slots still fills every slot, it just starts reusing once the fresh ones run
    out. Deterministic, so a resumed run rebuilds the same plan.
    """
    if not images:
        raise ValueError("dataset base vazio — nada para degradar")
    pool = sorted(str(p) for p in images)
    rng = random.Random(RESTORE_SEED)
    fresh = [p for p in pool if p not in used]
    reused = [p for p in pool if p in used]
    rng.shuffle(fresh)
    rng.shuffle(reused)
    order = fresh + reused  # untouched images first, converted ones as filler

    return [{"slot": f"restore_{i:03d}", "source": order[i % len(order)],
             "seed": RESTORE_SEED + i}
            for i in range(count)]


def _load_restore_manifest(restore_dir: Path) -> dict:
    try:
        return json.loads((restore_dir / RESTORE_MANIFEST_NAME).read_text())
    except (OSError, json.JSONDecodeError):
        return {"slots": {}}


def _save_restore_manifest(restore_dir: Path, manifest: dict) -> None:
    (restore_dir / RESTORE_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))


def build_restore_dataset(base_dir: Path, restore_dir: Path, job,
                          used: set[str] | None = None,
                          count: int = RESTORE_COUNT, workers: int = 4) -> dict:
    """Produce `count` (damaged control, clean target, caption) triples.

    control/<slot>.png  the degraded image — what the model is given
    <slot>.png          an untouched copy of the source (the target)
    <slot>.txt          RESTORE_CAPTION, the same for every slot

    Costs nothing but CPU, so unlike the conversion half a failed slot is
    retried on the next run rather than being paid for twice. Slots already
    recorded as ok are skipped so a cancelled run resumes.
    """
    images = base_images(base_dir)
    if not images:
        raise JobFailed("dataset base vazio — importe as imagens antes de treinar")

    slots = plan_restore(images, used or set(), count)
    control_dir = restore_dir / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_restore_manifest(restore_dir)

    lock = threading.Lock()
    totals = {"pairs": 0, "failed": 0, "reused": 0}
    totals["reused"] = sum(1 for s in slots if s["source"] in (used or set()))

    def done(name: str) -> bool:
        entry = manifest["slots"].get(name)
        return bool(entry and entry.get("status") == "ok"
                    and (restore_dir / f"{name}.png").exists()
                    and (control_dir / f"{name}.png").exists())

    pending = [s for s in slots if not done(s["slot"])]
    totals["pairs"] = len(slots) - len(pending)

    # the caption is a constant, but rewriting it every run keeps an existing
    # dataset correct when that constant changes
    for slot in slots:
        if (restore_dir / f"{slot['slot']}.png").exists():
            (restore_dir / f"{slot['slot']}.txt").write_text(RESTORE_CAPTION, encoding="utf-8")

    if pending:
        job.log(f"Degradando {len(pending)} imagens (severity {RESTORE_SEVERITY}, "
                f"local, sem custo)")
    else:
        job.log(f"Dataset de restauração já completo ({len(slots)} pares).")

    def record(name: str, entry: dict, **deltas) -> None:
        with lock:
            for k, v in deltas.items():
                totals[k] += v
            manifest["slots"][name] = entry
            _save_restore_manifest(restore_dir, manifest)

    def work(slot: dict) -> None:
        job.check_cancel()
        name, source = slot["slot"], Path(slot["source"])
        try:
            params = degrade.degrade_file(
                source, control_dir / f"{name}.png",
                severity=RESTORE_SEVERITY, seed=slot["seed"],
                multiplier=RESTORE_MULTIPLIER)
            shutil.copyfile(source, restore_dir / f"{name}.png")
            (restore_dir / f"{name}.txt").write_text(RESTORE_CAPTION, encoding="utf-8")
        except Cancelled:
            raise
        except Exception as exc:
            # one unreadable file must cost its own slot, never the phase
            record(name, {"status": "failed", "source": slot["source"],
                          "error": f"{type(exc).__name__}: {exc}"}, failed=1)
            job.log(f"⚠ {name}: {type(exc).__name__}: {exc}")
            return
        record(name, {"status": "ok", "source": slot["source"],
                      "seed": slot["seed"], "params": params}, pairs=1)

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(work, pending))

    if totals["pairs"] == 0:
        raise JobFailed(
            "o dataset de restauração ficou vazio — nenhuma imagem pôde ser degradada.")

    job.log(f"✔ Dataset de restauração: {totals['pairs']} pares, "
            f"{totals['failed']} falhas, {totals['reused']} reaproveitadas da conversão")
    return totals
