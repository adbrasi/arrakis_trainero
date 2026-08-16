"""Repeatable AI-like tiled grit degradation.

This is the damage half of the restoration dataset: the owner's proven script,
turned into a library so the pipeline can call it per image instead of shelling
out. The maths is unchanged — same tile construction, same warp, same
luminance-weighted amount, same over-sharpen — so an image degraded here is
byte-identical to one degraded by the standalone CLI with the same arguments.

Why tiled and not plain gaussian noise: the artefact this teaches a LoRA to
undo is the repeated micro-texture upscalers and generators leave behind, which
a random-per-pixel noise model would never reproduce.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SUPPORTED_OUTPUTS = {".jpg", ".jpeg", ".png", ".webp"}


def clamp(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def strength(severity: float, low: float, high: float) -> float:
    return low + severity * (high - low)


def apply_ai_grit_tiling_texture(
    image: np.ndarray,
    *,
    severity: float = 0.6,
    seed: int = 42,
    multiplier: float = 1.0,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Add muddy, repeated, crunchy micro-texture to an RGB image."""
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be between 0.0 and 1.0")
    if multiplier not in {1.0, 2.0}:
        raise ValueError("multiplier must be 1.0 or 2.0")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an RGB array")

    rng = np.random.default_rng(seed)
    height, width = image.shape[:2]
    tile_size = int(round(strength(severity, 72, 150)))

    tile = rng.normal(0, 1, (tile_size, tile_size)).astype(np.float32)
    tile = cv2.GaussianBlur(tile, (0, 0), strength(severity, 1.2, 2.8))
    sparse_grit = rng.random((tile_size, tile_size), dtype=np.float32)
    tile += 0.45 * (sparse_grit > strength(1 - severity, 0.92, 0.975))
    tile = cv2.GaussianBlur(tile, (0, 0), 0.45)
    tile -= tile.mean()
    tile /= max(tile.std(), 1e-6)

    tiled = np.tile(
        tile,
        (math.ceil(height / tile_size), math.ceil(width / tile_size)),
    )[:height, :width]
    warp = rng.normal(
        0,
        1,
        (max(2, height // 48), max(2, width // 48)),
    ).astype(np.float32)
    warp = cv2.resize(warp, (width, height), interpolation=cv2.INTER_CUBIC)
    texture = tiled * 0.72 + warp * 0.28

    amount = strength(severity, 5, 18) * multiplier
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    surface_mask = (0.25 + 0.75 * (1 - np.abs(gray - 0.55) * 1.5))[..., None]
    colored = np.stack(
        (texture * 0.9, texture * 0.72, texture * 0.48),
        axis=2,
    )
    degraded = image.astype(np.float32) + colored * amount * surface_mask

    sharpen_amount = strength(severity, 0.25, 0.75) * multiplier
    sharpened = degraded + sharpen_amount * (
        degraded - cv2.GaussianBlur(degraded, (0, 0), 0.8)
    )
    return clamp(sharpened), {
        "severity": severity,
        "seed": seed,
        "multiplier": multiplier,
        "tile_size": tile_size,
        "amount": round(amount, 3),
        "sharpen_amount": round(sharpen_amount, 3),
    }


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"))


def save_image_atomic(image: np.ndarray, path: Path, quality: int = 95) -> None:
    """Write through a dot-prefixed temp file in the same directory.

    Nothing partial may ever be visible under a dataset directory: the trainer
    globs it, and a half-written PNG is a crash three phases later.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.writing{path.suffix}")
    save_options = {}
    if path.suffix.lower() in {".jpg", ".jpeg", ".webp"}:
        save_options["quality"] = quality
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        save_options["subsampling"] = 0
    try:
        Image.fromarray(image).save(temporary, **save_options)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def degrade_file(source: Path, dest: Path, *, severity: float = 0.6, seed: int = 42,
                 multiplier: float = 1.0, quality: int = 95) -> dict[str, float | int]:
    """Degrade `source` into `dest`, preserving the source's exact dimensions.

    Preserving dimensions is what makes this pair-safe: musubi buckets the
    control image and the target independently from each own aspect ratio, so a
    degradation that resized would land the two in different buckets. The grit
    is applied in place on the array, so the ratio is identical by construction
    and no fit_control_to_target step is needed here.
    """
    if dest.suffix.lower() not in SUPPORTED_OUTPUTS:
        raise ValueError(f"unsupported output format: {dest.suffix}")
    image = load_rgb(source)
    result, parameters = apply_ai_grit_tiling_texture(
        image, severity=severity, seed=seed, multiplier=multiplier)
    save_image_atomic(result, dest, quality)
    return parameters
