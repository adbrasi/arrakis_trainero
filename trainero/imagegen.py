"""OpenRouter Images API client for the Style Rush synthetic dataset.

One model, one shape of request: gpt-image-2 restyling a single reference image
at `quality: low`, which is 1K native. The endpoint rejects `size`/`resolution`
outright, so framing is controlled through `aspect_ratio` alone.

HTTP is urllib on purpose: the server venv stays free of network dependencies.
"""

from __future__ import annotations

import base64
import json
import math
import os
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "openai/gpt-image-2"
API_URL = "https://openrouter.ai/api/v1/images"

# The ratios gpt-image-2 accepts, as (label, width/height). "auto" is left out
# on purpose: the framing has to follow the owner's image, not the model's guess.
SUPPORTED_RATIOS = [
    ("1:1", 1 / 1), ("3:2", 3 / 2), ("2:3", 2 / 3), ("4:3", 4 / 3),
    ("3:4", 3 / 4), ("16:9", 16 / 9), ("9:16", 9 / 16), ("21:9", 21 / 9),
]

# The reference image is tokenised in 32x32 patches and billed at 8e-6/token, so
# a 1024x1024 reference is 1024 tokens ($0.0082) and a 2048x2048 one costs four
# times that. quality "low" renders at 1K regardless, so a bigger reference buys
# nothing — it only inflates the bill and the upload.
REFERENCE_MAX_EDGE = 1024

# Measured, not derived. Three real calls at quality "low", after the reference
# was capped at REFERENCE_MAX_EDGE:
#   1024x1024 ref -> $0.014187   (1024 input-image + 23 prompt + 196 output tok)
#    796x1024 ref -> $0.011975
#    1024x683 ref -> $0.010517
# A square source is the ceiling: at a fixed longest edge it carries the most
# pixels, and the input image is what dominates the bill. Estimating with the
# ceiling keeps the number shown before TREINAR from ever being a surprise.
# (The published gpt-image-1 table would have said 0.011 — it does not apply.)
COST_PER_IMAGE = 0.0142

_REFUSAL_MARKERS = (
    "safety", "moderation", "content_policy", "content policy", "flagged", "rejected",
)


class ImageGenError(RuntimeError):
    """The image model failed for this slot."""


class RetriableError(ImageGenError):
    """Transient: rate limit or server-side error. Retry the same image."""


class RefusedError(ImageGenError):
    """Moderation refused this image. Retry with a different one."""


def aspect_ratio_for(width: int, height: int) -> str:
    """Nearest supported aspect ratio, in log space so 4:3 and 3:4 are symmetric."""
    if width <= 0 or height <= 0:
        return "1:1"
    target = math.log(width / height)
    return min(SUPPORTED_RATIOS, key=lambda r: abs(math.log(r[1]) - target))[0]


def to_data_url(path: Path) -> tuple[str, int, int]:
    """Read an image as a data URI, plus the dimensions actually sent.

    Anything above REFERENCE_MAX_EDGE is downscaled first: the token bill scales
    with the reference's pixel area, and the model renders at 1K anyway. Images
    already within budget are passed through byte for byte.
    """
    import io

    from PIL import Image

    with Image.open(path) as im:
        if max(im.size) <= REFERENCE_MAX_EDGE:
            fmt = (im.format or "PNG").lower()
            mime = "image/jpeg" if fmt in ("jpg", "jpeg") else f"image/{fmt}"
            encoded = base64.b64encode(path.read_bytes()).decode()
            return f"data:{mime};base64,{encoded}", im.size[0], im.size[1]

        if im.mode not in ("RGB", "RGBA", "L", "P"):
            im = im.convert("RGB")  # CMYK and friends cannot be written as PNG
        im.thumbnail((REFERENCE_MAX_EDGE, REFERENCE_MAX_EDGE), Image.LANCZOS)
        width, height = im.size
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{encoded}", width, height


def build_payload(prompt: str, data_url: str, aspect_ratio: str) -> dict:
    return {
        "model": MODEL,
        "prompt": prompt,
        "n": 1,
        "quality": "low",
        "moderation": "low",
        "aspect_ratio": aspect_ratio,
        "input_references": [{"type": "image_url", "image_url": {"url": data_url}}],
    }


def decode_first_image(body: dict) -> bytes:
    entries = body.get("data") or []
    for entry in entries:
        payload = entry.get("b64_json") or entry.get("image_base64")
        if payload:
            return base64.b64decode(payload)
        url = entry.get("url") or ""
        if url.startswith("data:"):
            return base64.b64decode(url.split(",", 1)[1])
    raise ImageGenError(f"resposta sem imagem decodificável: {str(body)[:400]}")


def classify_http_error(status: int, text: str) -> ImageGenError:
    low = text.lower()
    if status in (400, 403, 422) and any(m in low for m in _REFUSAL_MARKERS):
        return RefusedError(f"moderação recusou a imagem (HTTP {status})")
    if status == 429 or status >= 500:
        return RetriableError(f"HTTP {status}: {text[:300]}")
    return ImageGenError(f"HTTP {status}: {text[:300]}")


def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ImageGenError("defina OPENROUTER_API_KEY para gerar o dataset de conversão")
    return key


def generate(prompt: str, image_path: Path, timeout: float = 300.0) -> tuple[bytes, float]:
    """Restyle one image. Returns the PNG bytes and what the call cost in USD."""
    data_url, width, height = to_data_url(image_path)
    payload = build_payload(prompt, data_url, aspect_ratio_for(width, height))
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        raise classify_http_error(exc.code, exc.read().decode("utf-8", "replace")) from exc
    except urllib.error.URLError as exc:
        raise RetriableError(f"rede: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RetriableError("timeout na chamada da imagem") from exc

    # A 200 can still carry a refusal in the body instead of an image.
    if not body.get("data"):
        text = json.dumps(body)[:400]
        if any(m in text.lower() for m in _REFUSAL_MARKERS):
            raise RefusedError("moderação recusou a imagem (resposta 200 sem dados)")
        raise ImageGenError(f"resposta vazia: {text}")

    cost = float((body.get("usage") or {}).get("cost") or 0.0)
    return decode_first_image(body), cost
