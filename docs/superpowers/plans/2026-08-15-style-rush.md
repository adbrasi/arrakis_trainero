# Style Rush Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adicionar o modo Style Rush ao Arrakis Trainero — o dono manda um dataset de imagens e o trainer constrói sozinho um segundo dataset sintético de conversão de estilo (GPT Image via OpenRouter) e treina os dois juntos num LoRA de Flux Klein 9B, com sample images a cada época.

**Architecture:** Duas unidades novas e puras (`imagegen.py` = cliente HTTP da Images API; `style_rush.py` = montagem do dataset sintético) mais generalizações no que já existe: `write_dataset_toml` passa a receber uma lista de subsets (o modo LoRA normal passa uma lista de um), `comfy_convert` vira dado no preset em vez de código hardcoded no Anima, e o sampling entra em `build_train_config` para todo modelo. O modo novo é um `mode` a mais no `/api/train`, como o slider já é.

**Tech Stack:** Python 3.11+ stdlib (`urllib`, `ThreadPoolExecutor`, `unittest`), Pillow, HTML/CSS/JS vanilla, musubi-tuner (`kohya-ss/musubi-tuner@main`).

**Spec:** `docs/superpowers/specs/2026-08-15-style-rush-design.md`

## Global Constraints

- Respostas ao usuário e mensagens de commit em **pt-BR**; código, identificadores, comentários e docstrings em **inglês**.
- Nenhuma dependência de rede nova: HTTP é `urllib.request` da stdlib. A única dependência nova do venv do servidor é **Pillow**.
- Modelo de imagem: `openai/gpt-image-2`. Endpoint: `https://openrouter.ai/api/v1/images`. Chave: `OPENROUTER_API_KEY`.
- Parâmetros fixos da geração: `quality: "low"`, `moderation: "low"`, `n: 1`. **Nunca** enviar `size` ou `resolution` — o gpt-image-2 não aceita.
- `SLOT_COUNT = 50`. O dataset de conversão tem sempre 50 slots, independente do tamanho do dataset base.
- Schedule do Style Rush: `num_repeats = 1` nos dois subsets, `max_train_epochs = 5`, `save_every_n_epochs = 1`, `sample_every_n_epochs = 1`. Nada de `target_steps` nesse modo.
- `control_resolution = [1024, 1024]`.
- Caption do dataset de conversão: `convert the style of this image to the {trigger} style`, idêntica nos 50.
- Testes: `unittest`, rodados com `python -m unittest discover -s tests -v` a partir da raiz do repo. Sem GPU e sem rede.
- Erros: nada de gambiarra. Falha real para o job com a causa no log. Nenhum fallback silencioso.

---

### Task 1: Prompts de estilo e seleção de slots

**Files:**
- Create: `data/style_prompts.txt`
- Create: `trainero/style_rush.py`
- Create: `tests/test_style_rush.py`

**Interfaces:**
- Consumes: nada.
- Produces:
  - `trainero.style_rush.SLOT_COUNT: int` = 50
  - `trainero.style_rush.CAPTION_TEMPLATE: str` = `"convert the style of this image to the {trigger} style"`
  - `trainero.style_rush.load_style_prompts(path: Path | None = None) -> list[str]`
  - `trainero.style_rush.plan_slots(images: list[Path], prompts: list[str]) -> list[dict]` — retorna exatamente `SLOT_COUNT` dicts `{"slot": "slot_NN", "prompt": str, "sources": [str, str]}` (caminhos absolutos como `str`; `sources` tem 2 entradas distintas quando há ≥2 imagens, senão 1).

- [ ] **Step 1: Criar `data/style_prompts.txt` com as 50 linhas**

As 9 primeiras são as do dono (com `coloized` → `colorized` corrigido). Uma por linha, sem linha em branco:

```
Improve the quality of this image, remove noise, remove pixelation, make it HD, well-drawn, flat-color, cinematic anime style.
convert the art style of this artwork to sketch black and white manga japanese artstyle.
convert the art style of this artwork to sketch pastel colored, amateur draw on paper.
convert the art style of this artwork to 3d, pixar style, cgi league of legends, playstation 4.
convert the art style of this artwork to sketch colorized, colored manga anime japanese comic artstyle.
convert the art style of this artwork to fanart deviantart, cute, procreate ipad digital artwork, 2.5d, semi-realistic, realistic shadows, amazing detailed quality.
turn this image hd, keep same style, change the low details, fine details, enhance the quality to premium quality, amazing 4k hd image.
turn this image hd, keep same style, change the low details, fine details, enhance the quality to premium quality, amazing 4k hd image. 2D anime screencap style, fix skin, fix light. 2d.
change the style, fix the light, shadows and artwork. keep atmosphere.
convert the art style of this artwork to rough pencil sketch on textured paper, visible graphite strokes, no color.
convert the art style of this artwork to inked lineart only, thick black outlines, flat white fill, coloring book style.
convert the art style of this artwork to watercolor painting on cold press paper, soft bleeding edges, visible paper grain.
convert the art style of this artwork to thick oil painting, heavy impasto brush strokes, canvas texture, gallery lighting.
convert the art style of this artwork to gouache illustration, matte flat pigment, poster-like simplified shapes.
convert the art style of this artwork to 90s retro anime cel animation, film grain, slightly faded colors, VHS look.
convert the art style of this artwork to modern digital anime key visual, glossy highlights, sharp cel shading, vivid saturation.
convert the art style of this artwork to chibi super deformed style, big head, tiny body, simple cute shapes.
convert the art style of this artwork to american comic book style, bold ink, halftone dots, heavy shadow blocks.
convert the art style of this artwork to european bande dessinee ligne claire, clean uniform lines, flat color areas.
convert the art style of this artwork to low poly 3d render, faceted geometry, simple studio lighting.
convert the art style of this artwork to clay stop-motion figure, fingerprint texture, soft practical lighting.
convert the art style of this artwork to felt and fabric craft, stitched seams, soft plush texture.
convert the art style of this artwork to pixel art, 32-bit limited palette, hard aliased edges, retro game sprite.
convert the art style of this artwork to vector flat illustration, no gradients, geometric shapes, corporate flat design.
convert the art style of this artwork to photorealistic photography, real skin pores, natural depth of field, DSLR look.
convert the art style of this artwork to grainy 35mm film photograph, warm color cast, slight motion blur.
convert the art style of this artwork to black and white noir photography, hard directional light, deep crushed blacks.
convert the art style of this artwork to airbrushed 80s retro anime poster, soft gradients, muted pastel palette.
convert the art style of this artwork to heavy screentone manga page, black and white, dense hatching, dramatic speed lines.
convert the art style of this artwork to ukiyo-e japanese woodblock print, flat inked areas, visible paper texture.
convert the art style of this artwork to chinese ink wash painting, monochrome bleeding brush, large empty space.
convert the art style of this artwork to stained glass window, thick black leading, saturated translucent color panels.
convert the art style of this artwork to charcoal drawing, smudged shading, rough dark strokes on grey paper.
convert the art style of this artwork to colored pencil drawing, visible hatching, waxy layered pigment.
convert the art style of this artwork to marker illustration, alcohol marker blending, visible stroke banding.
convert the art style of this artwork to crayon drawing by a child, wobbly lines, uneven pressure, cheap paper.
convert the art style of this artwork to cutout paper collage, layered flat paper, drop shadows between layers.
convert the art style of this artwork to cel-shaded video game render, hard shadow terminator, thin outline, unreal engine toon shader.
convert the art style of this artwork to realistic 3d character render, subsurface scattering skin, cinematic rim light, octane render.
convert the art style of this artwork to concept art matte painting, loose painterly strokes, wide cinematic value range.
convert the art style of this artwork to soft light novel illustration, delicate lineart, pale skin, gentle pastel shading.
convert the art style of this artwork to gritty seinen manga art, harsh crosshatching, rough textured inking.
convert the art style of this artwork to vaporwave aesthetic, magenta and cyan glow, chromatic aberration, scanlines.
convert the art style of this artwork to blurry low quality webcam screenshot, jpeg artifacts, bad lighting, compressed.
convert the art style of this artwork to over-sharpened oversaturated amateur digital art, harsh contrast, muddy blending.
convert the art style of this artwork to monochrome blue-toned illustration, single hue palette, high contrast.
convert the art style of this artwork to sepia vintage engraving, fine etched lines, aged paper stains.
convert the art style of this artwork to graffiti spray paint on concrete wall, drips, stencil edges, urban texture.
convert the art style of this artwork to tattoo flash art, bold outlines, limited ink palette, traditional shading.
convert the art style of this artwork to soft blurred pastel dream, glowing bloom, low contrast, hazy diffuse light.
```

- [ ] **Step 2: Escrever os testes que falham**

Criar `tests/test_style_rush.py`:

```python
"""Unit tests for the Style Rush synthetic dataset pipeline (no GPU, no network)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainero.style_rush import (SLOT_COUNT, CAPTION_TEMPLATE, load_style_prompts,
                                 plan_slots)


class TestStylePrompts(unittest.TestCase):
    def test_fifty_distinct_prompts(self):
        prompts = load_style_prompts()
        self.assertEqual(len(prompts), SLOT_COUNT)
        self.assertEqual(len(set(prompts)), SLOT_COUNT, "prompts must be distinct")
        for p in prompts:
            self.assertTrue(p.strip(), "no blank prompt")

    def test_caption_template(self):
        self.assertEqual(
            CAPTION_TEMPLATE.format(trigger="makima"),
            "convert the style of this image to the makima style",
        )


class TestPlanSlots(unittest.TestCase):
    def _imgs(self, n):
        return [Path(f"/ds/img_{i:03d}.png") for i in range(n)]

    def test_always_fifty_slots(self):
        prompts = load_style_prompts()
        for n in (1, 2, 7, 50, 200):
            slots = plan_slots(self._imgs(n), prompts)
            self.assertEqual(len(slots), SLOT_COUNT, n)
            self.assertEqual([s["slot"] for s in slots],
                             [f"slot_{i:02d}" for i in range(SLOT_COUNT)], n)

    def test_each_slot_gets_a_distinct_prompt(self):
        prompts = load_style_prompts()
        slots = plan_slots(self._imgs(10), prompts)
        used = [s["prompt"] for s in slots]
        self.assertEqual(len(set(used)), SLOT_COUNT)
        self.assertEqual(set(used), set(prompts))

    def test_large_dataset_uses_distinct_images(self):
        slots = plan_slots(self._imgs(200), load_style_prompts())
        primaries = [s["sources"][0] for s in slots]
        self.assertEqual(len(set(primaries)), SLOT_COUNT)

    def test_small_dataset_wraps_around(self):
        slots = plan_slots(self._imgs(10), load_style_prompts())
        primaries = [s["sources"][0] for s in slots]
        self.assertEqual(len(set(primaries)), 10)
        # each image is reused 5 times, always with a different prompt
        self.assertEqual(len(set(s["prompt"] for s in slots)), SLOT_COUNT)

    def test_fallback_differs_from_primary(self):
        slots = plan_slots(self._imgs(10), load_style_prompts())
        for s in slots:
            self.assertEqual(len(s["sources"]), 2, s["slot"])
            self.assertNotEqual(s["sources"][0], s["sources"][1], s["slot"])

    def test_single_image_has_no_fallback(self):
        slots = plan_slots(self._imgs(1), load_style_prompts())
        for s in slots:
            self.assertEqual(len(s["sources"]), 1)

    def test_deterministic(self):
        prompts = load_style_prompts()
        a = plan_slots(self._imgs(37), prompts)
        b = plan_slots(self._imgs(37), prompts)
        self.assertEqual(a, b)

    def test_empty_dataset_raises(self):
        with self.assertRaises(ValueError):
            plan_slots([], load_style_prompts())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Rodar os testes e ver falhar**

Run: `python -m unittest discover -s tests -v -k StyleRush 2>&1 | tail -20`
Ou simplesmente: `python -m unittest tests.test_style_rush -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'trainero.style_rush'`

- [ ] **Step 4: Implementar `trainero/style_rush.py` (parte 1)**

```python
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
```

- [ ] **Step 5: Rodar os testes e ver passar**

Run: `python -m unittest tests.test_style_rush -v`
Expected: PASS, 9 testes.

- [ ] **Step 6: Commit**

```bash
git add data/style_prompts.txt trainero/style_rush.py tests/test_style_rush.py
git commit -m "feat(style-rush): 50 prompts de estilo e plano determinístico de slots"
```

---

### Task 2: Cliente da Images API do OpenRouter

**Files:**
- Create: `trainero/imagegen.py`
- Create: `tests/test_imagegen.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: nada de tasks anteriores.
- Produces:
  - `trainero.imagegen.ImageGenError(RuntimeError)` — falha definitiva do slot
  - `trainero.imagegen.RetriableError(ImageGenError)` — 429/5xx/timeout, vale tentar de novo na mesma imagem
  - `trainero.imagegen.RefusedError(ImageGenError)` — moderação recusou esta imagem
  - `trainero.imagegen.MODEL: str` = `"openai/gpt-image-2"`
  - `trainero.imagegen.API_URL: str` = `"https://openrouter.ai/api/v1/images"`
  - `trainero.imagegen.aspect_ratio_for(width: int, height: int) -> str`
  - `trainero.imagegen.to_data_url(path: Path) -> tuple[str, int, int]` — data URI, largura, altura
  - `trainero.imagegen.build_payload(prompt: str, data_url: str, aspect_ratio: str) -> dict`
  - `trainero.imagegen.decode_first_image(body: dict) -> bytes`
  - `trainero.imagegen.generate(prompt: str, image_path: Path, timeout: float = 300.0) -> tuple[bytes, float]` — bytes do PNG e custo em USD

- [ ] **Step 1: Escrever os testes que falham**

Criar `tests/test_imagegen.py`:

```python
"""Unit tests for the OpenRouter Images API client (payload shape only, no network)."""

import base64
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainero.imagegen import (MODEL, RefusedError, RetriableError, aspect_ratio_for,
                               build_payload, classify_http_error, decode_first_image)


class TestAspectRatio(unittest.TestCase):
    def test_square(self):
        self.assertEqual(aspect_ratio_for(1024, 1024), "1:1")
        self.assertEqual(aspect_ratio_for(900, 910), "1:1")

    def test_landscape_and_portrait(self):
        self.assertEqual(aspect_ratio_for(1536, 1024), "3:2")
        self.assertEqual(aspect_ratio_for(1024, 1536), "2:3")
        self.assertEqual(aspect_ratio_for(1920, 1080), "16:9")
        self.assertEqual(aspect_ratio_for(1080, 1920), "9:16")
        self.assertEqual(aspect_ratio_for(1024, 768), "4:3")
        self.assertEqual(aspect_ratio_for(768, 1024), "3:4")

    def test_extreme_ratio_clamps_to_nearest_supported(self):
        # 21:9 is not in our list; nearest supported is 16:9
        self.assertEqual(aspect_ratio_for(2560, 1080), "16:9")

    def test_zero_dimension_is_square(self):
        self.assertEqual(aspect_ratio_for(0, 0), "1:1")


class TestPayload(unittest.TestCase):
    def test_shape(self):
        p = build_payload("make it manga", "data:image/png;base64,AAAA", "3:2")
        self.assertEqual(p["model"], MODEL)
        self.assertEqual(p["prompt"], "make it manga")
        self.assertEqual(p["n"], 1)
        self.assertEqual(p["quality"], "low")
        self.assertEqual(p["moderation"], "low")
        self.assertEqual(p["aspect_ratio"], "3:2")
        self.assertEqual(p["input_references"],
                         [{"type": "image_url",
                           "image_url": {"url": "data:image/png;base64,AAAA"}}])

    def test_never_sends_size_or_resolution(self):
        p = build_payload("x", "data:image/png;base64,AAAA", "1:1")
        self.assertNotIn("size", p)
        self.assertNotIn("resolution", p)


class TestDecode(unittest.TestCase):
    def test_b64_json(self):
        raw = b"\x89PNG\r\n\x1a\nfake"
        body = {"data": [{"b64_json": base64.b64encode(raw).decode()}]}
        self.assertEqual(decode_first_image(body), raw)

    def test_data_url(self):
        raw = b"\x89PNG\r\n\x1a\nfake"
        url = "data:image/png;base64," + base64.b64encode(raw).decode()
        body = {"data": [{"url": url}]}
        self.assertEqual(decode_first_image(body), raw)

    def test_empty_data_raises(self):
        from trainero.imagegen import ImageGenError

        with self.assertRaises(ImageGenError):
            decode_first_image({"data": []})


class TestErrorClassification(unittest.TestCase):
    def test_moderation_is_refusal(self):
        exc = classify_http_error(400, '{"error":{"message":"rejected by the safety system"}}')
        self.assertIsInstance(exc, RefusedError)

    def test_moderation_variants(self):
        for msg in ("safety_violations detected", "Your request was flagged by moderation",
                    "content_policy_violation"):
            self.assertIsInstance(classify_http_error(400, msg), RefusedError, msg)

    def test_rate_limit_and_server_errors_are_retriable(self):
        for code in (429, 500, 502, 503, 504):
            self.assertIsInstance(classify_http_error(code, "boom"), RetriableError, code)

    def test_auth_error_is_fatal(self):
        exc = classify_http_error(401, "no key")
        self.assertNotIsInstance(exc, RetriableError)
        self.assertNotIsInstance(exc, RefusedError)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python -m unittest tests.test_imagegen -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'trainero.imagegen'`

- [ ] **Step 3: Implementar `trainero/imagegen.py`**

```python
"""OpenRouter Images API client for the Style Rush synthetic dataset.

One model, one shape of request: gpt-image-2 restyling a single reference image
at `quality: low`, which is 1K native. The endpoint rejects `size`/`resolution`
outright, so framing is controlled through `aspect_ratio` alone.

HTTP is urllib on purpose: the server venv stays free of network dependencies.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

MODEL = "openai/gpt-image-2"
API_URL = "https://openrouter.ai/api/v1/images"

# The ratios gpt-image-2 accepts, as (label, width/height).
SUPPORTED_RATIOS = [
    ("1:1", 1 / 1), ("3:2", 3 / 2), ("2:3", 2 / 3), ("4:3", 4 / 3),
    ("3:4", 3 / 4), ("16:9", 16 / 9), ("9:16", 9 / 16),
]

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
    import math

    target = math.log(width / height)
    return min(SUPPORTED_RATIOS, key=lambda r: abs(math.log(r[1]) - target))[0]


def to_data_url(path: Path) -> tuple[str, int, int]:
    """Read an image as a data URI, plus its pixel dimensions."""
    from PIL import Image

    with Image.open(path) as im:
        width, height = im.size
        fmt = (im.format or "PNG").lower()
    mime = "image/jpeg" if fmt in ("jpg", "jpeg") else f"image/{fmt}"
    encoded = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{encoded}", width, height


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
```

- [ ] **Step 4: Adicionar Pillow ao `requirements.txt`**

O arquivo passa a ser:

```
# Server venv only — training engines get their own venvs (see trainero/engines.py)
huggingface_hub[hf_xet]>=0.31.4,<2.0
pillow>=10.0
```

- [ ] **Step 5: Rodar e ver passar**

Run: `python -m unittest tests.test_imagegen -v`
Expected: PASS, 11 testes. Se `PIL` não estiver no venv local, os testes ainda passam — `to_data_url` é a única função que importa Pillow e ela não é exercida aqui.

- [ ] **Step 6: Commit**

```bash
git add trainero/imagegen.py tests/test_imagegen.py requirements.txt
git commit -m "feat(style-rush): cliente da Images API do OpenRouter (gpt-image-2, quality low)"
```

---

### Task 3: Montagem do dataset de conversão

**Files:**
- Modify: `trainero/style_rush.py`
- Modify: `tests/test_style_rush.py`

**Interfaces:**
- Consumes: `plan_slots`, `load_style_prompts`, `SLOT_COUNT`, `CAPTION_TEMPLATE` (Task 1); `imagegen.generate`, `imagegen.RefusedError`, `imagegen.RetriableError`, `imagegen.ImageGenError` (Task 2); `trainero.jobs.Job`, `trainero.jobs.JobFailed`.
- Produces:
  - `trainero.style_rush.MANIFEST_NAME: str` = `".style_rush.json"`
  - `trainero.style_rush.build_convert_dataset(base_dir: Path, convert_dir: Path, trigger: str, job, generate=imagegen.generate, workers: int = 4) -> dict` — retorna `{"pairs": int, "refused": int, "failed": int, "cost": float}`

- [ ] **Step 1: Escrever os testes que falham**

Acrescentar a `tests/test_style_rush.py` (mantendo os imports do topo e adicionando os novos):

```python
import json
import tempfile

from trainero.imagegen import RefusedError, RetriableError
from trainero.style_rush import MANIFEST_NAME, build_convert_dataset


class _FakeJob:
    """Minimal stand-in for trainero.jobs.Job — collects log lines."""

    def __init__(self):
        self.lines = []

    def log(self, msg):
        self.lines.append(msg)

    def check_cancel(self):
        pass


def _png_bytes(color=b"\x00"):
    """A 1x1 PNG, enough for the pipeline to copy bytes around."""
    from PIL import Image
    import io

    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    return buf.getvalue()


def _make_dataset(root: Path, n: int) -> Path:
    from PIL import Image

    base = root / "dataset"
    base.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (64, 48)).save(base / f"img_{i:03d}.png")
        (base / f"img_{i:03d}.txt").write_text("makima, a girl")
    return base


class TestBuildConvertDataset(unittest.TestCase):
    def test_happy_path_writes_fifty_pairs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 60)
            convert = root / "dataset_convert"
            calls = []

            def fake_generate(prompt, image_path, timeout=300.0):
                calls.append((prompt, str(image_path)))
                return _png_bytes(), 0.011

            result = build_convert_dataset(base, convert, "makima", _FakeJob(),
                                           generate=fake_generate, workers=2)

            self.assertEqual(result["pairs"], SLOT_COUNT)
            self.assertEqual(result["refused"], 0)
            self.assertAlmostEqual(result["cost"], 0.011 * SLOT_COUNT, places=4)
            self.assertEqual(len(calls), SLOT_COUNT)
            targets = sorted(p.name for p in convert.glob("slot_*.png"))
            controls = sorted(p.name for p in (convert / "control").glob("slot_*.png"))
            self.assertEqual(len(targets), SLOT_COUNT)
            self.assertEqual(targets, controls)
            caption = (convert / "slot_00.txt").read_text()
            self.assertEqual(caption, "convert the style of this image to the makima style")

    def test_refusal_retries_with_a_different_image(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 10)
            convert = root / "dataset_convert"
            seen = []

            def fake_generate(prompt, image_path, timeout=300.0):
                seen.append(str(image_path))
                # every slot_00 primary is refused, everything else succeeds
                if len([s for s in seen if s == str(image_path)]) == 1 and prompt.startswith("Improve"):
                    raise RefusedError("moderação recusou a imagem")
                return _png_bytes(), 0.011

            result = build_convert_dataset(base, convert, "makima", _FakeJob(),
                                           generate=fake_generate, workers=1)
            self.assertEqual(result["pairs"], SLOT_COUNT)
            self.assertEqual(result["refused"], 1)

    def test_two_refusals_drop_the_slot(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 10)
            convert = root / "dataset_convert"

            def fake_generate(prompt, image_path, timeout=300.0):
                raise RefusedError("moderação recusou a imagem")

            from trainero.jobs import JobFailed

            with self.assertRaises(JobFailed):
                build_convert_dataset(base, convert, "makima", _FakeJob(),
                                      generate=fake_generate, workers=1)

    def test_retriable_error_retries_the_same_image(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 10)
            convert = root / "dataset_convert"
            attempts = {}

            def fake_generate(prompt, image_path, timeout=300.0):
                key = str(image_path)
                attempts[key] = attempts.get(key, 0) + 1
                if attempts[key] == 1:
                    raise RetriableError("HTTP 503")
                return _png_bytes(), 0.011

            result = build_convert_dataset(base, convert, "makima", _FakeJob(),
                                           generate=fake_generate, workers=1)
            self.assertEqual(result["pairs"], SLOT_COUNT)
            self.assertEqual(result["refused"], 0)

    def test_resume_does_not_regenerate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 60)
            convert = root / "dataset_convert"

            def fake_generate(prompt, image_path, timeout=300.0):
                return _png_bytes(), 0.011

            build_convert_dataset(base, convert, "makima", _FakeJob(),
                                  generate=fake_generate, workers=2)

            second_calls = []

            def counting_generate(prompt, image_path, timeout=300.0):
                second_calls.append(prompt)
                return _png_bytes(), 0.011

            result = build_convert_dataset(base, convert, "makima", _FakeJob(),
                                           generate=counting_generate, workers=2)
            self.assertEqual(second_calls, [])
            self.assertEqual(result["pairs"], SLOT_COUNT)

    def test_manifest_records_each_slot(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 60)
            convert = root / "dataset_convert"

            def fake_generate(prompt, image_path, timeout=300.0):
                return _png_bytes(), 0.011

            build_convert_dataset(base, convert, "makima", _FakeJob(),
                                  generate=fake_generate, workers=2)
            manifest = json.loads((convert / MANIFEST_NAME).read_text())
            self.assertEqual(len(manifest["slots"]), SLOT_COUNT)
            entry = manifest["slots"]["slot_00"]
            self.assertEqual(entry["status"], "ok")
            self.assertTrue(entry["prompt"])
            self.assertTrue(entry["source"])

    def test_empty_base_dataset_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = root / "dataset"
            base.mkdir()
            from trainero.jobs import JobFailed

            with self.assertRaises(JobFailed):
                build_convert_dataset(base, root / "dataset_convert", "makima", _FakeJob(),
                                      generate=lambda *a, **k: (_png_bytes(), 0.0))
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python -m unittest tests.test_style_rush -v`
Expected: FAIL com `ImportError: cannot import name 'build_convert_dataset'`

- [ ] **Step 3: Implementar em `trainero/style_rush.py`**

Acrescentar aos imports do topo do arquivo:

```python
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import imagegen
from .config import IMAGE_EXTS, REPO_DIR
from .jobs import JobFailed
```

(o import de `REPO_DIR` já existe da Task 1 — junte na mesma linha)

E acrescentar ao final do arquivo:

```python
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


def build_convert_dataset(base_dir: Path, convert_dir: Path, trigger: str, job,
                          generate=imagegen.generate, workers: int = 4) -> dict:
    """Produce SLOT_COUNT (control, target, caption) triples under convert_dir.

    control/<slot>.png  the restyled image from GPT Image
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
    with lock:
        totals["pairs"] = len(slots) - len(pending)
    if pending:
        job.log(f"Gerando {len(pending)} imagens de conversão com {imagegen.MODEL} "
                f"(quality low) — ~${0.011 * len(pending):.2f}")
    else:
        job.log(f"Dataset de conversão já completo ({len(slots)} pares).")

    def work(slot: dict) -> None:
        job.check_cancel()
        name = slot["slot"]
        refusals = 0
        for source in slot["sources"]:  # primary, then the fallback image
            try:
                png, cost = _generate_with_retries(generate, slot["prompt"], Path(source))
            except imagegen.RefusedError as exc:
                refusals += 1
                with lock:
                    manifest["slots"][name] = {"status": "refused", "prompt": slot["prompt"],
                                               "source": source, "error": str(exc)}
                continue
            except imagegen.ImageGenError as exc:
                with lock:
                    totals["failed"] += 1
                    manifest["slots"][name] = {"status": "failed", "prompt": slot["prompt"],
                                               "source": source, "error": str(exc)}
                job.log(f"⚠ {name}: {exc}")
                return

            (control_dir / f"{name}.png").write_bytes(png)
            shutil.copyfile(source, convert_dir / f"{name}.png")
            (convert_dir / f"{name}.txt").write_text(caption, encoding="utf-8")
            with lock:
                totals["pairs"] += 1
                totals["refused"] += refusals
                totals["cost"] += cost
                manifest["slots"][name] = {"status": "ok", "prompt": slot["prompt"],
                                           "source": source, "cost": cost}
            return

        with lock:
            totals["refused"] += refusals
        job.log(f"⚠ {name}: recusado nas duas tentativas — slot descartado.")

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(work, pending))
        _save_manifest(convert_dir, manifest)

    if totals["pairs"] == 0:
        raise JobFailed(
            "o dataset de conversão ficou vazio — todas as imagens foram recusadas ou "
            "falharam. Sem ele o LoRA não aprende a converter estilo.")

    job.log(f"✔ Dataset de conversão: {totals['pairs']} pares, "
            f"{totals['refused']} recusas, custo ~${totals['cost']:.2f}")
    return totals
```

- [ ] **Step 4: Rodar e ver passar**

Run: `python -m unittest tests.test_style_rush -v`
Expected: PASS. Requer Pillow no venv local (`pip install pillow` se faltar).

- [ ] **Step 5: Commit**

```bash
git add trainero/style_rush.py tests/test_style_rush.py
git commit -m "feat(style-rush): montagem do dataset sintético com retomada e retry por imagem"
```

---

### Task 4: Presets — control, schedule, prompt de sample, comfy_convert como dado

**Files:**
- Modify: `trainero/presets.py`
- Modify: `tests/test_core.py`

**Interfaces:**
- Consumes: nada.
- Produces:
  - `trainero.presets.STYLE_RUSH_SCHEDULE: dict` = `{"num_repeats": 1, "epochs": 5, "save_every_n_epochs": 1}`
  - `trainero.presets.SAMPLE_PROMPT: str` — o texto padrão em prosa
  - `trainero.presets.SAMPLE_STEPS: int` = 28, `trainero.presets.SAMPLE_GUIDANCE: float` = 4.0, `trainero.presets.SAMPLE_SEED: int` = 42
  - `trainero.presets.CONTROL_RESOLUTION: list[int]` = `[1024, 1024]`
  - `trainero.presets.style_rush_models() -> list[str]`
  - chave `supports_control: True` em `flux-klein` e `qwen-image-edit`
  - chave `comfy_convert` como dict: `{"script": "networks/convert_anima_lora_to_comfy.py"}` no `anima`
  - `public_presets()` passa a expor `supports_control`

- [ ] **Step 1: Escrever os testes que falham**

Acrescentar a `tests/test_core.py`, dentro de `class TestPresets`:

```python
    def test_style_rush_models(self):
        from trainero.presets import style_rush_models

        keys = style_rush_models()
        self.assertIn("flux-klein", keys)
        self.assertIn("qwen-image-edit", keys)
        self.assertNotIn("wan-22", keys)
        self.assertNotIn("anima", keys)

    def test_style_rush_schedule_is_fixed(self):
        from trainero.presets import STYLE_RUSH_SCHEDULE

        self.assertEqual(STYLE_RUSH_SCHEDULE,
                         {"num_repeats": 1, "epochs": 5, "save_every_n_epochs": 1})

    def test_sample_prompt_is_prose(self):
        from trainero.presets import SAMPLE_PROMPT

        self.assertGreater(len(SAMPLE_PROMPT.split()), 60)
        self.assertNotIn("\n", SAMPLE_PROMPT)
        self.assertNotIn("--", SAMPLE_PROMPT)  # flags are added by write_sample_prompts

    def test_comfy_convert_is_data(self):
        self.assertEqual(MODELS["anima"]["comfy_convert"],
                         {"script": "networks/convert_anima_lora_to_comfy.py"})
        for key in MODEL_ORDER:
            cc = MODELS[key].get("comfy_convert")
            if cc is not None:
                self.assertIsInstance(cc, dict, key)
                self.assertTrue({"script", "convert_lora"} & set(cc), key)

    def test_public_presets_expose_control(self):
        pub = public_presets()
        self.assertTrue(pub["flux-klein"]["supports_control"])
        self.assertFalse(pub["krea2"]["supports_control"])
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python -m unittest tests.test_core -v`
Expected: FAIL com `ImportError: cannot import name 'style_rush_models'`

- [ ] **Step 3: Editar `trainero/presets.py`**

Trocar em `MODELS["anima"]` a linha:

```python
        "comfy_convert": True,  # sd-scripts LoRA needs convert_anima_lora_to_comfy.py
```

por:

```python
        # sd-scripts LoRA keys do not match what ComfyUI loads for Anima.
        "comfy_convert": {"script": "networks/convert_anima_lora_to_comfy.py"},
```

Em `MODELS["flux-klein"]`, logo após `"resolution": [1024, 1024],`, acrescentar:

```python
        "supports_control": True,   # FLUX.2 reference image = control_directory
```

Em `MODELS["qwen-image-edit"]`, logo após `"needs_control": True,`, acrescentar:

```python
        "supports_control": True,
```

Acrescentar ao final do arquivo, antes de `def net_types_for`:

```python
# ---------------------------------------------------------------------------
# Style Rush
# ---------------------------------------------------------------------------
# The synthetic conversion dataset is always SLOT_COUNT pairs and the owner
# cancels when the samples look right, so there is no target_steps math here.

STYLE_RUSH_SCHEDULE = {"num_repeats": 1, "epochs": 5, "save_every_n_epochs": 1}
CONTROL_RESOLUTION = [1024, 1024]

# ---------------------------------------------------------------------------
# Sampling during training
# ---------------------------------------------------------------------------
# One prompt, the same for every image model. It is written to exercise, in a
# single frame, everything that reveals a LoRA going wrong: face and gaze, hands
# holding an object, fabric, an animal, warm directional light with shadow, and
# a background with depth.

SAMPLE_PROMPT = (
    "A young woman sits alone at a tall window in a quiet apartment at golden hour, "
    "one knee drawn up onto the cushioned sill, both hands wrapped around a chipped "
    "ceramic mug. Late sunlight falls across her in long amber bars, warm on her cheek "
    "and throat, and fine dust turns slowly in the air. A grey cat lies curled asleep "
    "against her hip, one paw over its face. Her hair spills loose over one shoulder, "
    "her sweater slipping wide at the collar, and she looks out through the glass with "
    "a soft, unhurried gaze while the city beyond dissolves into hazy blue rooftops."
)
SAMPLE_STEPS = 28
SAMPLE_GUIDANCE = 4.0
SAMPLE_SEED = 42


def style_rush_models() -> list[str]:
    """Models that accept a control image, so they can learn style conversion."""
    return [k for k in MODEL_ORDER if MODELS[k].get("supports_control")]
```

Em `public_presets()`, dentro do dict `out[key]`, acrescentar após a linha `"needs_control": ...`:

```python
            "supports_control": m.get("supports_control", False),
```

- [ ] **Step 4: Rodar e ver passar**

Run: `python -m unittest tests.test_core -v`
Expected: PASS. `test_comfy_convert_is_data` cobre o formato novo; nenhum outro teste referencia `comfy_convert`.

- [ ] **Step 5: Commit**

```bash
git add trainero/presets.py tests/test_core.py
git commit -m "feat(style-rush): presets ganham supports_control, schedule fixo e prompt de sample"
```

---

### Task 5: `write_dataset_toml` por lista de subsets

**Files:**
- Modify: `trainero/training.py:63-138`
- Modify: `trainero/training.py:392-514` (as duas chamadas dentro de `run_training`)
- Modify: `trainero/sliders.py` (chamada do slider)
- Modify: `tests/test_core.py:108-161`

**Interfaces:**
- Consumes: `presets.CONTROL_RESOLUTION` (Task 4).
- Produces:
  - `trainero.training.write_dataset_toml(model_key: str, path: Path, subsets: list[dict], resolution: list[int], batch_size: int, ltx_cfg: dict | None = None) -> Path`
  - forma do subset: `{"dir": Path, "cache": Path, "num_repeats": int, "media": "image"|"video", "control_dir": Path|None, "control_resolution": list[int]|None}`
  - `trainero.training.image_subset(dataset_dir: Path, cache_dir: Path, num_repeats: int, control_dir: Path | None = None, control_resolution: list[int] | None = None) -> dict`

Nota: a assinatura nova recebe o **caminho final do toml** em vez de `pdir`, e não recebe mais `schedule` nem `stats` — o chamador monta os subsets.

- [ ] **Step 1: Reescrever os testes de TOML**

Substituir a classe `TestDatasetToml` inteira em `tests/test_core.py` por:

```python
class TestDatasetToml(unittest.TestCase):
    def _write(self, key, subsets, resolution=(1024, 1024), batch_size=1, ltx_cfg=None):
        import tempfile

        td = tempfile.mkdtemp()
        path = Path(td) / "dataset.toml"
        return write_dataset_toml(key, path, subsets, list(resolution), batch_size, ltx_cfg)

    def test_musubi_image(self):
        toml = self._write("qwen-image", [
            image_subset(Path("/ds"), Path("/cache"), 3),
        ])
        text = toml.read_text()
        self.assertIn("image_directory", text)
        self.assertIn("num_repeats = 3", text)
        self.assertIn("enable_bucket = true", text)
        self.assertNotIn("control_directory", text)

    def test_musubi_edit_has_control(self):
        toml = self._write("qwen-image-edit", [
            image_subset(Path("/ds"), Path("/cache"), 1,
                         control_dir=Path("/ds/control"), control_resolution=[1024, 1024]),
        ])
        text = toml.read_text()
        self.assertIn("control_directory", text)
        self.assertIn("control_resolution = [1024, 1024]", text)

    def test_two_subsets_only_one_has_control(self):
        toml = self._write("flux-klein", [
            image_subset(Path("/ds"), Path("/cache/images"), 1),
            image_subset(Path("/conv"), Path("/cache/convert"), 1,
                         control_dir=Path("/conv/control"), control_resolution=[1024, 1024]),
        ])
        text = toml.read_text()
        self.assertEqual(text.count("[[datasets]]"), 2)
        self.assertEqual(text.count("control_directory"), 1)
        self.assertEqual(text.count("control_resolution"), 1)
        self.assertIn('image_directory = "/ds"', text)
        self.assertIn('image_directory = "/conv"', text)

    def test_ltx_video(self):
        toml = self._write("ltx-23", [
            {"dir": Path("/ds"), "cache": Path("/cache"), "num_repeats": 1,
             "media": "video", "control_dir": None, "control_resolution": None},
        ], resolution=(768, 512),
            ltx_cfg={"resolution": "768x512x81", "fps": 25.0})
        text = toml.read_text()
        self.assertIn("video_directory", text)
        self.assertIn("target_frames = [81]", text)
        self.assertIn("target_fps = 25.0", text)

    def test_sdscripts_anima(self):
        toml = self._write("anima", [
            image_subset(Path("/ds"), Path("/cache"), 8),
        ], batch_size=8)
        text = toml.read_text()
        self.assertIn("[[datasets.subsets]]", text)
        self.assertIn("bucket_reso_steps = 64", text)
        self.assertIn("num_repeats = 8", text)
```

E ajustar o import no topo de `tests/test_core.py`:

```python
from trainero.training import (_cli_args, build_train_config, image_subset, slugify,
                               write_dataset_toml)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python -m unittest tests.test_core -v`
Expected: FAIL com `ImportError: cannot import name 'image_subset'`

- [ ] **Step 3: Substituir `write_dataset_toml` em `trainero/training.py`**

Trocar todo o bloco das linhas 59-138 (o comentário de seção `dataset.toml` e a função) por:

```python
# ---------------------------------------------------------------------------
# dataset.toml
# ---------------------------------------------------------------------------
# A subset is one [[datasets]] block. Style Rush writes two — the base dataset
# and the conversion dataset with its control_directory — and the musubi loader
# keeps their batches apart on its own (buckets are split by control count), so
# nothing else in the pipeline has to know there is more than one.


def image_subset(dataset_dir: Path, cache_dir: Path, num_repeats: int,
                 control_dir: Path | None = None,
                 control_resolution: list[int] | None = None) -> dict:
    return {"dir": dataset_dir, "cache": cache_dir, "num_repeats": num_repeats,
            "media": "image", "control_dir": control_dir,
            "control_resolution": control_resolution}


def video_subset(dataset_dir: Path, cache_dir: Path, num_repeats: int) -> dict:
    return {"dir": dataset_dir, "cache": cache_dir, "num_repeats": num_repeats,
            "media": "video", "control_dir": None, "control_resolution": None}


def write_dataset_toml(model_key: str, path: Path, subsets: list[dict],
                       resolution: list[int], batch_size: int,
                       ltx_cfg: dict | None = None) -> Path:
    model = MODELS[model_key]
    engine = model["engine"]
    path.parent.mkdir(parents=True, exist_ok=True)
    for sub in subsets:
        sub["cache"].mkdir(parents=True, exist_ok=True)

    if engine == "sd-scripts":
        lines = [
            "[general]",
            "shuffle_caption = false",
            "caption_extension = '.txt'",
            "caption_dropout_rate = 0.05",
            "enable_bucket = true",
            "bucket_no_upscale = true",
            "min_bucket_reso = 512",
            "max_bucket_reso = 1536",
            f"bucket_reso_steps = {model.get('bucket_reso_steps', 64)}",
            "",
            "[[datasets]]",
            f"resolution = {_toml_value(resolution)}",
            f"batch_size = {batch_size}",
            "",
        ]
        for sub in subsets:
            lines += [
                "  [[datasets.subsets]]",
                f"  image_dir = {_toml_value(str(sub['dir']))}",
                f"  num_repeats = {sub['num_repeats']}",
                "",
            ]
        path.write_text("\n".join(lines) + "\n")
        return path

    # musubi family
    lines = [
        "[general]",
        f"resolution = {_toml_value(resolution)}",
        'caption_extension = ".txt"',
        f"batch_size = {batch_size}",
        "enable_bucket = true",
        "bucket_no_upscale = true",
        "",
    ]
    for sub in subsets:
        if sub["media"] == "image":
            block = {
                "image_directory": str(sub["dir"]),
                "cache_directory": str(sub["cache"]),
                "num_repeats": sub["num_repeats"],
            }
            if sub.get("control_dir"):
                block["control_directory"] = str(sub["control_dir"])
                block["control_resolution"] = sub.get("control_resolution") or CONTROL_RESOLUTION
        else:
            block = {
                "video_directory": str(sub["dir"]),
                "cache_directory": str(sub["cache"]),
                "num_repeats": sub["num_repeats"],
            }
            if engine == "musubi-ltx":
                frames = int((ltx_cfg or {}).get("resolution", "768x512x81").split("x")[2])
                block["target_frames"] = [frames]
                block["target_fps"] = float((ltx_cfg or {}).get("fps", 25.0))
                block["frame_extraction"] = "full"
                block["max_frames"] = frames
            else:  # wan
                vd = model.get("video_dataset", {})
                block["target_frames"] = vd.get("target_frames", [1, 33, 65])
                block["frame_extraction"] = vd.get("frame_extraction", "full")
                block["max_frames"] = vd.get("max_frames", 81)
        lines += ["[[datasets]]"] + _toml_lines(block) + [""]

    path.write_text("\n".join(lines) + "\n")
    return path
```

Ajustar o import de presets no topo de `training.py` para incluir `CONTROL_RESOLUTION`:

```python
from .presets import (CONTROL_RESOLUTION, LORAPLUS_RATIO, MODELS, NETWORK_MODULES,
                      net_types_for, suggest_schedule, vram_tier)
```

- [ ] **Step 4: Atualizar a chamada dentro de `run_training`**

Em `trainero/training.py`, substituir a linha que chamava `write_dataset_toml(model_key, pdir, dataset_dir, pdir / "cache", schedule, resolution, batch_size, stats, ltx_cfg)` por:

```python
    subsets = []
    if stats.get("images"):
        subsets.append(image_subset(
            dataset_dir, pdir / "cache" / "images", schedule["num_repeats"],
            control_dir=(dataset_dir / "control") if model.get("needs_control") else None,
            control_resolution=model.get("control_resolution")))
    if stats.get("videos"):
        subsets.append(video_subset(dataset_dir, pdir / "cache" / "videos",
                                    schedule["num_repeats"]))
    dataset_toml = write_dataset_toml(model_key, pdir / "dataset.toml", subsets,
                                      resolution, batch_size, ltx_cfg)
```

- [ ] **Step 5: Atualizar a chamada em `trainero/sliders.py`**

Trocar o import da linha 27-28 por:

```python
from .training import (_cli_args, _script, build_train_config, image_subset,
                       launch_training, project_dir, run_caches, slugify,
                       video_subset, write_dataset_toml)
```

E substituir as linhas 173-174 (dentro do `for side, sdir, stats in (...)`) por:

```python
        subsets = []
        if stats.get("images"):
            subsets.append(image_subset(sdir, pdir / f"cache_{side}" / "images",
                                        schedule["num_repeats"]))
        if stats.get("videos"):
            subsets.append(video_subset(sdir, pdir / f"cache_{side}" / "videos",
                                        schedule["num_repeats"]))
        toml = write_dataset_toml(model_key, pdir / f"dataset_{side}.toml", subsets,
                                  resolution, batch_size, model.get("ltx"))
```

Isso também corrige um nome torto que existia antes: o lado negativo gerava `dataset_dataset_neg.toml` porque o nome vinha de `dataset_dir.name`. Agora é `dataset_pos.toml` / `dataset_neg.toml`.

- [ ] **Step 6: Rodar todos os testes**

Run: `python -m unittest discover -s tests -v`
Expected: PASS em tudo. Rodar também `python -m compileall trainero server.py -q`.

- [ ] **Step 7: Commit**

```bash
git add trainero/training.py trainero/sliders.py tests/test_core.py
git commit -m "refactor(training): dataset.toml por lista de subsets, um caminho só para todos os modos"
```

---

### Task 6: Sample prompts em todo treino

**Files:**
- Modify: `trainero/training.py` (nova função + `build_train_config`)
- Modify: `trainero/engines.py` (detecção de suporte)
- Modify: `tests/test_core.py`

**Interfaces:**
- Consumes: `presets.SAMPLE_PROMPT`, `SAMPLE_STEPS`, `SAMPLE_GUIDANCE`, `SAMPLE_SEED` (Task 4).
- Produces:
  - `trainero.engines.supports_sampling(engine: str) -> bool`
  - `trainero.training.sample_prompt_line(prompt_text: str, trigger: str, resolution: list[int], frames: int | None = None) -> str`
  - `trainero.training.write_sample_prompts(path: Path, prompt_text: str, trigger: str, resolution: list[int], frames: int | None = None) -> Path`
  - `build_train_config` ganha o parâmetro nomeado `sample_prompts: Path | None = None` e, quando presente, escreve `sample_prompts`, `sample_every_n_epochs = 1` e `sample_at_first = True` no config.

- [ ] **Step 1: Escrever os testes que falham**

Acrescentar a `tests/test_core.py`:

```python
class TestSamplePrompts(unittest.TestCase):
    def test_line_has_trigger_and_flags(self):
        from trainero.presets import SAMPLE_PROMPT
        from trainero.training import sample_prompt_line

        line = sample_prompt_line(SAMPLE_PROMPT, "makima", [1024, 1024])
        self.assertTrue(line.startswith("makima, A young woman"))
        self.assertIn("--w 1024", line)
        self.assertIn("--h 1024", line)
        self.assertIn("--d 42", line)
        self.assertIn("--s 28", line)
        self.assertIn("--g 4.0", line)
        self.assertNotIn("\n", line)

    def test_no_trigger_means_no_prefix(self):
        from trainero.training import sample_prompt_line

        line = sample_prompt_line("a cat", "", [1024, 1024])
        self.assertTrue(line.startswith("a cat --w"))

    def test_video_gets_frame_count(self):
        from trainero.training import sample_prompt_line

        line = sample_prompt_line("a cat", "trg", [768, 512], frames=81)
        self.assertIn("--f 81", line)

    def test_write_creates_single_line_file(self):
        import tempfile
        from trainero.training import write_sample_prompts

        with tempfile.TemporaryDirectory() as td:
            path = write_sample_prompts(Path(td) / "sample_prompts.txt",
                                        "a cat", "trg", [1024, 1024])
            lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
            self.assertEqual(len(lines), 1)


class TestSamplingInConfig(unittest.TestCase):
    def test_sampling_args_present(self):
        stats = {"items": 30, "images": 30, "videos": 0}
        sched = suggest_schedule("flux-klein", 30)
        cfg = build_train_config("flux-klein", {}, sched, stats, 48,
                                 Path("/tmp/ds.toml"), Path("/tmp/out"), "test", 1,
                                 sample_prompts=Path("/tmp/sp.txt"))
        self.assertEqual(cfg["sample_prompts"], "/tmp/sp.txt")
        self.assertEqual(cfg["sample_every_n_epochs"], 1)
        self.assertTrue(cfg["sample_at_first"])

    def test_no_sampling_when_not_requested(self):
        stats = {"items": 30, "images": 30, "videos": 0}
        sched = suggest_schedule("flux-klein", 30)
        cfg = build_train_config("flux-klein", {}, sched, stats, 48,
                                 Path("/tmp/ds.toml"), Path("/tmp/out"), "test", 1)
        self.assertNotIn("sample_prompts", cfg)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python -m unittest tests.test_core -v`
Expected: FAIL com `ImportError: cannot import name 'sample_prompt_line'`

- [ ] **Step 3: Implementar em `trainero/training.py`**

Acrescentar `SAMPLE_GUIDANCE, SAMPLE_PROMPT, SAMPLE_SEED, SAMPLE_STEPS` ao import de `.presets`, e acrescentar depois de `write_dataset_toml`:

```python
# ---------------------------------------------------------------------------
# Sample prompts
# ---------------------------------------------------------------------------
# One prompt, every model. The trainer writes the images to output_dir/sample/
# and the UI polls that folder — nothing here has to move files around.


def sample_prompt_line(prompt_text: str, trigger: str, resolution: list[int],
                       frames: int | None = None) -> str:
    text = prompt_text.strip()
    if trigger.strip():
        text = f"{trigger.strip()}, {text}"
    width, height = int(resolution[0]), int(resolution[1])
    parts = [text, f"--w {width}", f"--h {height}"]
    if frames:
        parts.append(f"--f {int(frames)}")
    parts += [f"--d {SAMPLE_SEED}", f"--s {SAMPLE_STEPS}", f"--g {SAMPLE_GUIDANCE}"]
    return " ".join(parts)


def write_sample_prompts(path: Path, prompt_text: str, trigger: str,
                         resolution: list[int], frames: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sample_prompt_line(prompt_text, trigger, resolution, frames) + "\n",
                    encoding="utf-8")
    return path
```

Na assinatura de `build_train_config`, acrescentar o parâmetro final:

```python
def build_train_config(model_key: str, overrides: dict, schedule: dict, stats: dict,
                       vram_gb: float, dataset_toml: Path, output_dir: Path,
                       output_name: str, batch_size: int,
                       sample_prompts: Path | None = None) -> dict:
```

E, logo antes do `if model["engine"] == "sd-scripts":` no final da função:

```python
    if sample_prompts is not None:
        cfg["sample_prompts"] = str(sample_prompts)
        cfg["sample_every_n_epochs"] = 1
        cfg["sample_at_first"] = True
```

- [ ] **Step 4: Implementar `supports_sampling` em `trainero/engines.py`**

Acrescentar ao final do arquivo:

```python
def supports_sampling(engine: str) -> bool:
    """Whether this engine's trainer accepts --sample_prompts.

    Confirmed present in musubi upstream. The LTX fork and sd-scripts are
    checked by reading their sources — cheaper and safer than running the
    trainer with a flag it may reject.
    """
    if engine == "musubi":
        return True
    root = engine_dir(engine)
    if not root.exists():
        return False
    for py in root.rglob("*.py"):
        try:
            if "--sample_prompts" in py.read_text(encoding="utf-8", errors="ignore"):
                return True
        except OSError:
            continue
    return False
```

- [ ] **Step 5: Ligar o sampling em `run_training`**

Em `run_training`, logo depois de calcular `resolution` e antes de montar `subsets`, acrescentar:

```python
    sample_path = None
    if overrides.get("sampling", True) and supports_sampling(engine):
        frames = None
        if model["media"] == "video":
            frames = int(ltx_cfg["resolution"].split("x")[2]) if engine == "musubi-ltx" \
                else (model.get("video_dataset", {}).get("max_frames") or 81)
        sample_path = write_sample_prompts(
            pdir / "sample_prompts.txt",
            overrides.get("sample_prompt") or SAMPLE_PROMPT,
            overrides.get("trigger", ""), resolution, frames)
        job.log(f"Samples a cada época: {sample_path}")
    elif overrides.get("sampling", True):
        job.log(f"⚠ engine {engine} não tem --sample_prompts — sampling desligado.")
```

E passar para `build_train_config`:

```python
    cfg = build_train_config(model_key, overrides, schedule, stats, vram_gb,
                             dataset_toml, output_dir, slug, batch_size,
                             sample_prompts=sample_path)
```

Acrescentar `supports_sampling` ao import de `.engines` no topo de `training.py`:

```python
from .engines import engine_dir, ensure_engine, supports_sampling, venv_bin, venv_python
```

- [ ] **Step 6: Rodar todos os testes**

Run: `python -m unittest discover -s tests -v` e `python -m compileall trainero server.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add trainero/training.py trainero/engines.py tests/test_core.py
git commit -m "feat(training): sample images a cada época em todo modelo com suporte"
```

---

### Task 7: Conversão pro formato ComfyUI como dado do preset

**Files:**
- Modify: `trainero/training.py:358-385` (substitui `anima_comfy_converter`)
- Modify: `trainero/training.py` (chamada dentro de `run_training`)
- Modify: `tests/test_core.py`

**Interfaces:**
- Consumes: chave `comfy_convert` do preset (Task 4).
- Produces:
  - `trainero.training.comfy_convert_command(model_key: str, src: Path, dst: Path, forced: bool = False) -> list[str] | None` — o comando a rodar, ou `None` quando o modelo não precisa
  - `trainero.training.comfy_converter(model_key: str, job: Job, forced: bool = False)` — devolve `fn(Path) -> Path | None` para o `UploadWatcher`, ou `None`

- [ ] **Step 1: Escrever os testes que falham**

Acrescentar a `tests/test_core.py`:

```python
class TestComfyConvert(unittest.TestCase):
    def test_anima_uses_its_script(self):
        from trainero.training import comfy_convert_command

        cmd = comfy_convert_command("anima", Path("/out/a.safetensors"),
                                    Path("/out/a_comfy.safetensors"))
        self.assertIsNotNone(cmd)
        self.assertTrue(any("convert_anima_lora_to_comfy.py" in str(c) for c in cmd))
        self.assertIn("/out/a.safetensors", [str(c) for c in cmd])

    def test_models_without_the_key_convert_nothing(self):
        from trainero.training import comfy_convert_command

        for key in ("flux-klein", "qwen-image", "krea2", "wan-22"):
            self.assertIsNone(
                comfy_convert_command(key, Path("/o/a.safetensors"),
                                      Path("/o/a_comfy.safetensors")), key)

    def test_forced_uses_convert_lora_on_musubi(self):
        from trainero.training import comfy_convert_command

        cmd = [str(c) for c in comfy_convert_command(
            "flux-klein", Path("/o/a.safetensors"), Path("/o/a_comfy.safetensors"),
            forced=True)]
        self.assertTrue(any(c.endswith("convert_lora.py") for c in cmd))
        self.assertIn("--target", cmd)
        self.assertIn("other", cmd)
        self.assertIn("--input", cmd)
        self.assertIn("--output", cmd)

    def test_forced_on_sdscripts_still_uses_the_script(self):
        from trainero.training import comfy_convert_command

        cmd = comfy_convert_command("anima", Path("/o/a.safetensors"),
                                    Path("/o/a_comfy.safetensors"), forced=True)
        self.assertTrue(any("convert_anima_lora_to_comfy.py" in str(c) for c in cmd))
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python -m unittest tests.test_core -v`
Expected: FAIL com `ImportError: cannot import name 'comfy_convert_command'`

- [ ] **Step 3: Substituir `anima_comfy_converter` em `trainero/training.py`**

Trocar o bloco inteiro (comentário de seção + função `anima_comfy_converter`) por:

```python
# ---------------------------------------------------------------------------
# ComfyUI format conversion (runs in the engine venv)
# ---------------------------------------------------------------------------
# ComfyUI's model_lora_keys_unet maps `lora_unet_<flattened key>` generically for
# every architecture, which is exactly what musubi saves — so musubi LoRAs load
# as-is. The exception is a backend whose module names differ from what ComfyUI
# loads: sd-scripts' Anima, which ships its own converter. That is why only the
# Anima preset carries `comfy_convert`. The advanced panel can force conversion
# on any musubi model via `convert_lora.py --target other` if a model turns out
# to need it in practice.


def comfy_convert_command(model_key: str, src: Path, dst: Path,
                          forced: bool = False) -> list[str] | None:
    model = MODELS[model_key]
    engine = model["engine"]
    spec = model.get("comfy_convert")
    if spec is None:
        if not forced or engine == "sd-scripts":
            return None
        spec = {"convert_lora": True}

    py = str(venv_python(engine))
    edir = engine_dir(engine)
    if "script" in spec:
        return [py, str(edir / spec["script"]), str(src), str(dst)]
    return [py, str(edir / _script(engine, "convert_lora.py")),
            "--input", str(src), "--output", str(dst), "--target", "other"]


def comfy_converter(model_key: str, job: Job, forced: bool = False):
    """A fn(ckpt) -> converted path|None for UploadWatcher, or None if unneeded."""
    probe = comfy_convert_command(model_key, Path("probe"), Path("probe_comfy"), forced)
    if probe is None:
        return None
    script = Path(probe[1])
    if not script.exists():
        job.log(f"⚠ {script.name} não existe neste engine — enviando só o formato nativo.")
        return None

    def convert(ckpt: Path) -> Path | None:
        # Runs inside the upload-watcher thread WHILE training runs in the main
        # job thread — must not touch job._proc (job.run would race the train
        # subprocess handle), so this uses subprocess directly.
        import subprocess

        if ckpt.name.endswith("_comfy.safetensors"):
            return None
        dest = ckpt.with_name(ckpt.stem + "_comfy.safetensors")
        if dest.exists():
            return None
        cmd = comfy_convert_command(model_key, ckpt, dest, forced)
        res = subprocess.run([str(c) for c in cmd], cwd=str(engine_dir(MODELS[model_key]["engine"])),
                             capture_output=True, text=True, timeout=600)
        if res.returncode != 0:
            job.log(f"⚠ conversão Comfy falhou: {res.stderr[-400:]}")
            return None
        job.log(f"✔ convertido p/ ComfyUI: {dest.name}")
        return dest

    return convert
```

- [ ] **Step 4: Atualizar a chamada em `run_training`**

Trocar:

```python
        convert = anima_comfy_converter(job) if model.get("comfy_convert") else None
```

por:

```python
        convert = comfy_converter(model_key, job, forced=bool(overrides.get("comfy_convert")))
```

- [ ] **Step 5: Rodar e ver passar**

Run: `python -m unittest discover -s tests -v`
Expected: PASS. Os testes de `comfy_convert_command` não tocam o disco porque só montam o comando.

- [ ] **Step 6: Commit**

```bash
git add trainero/training.py tests/test_core.py
git commit -m "refactor(hf): conversão ComfyUI vira dado do preset, com override no painel"
```

---

### Task 8: Pipeline `run_style_rush_training`

**Files:**
- Modify: `trainero/training.py`
- Modify: `tests/test_style_rush.py`

**Interfaces:**
- Consumes: `style_rush.build_convert_dataset` (Task 3), `presets.STYLE_RUSH_SCHEDULE`, `presets.CONTROL_RESOLUTION`, `presets.style_rush_models` (Task 4), `image_subset`/`write_dataset_toml` (Task 5), `write_sample_prompts` (Task 6), `comfy_converter` (Task 7).
- Produces:
  - `trainero.training.run_style_rush_training(job: Job, params: dict) -> None`, com `params = {"project", "model", "trigger", "overrides"}`
  - `run_training` passa a despachar `mode == "style-rush"` para ela.

- [ ] **Step 1: Escrever o teste de guarda que falha**

Acrescentar a `tests/test_style_rush.py`:

```python
class TestStyleRushGuards(unittest.TestCase):
    def test_unsupported_model_is_rejected(self):
        from trainero.jobs import JobFailed
        from trainero.training import run_style_rush_training

        job = _FakeJob()
        job.set_phases = lambda names: None
        job.start_phase = lambda name: None
        job.end_phase = lambda name, ok=True: None
        job.extra = {}
        with self.assertRaises(JobFailed) as ctx:
            run_style_rush_training(job, {"project": "p", "model": "wan-22",
                                          "trigger": "t", "overrides": {}})
        self.assertIn("control", str(ctx.exception).lower())

    def test_empty_trigger_is_rejected(self):
        from trainero.jobs import JobFailed
        from trainero.training import run_style_rush_training

        job = _FakeJob()
        job.set_phases = lambda names: None
        job.start_phase = lambda name: None
        job.end_phase = lambda name, ok=True: None
        job.extra = {}
        with self.assertRaises(JobFailed) as ctx:
            run_style_rush_training(job, {"project": "p", "model": "flux-klein",
                                          "trigger": "  ", "overrides": {}})
        self.assertIn("trigger", str(ctx.exception).lower())
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python -m unittest tests.test_style_rush -v`
Expected: FAIL com `ImportError: cannot import name 'run_style_rush_training'`

- [ ] **Step 3: Implementar em `trainero/training.py`**

Acrescentar aos imports do topo:

```python
from . import style_rush as sr
from .captioner import generate_captions
from .presets import (CONTROL_RESOLUTION, LORAPLUS_RATIO, MODELS, NETWORK_MODULES,
                      SAMPLE_GUIDANCE, SAMPLE_PROMPT, SAMPLE_SEED, SAMPLE_STEPS,
                      STYLE_RUSH_SCHEDULE, net_types_for, style_rush_models,
                      suggest_schedule, vram_tier)
```

E, antes de `run_training`, acrescentar:

```python
# ---------------------------------------------------------------------------
# Style Rush pipeline
# ---------------------------------------------------------------------------


def run_style_rush_training(job: Job, params: dict) -> None:
    """One dataset in, two datasets trained: the base (style) and the synthetic
    conversion pairs (style transfer). Fixed 5 epochs — the owner watches the
    samples and cancels when it looks right."""
    project = params["project"]
    model_key = params["model"]
    trigger = (params.get("trigger") or "").strip()
    overrides = params.get("overrides", {})

    if model_key not in style_rush_models():
        raise JobFailed(
            f"{MODELS[model_key]['label']} não aceita control image — o Style Rush precisa "
            f"de um modelo que suporte (Flux Klein ou Qwen Image Edit)")
    if not trigger:
        raise JobFailed("defina a trigger word antes de treinar no modo Style Rush")

    model = MODELS[model_key]
    engine = model["engine"]
    pdir = project_dir(project)
    slug = slugify(project)
    dataset_dir = pdir / "dataset"
    convert_dir = pdir / "dataset_convert"
    output_dir = pdir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = ds.inspect(dataset_dir)
    if stats["items"] == 0:
        raise JobFailed("importe o dataset antes de treinar")
    if stats["videos"]:
        raise JobFailed("Style Rush é só para datasets de imagem")

    job.set_phases(["Engine", "Modelos base", "Captions", "Dataset de conversão",
                    "Configuração", "Cache de latents", "Cache do text encoder",
                    "Treino", "Finalização"])

    gpu = gpu_info()
    vram_gb = gpu.get("vram_mb", 24576) / 1024
    job.extra["gpu"] = gpu
    job.log(f"GPU: {gpu.get('name', '?')} ({vram_gb:.0f} GB) · trigger: {trigger}")

    job.start_phase("Engine")
    ensure_engine(engine, job)
    job.end_phase("Engine")

    job.start_phase("Modelos base")
    ensure_models(model_key, job)
    job.end_phase("Modelos base")

    job.start_phase("Captions")
    if stats["missing_captions"]:
        job.log(f"{stats['missing_captions']} itens sem caption — gerando com "
                f"generic-style e trigger {trigger}")
        generate_captions(dataset_dir, "image", "generic-style", {"style_name": trigger}, job)
        stats = ds.inspect(dataset_dir)
        if stats["missing_captions"]:
            raise JobFailed(f"{stats['missing_captions']} itens continuam sem caption")
    else:
        job.log("Todas as imagens já têm caption.")
    job.end_phase("Captions")

    job.start_phase("Dataset de conversão")
    convert_stats = sr.build_convert_dataset(dataset_dir, convert_dir, trigger, job)
    job.extra["style_rush"] = convert_stats
    job.end_phase("Dataset de conversão")

    job.start_phase("Configuração")
    schedule = dict(STYLE_RUSH_SCHEDULE)
    for key in ("epochs", "num_repeats", "save_every_n_epochs"):
        if overrides.get(key):
            schedule[key] = int(overrides[key])
    tier = vram_tier(model_key, vram_gb)
    batch_size = tier.get("batch_size", 1)
    resolution = overrides.get("resolution") or model.get("resolution", [1024, 1024])
    subsets = [
        image_subset(dataset_dir, pdir / "cache" / "images", schedule["num_repeats"]),
        image_subset(convert_dir, pdir / "cache" / "convert", schedule["num_repeats"],
                     control_dir=convert_dir / "control",
                     control_resolution=CONTROL_RESOLUTION),
    ]
    dataset_toml = write_dataset_toml(model_key, pdir / "dataset.toml", subsets,
                                      resolution, batch_size)

    sample_path = None
    if overrides.get("sampling", True) and supports_sampling(engine):
        sample_path = write_sample_prompts(
            pdir / "sample_prompts.txt",
            overrides.get("sample_prompt") or SAMPLE_PROMPT, trigger, resolution)

    total_items = stats["items"] + convert_stats["pairs"]
    cfg = build_train_config(model_key, overrides, schedule, {"items": total_items},
                             vram_gb, dataset_toml, output_dir, slug, batch_size,
                             sample_prompts=sample_path)
    job.extra["schedule"] = schedule
    job.extra["config_summary"] = {
        "network_module": cfg.get("network_module"),
        "network_dim": cfg.get("network_dim"), "network_alpha": cfg.get("network_alpha"),
        "learning_rate": cfg.get("learning_rate"),
        "epochs": schedule["epochs"], "batch_size": batch_size,
        "base": stats["items"], "convert": convert_stats["pairs"],
        "fp8": bool(cfg.get("fp8_base")), "blocks_to_swap": cfg.get("blocks_to_swap", 0),
    }
    job.log(f"Config: {json.dumps(job.extra['config_summary'], ensure_ascii=False)}")
    job.end_phase("Configuração")

    run_caches(model_key, dataset_toml, vram_gb, job)

    watcher = None
    repo_id = None
    if overrides.get("hf_upload", True):
        user = hf_username()
        if user:
            repo_id = create_repo(f"{user}/{slug}", private=overrides.get("hf_private", True),
                                  job=job)
        else:
            job.log("⚠ Sem token HF (defina HF_TOKEN) — upload desativado.")
    if repo_id:
        job.extra["hf_repo"] = repo_id
        upload_text(repo_id, "README.md",
                    model_card(project, model["label"], stats, schedule, cfg), job)
        upload_text(repo_id, "trainero_config.json",
                    json.dumps({"model": model_key, "mode": "style-rush", "trigger": trigger,
                                "schedule": schedule, "style_rush": convert_stats,
                                "config": {k: v for k, v in cfg.items()
                                           if isinstance(v, (str, int, float, bool))}},
                               indent=2, ensure_ascii=False), job)
        convert = comfy_converter(model_key, job, forced=bool(overrides.get("comfy_convert")))
        watcher = UploadWatcher(repo_id, output_dir, job, convert=convert)
        watcher.start()

    job.start_phase("Treino")
    try:
        launch_training(model_key, cfg, pdir, job)
    finally:
        if watcher:
            job.log("Varredura final de checkpoints para o HF...")
            watcher.stop_and_sweep()
    job.end_phase("Treino")

    job.start_phase("Finalização")
    finals = sorted(output_dir.glob("*.safetensors"))
    if not finals:
        raise JobFailed("o treino terminou sem produzir checkpoints")
    job.extra["outputs"] = [f.name for f in finals]
    job.log(f"✔ {len(finals)} checkpoints em {output_dir}")
    if repo_id:
        job.log(f"✔ Tudo em https://huggingface.co/{repo_id}")
    job.end_phase("Finalização")
```

- [ ] **Step 4: Despachar o modo em `run_training`**

Logo depois do bloco `if mode == "slider":`, acrescentar:

```python
    if mode == "style-rush":
        return run_style_rush_training(job, params)
```

- [ ] **Step 5: Rodar e ver passar**

Run: `python -m unittest discover -s tests -v` e `python -m compileall trainero server.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add trainero/training.py tests/test_style_rush.py
git commit -m "feat(style-rush): pipeline completo de treino com dois datasets"
```

---

### Task 9: API — trigger, samples e o novo modo

**Files:**
- Modify: `server.py`
- Create: `tests/test_server_samples.py`

**Interfaces:**
- Consumes: `presets.style_rush_models` (Task 4).
- Produces:
  - `POST /api/project` aceita `{"name": str, "trigger": str}` e persiste os dois no state
  - `GET /api/status` inclui `"trigger"`, `"dataset_convert"` e `"style_rush_models"`
  - `GET /api/samples` → `{"samples": [{"name": str, "epoch": int, "idx": int}]}`, mais recente primeiro
  - `GET /api/sample?name=<basename>` → o PNG (`image/png`), 404 se não existir, 400 se o nome não for um basename simples
  - `POST /api/train` aceita `mode: "style-rush"` e repassa `trigger`
  - `server.parse_sample_name(name: str) -> tuple[int, int]` — `(epoch, idx)`, `(-1, -1)` quando não bate

- [ ] **Step 1: Escrever os testes que falham**

Criar `tests/test_server_samples.py`:

```python
"""Unit tests for the sample-listing helpers in server.py (no HTTP)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server


class TestParseSampleName(unittest.TestCase):
    def test_musubi_pattern(self):
        # <output_name>_e{epoch:06d}_{idx:02d}_{timestamp}_{seed}.png
        self.assertEqual(server.parse_sample_name("makima_e000003_01_20260815120000_42.png"),
                         (3, 1))

    def test_step_based_pattern_without_epoch(self):
        self.assertEqual(server.parse_sample_name("makima_000500_00_20260815120000.png"),
                         (-1, 0))

    def test_unknown_name(self):
        self.assertEqual(server.parse_sample_name("whatever.png"), (-1, -1))


class TestListSamples(unittest.TestCase):
    def test_newest_first(self):
        with tempfile.TemporaryDirectory() as td:
            sample_dir = Path(td) / "sample"
            sample_dir.mkdir()
            for epoch in (1, 2, 3):
                (sample_dir / f"m_e{epoch:06d}_00_2026081512000{epoch}_42.png").write_bytes(b"x")
            names = [s["name"] for s in server.list_samples(sample_dir)]
            self.assertEqual(len(names), 3)
            self.assertIn("e000003", names[0])
            self.assertIn("e000001", names[-1])

    def test_missing_dir_is_empty(self):
        self.assertEqual(server.list_samples(Path("/nope/sample")), [])


class TestSafeSampleName(unittest.TestCase):
    def test_rejects_traversal(self):
        for bad in ("../secret.png", "a/b.png", "", "..", "x.txt"):
            self.assertFalse(server.safe_sample_name(bad), bad)

    def test_accepts_plain_png(self):
        self.assertTrue(server.safe_sample_name("m_e000001_00_x_42.png"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python -m unittest tests.test_server_samples -v`
Expected: FAIL com `AttributeError: module 'server' has no attribute 'parse_sample_name'`

- [ ] **Step 3: Implementar os helpers em `server.py`**

Acrescentar `import re` ao topo e, logo depois de `WEB_DIR = ...`:

```python
# musubi writes samples as <output_name>_e{epoch:06d}_{idx:02d}_{ts}[_{seed}].png
# and falls back to a bare step counter when the run is step-based.
_SAMPLE_RE = re.compile(r"_(e?)(\d{6})_(\d{2})_")


def parse_sample_name(name: str) -> tuple[int, int]:
    """(epoch, index) from a sample filename; (-1, -1) when it does not match."""
    m = _SAMPLE_RE.search(name)
    if not m:
        return (-1, -1)
    epoch = int(m.group(2)) if m.group(1) == "e" else -1
    return (epoch, int(m.group(3)))


def list_samples(sample_dir: Path) -> list[dict]:
    """Samples newest first — the UI shows the freshest epoch at the top."""
    try:
        files = [p for p in sample_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    except OSError:
        return []
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in files:
        epoch, idx = parse_sample_name(p.name)
        out.append({"name": p.name, "epoch": epoch, "idx": idx})
    return out


def safe_sample_name(name: str) -> bool:
    """A plain .png basename — no separators, no traversal."""
    return bool(name) and name == Path(name).name and name.lower().endswith(".png") \
        and ".." not in name


def sample_dir() -> Path:
    return project_dir(current_project() or "projeto") / "output" / "sample"
```

- [ ] **Step 4: Ligar os endpoints e o trigger**

Em `do_GET`, antes do `return super().do_GET()`:

```python
        if url.path == "/api/samples":
            return self._json({"samples": list_samples(sample_dir())})
        if url.path == "/api/sample":
            name = parse_qs(url.query).get("name", [""])[0]
            return self._serve_sample(name)
```

Acrescentar o método:

```python
    def _serve_sample(self, name: str):
        if not safe_sample_name(name):
            return self._error("nome de sample inválido")
        path = sample_dir() / name
        try:
            data = path.read_bytes()
        except OSError:
            return self._error("sample não encontrado", 404)
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(data)
```

Em `dataset_dir(side)`, aceitar o lado novo:

```python
def dataset_dir(side: str = "pos") -> Path:
    pdir = project_dir(current_project() or "projeto")
    return pdir / {"neg": "dataset_neg", "convert": "dataset_convert"}.get(side, "dataset")
```

Em `_set_project`, persistir a trigger:

```python
    def _set_project(self):
        body = self._body_json()
        name = (body.get("name") or "").strip()
        if not name:
            return self._error("nome vazio")
        update_state(project=name, trigger=(body.get("trigger") or "").strip())
        pdir = project_dir(name)
        (pdir / "dataset").mkdir(parents=True, exist_ok=True)
        self._json({"ok": True, "slug": slugify(name)})
```

Em `_status`, acrescentar ao payload:

```python
            "trigger": load_state().get("trigger", ""),
            "dataset_convert": ds.inspect(dataset_dir("convert")) if project else {},
            "style_rush_models": style_rush_models(),
```

E o import: `from trainero.presets import (CAPTION_PROFILES, MODEL_ORDER, MODELS, public_presets, style_rush_models, suggest_schedule)`

Em `_train`, repassar a trigger:

```python
        params = {
            "project": project,
            "model": model_key,
            "mode": body.get("mode", "lora"),
            "trigger": (body.get("trigger") or load_state().get("trigger") or "").strip(),
            "overrides": body.get("overrides") or {},
            "slider_targets": body.get("slider_targets") or [],
        }
```

- [ ] **Step 5: Rodar e ver passar**

Run: `python -m unittest discover -s tests -v` e `python -m compileall trainero server.py -q`
Expected: PASS.

- [ ] **Step 6: Smoke do servidor**

```bash
python server.py & sleep 2
curl -s localhost:8090/api/samples
curl -s "localhost:8090/api/sample?name=../etc/passwd" -o /dev/null -w '%{http_code}\n'
curl -s localhost:8090/api/status | head -c 300
kill %1
```
Expected: `{"samples": []}`, `400` na travessia, e o status trazendo `"trigger"` e `"style_rush_models"`.

- [ ] **Step 7: Commit**

```bash
git add server.py tests/test_server_samples.py
git commit -m "feat(api): trigger no projeto, modo style-rush e endpoints de sample"
```

---

### Task 10: UI — modo, trigger e galeria de samples

**Files:**
- Modify: `web/index.html`
- Modify: `web/app.js`
- Modify: `web/styles.css`

**Interfaces:**
- Consumes: `/api/status` (`trigger`, `dataset_convert`, `style_rush_models`), `/api/samples`, `/api/sample`, `/api/presets` (`supports_control`), `POST /api/project` com `trigger`, `POST /api/train` com `mode: "style-rush"` e `trigger`.
- Produces: nada consumido por outras tasks.

- [ ] **Step 1: Botão do modo e campo de trigger no card 1**

Em `web/index.html`, substituir o bloco `<div class="row">` da seção `#card-project` por:

```html
    <div class="row">
      <input id="project-name" type="text" placeholder="nome do LoRA (ex.: makima_v1)" autocomplete="off">
      <input id="trigger-word" type="text" placeholder="trigger word (ex.: makima_style)" autocomplete="off">
      <div class="mode-toggle" id="mode-toggle">
        <button data-mode="lora" class="active">LoRA</button>
        <button data-mode="slider">Concept Slider</button>
        <button data-mode="style-rush">Style Rush</button>
      </div>
    </div>
    <p class="hint" id="style-rush-hint" hidden>
      Style Rush: manda só o dataset de estilo. O trainer gera 50 pares de conversão com
      GPT Image (~$0.55), treina 5 epochs salvando checkpoint e sample a cada época, e você
      cancela quando achar bom.
    </p>
```

- [ ] **Step 2: Campos novos no painel avançado**

Em `web/index.html`, dentro de `.adv-grid`, antes do checkbox `adv-hf`, acrescentar:

```html
      <label class="check"><input type="checkbox" id="adv-sampling" checked> Gerar samples a cada época</label>
      <label class="check"><input type="checkbox" id="adv-comfy"> Converter LoRA p/ formato ComfyUI</label>
      <label class="wide">Prompt do sample <input type="text" id="adv-sample-prompt" placeholder="(padrão do trainer)"></label>
```

- [ ] **Step 3: Galeria no card de progresso**

Em `web/index.html`, dentro de `#card-progress`, entre `#hf-link-row` e `<pre class="console">`:

```html
    <div class="samples" id="samples" hidden></div>
```

- [ ] **Step 4: CSS da galeria**

Acrescentar ao final de `web/styles.css`:

```css
/* sample gallery -------------------------------------------------------- */
.samples {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 10px;
  margin: 14px 0;
}
.samples figure { margin: 0; }
.samples img {
  width: 100%;
  aspect-ratio: 1 / 1;
  object-fit: cover;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--bg-primary);
  display: block;
}
.samples figcaption {
  font-size: 11px;
  color: var(--text-muted);
  margin-top: 4px;
  font-family: "JetBrains Mono", monospace;
}
.adv-grid label.wide { grid-column: 1 / -1; }
```

Os tokens `--border`, `--bg-primary` e `--text-muted` já existem no `:root` de `web/styles.css`.

- [ ] **Step 5: Lógica no `app.js`**

Trocar em `web/app.js`:

`state` ganha `samples: []`.

O listener do projeto passa a incluir a trigger:

```javascript
let projectTimer = null;
for (const id of ["#project-name", "#trigger-word"]) {
  $(id).addEventListener("input", () => {
    clearTimeout(projectTimer);
    projectTimer = setTimeout(saveProject, 600);
  });
}
async function saveProject() {
  const name = $("#project-name").value.trim();
  if (!name) return;
  try {
    await post("/api/project", { name, trigger: $("#trigger-word").value.trim() });
    state.projectSet = true;
    refreshTrainButton();
  } catch (e) { toast(e.message, "error"); }
}
```

`renderModeUI` passa a tratar o modo novo:

```javascript
function styleRush() { return state.mode === "style-rush"; }

function renderModeUI() {
  const slider = state.mode === "slider";
  const native = sliderIsNative();
  const rush = styleRush();
  $("#dataset-title").textContent = slider
    ? (native ? "Pares de prompt do slider" : "Datasets do slider")
    : rush ? "Dataset de estilo" : "Dataset";
  $("#dataset-panels").hidden = slider && native;
  $("#slider-prompts").hidden = !(slider && native);
  $(".ds-panel[data-side=neg]").hidden = !slider || native;
  $(".ds-panel[data-side=pos] .ds-label").hidden = !slider || native;
  $("#style-rush-hint").hidden = !rush;
  if (slider && native && !$("#pair-list").children.length) addPair();
  renderModelAvailability();
  refreshTrainButton();
}

function renderModelAvailability() {
  const allowed = styleRush() ? (state.status?.style_rush_models || []) : null;
  $$(".model-btn").forEach((b) => {
    const ok = !allowed || allowed.includes(b.dataset.key);
    b.disabled = !ok;
    b.classList.toggle("unavailable", !ok);
  });
  if (allowed && state.model && !allowed.includes(state.model)) {
    state.model = null;
    $$(".model-btn").forEach((b) => b.classList.remove("selected"));
    $("#preset-line").hidden = true;
  }
}
```

`collectOverrides` ganha os campos novos:

```javascript
  o.sampling = $("#adv-sampling").checked;
  o.comfy_convert = $("#adv-comfy").checked;
  const sp = $("#adv-sample-prompt").value.trim();
  if (sp) o.sample_prompt = sp;
  const trig = $("#trigger-word").value.trim();
  if (trig) o.trigger = trig;
```

O handler do TREINAR manda a trigger e valida o modo:

```javascript
$("#btn-train").addEventListener("click", async () => {
  if (!requireProject() || !state.model) return;
  if (styleRush() && !$("#trigger-word").value.trim()) {
    toast("Style Rush precisa de uma trigger word", "error");
    $("#trigger-word").focus();
    return;
  }
  await saveProject();
  try {
    await post("/api/train", {
      model: state.model,
      mode: state.mode,
      trigger: $("#trigger-word").value.trim(),
      overrides: collectOverrides(),
      slider_targets: sliderTargets(),
    });
    toast("Treino iniciado ⚔", "success");
    $("#card-progress").hidden = false;
  } catch (e) { toast(e.message, "error"); }
});
```

`refreshTrainButton` exige OpenRouter e trigger no modo novo:

```javascript
function refreshTrainButton() {
  const s = state.status || {};
  const busy = s.job && s.job.status === "running";
  const hasData = sliderIsNative()
    ? sliderTargets().length > 0 || state.mode !== "slider"
    : (s.dataset?.items || 0) > 0;
  const needNeg = state.mode === "slider" && !sliderIsNative();
  const negOk = !needNeg || (s.dataset_neg?.items || 0) > 0;
  const rushOk = !styleRush() || (!!s.openrouter && !!$("#trigger-word").value.trim());
  $("#btn-train").disabled = !!busy || !state.model || !hasData || !negOk || !rushOk;
  $("#btn-cancel").hidden = !busy;
}
```

Preencher a trigger vinda do servidor dentro de `poll()`, junto do bloco que preenche o nome:

```javascript
  if (s.trigger && !$("#trigger-word").value) $("#trigger-word").value = s.trigger;
```

E, no final de `renderJob(job)`, antes de `fetchLog()`:

```javascript
  fetchSamples();
```

Acrescentar a função:

```javascript
let lastSampleKey = "";
async function fetchSamples() {
  try {
    const { samples } = await api("/api/samples");
    const key = samples.map((s) => s.name).join("|");
    if (key === lastSampleKey) return;
    lastSampleKey = key;
    const box = $("#samples");
    box.hidden = samples.length === 0;
    box.innerHTML = samples.slice(0, 12).map((s) => {
      const label = s.epoch >= 0 ? `epoch ${s.epoch}` : s.name;
      return `<figure>
        <a href="/api/sample?name=${encodeURIComponent(s.name)}" target="_blank">
          <img src="/api/sample?name=${encodeURIComponent(s.name)}" alt="${label}" loading="lazy">
        </a>
        <figcaption>${label}</figcaption>
      </figure>`;
    }).join("");
  } catch { /* transient */ }
}
```

- [ ] **Step 6: Verificar no navegador**

```bash
python server.py &
sleep 2
```
Abrir `http://localhost:8090`. Conferir, sem GPU:
1. O toggle mostra três modos e clicar em **Style Rush** exibe a dica e desabilita todos os modelos exceto Flux Klein e Qwen Image Edit.
2. O campo trigger aparece no card 1 e persiste após recarregar a página (o `/api/status` devolve).
3. Com Style Rush ativo e trigger vazia, o botão TREINAR fica desabilitado.
4. O painel avançado mostra os três controles novos.
5. `#samples` fica escondido quando não há amostra.

`kill %1` ao terminar.

- [ ] **Step 7: Commit**

```bash
git add web/index.html web/app.js web/styles.css
git commit -m "feat(ui): modo Style Rush, trigger no topo e galeria de samples"
```

---

### Task 11: Documentação e gate final

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: tudo.
- Produces: nada.

- [ ] **Step 1: Documentar o modo no `README.md`**

Acrescentar, logo depois da seção "Concept Sliders":

```markdown
## Style Rush

Toggle "Style Rush" no topo, para **Flux Klein** ou **Qwen Image Edit**. Você manda só o
dataset de estilo e preenche a trigger word; o trainer faz o resto:

1. captions do dataset via OpenRouter (profile `generic-style`, trigger na primeira palavra);
2. **50 pares de conversão** gerados com `openai/gpt-image-2` (quality low, ~$0.55): a saída
   vira a control image, a sua imagem original vira o target, e a caption é a mesma nos 50 —
   `convert the style of this image to the <trigger> style`;
3. treino com os **dois datasets no mesmo `dataset.toml`** (só o de conversão tem
   `control_directory`) — o musubi mantém os batches separados sozinho;
4. 5 epochs, checkpoint e sample a cada época. Você olha as amostras e cancela quando quiser.

Se o dataset tiver menos de 50 imagens, elas se repetem entre os slots, sempre com estilos
diferentes. Imagem recusada pela moderação é tentada uma segunda vez com **outra** imagem;
falhou nas duas, o slot é descartado e o log diz quantos caíram.

Precisa de `OPENROUTER_API_KEY`.
```

Na seção "Epochs, nunca max steps", acrescentar ao final:

```markdown
No **Style Rush** isso não se aplica: são 5 epochs fixos, repeats 1, e o dono cancela quando
os samples ficam bons.
```

Acrescentar uma seção nova antes de "Layout no pod":

```markdown
## Samples durante o treino

Todo treino gera uma imagem de amostra por época (`--sample_prompts`), mostrada na galeria do
card de progresso e salva em `output/sample/`. O prompt padrão é o mesmo para todos os modelos
de imagem e começa pela trigger word; dá para trocar no painel ⚙.
```

- [ ] **Step 2: Gate completo**

```bash
python -m compileall trainero server.py -q && echo "compile OK"
python -m unittest discover -s tests -v
```
Expected: compile OK e todos os testes passando.

- [ ] **Step 3: Conferir que nada ficou pendurado**

```bash
grep -rn "anima_comfy_converter\|comfy_convert.*True\b" trainero/ | grep -v "convert_lora"
grep -rn "write_dataset_toml" trainero/ tests/
```
Expected: nenhuma referência ao nome antigo `anima_comfy_converter`; toda chamada de
`write_dataset_toml` usando a assinatura nova (path + lista de subsets).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: README documenta Style Rush e os samples por época"
```

---

## Self-Review

**Cobertura do spec:**

| Seção do spec | Task |
|---|---|
| Modelos elegíveis (`supports_control`) | 4, 8, 10 |
| Captions do base (`generic-style` + trigger) | 8 |
| 50 slots, dataset menor que 50, prompts distintos | 1 |
| Chamada gpt-image-2 (quality/moderation/aspect_ratio, sem size) | 2 |
| Recusa → outra imagem, 2 tentativas; manifest idempotente; 4 workers | 3 |
| `dataset.toml` de dois subsets, `control_resolution [1024,1024]` | 5, 8 |
| Schedule fixo (repeats 1, 5 epochs, save 1) | 4, 8 |
| Sampling em todo modo, prompt único em prosa, `sample_at_first` | 4, 6 |
| Detecção de suporte a sampling por engine | 6 |
| `comfy_convert` como dado + checkbox de override | 4, 7, 10 |
| Endpoints `/api/samples` e `/api/sample` | 9 |
| UI: modo, trigger no topo, galeria, campos avançados | 10 |
| Erros (sem chave, trigger vazia, 50 recusados, falha parcial) | 2, 3, 8, 10 |
| Pillow como única dependência nova | 2 |
| Testes listados no spec | 1, 2, 3, 4, 5, 6, 7, 8, 9 |

**Consistência de tipos:** `write_dataset_toml(model_key, path, subsets, resolution, batch_size, ltx_cfg)` é usada com essa assinatura nas Tasks 5, 8 e nos testes. `image_subset`/`video_subset` produzem a forma que `write_dataset_toml` consome. `build_convert_dataset` devolve `{"pairs","refused","failed","cost"}`, lido em `job.extra["style_rush"]` e no `trainero_config.json` (Task 8). `comfy_convert_command` devolve `list[str] | None` e `comfy_converter` devolve `fn|None`, que é o que o `UploadWatcher(convert=...)` já espera. `parse_sample_name` devolve `(epoch, idx)` consumido por `list_samples`, que alimenta `/api/samples` e a galeria.

**Sem placeholders:** todo passo traz o código real, incluindo a substituição exata em `sliders.py:173-174` e os tokens de cor que já existem no `:root` de `web/styles.css`.
