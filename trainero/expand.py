"""Expand mode: a few reference images of one character -> a training dataset.

The owner drops 1-5 images of a character. An image model on OpenRouter draws
that same character in new angles, framings, expressions, outfits, places and
lights; a cheap VLM judge compares every candidate with the references and
only what keeps the identity, the medium and a clean render is kept. Accepted
images land in dataset/ as gen_*.png and go through the normal caption + train
flow — the Krea 2 recipe counts them like any other image.

What poisons a synthetic character set, and what this guards against:
  identity drift ("average face")  judge score + the prompt pins every trait
  the generator's own look          the prompt pins the reference's medium
  repeated poses                    one curated variation per attempt, rotated
  copies of the reference           perceptual-hash dedupe against refs too

Needs OPENROUTER_API_KEY. Model, judge and thresholds are env-configurable.
"""

from __future__ import annotations

import io
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import imagegen
from .config import IMAGE_EXTS
from .jobs import Cancelled, JobFailed

# gpt-image-2 is the model whose request shape and billing were verified on
# this endpoint (Style Rush). Any other image model listed at
# /api/v1/images/models works through EXPAND_IMAGE_MODEL — the payload follows
# that model's declared parameters (e.g. google/gemini-nano-banana-2.1).
IMAGE_MODEL = os.environ.get("EXPAND_IMAGE_MODEL", "openai/gpt-image-2")
# only sent to models that declare `quality`. "low" is what Style Rush uses for
# throwaway controls; these images ARE the training set, so medium by default.
IMAGE_QUALITY = os.environ.get("EXPAND_IMAGE_QUALITY", "medium")
# the judge: a vision model on chat/completions. The caption cascade already
# runs this one with a strict json_schema on this account.
JUDGE_MODEL = os.environ.get("EXPAND_JUDGE_MODEL", "google/gemini-3.7-flash")
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_TARGET = 20
MAX_REFERENCES = 5          # the owner gives 1-5; more are ignored
REFS_PER_CALL = int(os.environ.get("EXPAND_REFS_PER_CALL", "3"))
WORKERS = int(os.environ.get("EXPAND_WORKERS", "4"))
MIN_IDENTITY = int(os.environ.get("EXPAND_MIN_IDENTITY", "8"))
MIN_CLEAN = int(os.environ.get("EXPAND_MIN_CLEAN", "7"))
MIN_STYLE = int(os.environ.get("EXPAND_MIN_STYLE", "7"))
# perceptual-hash distance (of 64 bits) at or below which two images are the
# same picture
DEDUPE_DISTANCE = 6
RETRIABLE_ATTEMPTS = 3
GEN_PREFIX = "gen_"
MANIFEST_NAME = ".expand.json"


def max_attempts(needed: int) -> int:
    """Paid-generation ceiling for one run: EXPAND_MAX_ATTEMPTS, else 2x."""
    env = os.environ.get("EXPAND_MAX_ATTEMPTS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    return max(2 * needed, needed + 2)


# ---------------------------------------------------------------------------
# Variations
# ---------------------------------------------------------------------------
# One per attempt, rotated. Twelve of thirty (40%) keep the canonical outfit so
# the LoRA still learns it; the rest change clothes so it is not welded to the
# character. Every axis the research lists is covered: angle, framing,
# expression, outfit, background, light, action/props.
# outfit None = keep the outfit of the reference.

VARIATIONS = [
    {"text": "front view, head-and-shoulders close-up portrait, neutral relaxed expression, plain light grey studio background, soft even light", "aspect": "3:4", "outfit": None},
    {"text": "three-quarter view from the left, waist-up shot, gentle smile, standing in a sunlit park with blurred trees behind", "aspect": "3:4", "outfit": None},
    {"text": "full body shot, standing naturally with weight on one leg, front view, city sidewalk in daytime", "aspect": "2:3", "outfit": None},
    {"text": "side profile view facing right, close-up of the face and shoulders, calm expression, dark background with a single warm rim light", "aspect": "1:1", "outfit": None},
    {"text": "seen from behind and slightly to the side, looking back over the shoulder at the viewer, waist-up, evening street with bokeh lights", "aspect": "3:4", "outfit": None},
    {"text": "high-angle shot looking down at the character, full body, sitting on the floor of a cozy bedroom, soft lamp light", "aspect": "3:4", "outfit": None},
    {"text": "low-angle shot looking up at the character, waist-up, confident expression, blue sky with clouds behind", "aspect": "3:4", "outfit": None},
    {"text": "laughing openly with eyes partly closed, waist-up, three-quarter view from the right, bright cafe interior", "aspect": "4:3", "outfit": None},
    {"text": "walking toward the camera, full body, mid-stride, busy shopping street, overcast daylight", "aspect": "2:3", "outfit": None},
    {"text": "close-up of the face, surprised expression with raised eyebrows, front view, colourful neon-lit night background", "aspect": "1:1", "outfit": None},
    {"text": "sitting at a desk reading a book, waist-up, three-quarter view, library with warm afternoon light through tall windows", "aspect": "4:3", "outfit": None},
    {"text": "full body, side view, leaning against a brick wall with arms crossed, golden hour sunlight", "aspect": "2:3", "outfit": None},
    {"text": "front view, waist-up, serious determined expression, rainy night street, wet reflections, cool blue light", "aspect": "3:4", "outfit": "a dark raincoat over casual clothes"},
    {"text": "full body, three-quarter view, standing on a beach at sunset, wind in the hair, warm orange backlight", "aspect": "2:3", "outfit": "light summer clothes"},
    {"text": "close-up portrait, soft smile, front view, autumn forest background with orange leaves, diffuse light", "aspect": "3:4", "outfit": "a knitted autumn sweater"},
    {"text": "waist-up, three-quarter view from the left, holding a cup of coffee with both hands, kitchen in the morning light", "aspect": "4:3", "outfit": "comfortable home clothes"},
    {"text": "full body, dynamic action pose jumping or running, low angle, urban rooftop, dramatic sky", "aspect": "2:3", "outfit": "athletic sportswear"},
    {"text": "side profile facing left, waist-up, looking out of a train window, moody overcast light", "aspect": "4:3", "outfit": "a casual jacket"},
    {"text": "front view, full body, standing in a snowy street at night, warm street lamps, breath visible in the cold air", "aspect": "2:3", "outfit": "a warm winter coat and scarf"},
    {"text": "waist-up, sad or pensive expression looking down, three-quarter view, dim room lit by a window, soft shadows", "aspect": "3:4", "outfit": "a plain oversized t-shirt"},
    {"text": "close-up of the face, angry frown, front view, dark dramatic background with hard side light", "aspect": "1:1", "outfit": None},
    {"text": "full body, seen from behind, walking away down a forest path, dappled sunlight", "aspect": "2:3", "outfit": "hiking clothes"},
    {"text": "waist-up, three-quarter view from the right, winking playfully, colourful amusement park in the background", "aspect": "3:4", "outfit": "a casual hoodie"},
    {"text": "sitting on a park bench, full body, side view, relaxed, spring flowers around, bright daylight", "aspect": "3:2", "outfit": "a light spring outfit"},
    {"text": "close-up, eyes closed, peaceful expression, top light, simple dark background", "aspect": "1:1", "outfit": None},
    {"text": "full body, three-quarter view, standing in a modern office, arms relaxed, cool fluorescent light", "aspect": "2:3", "outfit": "smart business attire"},
    {"text": "waist-up, high-angle shot, looking up at the camera with a curious expression, cluttered workshop background", "aspect": "3:4", "outfit": "work clothes with an apron"},
    {"text": "full body, elegant pose, front view, evening party in a grand hall, warm chandelier light", "aspect": "2:3", "outfit": "formal evening wear"},
    {"text": "waist-up, side view, playing a guitar, small stage with coloured spotlights", "aspect": "4:3", "outfit": "a stage outfit"},
    {"text": "front view, waist-up, shy smile, cherry blossom trees background, soft pastel daylight", "aspect": "3:4", "outfit": "a light dress or light shirt, whichever suits the character"},
]


def build_prompt(variation: dict, char_class: str = "", n_refs: int = 1) -> str:
    """The instruction for one attempt: copy the identity, change only this."""
    who = char_class.strip() or "character"
    refs = "the reference image" if n_refs <= 1 else f"the {n_refs} reference images (all the same {who})"
    outfit = ("Outfit: keep exactly the outfit from the reference — same garments, colours and accessories."
              if variation.get("outfit") is None
              else f"Outfit: {variation['outfit']} (a new outfit; everything about the body and face stays the same).")
    return (
        f"Create a brand-new image of the exact same {who} shown in {refs}.\n\n"
        "The identity must be identical to the reference: same face and facial structure, same eyes "
        "(shape and colour), same eyebrows, same nose and mouth, same hairstyle, hair length and hair "
        "colour, same skin tone, same body type and proportions, same apparent age, and every "
        "permanent mark (scars, tattoos, freckles, moles, piercings).\n"
        "Keep the medium and art style of the reference exactly: a photograph stays a realistic "
        "photograph with natural skin texture and real camera optics; an anime, cartoon, game or "
        "illustrated character stays in the same drawing style, line work, shading and colouring. "
        "Do not beautify, smooth the skin, change the face shape or restyle the character.\n\n"
        f"Change only this — {variation['text']}.\n"
        f"{outfit}\n\n"
        f"Exactly one {who} in the image. No collage, no split panels, no character sheet, no text, "
        "no watermark, no border. Correct anatomy and hands."
    )


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

class JudgeError(RuntimeError):
    """The judge did not return a usable verdict."""


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "same_identity": {"type": "integer", "description": "0-10, 10 = unmistakably the same character"},
        "clean": {"type": "integer", "description": "0-10, 10 = no artifacts at all"},
        "same_style": {"type": "integer", "description": "0-10, 10 = same medium and art style"},
        "follows_prompt": {"type": "integer", "description": "0-10"},
        "single_subject": {"type": "boolean"},
        "issues": {"type": "string"},
    },
    "required": ["same_identity", "clean", "same_style", "follows_prompt", "single_subject", "issues"],
    "additionalProperties": False,
}

JUDGE_INSTRUCTIONS = (
    "You are a strict quality judge for a character LoRA training dataset. The first {n} image(s) "
    "are REFERENCES of one character. The last image is a CANDIDATE that was asked to show the same "
    "character with this change: \"{change}\".\n\n"
    "Score the CANDIDATE from 0 to 10 on:\n"
    "- same_identity: is it unmistakably the same character? Compare face structure, eyes, hair "
    "(style, length, colour), skin tone, body proportions, age and permanent marks. Ignore changes "
    "of pose, expression, outfit, background and lighting — those were requested. A similar-looking "
    "but different person, an 'average' face or a changed hair colour scores 5 or lower.\n"
    "- clean: rendering quality. Deduct for extra or fused fingers, broken limbs, warped faces, "
    "melted details, garbled text, watermarks, collage or panel layouts. 10 = flawless.\n"
    "- same_style: same medium and art style as the references (photo vs anime vs 3D, line work, "
    "shading). An anime character turned photoreal, or a photo turned plastic/airbrushed, scores low.\n"
    "- follows_prompt: how well it shows the requested change.\n"
    "- single_subject: true if exactly one instance of the character is in the image.\n"
    "- issues: one short sentence naming the main problem, or \"none\".\n"
    "Answer only with the JSON object."
)


def build_judge_payload(ref_urls: list[str], candidate_url: str, change: str,
                        model: str | None = None) -> dict:
    content = [{"type": "text", "text": JUDGE_INSTRUCTIONS.format(n=len(ref_urls), change=change)}]
    for i, url in enumerate(ref_urls, 1):
        content.append({"type": "text", "text": f"REFERENCE {i}:"})
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": "CANDIDATE:"})
    content.append({"type": "image_url", "image_url": {"url": candidate_url}})
    return {
        "model": model or JUDGE_MODEL,
        "messages": [{"role": "user", "content": content}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "verdict", "strict": True, "schema": JUDGE_SCHEMA}},
        # without a reasoning cap the Gemini flash models spend the budget
        # thinking and return an empty message (measured on the caption path)
        "reasoning": {"effort": "low"},
        "plugins": [{"id": "response-healing"}],
        "temperature": 0,
        "usage": {"include": True},
    }


def _clamp_score(value) -> int:
    try:
        return max(0, min(10, int(round(float(value)))))
    except (TypeError, ValueError):
        raise JudgeError(f"nota inválida: {value!r}")


def parse_judge(body: dict) -> dict:
    """The verdict from a chat/completions response, normalised.

    Tolerates the usual wrappers (```json fences, prose around the object) —
    the schema asks for bare JSON, but a judge that adds a fence is still a
    judge that answered.
    """
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise JudgeError(f"resposta do juiz sem conteúdo: {str(body)[:300]}")
    if isinstance(content, list):  # some providers return content parts
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    text = (content or "").strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise JudgeError(f"juiz não devolveu JSON: {text[:200]}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeError(f"JSON inválido do juiz: {exc}")
    verdict = {key: _clamp_score(data.get(key))
               for key in ("same_identity", "clean", "same_style", "follows_prompt")}
    single = data.get("single_subject", True)
    verdict["single_subject"] = single if isinstance(single, bool) else str(single).lower() == "true"
    verdict["issues"] = str(data.get("issues") or "")[:300]
    verdict["cost"] = float((body.get("usage") or {}).get("cost") or 0.0)
    return verdict


def accept(verdict: dict, min_identity: int = MIN_IDENTITY, min_clean: int = MIN_CLEAN,
           min_style: int = MIN_STYLE) -> tuple[bool, str]:
    """(kept?, why not). Identity first: it is the whole point of the dataset."""
    if verdict["same_identity"] < min_identity:
        return False, f"identidade {verdict['same_identity']}/10"
    if verdict["same_style"] < min_style:
        return False, f"estilo {verdict['same_style']}/10"
    if verdict["clean"] < min_clean:
        return False, f"artefatos (limpeza {verdict['clean']}/10)"
    if not verdict["single_subject"]:
        return False, "mais de um personagem"
    return True, ""


def judge(ref_paths: list[Path], candidate_png: bytes, change: str,
          timeout: float = 120.0) -> dict:
    """Ask the VLM judge. Raises JudgeError / imagegen errors."""
    import base64

    refs = [imagegen.to_data_url(Path(p))[0] for p in ref_paths]
    cand = "data:image/png;base64," + base64.b64encode(_shrink_png(candidate_png)).decode()
    payload = build_judge_payload(refs, cand, change)
    request = urllib.request.Request(
        CHAT_URL, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {imagegen.api_key()}",
                 "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        raise imagegen.classify_http_error(exc.code, exc.read().decode("utf-8", "replace")) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise imagegen.RetriableError(f"juiz: {exc}") from exc
    return parse_judge(body)


def _shrink_png(png: bytes, edge: int = 1024) -> bytes:
    from PIL import Image

    with Image.open(io.BytesIO(png)) as im:
        if max(im.size) <= edge:
            return png
        im = im.convert("RGB")
        im.thumbnail((edge, edge), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------------

def dhash(image) -> int:
    """64-bit difference hash of a PIL image, a path or PNG bytes."""
    from PIL import Image

    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    elif isinstance(image, (str, Path)):
        image = Image.open(image)
    small = image.convert("L").resize((9, 8), Image.LANCZOS)
    px = small.tobytes()
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


def is_duplicate(h: int, known: list[int], distance: int = DEDUPE_DISTANCE) -> bool:
    return any(bin(h ^ k).count("1") <= distance for k in known)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def reference_images(dataset_dir: Path) -> list[Path]:
    """The owner's images: everything in dataset/ that this module did not make."""
    try:
        return sorted(f for f in Path(dataset_dir).iterdir()
                      if f.is_file() and f.suffix.lower() in IMAGE_EXTS
                      and not f.name.startswith(GEN_PREFIX))
    except OSError:
        return []


def generated_images(dataset_dir: Path) -> list[Path]:
    try:
        return sorted(f for f in Path(dataset_dir).iterdir()
                      if f.is_file() and f.name.startswith(GEN_PREFIX)
                      and f.suffix.lower() in IMAGE_EXTS)
    except OSError:
        return []


def refs_for_attempt(refs: list[Path], index: int, per_call: int = REFS_PER_CALL) -> list[Path]:
    """A rotating window over the references, so each one leads some attempts."""
    if len(refs) <= per_call:
        return list(refs)
    return [refs[(index + k) % len(refs)] for k in range(per_call)]


def _load_manifest(dataset_dir: Path) -> dict:
    try:
        data = json.loads((dataset_dir / MANIFEST_NAME).read_text())
        if isinstance(data, dict) and isinstance(data.get("attempts"), list):
            return data
    except (OSError, ValueError):
        pass
    return {"attempts": [], "cost": 0.0}


def _default_generate(prompt: str, refs: list[Path], aspect: str) -> tuple[bytes, float]:
    params = imagegen.model_parameters(IMAGE_MODEL)
    return imagegen.generate_from_references(prompt, refs, aspect, IMAGE_MODEL,
                                             quality=IMAGE_QUALITY, params=params)


def _with_retries(fn, *args):
    last: Exception | None = None
    for attempt in range(RETRIABLE_ATTEMPTS):
        try:
            return fn(*args)
        except imagegen.RetriableError as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise last


def expand_dataset(dataset_dir: Path, job, target: int = DEFAULT_TARGET,
                   char_class: str = "", generate=_default_generate, judge_fn=judge,
                   workers: int = WORKERS) -> dict:
    """Fill dataset/ with `target` accepted gen_*.png images of the character.

    Images already generated count toward the target, so a cancelled run
    resumes without paying for them again. Stops at the target or at the
    attempt ceiling, whichever comes first.
    """
    dataset_dir = Path(dataset_dir)
    refs = reference_images(dataset_dir)
    if not refs:
        raise JobFailed("mande de 1 a 5 imagens do personagem antes de expandir")
    if len(refs) > MAX_REFERENCES:
        job.log(f"{len(refs)} imagens no dataset — usando as {MAX_REFERENCES} primeiras como referência.")
        refs = refs[:MAX_REFERENCES]

    manifest = _load_manifest(dataset_dir)
    have = len(generated_images(dataset_dir))
    needed = max(0, int(target) - have)
    totals = {"accepted": 0, "rejected": 0, "refused": 0, "failed": 0, "duplicates": 0,
              "attempts": 0, "cost_images": 0.0, "cost_judge": 0.0,
              "existing": have, "target": int(target), "model": IMAGE_MODEL, "judge": JUDGE_MODEL}
    if needed == 0:
        job.log(f"Já existem {have} imagens geradas — nada a fazer.")
        return totals

    ceiling = max_attempts(needed)
    job.log(f"Expandindo {len(refs)} referência(s) → +{needed} imagens com {IMAGE_MODEL} "
            f"(juiz {JUDGE_MODEL}, identidade ≥ {MIN_IDENTITY}, no máximo {ceiling} gerações).")

    lock = threading.Lock()
    known = [dhash(p) for p in refs] + [dhash(p) for p in generated_images(dataset_dir)]
    start = len(manifest["attempts"])
    account_dead: list[Exception] = []

    def record(entry: dict, **deltas) -> None:
        with lock:
            for k, v in deltas.items():
                totals[k] += v
            manifest["attempts"].append(entry)
            manifest["cost"] = round(manifest.get("cost", 0.0) + entry.get("cost", 0.0), 6)
            (dataset_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    def work(index: int) -> None:
        if account_dead:
            return
        job.check_cancel()
        variation = VARIATIONS[index % len(VARIATIONS)]
        attempt_refs = refs_for_attempt(refs, index)
        prompt = build_prompt(variation, char_class, len(attempt_refs))
        entry = {"index": index, "variation": variation["text"], "refs": [p.name for p in attempt_refs]}
        try:
            png, cost = _with_retries(generate, prompt, attempt_refs, variation["aspect"])
        except imagegen.AccountError as exc:
            with lock:
                account_dead.append(exc)
            return
        except imagegen.RefusedError as exc:
            record({**entry, "status": "refused", "error": str(exc)}, refused=1)
            return
        except Cancelled:
            raise
        except Exception as exc:
            record({**entry, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}, failed=1)
            job.log(f"⚠ geração {index}: {type(exc).__name__}: {exc}")
            return
        entry["cost"] = cost
        try:
            verdict = _with_retries(judge_fn, attempt_refs, png, variation["text"])
        except imagegen.AccountError as exc:
            record({**entry, "status": "failed", "error": str(exc)}, failed=1, cost_images=cost)
            with lock:
                account_dead.append(exc)
            return
        except Cancelled:
            raise
        except Exception as exc:
            # paid image, no verdict: kept out — an unjudged image is exactly
            # what this mode exists to keep out of the dataset
            record({**entry, "status": "failed", "error": f"juiz: {type(exc).__name__}: {exc}"},
                   failed=1, cost_images=cost)
            job.log(f"⚠ juiz falhou na geração {index}: {exc}")
            return
        entry["verdict"] = verdict
        entry["cost"] = cost + verdict.get("cost", 0.0)
        ok, why = accept(verdict)
        if not ok:
            record({**entry, "status": "rejected", "reason": why}, rejected=1,
                   cost_images=cost, cost_judge=verdict.get("cost", 0.0))
            job.log(f"✗ geração {index} recusada pelo juiz: {why} — {verdict['issues']}")
            return
        h = dhash(png)
        with lock:
            duplicate = is_duplicate(h, known)
            if not duplicate:
                known.append(h)
        if duplicate:
            record({**entry, "status": "duplicate"}, duplicates=1,
                   cost_images=cost, cost_judge=verdict.get("cost", 0.0))
            job.log(f"✗ geração {index}: quase idêntica a uma imagem existente")
            return
        name = f"{GEN_PREFIX}{index:04d}.png"
        _write_png(png, dataset_dir / name)
        record({**entry, "status": "ok", "file": name}, accepted=1,
               cost_images=cost, cost_judge=verdict.get("cost", 0.0))
        job.log(f"✔ {name} (identidade {verdict['same_identity']}, limpeza {verdict['clean']}, "
                f"estilo {verdict['same_style']}) — {variation['text'][:60]}")

    index = start
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        while not account_dead:
            job.check_cancel()
            with lock:
                missing = needed - totals["accepted"]
                left = ceiling - totals["attempts"]
            if missing <= 0 or left <= 0:
                break
            # never more in flight than still needed: no paid overshoot
            wave = list(range(index, index + min(missing, left, max(1, workers))))
            index += len(wave)
            with lock:
                totals["attempts"] += len(wave)
            list(pool.map(work, wave))

    totals["cost"] = round(totals["cost_images"] + totals["cost_judge"], 4)
    job.extra["expand"] = totals
    if account_dead:
        raise JobFailed(f"{account_dead[0]} — {totals['accepted']} imagens novas aceitas. "
                        f"Resolva e rode de novo: as aceitas ficam.")
    msg = (f"{totals['accepted']} aceitas, {totals['rejected']} recusadas pelo juiz, "
           f"{totals['duplicates']} duplicadas, {totals['refused']} barradas na moderação, "
           f"{totals['failed']} falhas — custo ~${totals['cost']:.2f}")
    if totals["accepted"] < needed:
        job.log(f"⚠ meta não atingida ({have + totals['accepted']} de {target}) depois de "
                f"{totals['attempts']} gerações: {msg}. Rode de novo para continuar.")
    else:
        job.log(f"✔ Expansão concluída: {msg}")
    return totals


def _write_png(data: bytes, dest: Path) -> None:
    """Re-encode as PNG (models return PNG, JPEG or WebP) via a temp name, so a
    half-written file never carries a media extension in the dataset."""
    from PIL import Image

    tmp = dest.with_name(f".{dest.name}.part")
    with Image.open(io.BytesIO(data)) as im:
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        im.save(tmp, format="PNG")
    os.replace(tmp, dest)
