"""DiffSynth-Studio engine: Qwen Image 2.1 (text-to-image and edit).

The dataset decides the task. A `control/` folder beside the images means
editing: `control/<stem>.<ext>` (or `<stem>_0`, `<stem>_1`… for several
references) is the input, `<stem>.<ext>` the result and `<stem>.txt` the
instruction. Without one, it is plain text-to-image.

The two cannot share a run — DiffSynth's prompt encoder switches to the edit
template whenever an edit image is present, and an empty list breaks it — so a
half-paired dataset is refused before anything is downloaded.

Training runs through trainero/tools/diffsynth_train.py inside the engine venv:
DiffSynth's own module and loop, plus per-epoch samples and the progress lines
the UI reads. Models come from MODELS_DIR (downloaded from Hugging Face by
models_download), never from ModelScope: DIFFSYNTH_SKIP_DOWNLOAD points
DiffSynth at the local copy.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from . import dataset as ds
from .config import IMAGE_EXTS, REPO_DIR
from .jobs import Job, JobFailed
from .models_download import model_root
from .presets import MODELS, vram_tier

WRAPPER = REPO_DIR / "trainero" / "tools" / "diffsynth_train.py"
METADATA = "diffsynth_metadata.json"


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

def control_images(dataset_dir: Path) -> dict[str, list[Path]]:
    """stem -> its reference images, in order (`a.png`, then `a_0`, `a_1`…)."""
    cdir = Path(dataset_dir) / "control"
    if not cdir.is_dir():
        return {}
    target_stems = {t.stem for t in targets(dataset_dir)}
    found: dict[str, list[tuple[int, Path]]] = {}
    for f in sorted(cdir.iterdir()):
        if not ds.is_media_item(f) or f.suffix.lower() not in IMAGE_EXTS:
            continue
        m = re.fullmatch(r"(.+)_(\d+)", f.stem)
        # `a_0.png` is a second reference of `a` only when `a` is a target and
        # `a_0` is not: a target genuinely named `img_01` keeps `img_01.png`
        if m and f.stem not in target_stems and m.group(1) in target_stems:
            found.setdefault(m.group(1), []).append((int(m.group(2)) + 1, f))
        else:
            found.setdefault(f.stem, []).append((0, f))
    return {stem: [p for _, p in sorted(items)] for stem, items in found.items()}


def targets(dataset_dir: Path) -> list[Path]:
    return [f for f in sorted(Path(dataset_dir).iterdir())
            if ds.is_media_item(f) and f.suffix.lower() in IMAGE_EXTS]


def is_edit(dataset_dir: Path) -> bool:
    return bool(control_images(dataset_dir))


def preflight(dataset_dir: Path, stats: dict) -> bool:
    """Refuse what would fail after the download. Returns True for edit mode."""
    if stats.get("videos"):
        raise JobFailed("Qwen Image 2.1 é só imagem — tire os vídeos do dataset")
    controls = control_images(dataset_dir)
    if not controls:
        return False
    items = targets(dataset_dir)
    unpaired = [p.name for p in items if p.stem not in controls]
    if unpaired:
        raise JobFailed(
            f"dataset de edição: {len(unpaired)} de {len(items)} imagens sem imagem de controle "
            f"em control/ com o mesmo nome (ex.: {', '.join(unpaired[:3])}). Um treino é de "
            f"edição (todas com controle) ou texto→imagem (pasta control/ vazia), nunca os dois")
    if stats.get("missing_captions"):
        # the caption profiles describe images; an edit LoRA learns instructions
        raise JobFailed(
            f"dataset de edição: {stats['missing_captions']} imagens sem .txt — em edição o "
            f".txt é a instrução (ex.: \"deixe no estilo aquarela\") e precisa vir de você "
            f"(ex.: {', '.join(stats['missing_sample'][:3])})")
    return True


def write_metadata(dataset_dir: Path, path: Path, edit: bool) -> Path:
    """DiffSynth's UnifiedDataset metadata: one record per training image."""
    controls = control_images(dataset_dir) if edit else {}
    records = []
    for img in targets(dataset_dir):
        caption = img.with_suffix(".txt").read_text(encoding="utf-8", errors="replace").strip()
        record = {"image": str(img), "prompt": caption}
        if edit:
            record["edit_image"] = [str(c) for c in controls[img.stem]]
        records.append(record)
    if not records:
        raise JobFailed("nenhuma imagem no dataset")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def _snap32(width: float, height: float, pixels: int) -> list[int]:
    scale = math.sqrt(pixels / max(1.0, width * height))
    return [max(32, int(width * scale) // 32 * 32), max(32, int(height * scale) // 32 * 32)]


def edit_sample(dataset_dir: Path, max_pixels: int) -> tuple[list[Path], str, list[int]] | None:
    """The first pair: its references, its instruction, and a frame of the
    target's shape. Sampling an edit LoRA means asking it to do the edit."""
    from PIL import Image

    controls = control_images(dataset_dir)
    for img in targets(dataset_dir):
        if img.stem in controls:
            try:
                with Image.open(img) as im:
                    size = _snap32(im.size[0], im.size[1], max_pixels)
            except OSError:
                continue
            prompt = img.with_suffix(".txt").read_text(encoding="utf-8", errors="replace").strip()
            return controls[img.stem], prompt, size
    return None


def build_config(model_key: str, overrides: dict, schedule: dict, vram_gb: float,
                 dataset_dir: Path, metadata: Path, output_dir: Path, output_name: str,
                 edit: bool, sample: dict | None, job: Job | None = None) -> dict:
    """CLI arguments for trainero/tools/diffsynth_train.py (DiffSynth names)."""
    model = MODELS[model_key]
    tier = vram_tier(model_key, vram_gb).get("train", {})
    patterns = model["model_args"]

    def note(msg: str) -> None:
        if job:
            job.log(msg)

    for key, why in (("network_alpha", "o DiffSynth fixa alpha = rank"),
                     ("blocks_to_swap", "o DiffSynth não troca blocos; o tier de VRAM usa FP8/offload"),
                     ("loraplus", "o DiffSynth não tem LoRA+")):
        if overrides.get(key) not in (None, "", False):
            note(f"⚠ {key} ignorado no Qwen Image 2.1: {why}")

    try:
        rank = int(overrides.get("network_dim") or model["train"]["network_dim"])
        lr = float(overrides.get("learning_rate") or model["train"]["learning_rate"])
    except (TypeError, ValueError):
        raise JobFailed("rank/learning rate inválidos")

    cfg = {
        "dataset_base_path": str(dataset_dir),
        "dataset_metadata_path": str(metadata),
        "data_file_keys": "image,edit_image" if edit else "image",
        "max_pixels": int(model.get("max_pixels", 1024 * 1024)),
        "dataset_repeat": int(schedule["num_repeats"]),
        "dataset_num_workers": 2,
        "model_id_with_origin_paths": ",".join(patterns[k] for k in ("dit", "text_encoder", "vae")),
        "processor_path": str(model_root(model_key) / model["processor"]),
        "learning_rate": lr,
        "num_epochs": int(schedule["epochs"]),
        "remove_prefix_in_ckpt": "pipe.dit.",
        "output_path": str(output_dir),
        "output_name": output_name,
        "lora_base_model": "dit",
        # empty = DiffSynth auto-detects every large Linear in each block, the
        # official recipe; ComfyUI maps all of them (incl. the fused MLP)
        "lora_target_modules": "",
        "lora_rank": rank,
        "use_gradient_checkpointing": True,
    }
    if edit:
        cfg["extra_inputs"] = "edit_image"
    fp8 = [patterns[name] for name in tier.get("fp8_models", [])]
    if fp8:
        cfg["fp8_models"] = ",".join(fp8)
    for flag in ("use_gradient_checkpointing_offload", "enable_model_cpu_offload"):
        if tier.get(flag):
            cfg[flag] = True
    if edit and vram_gb < 40:
        # every reference is a full image of tokens in the same sequence: two
        # controls + the target is ~3x the activations of a T2I step. Keeping
        # the checkpointed activations in RAM is what makes that fit in 32 GB.
        cfg["use_gradient_checkpointing_offload"] = True
    if sample:
        cfg.update({f"sample_{k}": v for k, v in sample.items()})
    return cfg


def summary(cfg: dict, schedule: dict, edit: bool) -> dict:
    return {
        "engine": "diffsynth", "task": "edit" if edit else "t2i",
        "network_dim": cfg["lora_rank"], "network_alpha": cfg["lora_rank"],
        "learning_rate": cfg["learning_rate"],
        "epochs": schedule["epochs"], "repeats": schedule["num_repeats"], "batch_size": 1,
        "fp8": cfg.get("fp8_models", "").count(":") or 0,
        "offload": bool(cfg.get("enable_model_cpu_offload")),
    }


def env_for(model_key: str) -> dict:
    return {
        "DIFFSYNTH_MODEL_BASE_PATH": str(model_root(model_key)),
        "DIFFSYNTH_SKIP_DOWNLOAD": "true",
        "TOKENIZERS_PARALLELISM": "false",
    }


def command(model_key: str, cfg: dict, accelerate_bin: str, engine_root: Path) -> list[str]:
    from .training import _cli_args

    script = engine_root / MODELS[model_key]["train_script"]
    return [accelerate_bin, "launch", "--num_cpu_threads_per_process", "1",
            "--mixed_precision", "bf16", str(WRAPPER),
            "--diffsynth_script", str(script)] + _cli_args(cfg)
