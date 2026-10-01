"""Model registry: every architecture Arrakis Trainero can train.

Pure data + tiny pure functions. Values come from the owner's proven trainers
(image_lora_arrakis, ltx23, research_anima_train) and the upstream musubi-tuner
docs. There is exactly ONE preset per model — the standard LoRA recipe. The
only user-facing knobs live in the collapsible advanced panel and are applied
as overrides on top of these dicts.

Engines:
  musubi     kohya-ss/musubi-tuner@main            (src/musubi_tuner/<arch>_*.py)
  musubi-ltx AkaneTendo25/musubi-tuner@ltx-2-dev   (scripts at repo root)
  sd-scripts kohya-ss/sd-scripts@main              (anima_train_network.py)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

ENGINES = {
    "musubi": {
        "repo": "https://github.com/kohya-ss/musubi-tuner.git",
        "branch": "main",
        "dir": "musubi-tuner",
        "install": "package",  # pip install -e .
        "script_prefix": "src/musubi_tuner/",
    },
    "musubi-ltx": {
        "repo": "https://github.com/AkaneTendo25/musubi-tuner.git",
        "branch": "ltx-2-dev",
        "dir": "musubi-ltx",
        "install": "package",
        "script_prefix": "",  # scripts live at repo root in this fork
    },
    "sd-scripts": {
        "repo": "https://github.com/kohya-ss/sd-scripts.git",
        "branch": "main",
        "dir": "sd-scripts",
        "install": "requirements",  # pip install -r requirements.txt (installs '.')
        "script_prefix": "",
    },
    "captioner": {
        "repo": "https://github.com/adbrasi/captionkreator.git",
        "branch": "main",
        "dir": "captionkreator",
        # `pip install -e .[wd14]`: the CLI plus the local booru tagger. No torch
        # and no accelerate — it is an HTTP client plus an ONNX model on CPU.
        "install": "package",
        "extras": "wd14",
        "torch": False,
        # The engine's code changes between runs while its dependencies do not;
        # a long-lived pod would otherwise caption with a stale engine.
        "pull": True,
    },
}

# ---------------------------------------------------------------------------
# Caption profiles (caption_profiles/<media>/<key>/, run by captionkreator)
# ---------------------------------------------------------------------------
# `trigger`: the profile writes the trigger word into every caption and refuses
# to run without one. Kept here as data because the UI and the preflight need it
# without parsing YAML; a test checks it against each profile.yaml.

CAPTION_PROFILES = {
    "image": [
        {"key": "generic-style", "label": "Estilo", "trigger": True},
        {"key": "generic-character", "label": "Personagem", "trigger": True},
        {"key": "default", "label": "Descrição em prosa (conceito, NSFW)", "trigger": False},
        {"key": "anima-style", "label": "Estilo (Anima)", "trigger": True},
        {"key": "anima-character", "label": "Personagem (Anima)", "trigger": True},
        {"key": "anima-concept", "label": "Conceito (Anima)", "trigger": False},
        {"key": "anima-outfit", "label": "Roupa (Anima)", "trigger": True},
    ],
    "video": [
        {"key": "grok_looping_animation_template", "label": "Loop (LTX 2.3)", "trigger": False},
        {"key": "nsfw_caption_video", "label": "Animação em loop NSFW", "trigger": False},
        {"key": "live-wallpaper", "label": "Live wallpaper", "trigger": False},
        {"key": "live_wallpaper_game_cg", "label": "Live wallpaper game CG", "trigger": False},
    ],
}
#: not offered in the UI: Style Rush captions its own dataset with it
STYLE_RUSH_CAPTION_PROFILE = {"key": "style-rush", "label": "Style Rush", "trigger": True}

#: what the owner is training, in his words — the one caption question a
#: non-expert can answer. The profile follows from it and the model family.
LORA_KINDS = {
    "style": "Estilo",
    "character": "Personagem",
    "concept": "Conceito / outro",
    "outfit": "Roupa",
}

_KIND_PROFILE = {
    "anima": {"style": "anima-style", "character": "anima-character",
              "concept": "anima-concept", "outfit": "anima-outfit"},
    "image": {"style": "generic-style", "character": "generic-character",
              "concept": "default", "outfit": "default"},
}


def caption_profile(key: str) -> dict | None:
    if key == STYLE_RUSH_CAPTION_PROFILE["key"]:
        return dict(STYLE_RUSH_CAPTION_PROFILE, media="image")
    for media, profiles in CAPTION_PROFILES.items():
        for p in profiles:
            if p["key"] == key:
                return dict(p, media=media)
    return None


def default_caption_profile(model_key: str | None, kind: str | None, media: str) -> str:
    """The profile a non-expert gets: chosen by what he trains and on what.

    Anima learns from booru tags mixed with prose, everything else here from
    prose; video has its own family. Without a model yet, prose is the safe
    default — every image model reads it.
    """
    if media == "video":
        if model_key == "ltx-23":
            return "grok_looping_animation_template"
        return "nsfw_caption_video"
    family = "anima" if model_key == "anima" else "image"
    return _KIND_PROFILE[family].get(kind or "", _KIND_PROFILE[family]["concept"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# downloads: (repo, file_in_repo, local_subpath). local paths relative to
#   MODELS_DIR/<model_key>/. A trailing "/" in file means snapshot of that dir.
# train: keys are the training-script argument names (written to train.toml
#   for engines that support --config_file, or passed as CLI otherwise).
# vram_tiers: first entry whose min_gb <= detected VRAM wins; extra args merge
#   into the train config (and flagged ones into the cache commands).

MODELS = {
    # ------------------------------------------------------------------ IMAGEM
    "krea2": {
        "label": "Krea 2",
        "base_model": "krea/Krea-2-Raw",
        "group": "imagem",
        "media": "image",
        "engine": "musubi",
        "arch": "krea2",
        "downloads": [
            ("krea/Krea-2-Raw", "raw.safetensors", "raw.safetensors"),
            # Qwen-Image-Edit_ComfyUI ships no vae/ at all — the VAE lives in the
            # base Qwen-Image repo (same file the qwen-image presets pull).
            ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors", "qwen_image_vae.safetensors"),
            ("Comfy-Org/Qwen3-VL", "text_encoders/qwen3vl_4b_bf16.safetensors", "qwen3vl_4b_bf16.safetensors"),
        ],
        "model_args": {"dit": "raw.safetensors", "vae": "qwen_image_vae.safetensors", "text_encoder": "qwen3vl_4b_bf16.safetensors"},
        "cache_latents_args": {"vae": "qwen_image_vae.safetensors"},
        "cache_te_args": {"text_encoder": "qwen3vl_4b_bf16.safetensors", "batch_size": 4},
        "network_module": "networks.lora_krea2",
        "train": {
            "network_dim": 32, "network_alpha": 32,
            "learning_rate": 1e-4, "optimizer_type": "adamw8bit", "lr_scheduler": "cosine",
            "timestep_sampling": "krea2_shift", "weighting_scheme": "none",
        },
        # Sampling on the K2 RAW model is CFG-based, and musubi only turns CFG
        # on when the prompt line carries a negative prompt (--n); the scale is
        # --l, NOT --g (--g is embedded guidance, which K2 ignores). Without
        # these two the samples come out blurry BY DESIGN (docs/krea2.md).
        # 5.5 is the official default (K2 "guidance 4.5" + 1). Anima has the
        # same --l rule (see its preset); the others ignore --g or set their own.
        "sample_args": {"l": 5.5, "n": "blurry, low quality, jpeg artifacts, watermark, text"},
        "max_blocks_to_swap": 26,  # docs/krea2.md: 28 blocks − 2
        "resolution": [1024, 1024],
        "target_steps": 2000,
        "vram_tiers": [
            {"min_gb": 70, "train": {}},
            # fp8 halves the 28 main blocks to ~12 GB (docs/krea2.md), leaving
            # a 32 GB card (RTX 5090) room for activations plus the ~8 GB text
            # encoder that visits at sample time — swapping blocks there only
            # costs step time. Swap starts at 24 GB cards, where fp8 alone sits
            # exactly at the limit during sampling.
            {"min_gb": 30, "train": {"fp8_base": True, "fp8_scaled": True}},
            {"min_gb": 22, "train": {"fp8_base": True, "fp8_scaled": True, "blocks_to_swap": 8}},
            {"min_gb": 0, "train": {"fp8_base": True, "fp8_scaled": True, "blocks_to_swap": 20}},
        ],
    },
    "anima": {
        "label": "Anima",
        "base_model": "circlestone-labs/Anima",
        "group": "imagem",
        "media": "image",
        "engine": "sd-scripts",
        "arch": "anima",
        "downloads": [
            ("circlestone-labs/Anima", "split_files/diffusion_models/anima-preview3-base.safetensors", "anima-preview3-base.safetensors"),
            ("circlestone-labs/Anima", "split_files/vae/qwen_image_vae.safetensors", "qwen_image_vae.safetensors"),
            ("circlestone-labs/Anima", "split_files/text_encoders/qwen_3_06b_base.safetensors", "qwen_3_06b_base.safetensors"),
        ],
        "model_args": {
            "pretrained_model_name_or_path": "anima-preview3-base.safetensors",
            "qwen3": "qwen_3_06b_base.safetensors",
            "vae": "qwen_image_vae.safetensors",
        },
        "train_script": "anima_train_network.py",
        "network_module": "networks.lora_anima",
        "train": {
            "network_dim": 32, "network_alpha": 16,
            "learning_rate": 2e-5, "optimizer_type": "AdamW8bit", "lr_scheduler": "cosine",
            "timestep_sampling": "sigmoid", "weighting_scheme": "uniform",
            "discrete_flow_shift": 3.0, "sigmoid_scale": 1.0,
            "vae_chunk_size": 64, "vae_disable_cache": True,
            "network_train_unet_only": True, "full_bf16": True,
            "cache_latents_to_disk": True, "cache_text_encoder_outputs_to_disk": True,
            "save_model_as": "safetensors", "save_precision": "bf16",
            # DO NOT add noise_offset: green-tint artifacts on Anima.
        },
        # sd-scripts' Anima sampler reads CFG from --l (prompt "scale") and
        # ignores --g; without this every sample ran at its 7.5 default.
        "sample_args": {"l": 4.0},
        "resolution": [1024, 1024],
        "bucket_reso_steps": 64,
        "exposures_per_image": 660,
        # sd-scripts LoRA keys do not match what ComfyUI loads for Anima.
        "comfy_convert": {"script": "networks/convert_anima_lora_to_comfy.py"},
        "vram_tiers": [
            {"min_gb": 44, "train": {}, "batch_size": 16},
            {"min_gb": 30, "train": {}, "batch_size": 8},
            {"min_gb": 22, "train": {}, "batch_size": 4},
            {"min_gb": 0, "train": {"blocks_to_swap": 16}, "batch_size": 2},
        ],
    },
    "flux-klein": {
        "label": "Flux Klein",
        "base_model": "black-forest-labs/FLUX.2-klein-base-9B",
        "group": "imagem",
        "media": "image",
        "engine": "musubi",
        "arch": "flux_2",
        "downloads": [
            ("black-forest-labs/FLUX.2-klein-base-9B", "flux-2-klein-base-9b.safetensors", "flux-2-klein-base-9b.safetensors"),
            ("black-forest-labs/FLUX.2-dev", "ae.safetensors", "ae.safetensors"),
            ("black-forest-labs/FLUX.2-klein-9B", "text_encoder/", "text_encoder/"),
        ],
        "model_args": {
            "dit": "flux-2-klein-base-9b.safetensors",
            "vae": "ae.safetensors",
            "text_encoder": "text_encoder/model-00001-of-00004.safetensors",
        },
        "extra_args": {"model_version": "klein-base-9b"},
        "cache_latents_args": {"vae": "ae.safetensors", "model_version": "klein-base-9b"},
        "cache_te_args": {
            "text_encoder": "text_encoder/model-00001-of-00004.safetensors",
            "model_version": "klein-base-9b", "batch_size": 8, "fp8_text_encoder": True,
        },
        "network_module": "networks.lora_flux_2",
        "train": {
            "network_dim": 128, "network_alpha": 128,
            "learning_rate": 1e-4, "optimizer_type": "adamw8bit", "lr_scheduler": "cosine",
            "timestep_sampling": "flux2_shift", "weighting_scheme": "none",
        },
        "max_blocks_to_swap": 16,  # docs/flux_2.md, klein-9b with fp8
        "resolution": [1024, 1024],
        "supports_control": True,   # FLUX.2 reference image = control_directory
        "target_steps": 2000,
        "vram_tiers": [
            {"min_gb": 70, "train": {}},
            {"min_gb": 30, "train": {"fp8_base": True, "fp8_scaled": True}},
            {"min_gb": 22, "train": {"fp8_base": True, "fp8_scaled": True, "blocks_to_swap": 8}},
            {"min_gb": 0, "train": {"fp8_base": True, "fp8_scaled": True, "blocks_to_swap": 16}},
        ],
    },
    "qwen-image": {
        "label": "Qwen Image",
        "base_model": "Comfy-Org/Qwen-Image_ComfyUI",
        "group": "imagem",
        "media": "image",
        "engine": "musubi",
        "arch": "qwen_image",
        "downloads": [
            ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/diffusion_models/qwen_image_bf16.safetensors", "qwen_image_bf16.safetensors"),
            ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors", "qwen_image_vae.safetensors"),
            ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b.safetensors", "qwen_2.5_vl_7b.safetensors"),
        ],
        "model_args": {"dit": "qwen_image_bf16.safetensors", "vae": "qwen_image_vae.safetensors", "text_encoder": "qwen_2.5_vl_7b.safetensors"},
        "cache_latents_args": {"vae": "qwen_image_vae.safetensors"},
        "cache_te_args": {"text_encoder": "qwen_2.5_vl_7b.safetensors", "batch_size": 8},
        "network_module": "networks.lora_qwen_image",
        "train": {
            "network_dim": 64, "network_alpha": 64,
            "learning_rate": 5e-5, "optimizer_type": "adamw8bit", "lr_scheduler": "cosine",
            "timestep_sampling": "shift", "weighting_scheme": "none", "discrete_flow_shift": 2.2,
        },
        "resolution": [1024, 1024],
        "target_steps": 2000,
        "vram_tiers": [
            {"min_gb": 70, "train": {}},
            {"min_gb": 38, "train": {"fp8_base": True, "fp8_scaled": True}},
            # 32 GB (RTX 5090) used to fall to the 22 GB tier and swap 16 blocks
            # while sitting at 18 GB of 32. The DiT is 40.9 GB in bf16, so ~20.5
            # at fp8; measured overhead outside the blocks is ~4.3 GB, which puts
            # training near 25 GB with nothing swapped. The margin that is left
            # is for the 7B text encoder that loads to encode the sample prompt,
            # so a few blocks still travel — 8, not 16. Raise or zero it from the
            # advanced panel if sampling is off.
            {"min_gb": 30, "train": {"fp8_base": True, "fp8_scaled": True, "blocks_to_swap": 8}, "cache_te": {"fp8_vl": True}},
            {"min_gb": 22, "train": {"fp8_base": True, "fp8_scaled": True, "blocks_to_swap": 16}, "cache_te": {"fp8_vl": True}},
            {"min_gb": 0, "train": {"fp8_base": True, "fp8_scaled": True, "blocks_to_swap": 45}, "cache_te": {"fp8_vl": True}},
        ],
    },
    "ideogram": {
        "label": "Ideogram",
        "base_model": "Comfy-Org/Ideogram-4",
        "group": "imagem",
        "media": "image",
        "engine": "musubi",
        "arch": "ideogram4",
        "downloads": [
            ("Comfy-Org/Ideogram-4", "diffusion_models/ideogram4_fp8_scaled.safetensors", "ideogram4_fp8_scaled.safetensors"),
            ("Comfy-Org/Ideogram-4", "text_encoders/qwen3vl_8b_fp8_scaled.safetensors", "qwen3vl_8b_fp8_scaled.safetensors"),
            ("Comfy-Org/Ideogram-4", "vae/flux2-vae.safetensors", "flux2-vae.safetensors"),
        ],
        "model_args": {
            "dit": "ideogram4_fp8_scaled.safetensors",
            # sampling reads args.vae / args.text_encoder
            "vae": "flux2-vae.safetensors",
            "text_encoder": "qwen3vl_8b_fp8_scaled.safetensors",
        },
        "cache_latents_args": {"vae": "flux2-vae.safetensors", "vae_dtype": "bfloat16"},
        "cache_te_args": {"text_encoder": "qwen3vl_8b_fp8_scaled.safetensors"},
        "network_module": "networks.lora_ideogram4",
        "train": {
            "network_dim": 32, "network_alpha": 32,
            "learning_rate": 1e-4, "optimizer_type": "adamw8bit", "lr_scheduler": "cosine",
            # timestep_sampling defaults to ideogram4_shift; weighting must stay none.
        },
        "max_blocks_to_swap": 33,  # ideogram4_train_network.py raises above it
        "resolution": [1024, 1024],
        "target_steps": 2000,
        "vram_tiers": [
            {"min_gb": 44, "train": {}},
            {"min_gb": 22, "train": {"blocks_to_swap": 16}},
            {"min_gb": 0, "train": {"blocks_to_swap": 33}},
        ],
    },
    # ------------------------------------------------------------------- VÍDEO
    "ltx-23": {
        "label": "LTX 2.3",
        "base_model": "Lightricks/LTX-2.3",
        "group": "video",
        "media": "video",
        "engine": "musubi-ltx",
        "arch": "ltx2",
        "downloads": [
            ("Lightricks/LTX-2.3", "ltx-2.3-22b-dev.safetensors", "ltx-2.3-22b-dev.safetensors"),
            ("google/gemma-3-12b-it-qat-q4_0-unquantized", "/", "text_encoder/"),
        ],
        "model_args": {"ltx2_checkpoint": "ltx-2.3-22b-dev.safetensors", "gemma_root": "text_encoder"},
        "network_module": "networks.lora_ltx2",
        "train": {
            "network_dim": 256, "network_alpha": 256,
            "learning_rate": 1e-4, "optimizer_type": "adamw", "lr_scheduler": "cosine",
            "timestep_sampling": "shifted_logit_normal",
            "caption_dropout_rate": 0.05, "max_grad_norm": 1.0,
        },
        "ltx": {
            "version": "2.3", "mode": "v", "first_frame_p": "1.0",
            "resolution": "768x512x81", "fps": 25.0,
            "lora_target_preset": "t2v",
        },
        "resolution": [768, 512],
        "target_steps": 3000,
        "supports_slider_native": True,
        "vram_tiers": [
            {"min_gb": 140, "train": {}},
            {"min_gb": 70, "train": {"fp8_base": True, "fp8_scaled": True}},
            # gemma_load_in_8bit in train too: the trainer loads Gemma-12B to
            # encode the sample prompt (ltx2_sampling.py), ~24 GB in bf16
            {"min_gb": 44, "train": {"fp8_base": True, "fp8_scaled": True, "blocks_to_swap": 10, "gemma_load_in_8bit": True}, "cache_te": {"gemma_load_in_8bit": True}},
            {"min_gb": 30, "train": {"fp8_base": True, "fp8_scaled": True, "blocks_to_swap": 16, "gemma_load_in_8bit": True}, "cache_te": {"gemma_load_in_8bit": True}},
            {"min_gb": 0, "train": {"fp8_base": True, "fp8_scaled": True, "blocks_to_swap": 24, "gemma_load_in_8bit": True}, "cache_te": {"gemma_load_in_8bit": True}},
        ],
    },
}

MODEL_ORDER = ["krea2", "anima", "flux-klein", "qwen-image", "ideogram", "ltx-23"]

# Only standard LoRA is offered (LoHa/LoKr were dropped on purpose: one network
# type keeps every preset proven). LoRA+ rides on it as an optional ratio.
LORAPLUS_RATIO = 16  # owner's value in image_lora_arrakis


# ---------------------------------------------------------------------------
# Style Rush
# ---------------------------------------------------------------------------
# The synthetic conversion dataset chases a fixed pair count and the owner
# cancels when the samples look right, so there is no target_steps math here.

STYLE_RUSH_SCHEDULE = {"num_repeats": 2, "epochs": 5, "save_every_n_epochs": 1}
CONTROL_RESOLUTION = [1024, 1024]

# ---------------------------------------------------------------------------
# Sampling during training
# ---------------------------------------------------------------------------
# One prompt, the same for every image model. It is written to exercise, in a
# single frame, everything that reveals a LoRA going wrong: face and eye contact,
# hands holding an object, fabric, an animal, directional light with shadow, and
# a background with depth.

SAMPLE_PROMPT = (
    "A young woman sits alone at a tall window in a quiet apartment, one knee drawn "
    "up onto the cushioned sill, both hands wrapped around a chipped ceramic mug. "
    "Daylight falls across her in long bars, bright on her cheek and throat, and fine "
    "dust turns slowly in the air. A grey cat lies curled asleep against her hip, one "
    "paw over its face. Her hair spills loose over one shoulder, her sweater slipping "
    "wide at the collar, and she looks straight at the viewer with a soft, unhurried "
    "gaze, eyes meeting the camera, while the city beyond dissolves into hazy blue "
    "rooftops."
)
SAMPLE_STEPS = 28
SAMPLE_GUIDANCE = 4.0
SAMPLE_SEED = 42
SAMPLE_ASPECT = (16, 9)


def sample_resolution(resolution: list[int]) -> list[int]:
    """The frame a sample is rendered at: SAMPLE_ASPECT, same pixel budget.

    Only the sample. The training resolution is untouched — this sets --w/--h on
    the prompt line and nothing else. A square sample crops away most of what a
    style LoRA is actually judged on (full figure, room, background depth), so
    the frame the owner stares at every epoch is the wide one.

    Snapped to multiples of 16 because the VAE strides 8 and the DiT patches 2;
    that snapping is why the ratio lands near 16:9 rather than exactly on it.
    """
    import math

    w_ratio, h_ratio = SAMPLE_ASPECT
    pixels = max(1, int(resolution[0]) * int(resolution[1]))
    height = max(16, int(round(math.sqrt(pixels * h_ratio / w_ratio) / 16)) * 16)
    width = max(16, int(round(height * w_ratio / h_ratio / 16)) * 16)
    return [width, height]


def style_rush_models() -> list[str]:
    """Models that accept a control image, so they can learn style conversion."""
    return [k for k in MODEL_ORDER if MODELS[k].get("supports_control")]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def suggest_schedule(model_key: str, n_items: int) -> dict:
    """Epochs-first schedule: pick num_repeats so an epoch is a meaningful
    unit, then epochs so the TOTAL work lands in the model's sweet spot,
    whatever the dataset size. save_every targets ~10 checkpoints."""
    model = MODELS[model_key]
    n = max(1, n_items)
    if "exposures_per_image" in model:  # Anima: 600-720 exposures per image
        num_repeats = clamp(round(250 / n), 1, 20)
        epochs = clamp(round(model["exposures_per_image"] / num_repeats), 4, 400)
    else:
        target = model.get("target_steps", 2000)
        num_repeats = clamp(round(250 / n), 1, 20)
        epochs = clamp(round(target / (n * num_repeats)), 4, 40)
    save_every = max(1, round(epochs / 10))
    return {"num_repeats": num_repeats, "epochs": epochs, "save_every_n_epochs": save_every}


def vram_tier(model_key: str, vram_gb: float) -> dict:
    for tier in MODELS[model_key]["vram_tiers"]:
        if vram_gb >= tier["min_gb"]:
            return tier
    return MODELS[model_key]["vram_tiers"][-1]


def public_presets() -> dict:
    """JSON-safe snapshot for the frontend."""
    out = {}
    for key in MODEL_ORDER:
        m = MODELS[key]
        out[key] = {
            "key": key,
            "label": m["label"],
            "group": m["group"],
            "media": m["media"],
            "engine": m["engine"],
            "train": {k: v for k, v in m["train"].items()
                      if k in ("network_dim", "network_alpha", "learning_rate", "optimizer_type",
                               "lr_scheduler", "timestep_sampling")},
            "resolution": m.get("resolution"),
            "needs_control": m.get("needs_control", False),
            "supports_control": m.get("supports_control", False),
            "supports_slider_native": m.get("supports_slider_native", False),
            "ltx": m.get("ltx"),
        }
    return out
