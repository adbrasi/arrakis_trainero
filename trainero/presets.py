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
        "repo": "https://github.com/adbrasi/data_araknideo.git",
        "branch": "main",
        "dir": "data_araknideo",
        "install": "requirements",
        "script_prefix": "",
    },
}

# ---------------------------------------------------------------------------
# Caption presets (data_araknideo prompt profiles)
# ---------------------------------------------------------------------------

CAPTION_PROFILES = {
    "image": [
        {"key": "anima-character", "label": "Personagem (Anima)", "var": "trigger_character"},
        {"key": "anima-style", "label": "Estilo (Anima)", "var": "trigger_style"},
        {"key": "anima-concept", "label": "Conceito (Anima)", "var": None},
        {"key": "anima-outfit", "label": "Outfit (Anima)", "var": "trigger_outfit"},
        {"key": "default", "label": "NSFW prosa (Flux/Qwen/Krea)", "var": None},
        {"key": "generic-character", "label": "Personagem genérico", "var": "character_name"},
        {"key": "generic-style", "label": "Estilo genérico", "var": "style_name"},
    ],
    "video": [
        {"key": "nsfw_caption_video", "label": "Animação em loop NSFW", "var": None},
        {"key": "grok_looping_animation_template", "label": "Loop LTX 2.3 (template novo)", "var": None},
        {"key": "live-wallpaper", "label": "Live wallpaper", "var": None},
        {"key": "live_wallpaper_game_cg", "label": "Live wallpaper game CG", "var": None},
    ],
}

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
        "resolution": [1024, 1024],
        "target_steps": 2000,
        "vram_tiers": [
            {"min_gb": 70, "train": {}},
            {"min_gb": 44, "train": {"fp8_base": True, "fp8_scaled": True}},
            {"min_gb": 30, "train": {"fp8_base": True, "fp8_scaled": True, "blocks_to_swap": 8}},
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
            {"min_gb": 22, "train": {"fp8_base": True, "fp8_scaled": True, "blocks_to_swap": 16}, "cache_te": {"fp8_vl": True}},
            {"min_gb": 0, "train": {"fp8_base": True, "fp8_scaled": True, "blocks_to_swap": 45}, "cache_te": {"fp8_vl": True}},
        ],
    },
    "qwen-image-edit": {
        "label": "Qwen Image Edit",
        "base_model": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
        "group": "imagem",
        "media": "image",
        "engine": "musubi",
        "arch": "qwen_image",
        "downloads": [
            ("Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors", "qwen_image_edit_2511_bf16.safetensors"),
            ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors", "qwen_image_vae.safetensors"),
            ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b.safetensors", "qwen_2.5_vl_7b.safetensors"),
        ],
        "model_args": {"dit": "qwen_image_edit_2511_bf16.safetensors", "vae": "qwen_image_vae.safetensors", "text_encoder": "qwen_2.5_vl_7b.safetensors"},
        "extra_args": {"model_version": "edit-2511"},
        "cache_latents_args": {"vae": "qwen_image_vae.safetensors", "model_version": "edit-2511"},
        "cache_te_args": {"text_encoder": "qwen_2.5_vl_7b.safetensors", "batch_size": 8, "model_version": "edit-2511"},
        "network_module": "networks.lora_qwen_image",
        "train": {
            "network_dim": 64, "network_alpha": 64,
            "learning_rate": 5e-5, "optimizer_type": "adamw8bit", "lr_scheduler": "cosine",
            "timestep_sampling": "shift", "weighting_scheme": "none", "discrete_flow_shift": 2.2,
        },
        "resolution": [1024, 1024],
        "needs_control": True,  # dataset must ship a control/ folder
        "supports_control": True,
        "control_resolution": [1024, 1024],
        "target_steps": 2000,
        "vram_tiers": [
            {"min_gb": 70, "train": {}},
            {"min_gb": 38, "train": {"fp8_base": True, "fp8_scaled": True}},
            {"min_gb": 22, "train": {"fp8_base": True, "fp8_scaled": True, "blocks_to_swap": 20}, "cache_te": {"fp8_vl": True}},
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
            # same as Wan: sampling reads args.vae / args.text_encoder
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
        "resolution": [1024, 1024],
        "target_steps": 2000,
        "net_types": ["lora"],  # DiT base is FP8; LoHa/LoKr unverified here
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
            {"min_gb": 44, "train": {"fp8_base": True, "fp8_scaled": True, "blocks_to_swap": 10}, "cache_te": {"gemma_load_in_8bit": True}},
            {"min_gb": 30, "train": {"fp8_base": True, "fp8_scaled": True, "blocks_to_swap": 16}, "cache_te": {"gemma_load_in_8bit": True}},
            {"min_gb": 0, "train": {"fp8_base": True, "fp8_scaled": True, "blocks_to_swap": 24}, "cache_te": {"gemma_load_in_8bit": True}},
        ],
    },
    "wan-22": {
        "label": "Wan 2.2",
        "base_model": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "group": "video",
        "media": "video",
        "engine": "musubi",
        "arch": "wan",
        "downloads": [
            ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors", "wan2.2_t2v_low_noise_14B_fp16.safetensors"),
            ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors", "wan2.2_t2v_high_noise_14B_fp16.safetensors"),
            ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/vae/wan_2.1_vae.safetensors", "wan_2.1_vae.safetensors"),
            ("Wan-AI/Wan2.1-I2V-14B-720P", "models_t5_umt5-xxl-enc-bf16.pth", "models_t5_umt5-xxl-enc-bf16.pth"),
        ],
        "model_args": {
            "dit": "wan2.2_t2v_low_noise_14B_fp16.safetensors",
            "dit_high_noise": "wan2.2_t2v_high_noise_14B_fp16.safetensors",
            # sampling loads the VAE and T5 from these same args (trainer_base
            # _prepare_sampling); without them --sample_prompts crashes the run
            "vae": "wan_2.1_vae.safetensors",
            "t5": "models_t5_umt5-xxl-enc-bf16.pth",
        },
        "extra_args": {"task": "t2v-A14B"},
        "cache_latents_args": {"vae": "wan_2.1_vae.safetensors"},
        "cache_te_args": {"t5": "models_t5_umt5-xxl-enc-bf16.pth", "batch_size": 8},
        "cache_te_key": "t5",
        "network_module": "networks.lora_wan",
        "train": {
            "network_dim": 32, "network_alpha": 32,
            "learning_rate": 2e-4, "optimizer_type": "adamw8bit", "lr_scheduler": "cosine",
            "timestep_sampling": "shift", "discrete_flow_shift": 3.0,
        },
        "video_dataset": {"target_frames": [1, 33, 65], "frame_extraction": "full", "max_frames": 81, "source_fps": None},
        "resolution": [832, 480],
        "target_steps": 3000,
        "vram_tiers": [
            {"min_gb": 70, "train": {"fp8_base": True, "fp8_scaled": True}},
            {"min_gb": 44, "train": {"fp8_base": True, "fp8_scaled": True, "offload_inactive_dit": True}, "cache_te": {"fp8_t5": True}},
            {"min_gb": 30, "train": {"fp8_base": True, "fp8_scaled": True, "blocks_to_swap": 20}, "cache_te": {"fp8_t5": True}},
            {"min_gb": 0, "train": {"fp8_base": True, "fp8_scaled": True, "blocks_to_swap": 30}, "cache_te": {"fp8_t5": True}},
        ],
    },
}

MODEL_ORDER = ["krea2", "anima", "flux-klein", "qwen-image", "qwen-image-edit", "ideogram", "ltx-23", "wan-22"]

# Network-type availability. LoRA+ rides on standard LoRA only.
# DoRA is not implemented by any of the chosen backends (musubi has LoRA/LoHa/
# LoKr; sd-scripts' networks.lora_anima has no dora_wd) — documented in README.
NET_TYPES_MUSUBI = ["lora", "loha", "lokr"]
NET_TYPES_SDSCRIPTS = ["lora"]

NETWORK_MODULES = {
    "musubi": {"loha": "networks.loha", "lokr": "networks.lokr"},
    "musubi-ltx": {"loha": "networks.loha", "lokr": "networks.lokr"},
}

LORAPLUS_RATIO = 16  # owner's value in image_lora_arrakis


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


def net_types_for(model_key: str) -> list[str]:
    model = MODELS[model_key]
    if "net_types" in model:
        return model["net_types"]
    if model["engine"] == "sd-scripts":
        return NET_TYPES_SDSCRIPTS
    return NET_TYPES_MUSUBI


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
            "net_types": net_types_for(key),
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
