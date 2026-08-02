"""Training pipeline: dataset.toml + train config + phased execution.

One entrypoint (`run_training`) drives everything the TREINAR button needs:
engine install, base-model download, config generation, latent/TE caching,
epoch-based training with live progress, and continuous HF upload.

Engines differ in invocation only:
  musubi      accelerate launch src/musubi_tuner/<arch>_train_network.py --config_file
  musubi-ltx  CLI args at repo root (the fork predates --config_file everywhere)
  sd-scripts  accelerate launch anima_train_network.py --config_file (cache built in)
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

from . import dataset as ds
from .config import PROJECTS_DIR, gpu_info
from .engines import engine_dir, ensure_engine, venv_bin, venv_python
from .hf_upload import UploadWatcher, create_repo, hf_username, model_card, upload_text
from .jobs import Job, JobFailed
from .models_download import ensure_models, resolve
from .presets import (LORAPLUS_RATIO, MODELS, NETWORK_MODULES, net_types_for,
                      suggest_schedule, vram_tier)


def slugify(name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", name.strip()).strip("_").lower()
    return slug or "projeto"


def project_dir(name: str) -> Path:
    return PROJECTS_DIR / slugify(name)


# ---------------------------------------------------------------------------
# TOML helpers (values only ever str/int/float/bool/list — no escaping needed
# beyond json.dumps for strings)
# ---------------------------------------------------------------------------

def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return json.dumps(str(v), ensure_ascii=False)


def _toml_lines(d: dict) -> list[str]:
    return [f"{k} = {_toml_value(v)}" for k, v in d.items() if v is not None]


# ---------------------------------------------------------------------------
# dataset.toml
# ---------------------------------------------------------------------------

def write_dataset_toml(model_key: str, pdir: Path, dataset_dir: Path, cache_dir: Path,
                       schedule: dict, resolution: list[int], batch_size: int,
                       stats: dict, ltx_cfg: dict | None = None) -> Path:
    model = MODELS[model_key]
    engine = model["engine"]
    path = pdir / ("dataset.toml" if dataset_dir.name == "dataset" else f"dataset_{dataset_dir.name}.toml")
    cache_dir.mkdir(parents=True, exist_ok=True)

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
            "  [[datasets.subsets]]",
            f"  image_dir = {_toml_value(str(dataset_dir))}",
            f"  num_repeats = {schedule['num_repeats']}",
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
    has_videos = stats.get("videos", 0) > 0
    has_images = stats.get("images", 0) > 0

    if has_images:
        block = {
            "image_directory": str(dataset_dir),
            "cache_directory": str(cache_dir / "images"),
            "num_repeats": schedule["num_repeats"],
        }
        if model.get("needs_control"):
            block["control_directory"] = str(dataset_dir / "control")
            block["control_resolution"] = model.get("control_resolution", [1024, 1024])
        lines += ["[[datasets]]"] + _toml_lines(block) + [""]

    if has_videos:
        block = {
            "video_directory": str(dataset_dir),
            "cache_directory": str(cache_dir / "videos"),
            "num_repeats": schedule["num_repeats"],
        }
        if engine == "musubi-ltx":
            frames = int((ltx_cfg or {}).get("resolution", "768x512x81").split("x")[2])
            block["target_frames"] = [frames]
            block["target_fps"] = float((ltx_cfg or {}).get("fps", 25.0))
            block["frame_extraction"] = "full"
            block["max_frames"] = frames
        else:  # wan
            vd = MODELS[model_key].get("video_dataset", {})
            block["target_frames"] = vd.get("target_frames", [1, 33, 65])
            block["frame_extraction"] = vd.get("frame_extraction", "full")
            block["max_frames"] = vd.get("max_frames", 81)
        lines += ["[[datasets]]"] + _toml_lines(block) + [""]

    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Train config
# ---------------------------------------------------------------------------

def build_train_config(model_key: str, overrides: dict, schedule: dict, stats: dict,
                       vram_gb: float, dataset_toml: Path, output_dir: Path,
                       output_name: str, batch_size: int) -> dict:
    """Resolved flat dict of training args (musubi/sd-scripts arg names)."""
    model = MODELS[model_key]
    tier = vram_tier(model_key, vram_gb)
    cfg: dict = {}

    for rel_key, rel in model["model_args"].items():
        cfg[rel_key] = str(resolve(model_key, rel))
    cfg.update(model.get("extra_args", {}))
    cfg.update(model["train"])
    cfg.update(tier.get("train", {}))

    net_type = overrides.get("net_type", "lora")
    if net_type not in net_types_for(model_key):
        raise JobFailed(f"{model['label']} não suporta rede {net_type}")
    if net_type == "lora":
        cfg["network_module"] = model["network_module"]
    else:
        cfg["network_module"] = NETWORK_MODULES[model["engine"]][net_type]
        if "network_alpha" not in overrides:
            cfg["network_alpha"] = max(1, int(cfg["network_dim"]) // 2)

    for key in ("network_dim", "network_alpha", "learning_rate"):
        if key in overrides and overrides[key] not in (None, ""):
            # learning_rate chega como texto do painel; se ficar str, o TOML sai
            # com aspas e o argparse do trainer nunca converte (crash no optimizer)
            try:
                cfg[key] = float(overrides[key]) if key == "learning_rate" else int(overrides[key])
            except (TypeError, ValueError):
                raise JobFailed(f"valor inválido para {key}: {overrides[key]!r}")

    network_args = []
    if overrides.get("loraplus") and net_type == "lora":
        network_args.append(f"loraplus_lr_ratio={LORAPLUS_RATIO}")
    if network_args:
        cfg["network_args"] = network_args

    n_items = max(1, stats.get("items", 1))
    est_steps = math.ceil(n_items * schedule["num_repeats"] / batch_size) * schedule["epochs"]
    cfg.update({
        "dataset_config": str(dataset_toml),
        "mixed_precision": "bf16",
        "gradient_checkpointing": True,
        "max_train_epochs": schedule["epochs"],
        "save_every_n_epochs": schedule["save_every_n_epochs"],
        "seed": 42,
        "output_dir": str(output_dir),
        "output_name": output_name,
        "max_data_loader_n_workers": 2,
        "persistent_data_loader_workers": True,
    })
    if model["engine"] == "sd-scripts":
        cfg["attn_mode"] = "torch"
        cfg["lr_warmup_steps"] = 0.1  # ratio
    else:
        cfg["sdpa"] = True
        cfg["lr_warmup_steps"] = max(0, round(0.05 * est_steps))
    return cfg


def write_train_toml(cfg: dict, path: Path) -> Path:
    path.write_text("\n".join(_toml_lines(cfg)) + "\n")
    return path


def _cli_args(cfg: dict) -> list[str]:
    out: list[str] = []
    for k, v in cfg.items():
        if v is None:
            continue
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                out.append(flag)
        elif isinstance(v, list):
            out.append(flag)
            out += [str(x) for x in v]
        else:
            out += [flag, str(v)]
    return out


# ---------------------------------------------------------------------------
# Cache phases (musubi family)
# ---------------------------------------------------------------------------

def _script(engine: str, name: str) -> str:
    from .presets import ENGINES

    return ENGINES[engine]["script_prefix"] + name


def run_caches(model_key: str, dataset_toml: Path, vram_gb: float, job: Job) -> None:
    model = MODELS[model_key]
    engine = model["engine"]
    if engine == "sd-scripts":
        return  # cache is built into the training run
    tier = vram_tier(model_key, vram_gb)
    edir = engine_dir(engine)
    py = str(venv_python(engine))
    arch = model["arch"]

    if engine == "musubi-ltx":
        ckpt = str(resolve(model_key, model["model_args"]["ltx2_checkpoint"]))
        gemma = str(resolve(model_key, model["model_args"]["gemma_root"]))
        mode = model["ltx"]["mode"]
        lat = [py, _script(engine, "ltx2_cache_latents.py"),
               "--dataset_config", str(dataset_toml), "--ltx2_checkpoint", ckpt,
               "--vae_dtype", "bf16", "--ltx2_mode", mode,
               "--batch_size", "4", "--num_workers", "4"]
        if vram_gb < 80:
            lat += ["--vae_spatial_tile_size", "512", "--vae_spatial_tile_overlap", "64"]
        job.start_phase("Cache de latents")
        _run_cache_with_retry(job, lat, cwd=edir)
        job.end_phase("Cache de latents")

        te = [py, _script(engine, "ltx2_cache_text_encoder_outputs.py"),
              "--dataset_config", str(dataset_toml), "--ltx2_checkpoint", ckpt,
              "--gemma_root", gemma, "--ltx2_mode", mode,
              "--batch_size", "8", "--num_workers", "4"]
        for flag, on in tier.get("cache_te", {}).items():
            if on:
                te.append(f"--{flag}")
        job.start_phase("Cache do text encoder")
        _run_cache_with_retry(job, te, cwd=edir)
        job.end_phase("Cache do text encoder")
        return

    # upstream musubi
    def _append(cmd: list[str], key: str, value) -> None:
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        elif key in ("vae", "text_encoder", "t5"):
            cmd += [f"--{key}", str(resolve(model_key, value))]
        else:
            cmd += [f"--{key}", str(value)]

    lat = [py, _script(engine, f"{arch}_cache_latents.py"), "--dataset_config", str(dataset_toml)]
    for k, v in model.get("cache_latents_args", {}).items():
        _append(lat, k, v)
    lat.append("--skip_existing")
    job.start_phase("Cache de latents")
    _run_cache_with_retry(job, lat, cwd=edir)
    job.end_phase("Cache de latents")

    te = [py, _script(engine, f"{arch}_cache_text_encoder_outputs.py"), "--dataset_config", str(dataset_toml)]
    for k, v in model.get("cache_te_args", {}).items():
        _append(te, k, v)
    for flag, on in tier.get("cache_te", {}).items():
        if on:
            te.append(f"--{flag}")
    te.append("--skip_existing")
    job.start_phase("Cache do text encoder")
    _run_cache_with_retry(job, te, cwd=edir)
    job.end_phase("Cache do text encoder")


def _run_cache_with_retry(job: Job, cmd: list[str], cwd: Path) -> None:
    """SIGKILL during caching is almost always RAM OOM — retry with half the
    batch/workers once (pattern from the owner's ltx23 trainer)."""
    try:
        job.run(cmd, cwd=cwd)
    except JobFailed as exc:
        if "-9" not in str(exc) and "137" not in str(exc):
            raise
        job.log("⚠ Cache morto (provável OOM de RAM). Retry conservador...")
        retry = list(cmd)
        for flag in ("--batch_size", "--num_workers"):
            if flag in retry:
                i = retry.index(flag) + 1
                retry[i] = str(max(1, int(retry[i]) // 2))
        job.run(retry, cwd=cwd)


# ---------------------------------------------------------------------------
# Train phase
# ---------------------------------------------------------------------------

def launch_training(model_key: str, cfg: dict, pdir: Path, job: Job, toml_name: str = "train.toml") -> None:
    model = MODELS[model_key]
    engine = model["engine"]
    edir = engine_dir(engine)
    accel = str(venv_bin(engine, "accelerate"))

    if engine == "musubi-ltx":
        script = _script(engine, "ltx2_train_network.py")
        ltx = model["ltx"]
        cfg = dict(cfg)
        cfg.update({
            "ltx_version": ltx["version"], "ltx_version_check_mode": "error",
            "ltx2_mode": ltx["mode"],
            "ltx2_first_frame_conditioning_p": ltx["first_frame_p"],
            "lora_target_preset": ltx["lora_target_preset"],
        })
        cmd = [accel, "launch", "--num_cpu_threads_per_process", "1",
               "--mixed_precision", "bf16", script] + _cli_args(cfg)
    else:
        toml_path = write_train_toml(cfg, pdir / toml_name)
        script = _script(engine, f"{model['arch']}_train_network.py") \
            if engine == "musubi" else model["train_script"]
        cmd = [accel, "launch", "--num_cpu_threads_per_process", "1",
               "--mixed_precision", "bf16", script, "--config_file", str(toml_path)]

    job.run(cmd, cwd=edir, parse_progress=True)


# ---------------------------------------------------------------------------
# Anima -> ComfyUI conversion (runs in the sd-scripts venv)
# ---------------------------------------------------------------------------

def anima_comfy_converter(job: Job):
    edir = engine_dir("sd-scripts")
    py = str(venv_python("sd-scripts"))
    script = edir / "networks" / "convert_anima_lora_to_comfy.py"
    if not script.exists():
        job.log("⚠ convert_anima_lora_to_comfy.py não existe neste sd-scripts — enviando só o formato nativo.")
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
        res = subprocess.run([py, str(script), str(ckpt), str(dest)], cwd=str(edir),
                             capture_output=True, text=True, timeout=600)
        if res.returncode != 0:
            job.log(f"⚠ conversão Comfy falhou: {res.stderr[-400:]}")
            return None
        job.log(f"✔ convertido p/ ComfyUI: {dest.name}")
        return dest

    return convert


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_training(job: Job, params: dict) -> None:
    project = params["project"]
    model_key = params["model"]
    overrides = params.get("overrides", {})
    mode = params.get("mode", "lora")
    model = MODELS[model_key]
    engine = model["engine"]
    pdir = project_dir(project)
    slug = slugify(project)
    dataset_dir = pdir / "dataset"
    output_dir = pdir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "slider":
        from .sliders import run_slider_training

        return run_slider_training(job, params)

    stats = ds.inspect(dataset_dir)
    if stats["items"] == 0:
        raise JobFailed("importe o dataset antes de treinar")
    if stats["missing_captions"] > 0:
        raise JobFailed(
            f"{stats['missing_captions']} itens sem caption — gere as captions primeiro "
            f"(ex.: {', '.join(stats['missing_sample'][:3])})")
    if model.get("needs_control") and stats["control_images"] == 0:
        raise JobFailed("Qwen Image Edit precisa de imagens de controle em dataset/control/")
    if model["media"] == "video" and stats["videos"] == 0 and stats["images"] == 0:
        raise JobFailed("dataset vazio")

    phases = ["Engine", "Modelos base", "Configuração"]
    if engine != "sd-scripts":
        phases += ["Cache de latents", "Cache do text encoder"]
    phases += ["Treino", "Finalização"]
    job.set_phases(phases)

    gpu = gpu_info()
    vram_gb = gpu.get("vram_mb", 24576) / 1024
    job.extra["gpu"] = gpu
    job.log(f"GPU: {gpu.get('name', '?')} ({vram_gb:.0f} GB)")

    job.start_phase("Engine")
    ensure_engine(engine, job)
    job.end_phase("Engine")

    job.start_phase("Modelos base")
    ensure_models(model_key, job)
    job.end_phase("Modelos base")

    job.start_phase("Configuração")
    schedule = suggest_schedule(model_key, stats["items"])
    for key in ("epochs", "num_repeats", "save_every_n_epochs"):
        if overrides.get(key):
            schedule[key] = int(overrides[key])
    tier = vram_tier(model_key, vram_gb)
    batch_size = tier.get("batch_size", 1)
    ltx_cfg = dict(model.get("ltx") or {})
    if overrides.get("ltx_resolution"):
        ltx_cfg["resolution"] = overrides["ltx_resolution"]
    resolution = overrides.get("resolution") or model.get("resolution", [1024, 1024])
    if engine == "musubi-ltx":
        try:
            w, h, _f = (int(x) for x in ltx_cfg["resolution"].lower().split("x"))
            resolution = [w, h]
        except (ValueError, KeyError):
            raise JobFailed(f"resolução LTX inválida: {ltx_cfg.get('resolution')!r} (use LxAxF, ex. 768x512x81)")
    dataset_toml = write_dataset_toml(model_key, pdir, dataset_dir, pdir / "cache",
                                      schedule, resolution, batch_size, stats, ltx_cfg)
    cfg = build_train_config(model_key, overrides, schedule, stats, vram_gb,
                             dataset_toml, output_dir, slug, batch_size)
    job.extra["schedule"] = schedule
    job.extra["config_summary"] = {
        "network_module": cfg.get("network_module"),
        "network_dim": cfg.get("network_dim"), "network_alpha": cfg.get("network_alpha"),
        "learning_rate": cfg.get("learning_rate"),
        "epochs": schedule["epochs"], "batch_size": batch_size,
        "fp8": bool(cfg.get("fp8_base")), "blocks_to_swap": cfg.get("blocks_to_swap", 0),
    }
    job.log(f"Config: {json.dumps(job.extra['config_summary'], ensure_ascii=False)}")
    job.end_phase("Configuração")

    run_caches(model_key, dataset_toml, vram_gb, job)

    # HuggingFace repo + watcher (default ON)
    watcher = None
    repo_id = None
    if overrides.get("hf_upload", True):
        user = hf_username()
        if user:
            repo_id = create_repo(f"{user}/{slug}", private=overrides.get("hf_private", True), job=job)
        else:
            job.log("⚠ Sem token HF (defina HF_TOKEN) — upload desativado.")
    if repo_id:
        job.extra["hf_repo"] = repo_id
        upload_text(repo_id, "README.md",
                    model_card(project, model["label"], stats, schedule, cfg), job)
        upload_text(repo_id, "trainero_config.json",
                    json.dumps({"model": model_key, "schedule": schedule,
                                "config": {k: v for k, v in cfg.items() if isinstance(v, (str, int, float, bool))}},
                               indent=2, ensure_ascii=False), job)
        convert = anima_comfy_converter(job) if model.get("comfy_convert") else None
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
