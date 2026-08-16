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
from . import style_rush as sr
from .captioner import generate_captions
from .config import PROJECTS_DIR, gpu_info
from .engines import (engine_dir, ensure_engine, supports_sampling, venv_bin,
                      venv_python)
from .hf_upload import UploadWatcher, create_repo, hf_username, upload_run_files
from .jobs import Job, JobFailed
from .models_download import ensure_models, resolve
from .presets import (CONTROL_RESOLUTION, LORAPLUS_RATIO, MODELS, NETWORK_MODULES,
                      SAMPLE_GUIDANCE, SAMPLE_PROMPT, SAMPLE_SEED, SAMPLE_STEPS,
                      STYLE_RUSH_SCHEDULE, net_types_for, sample_resolution,
                      style_rush_models, suggest_schedule, vram_tier)


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
    """Still images get the wide sample frame; video keeps the shape it trains
    on, where frame size is tied to what the model learned."""
    if frames is None:
        resolution = sample_resolution(resolution)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sample_prompt_line(prompt_text, trigger, resolution, frames) + "\n",
                    encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Train config
# ---------------------------------------------------------------------------

def build_train_config(model_key: str, overrides: dict, schedule: dict, stats: dict,
                       vram_gb: float, dataset_toml: Path, output_dir: Path,
                       output_name: str, batch_size: int,
                       sample_prompts: Path | None = None) -> dict:
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
    if sample_prompts is not None:
        cfg["sample_prompts"] = str(sample_prompts)
        cfg["sample_every_n_epochs"] = 1
        cfg["sample_at_first"] = True
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
    drop_stale_te_cache(subset_dirs(dataset_toml), job)
    _run_cache_with_retry(job, te, cwd=edir)
    job.end_phase("Cache do text encoder")


CAPTION_STAMP = ".captions.sha"


def subset_dirs(dataset_toml: Path) -> list[tuple[Path, Path]]:
    """(image dir, cache dir) for every [[datasets]] block, read back from the
    TOML that was just written — the one place that already knows the pairing."""
    pairs, image_dir = [], None
    for line in dataset_toml.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"')
        if key in ("image_directory", "video_directory"):
            image_dir = Path(value)
        elif key == "cache_directory" and image_dir is not None:
            pairs.append((image_dir, Path(value)))
            image_dir = None
    return pairs


def drop_stale_te_cache(pairs: list[tuple[Path, Path]], job: Job) -> None:
    """Delete text-encoder caches whose captions have changed since they were built.

    musubi names a TE cache after the media file alone
    (`<basename>_<arch>_te.safetensors`), so nothing about it changes when the
    caption does — and `--skip_existing` then reuses embeddings of the old text.
    Change the trigger word on a project that already trained and the run uses
    the previous trigger end to end, while the samples the owner judges it by
    use the new one. Silent, and it costs a whole training.
    """
    for media_dir, cache_dir in pairs:
        if not cache_dir.exists():
            continue
        digest = ds.captions_digest(media_dir)
        stamp = cache_dir / CAPTION_STAMP
        try:
            previous = stamp.read_text(encoding="utf-8").strip()
        except OSError:
            previous = ""
        if previous == digest:
            continue
        if previous:
            stale = list(cache_dir.glob("*_te.safetensors"))
            for f in stale:
                f.unlink(missing_ok=True)
            if stale:
                job.log(f"As captions de {media_dir.name} mudaram — {len(stale)} caches "
                        f"de text encoder refeitos.")
        # No stamp yet means a cache built before this check existed. Throwing it
        # away would cost minutes of GPU on every upgrade to prove a staleness
        # nobody has evidence of; stamping it makes every run after this one safe.
        stamp.write_text(digest, encoding="utf-8")


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
# ComfyUI format conversion (runs in the engine venv)
# ---------------------------------------------------------------------------
# ComfyUI's model_lora_keys_unet maps `lora_unet_<flattened key>` generically for
# every architecture, which is exactly what musubi saves — so musubi LoRAs load
# as-is and conversion would be a no-op. The exception is a backend whose module
# names differ from what ComfyUI loads: sd-scripts' Anima, which ships its own
# converter. That is why only the Anima preset carries `comfy_convert`, and why
# there is no switch for this in the UI: the preset already knows. A model later
# found to need it gets `comfy_convert` added to its preset.


def comfy_convert_command(model_key: str, src: Path, dst: Path) -> list[str] | None:
    spec = MODELS[model_key].get("comfy_convert")
    if spec is None:
        return None
    engine = MODELS[model_key]["engine"]
    py = str(venv_python(engine))
    edir = engine_dir(engine)
    if "script" in spec:
        return [py, str(edir / spec["script"]), str(src), str(dst)]
    return [py, str(edir / _script(engine, "convert_lora.py")),
            "--input", str(src), "--output", str(dst), "--target", "other"]


def comfy_converter(model_key: str, job: Job):
    """A fn(ckpt) -> converted path|None for UploadWatcher, or None if unneeded."""
    probe = comfy_convert_command(model_key, Path("probe"), Path("probe_comfy"))
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
        cmd = comfy_convert_command(model_key, ckpt, dest)
        res = subprocess.run([str(c) for c in cmd],
                             cwd=str(engine_dir(MODELS[model_key]["engine"])),
                             capture_output=True, text=True, timeout=600)
        if res.returncode != 0:
            job.log(f"⚠ conversão Comfy falhou: {res.stderr[-400:]}")
            return None
        job.log(f"✔ convertido p/ ComfyUI: {dest.name}")
        return dest

    return convert


def finalise(job: Job, output_dir: Path, repo_id: str | None) -> None:
    """Report what actually landed, not what was attempted.

    The old message said "✔ Tudo em huggingface.co/<repo>" purely because a
    repo id existed, while the only thing it had looked at was the local
    directory. An upload disabled by a 403 mid-run, or a checkpoint the final
    sweep missed, still read as a complete success — and the owner destroys the
    pod on the strength of that line.
    """
    finals = sorted(output_dir.glob("*.safetensors"))
    if not finals:
        raise JobFailed("o treino terminou sem produzir checkpoints")
    job.extra["outputs"] = [f.name for f in finals]
    job.log(f"✔ {len(finals)} checkpoints em {output_dir}")
    if not repo_id:
        return
    sent = set(job.extra.get("hf_files") or [])
    absent = [f.name for f in finals if f.name not in sent]
    if absent:
        job.log(f"⚠ {len(absent)} checkpoints NÃO subiram para o HF: "
                f"{', '.join(absent[:5])}{'…' if len(absent) > 5 else ''}")
        job.log(f"⚠ o que subiu está em https://huggingface.co/{repo_id} — "
                f"o resto só existe neste pod, não destrua sem copiar.")
    else:
        job.log(f"✔ Tudo em https://huggingface.co/{repo_id}")


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
    restore_dir = pdir / "dataset_restore"
    output_dir = pdir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = ds.inspect(dataset_dir)
    if stats["items"] == 0:
        raise JobFailed("importe o dataset antes de treinar")
    if stats["videos"]:
        raise JobFailed("Style Rush é só para datasets de imagem")

    job.set_phases(["Engine", "Modelos base", "Captions", "Dataset de conversão",
                    "Dataset de restauração", "Configuração", "Cache de latents",
                    "Cache do text encoder", "Treino", "Finalização"])

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

    job.start_phase("Dataset de restauração")
    restore_stats = sr.build_restore_dataset(
        dataset_dir, restore_dir, job, used=sr.convert_sources(convert_dir))
    job.extra["restore"] = restore_stats
    job.end_phase("Dataset de restauração")

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
        image_subset(restore_dir, pdir / "cache" / "restore", schedule["num_repeats"],
                     control_dir=restore_dir / "control",
                     control_resolution=CONTROL_RESOLUTION),
    ]
    dataset_toml = write_dataset_toml(model_key, pdir / "dataset.toml", subsets,
                                      resolution, batch_size)

    sample_path = None
    if overrides.get("sampling", True) and supports_sampling(engine):
        sample_path = write_sample_prompts(
            pdir / "sample_prompts.txt",
            overrides.get("sample_prompt") or SAMPLE_PROMPT, trigger, resolution)

    total_items = stats["items"] + convert_stats["pairs"] + restore_stats["pairs"]
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
        "restore": restore_stats["pairs"],
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
        upload_run_files(repo_id, job, info={
            "project": project, "model": model_key, "model_label": model["label"],
            "mode": "style-rush", "trigger": trigger, "schedule": schedule,
            "dataset": stats, "style_rush": convert_stats, "restore": restore_stats,
            "config": {k: v for k, v in cfg.items() if isinstance(v, (str, int, float, bool))},
        }, captions=ds.captions_map(dataset_dir))
        convert = comfy_converter(model_key, job)
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
    finalise(job, output_dir, repo_id)
    job.end_phase("Finalização")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_training(job: Job, params: dict) -> None:
    project = params["project"]
    model_key = params["model"]
    overrides = params.get("overrides", {})
    mode = params.get("mode", "lora")
    # one source of truth, the same one run_style_rush_training uses: the server
    # already resolved it from the request body or the stored state. Reading it
    # back out of `overrides` instead made it a second channel that only the UI
    # ever filled — an API call without it silently sampled without the trigger.
    trigger = (params.get("trigger") or "").strip()
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

    if mode == "style-rush":
        return run_style_rush_training(job, params)

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
    sample_path = None
    if overrides.get("sampling", True) and supports_sampling(engine):
        frames = None
        if model["media"] == "video":
            frames = int(ltx_cfg["resolution"].split("x")[2]) if engine == "musubi-ltx" \
                else (model.get("video_dataset", {}).get("max_frames") or 81)
        sample_path = write_sample_prompts(
            pdir / "sample_prompts.txt",
            overrides.get("sample_prompt") or SAMPLE_PROMPT,
            trigger, resolution, frames)
        job.log(f"Samples a cada época: {sample_path}")
    elif overrides.get("sampling", True):
        job.log(f"⚠ engine {engine} não tem --sample_prompts — sampling desligado.")

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
    cfg = build_train_config(model_key, overrides, schedule, stats, vram_gb,
                             dataset_toml, output_dir, slug, batch_size,
                             sample_prompts=sample_path)
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
        upload_run_files(repo_id, job, info={
            "project": project, "model": model_key, "model_label": model["label"],
            "mode": mode, "trigger": trigger, "schedule": schedule, "dataset": stats,
            "config": {k: v for k, v in cfg.items() if isinstance(v, (str, int, float, bool))},
        }, captions=ds.captions_map(dataset_dir))
        convert = comfy_converter(model_key, job)
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
    finalise(job, output_dir, repo_id)
    job.end_phase("Finalização")
