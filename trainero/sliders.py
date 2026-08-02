"""Concept sliders.

Two flavors, chosen by the selected model:

  LTX 2.3  → native `ltx2_train_slider.py` from the AkaneTendo fork, text-only
             mode: positive/negative PROMPT pairs, no dataset at all.

  Others   → dataset-pair slider: train a standard LoRA on the "high" dataset
             and another on the "low" dataset with identical configs, then
             build the slider by rank concatenation with a negated B factor
             (ΔW = ΔW_pos − ΔW_neg, exact for linear LoRA). Apply with
             +strength to push toward the concept, −strength to push away.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import dataset as ds
from .config import REPO_DIR, gpu_info
from .engines import engine_dir, ensure_engine, venv_bin, venv_python
from .hf_upload import create_repo, hf_username, upload_text
from .jobs import Job, JobFailed
from .models_download import ensure_models, resolve
from .presets import MODELS, suggest_schedule, vram_tier
from .training import (_cli_args, _script, build_train_config, launch_training,
                       project_dir, run_caches, slugify, write_dataset_toml)

MAKE_SLIDER = REPO_DIR / "trainero" / "tools" / "make_slider.py"


def run_slider_training(job: Job, params: dict) -> None:
    model_key = params["model"]
    if MODELS[model_key].get("supports_slider_native"):
        _ltx_native_slider(job, params)
    else:
        _dataset_pair_slider(job, params)


# ---------------------------------------------------------------------------
# LTX 2.3 — native, prompt pairs
# ---------------------------------------------------------------------------

def _ltx_native_slider(job: Job, params: dict) -> None:
    project = params["project"]
    model_key = params["model"]
    overrides = params.get("overrides", {})
    targets = [t for t in params.get("slider_targets", [])
               if t.get("positive", "").strip() and t.get("negative", "").strip()]
    if not targets:
        raise JobFailed("adicione pelo menos um par de prompts (positivo/negativo)")

    model = MODELS[model_key]
    pdir = project_dir(project)
    pdir.mkdir(parents=True, exist_ok=True)
    slug = slugify(project)
    output_dir = pdir / "output"
    output_dir.mkdir(exist_ok=True)

    job.set_phases(["Engine", "Modelos base", "Configuração", "Treino do slider", "Finalização"])
    gpu = gpu_info()
    vram_gb = gpu.get("vram_mb", 24576) / 1024
    job.extra["gpu"] = gpu

    job.start_phase("Engine")
    ensure_engine("musubi-ltx", job)
    job.end_phase("Engine")

    job.start_phase("Modelos base")
    ensure_models(model_key, job)
    job.end_phase("Modelos base")

    job.start_phase("Configuração")
    slider_toml = pdir / "slider.toml"
    lines = ['mode = "text"', "guidance_strength = 1.0", "frame_rate = 25", ""]
    for t in targets:
        lines += ["[[targets]]",
                  f"positive = {json.dumps(t['positive'].strip(), ensure_ascii=False)}",
                  f"negative = {json.dumps(t['negative'].strip(), ensure_ascii=False)}", ""]
    slider_toml.write_text("\n".join(lines))
    tier = vram_tier(model_key, vram_gb)
    steps = int(overrides.get("slider_steps") or 600)
    cfg = {
        "slider_config": str(slider_toml),
        "ltx2_checkpoint": str(resolve(model_key, model["model_args"]["ltx2_checkpoint"])),
        "gemma_root": str(resolve(model_key, model["model_args"]["gemma_root"])),
        "ltx_version": model["ltx"]["version"], "ltx_version_check_mode": "error",
        "ltx2_mode": model["ltx"]["mode"],
        "network_module": "networks.lora_ltx2",
        "network_dim": int(overrides.get("network_dim") or 16),
        "network_alpha": int(overrides.get("network_alpha") or 16),
        "learning_rate": float(overrides.get("learning_rate") or 2e-4),
        "optimizer_type": "adamw8bit", "lr_scheduler": "constant",
        "max_train_steps": steps, "save_every_n_steps": max(100, steps // 4),
        "latent_width": 768, "latent_height": 512, "latent_frames": 1,
        "mixed_precision": "bf16", "sdpa": True, "gradient_checkpointing": True,
        "seed": 42, "output_dir": str(output_dir), "output_name": f"{slug}_slider",
        "max_data_loader_n_workers": 2,
    }
    cfg.update(tier.get("train", {}))
    job.extra["config_summary"] = {"slider": "ltx-native", "targets": len(targets), "steps": steps}
    job.end_phase("Configuração")

    job.start_phase("Treino do slider")
    edir = engine_dir("musubi-ltx")
    accel = str(venv_bin("musubi-ltx", "accelerate"))
    cmd = [accel, "launch", "--num_cpu_threads_per_process", "1", "--mixed_precision", "bf16",
           _script("musubi-ltx", "ltx2_train_slider.py")] + _cli_args(cfg)
    job.run(cmd, cwd=edir, parse_progress=True)
    job.end_phase("Treino do slider")

    job.start_phase("Finalização")
    _finish_upload(job, params, output_dir, slug)
    job.end_phase("Finalização")


# ---------------------------------------------------------------------------
# Dataset-pair slider (image models + wan)
# ---------------------------------------------------------------------------

def _dataset_pair_slider(job: Job, params: dict) -> None:
    project = params["project"]
    model_key = params["model"]
    overrides = dict(params.get("overrides", {}))
    if overrides.get("net_type", "lora") != "lora":
        raise JobFailed("concept slider por dataset só funciona com LoRA padrão")
    model = MODELS[model_key]
    engine = model["engine"]
    pdir = project_dir(project)
    slug = slugify(project)
    pos_dir, neg_dir = pdir / "dataset", pdir / "dataset_neg"

    pos = ds.inspect(pos_dir)
    neg = ds.inspect(neg_dir)
    if pos["items"] == 0 or neg["items"] == 0:
        raise JobFailed("importe os dois datasets do slider (conceito ALTO e conceito BAIXO)")
    for name, st in (("ALTO", pos), ("BAIXO", neg)):
        if st["missing_captions"] > 0:
            raise JobFailed(f"dataset {name}: {st['missing_captions']} itens sem caption")

    phases = ["Engine", "Modelos base", "Configuração"]
    if engine != "sd-scripts":
        phases += ["Cache de latents", "Cache do text encoder"]
    phases += ["Treino conceito ALTO", "Treino conceito BAIXO", "Montar slider", "Finalização"]
    job.set_phases(phases)

    gpu = gpu_info()
    vram_gb = gpu.get("vram_mb", 24576) / 1024
    job.extra["gpu"] = gpu

    job.start_phase("Engine")
    ensure_engine(engine, job)
    job.end_phase("Engine")
    job.start_phase("Modelos base")
    ensure_models(model_key, job)
    job.end_phase("Modelos base")

    job.start_phase("Configuração")
    schedule = suggest_schedule(model_key, max(pos["items"], neg["items"]))
    for key in ("epochs", "num_repeats", "save_every_n_epochs"):
        if overrides.get(key):
            schedule[key] = int(overrides[key])
    schedule["save_every_n_epochs"] = schedule["epochs"]  # only the final matters
    tier = vram_tier(model_key, vram_gb)
    batch_size = tier.get("batch_size", 1)
    resolution = overrides.get("resolution") or model.get("resolution", [1024, 1024])

    runs = []
    for side, sdir, stats in (("pos", pos_dir, pos), ("neg", neg_dir, neg)):
        out = pdir / f"output_{side}"
        out.mkdir(parents=True, exist_ok=True)
        toml = write_dataset_toml(model_key, pdir, sdir, pdir / f"cache_{side}",
                                  schedule, resolution, batch_size, stats, model.get("ltx"))
        cfg = build_train_config(model_key, overrides, schedule, stats, vram_gb,
                                 toml, out, f"{slug}_{side}", batch_size)
        runs.append((side, toml, cfg, out))
    job.extra["schedule"] = schedule
    job.end_phase("Configuração")

    for side, toml, cfg, out in runs:
        if engine != "sd-scripts":
            run_caches(model_key, toml, vram_gb, job)
        job.start_phase("Treino conceito " + ("ALTO" if side == "pos" else "BAIXO"))
        launch_training(model_key, cfg, pdir, job, toml_name=f"train_{side}.toml")
        job.end_phase("Treino conceito " + ("ALTO" if side == "pos" else "BAIXO"))

    job.start_phase("Montar slider")
    pos_final = _final_ckpt(pdir / "output_pos", f"{slug}_pos")
    neg_final = _final_ckpt(pdir / "output_neg", f"{slug}_neg")
    output_dir = pdir / "output"
    output_dir.mkdir(exist_ok=True)
    slider_path = output_dir / f"{slug}_slider.safetensors"
    py = str(venv_python(engine))
    job.run([py, str(MAKE_SLIDER), str(pos_final), str(neg_final), str(slider_path)])
    if not slider_path.exists():
        raise JobFailed("merge do slider não produziu arquivo")
    job.log(f"✔ Slider: {slider_path.name} (use força positiva p/ mais conceito, negativa p/ menos)")
    job.end_phase("Montar slider")

    job.start_phase("Finalização")
    _finish_upload(job, params, output_dir, slug)
    job.end_phase("Finalização")


def _final_ckpt(output_dir: Path, name: str) -> Path:
    exact = output_dir / f"{name}.safetensors"
    if exact.exists():
        return exact
    ckpts = sorted(output_dir.glob(f"{name}*.safetensors"))
    if not ckpts:
        raise JobFailed(f"nenhum checkpoint em {output_dir}")
    return ckpts[-1]


def _finish_upload(job: Job, params: dict, output_dir: Path, slug: str) -> None:
    overrides = params.get("overrides", {})
    outputs = sorted(output_dir.glob("*.safetensors"))
    job.extra["outputs"] = [f.name for f in outputs]
    if not overrides.get("hf_upload", True):
        return
    user = hf_username()
    if not user:
        job.log("⚠ Sem token HF — upload desativado.")
        return
    repo_id = create_repo(f"{user}/{slug}", private=overrides.get("hf_private", True), job=job)
    if not repo_id:
        return
    job.extra["hf_repo"] = repo_id
    from huggingface_hub import HfApi

    from .config import hf_token

    api = HfApi(token=hf_token())
    for f in outputs:
        job.log(f"↑ HF: {f.name}")
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                        repo_id=repo_id, repo_type="model")
    upload_text(repo_id, "README.md",
                f"# {params['project']}\n\nConcept slider treinado com Arrakis Trainero.\n"
                "Aplique com multiplicador positivo para MAIS do conceito, negativo para MENOS.\n",
                job)
    job.log(f"✔ https://huggingface.co/{repo_id}")
