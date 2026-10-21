"""Arrakis wrapper around DiffSynth-Studio's Qwen-Image-2.1 LoRA trainer.

Runs inside the DiffSynth engine venv, launched by `accelerate launch`. The
training itself is DiffSynth's, unchanged: the official training module and
parser are imported from examples/qwen_image_21/model_training/train.py and
the official `launch_training_task` runs the loop. What this adds is what the
Arrakis UI needs and DiffSynth does not do:

- a sample image every epoch (and one before the first step), rendered with
  the pipeline already in memory — the LoRA is live in it, so there is no
  second model load. DiffSynth has no in-training sampling at all;
- checkpoint names in the musubi convention (`<name>-000001.safetensors`) and
  sample names the gallery parses (`<name>_e000001_00_<ts>.png`);
- one global `steps:` progress bar with `avr_loss=`, and `epoch N/M` lines,
  which is what the UI's progress parser reads.

A sample that fails is a warning, never a failed training.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path


def load_official(script: str):
    spec = importlib.util.spec_from_file_location("diffsynth_official_train", script)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(script).parent))
    spec.loader.exec_module(module)  # definitions only: its main is guarded
    return module


class GlobalProgress:
    """Replaces the runner's per-epoch tqdm with one bar over the whole run."""

    def __init__(self, total: int, epochs: int):
        from tqdm import tqdm

        self.bar = tqdm(total=total, desc="steps", smoothing=0, dynamic_ncols=True)
        self.epochs = epochs
        self.epoch = 0
        self.ema = None

    def __call__(self, iterable, *args, **kwargs):
        self.epoch += 1
        self.bar.write(f"epoch {self.epoch}/{self.epochs}")
        for item in iterable:
            yield item
            self.bar.update(1)

    def record_loss(self, loss) -> None:
        try:
            value = float(loss.detach().float().mean().item())
        except Exception:  # noqa: BLE001 — a loss we cannot read is not worth a crash
            return
        self.ema = value if self.ema is None else 0.98 * self.ema + 0.02 * value
        self.bar.set_postfix(avr_loss=f"{self.ema:.4f}", refresh=False)


def make_sampler(args):
    if not args.sample_prompt or not args.sample_dir:
        return None
    if args.enable_model_cpu_offload:
        print("[arrakis] offload de modelo ligado — samples durante o treino desligados", flush=True)
        return None
    from PIL import Image

    edit = None
    if args.sample_edit_image:
        edit = [Image.open(p).convert("RGBA") for p in args.sample_edit_image.split(",")]
    sample_dir = Path(args.sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    def sample(pipe, epoch: int) -> None:
        import torch

        dit = getattr(pipe, "dit", None)
        was_training = bool(dit is not None and dit.training)
        try:
            if dit is not None:
                dit.eval()
            with torch.no_grad():
                image = pipe(args.sample_prompt, edit_image=edit,
                             height=args.sample_height, width=args.sample_width,
                             seed=args.sample_seed, num_inference_steps=args.sample_steps,
                             progress_bar_cmd=lambda x: x)
            stamp = time.strftime("%Y%m%d%H%M%S")
            path = sample_dir / f"{args.output_name}_e{epoch:06d}_00_{stamp}.png"
            image.save(path)
            print(f"[arrakis] sample da época {epoch}: {path.name}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[arrakis] ⚠ sample da época {epoch} falhou ({type(exc).__name__}: {exc}) "
                  f"— o treino continua", flush=True)
        finally:
            # inference re-timesteps the shared scheduler (40 steps); the loss
            # samples its timestep from it, so training must get the 1000 back
            pipe.scheduler.set_timesteps(1000, training=True)
            if dit is not None and was_training:
                dit.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return sample


def main(argv: list[str]) -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--diffsynth_script", required=True)
    known, rest = pre.parse_known_args(argv)
    official = load_official(known.diffsynth_script)

    parser = official.qwen_image_21_parser()
    parser.add_argument("--output_name", required=True)
    parser.add_argument("--sample_prompt", default=None)
    parser.add_argument("--sample_width", type=int, default=1024)
    parser.add_argument("--sample_height", type=int, default=1024)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--sample_steps", type=int, default=40)
    parser.add_argument("--sample_edit_image", default=None)
    parser.add_argument("--sample_dir", default=None)
    args = parser.parse_args(rest)

    import accelerate
    from diffsynth.core import UnifiedDataset
    from diffsynth.diffusion import ModelLogger
    from diffsynth.diffusion import runner

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(
            find_unused_parameters=args.find_unused_parameters)],
    )
    # the same dataset and module construction as the official main
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=32,
            width_division_factor=32,
            convert_RGB=False,
            convert_RGBA=True,
        ),
    )
    model = official.QwenImage21TrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        processor_path=args.processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        quant_options=args.quant_options,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload)
        else accelerator.device,
    )

    progress = GlobalProgress(total=len(dataset) * args.num_epochs // accelerator.num_processes,
                              epochs=args.num_epochs)
    runner.tqdm = progress  # the loop calls tqdm(dataloader) once per epoch
    sampler = make_sampler(args)

    class ArrakisLogger(ModelLogger):
        def on_step_end(self, accelerator, model, save_steps=None, **kwargs):
            super().on_step_end(accelerator, model, save_steps, **kwargs)
            if kwargs.get("loss") is not None:
                progress.record_loss(kwargs["loss"])

        def on_epoch_end(self, accelerator, model, epoch_id):
            self.save_model(accelerator, model, f"{args.output_name}-{epoch_id + 1:06d}.safetensors")
            if sampler and accelerator.is_main_process:
                sampler(accelerator.unwrap_model(model).pipe, epoch_id + 1)
            accelerator.wait_for_everyone()

    logger = ArrakisLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt)

    if sampler and accelerator.is_main_process:
        sampler(model.pipe, 0)  # the base model, before any step: the baseline
    runner.launch_training_task(accelerator, dataset, model, logger, args=args)
    progress.bar.close()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main(sys.argv[1:])
