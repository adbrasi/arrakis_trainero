"""Stand-in for DiffSynth's examples/qwen_image_21/model_training/train.py.

The parser is the REAL one (imported from the official script named by
DIFFSYNTH_REAL_TRAIN), so every flag Arrakis generates is checked by DiffSynth
itself. Only the 33 GB model is replaced: a 4x4 Linear with a fake pipeline,
so the real runner, logger and dataset run on CPU.
"""

import importlib.util
import json
import os

import torch
from diffsynth.diffusion import DiffusionTrainingModule
from PIL import Image

_spec = importlib.util.spec_from_file_location("real_train", os.environ["DIFFSYNTH_REAL_TRAIN"])
_real = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_real)
qwen_image_21_parser = _real.qwen_image_21_parser

RECORD = os.environ["FAKE_RECORD"]


def _record(**entry):
    with open(RECORD, "a") as f:
        f.write(json.dumps(entry) + "\n")


class FakeScheduler:
    def set_timesteps(self, n, training=False, **kwargs):
        _record(event="set_timesteps", n=n, training=training)


class FakePipe:
    def __init__(self, dit):
        self.dit = dit
        self.scheduler = FakeScheduler()

    def __call__(self, prompt, negative_prompt=" ", cfg_scale=1.0, edit_image=None,
                 height=1024, width=1024, seed=None, num_inference_steps=40,
                 progress_bar_cmd=None, **kwargs):
        _record(event="sample", prompt=prompt, refs=len(edit_image or []),
                size=[width, height], steps=num_inference_steps, cfg=cfg_scale)
        self.scheduler.set_timesteps(num_inference_steps)  # what the real pipe does
        if "FAIL" in prompt:
            raise RuntimeError("CUDA out of memory (simulado)")
        return Image.new("RGBA", (width // 16, height // 16))


class QwenImage21TrainingModule(DiffusionTrainingModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.dit = torch.nn.Linear(4, 4)
        self.pipe = FakePipe(self.dit)
        _record(event="init", **{k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool, type(None)))})

    def forward(self, data, inputs=None):
        edit = data.get("edit_image")
        _record(event="step", image=list(data["image"].size), mode=data["image"].mode,
                refs=len(edit) if edit is not None else None, prompt=data["prompt"])
        return (self.dit(torch.ones(1, 4)) ** 2).mean()
