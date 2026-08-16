"""End-to-end wiring of run_training, the plain LoRA path (no GPU, no network).

The style-rush path has its own harness; this one exists because that coverage
gap let run_training ship referencing a `trigger` name it never bound — every
LoRA run died right after the Configuração phase, and no unit test noticed
because none of them ever called run_training.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _FakeJob:
    def __init__(self):
        self.lines = []
        self.extra = {}
        self.phase = ""

    def log(self, msg):
        self.lines.append(msg)

    def check_cancel(self):
        pass

    def set_phases(self, names):
        self.phases = list(names)

    def start_phase(self, name):
        self.phase = name

    def end_phase(self, name, ok=True):
        pass


def _make_dataset(pdir: Path, n: int = 6) -> Path:
    base = pdir / "dataset"
    base.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    for i in range(n):
        Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(
            base / f"img_{i:03d}.png")
        (base / f"img_{i:03d}.txt").write_text("makima, a girl")
    return base


class TestRunTrainingLora(unittest.TestCase):
    def _run(self, td: Path, model_key: str, params_extra: dict | None = None):
        from trainero import training

        pdir = td / "projects" / "p"
        _make_dataset(pdir)
        uploaded = {}

        def fake_launch(mk, cfg, project_dir, job, toml_name="train.toml"):
            (Path(cfg["output_dir"]) / "p-000001.safetensors").write_bytes(b"ckpt")

        def fake_upload_run_files(repo_id, job, info, captions=None):
            uploaded["info"] = info

        class _Watcher:
            def __init__(self, *a, **kw):
                pass

            def start(self):
                pass

            def stop_and_sweep(self):
                pass

        job = _FakeJob()
        params = {"project": "p", "model": model_key, "mode": "lora",
                  "trigger": "anime loven", "overrides": {}}
        params.update(params_extra or {})

        with mock.patch.object(training, "PROJECTS_DIR", td / "projects"), \
             mock.patch.object(training, "ensure_engine"), \
             mock.patch.object(training, "ensure_models"), \
             mock.patch.object(training, "run_caches"), \
             mock.patch.object(training, "resolve", lambda mk, rel: Path("/models") / rel), \
             mock.patch.object(training, "supports_sampling", return_value=True), \
             mock.patch.object(training, "hf_username", return_value="dono"), \
             mock.patch.object(training, "create_repo", return_value="dono/p"), \
             mock.patch.object(training, "upload_run_files", fake_upload_run_files), \
             mock.patch.object(training, "UploadWatcher", _Watcher), \
             mock.patch.object(training, "comfy_converter", return_value=None), \
             mock.patch.object(training, "gpu_info",
                               return_value={"name": "T", "vram_mb": 32768}), \
             mock.patch.object(training, "launch_training", fake_launch):
            training.run_training(job, params)
        return pdir, job, uploaded

    def test_a_lora_run_reaches_the_end(self):
        """Guards the exact break: the HF-upload block referenced an unbound
        name, so every run with a token died after Configuração."""
        with tempfile.TemporaryDirectory() as td:
            pdir, job, uploaded = self._run(Path(td), "qwen-image")
            self.assertTrue(list((pdir / "output").glob("*.safetensors")))
            self.assertEqual(uploaded["info"]["trigger"], "anime loven")
            self.assertEqual(uploaded["info"]["mode"], "lora")

    def test_the_trigger_reaches_the_sample_prompt(self):
        """A sample rendered without the trigger does not exercise the LoRA, so
        the owner watches five epochs of images that cannot show progress."""
        with tempfile.TemporaryDirectory() as td:
            pdir, _, _ = self._run(Path(td), "qwen-image")
            prompt = (pdir / "sample_prompts.txt").read_text()
            self.assertTrue(prompt.startswith("anime loven, "), prompt[:60])

    def test_the_trigger_survives_an_api_call_that_omits_overrides(self):
        """The UI happens to copy the trigger into overrides too. Reading it
        from there made a direct API call sample without it — silently."""
        with tempfile.TemporaryDirectory() as td:
            pdir, _, _ = self._run(Path(td), "qwen-image",
                                   {"overrides": {"hf_upload": False}})
            self.assertIn("anime loven", (pdir / "sample_prompts.txt").read_text())

    def test_sd_scripts_path_also_completes(self):
        """Anima is the only sd-scripts model and skips the cache phases; its
        config goes through a different branch of write_dataset_toml."""
        with tempfile.TemporaryDirectory() as td:
            pdir, job, _ = self._run(Path(td), "anima")
            self.assertTrue(list((pdir / "output").glob("*.safetensors")))
            toml = (pdir / "dataset.toml").read_text()
            self.assertIn("[[datasets.subsets]]", toml)
            self.assertNotIn("Cache de latents", job.phases)


if __name__ == "__main__":
    unittest.main()
