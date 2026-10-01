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



class TestStyleRushOverrides(unittest.TestCase):
    """Estes overrides existem para o dono mexer neles pela UI; um deles não
    ligado ao pipeline é um campo que mente sobre o que vai acontecer."""

    def _source(self):
        from pathlib import Path as P
        return (P(__file__).resolve().parent.parent / "trainero" / "training.py").read_text()

    def test_the_convert_target_reaches_the_builder(self):
        src = self._source()
        self.assertIn('overrides.get("convert_target")', src)
        self.assertIn("target=convert_target", src)

    def test_the_default_target_comes_from_style_rush(self):
        self.assertIn("sr.DEFAULT_CONVERT_TARGET", self._source())

    def test_style_rush_captions_with_its_own_profile_and_honours_redo(self):
        """Style Rush captions with the style-rush profile (Gemini first: its
        refusals feed the paid phase's content_flagged) and redo reaches it."""
        from unittest import mock

        from trainero import training
        from trainero.jobs import JobFailed

        calls = []

        def fake_generate(dataset_dir, profile, job, trigger="", redo=False):
            calls.append((profile, trigger, redo))
            raise JobFailed("stop here")

        with tempfile.TemporaryDirectory() as td:
            pdir = Path(td) / "p"
            (pdir / "dataset").mkdir(parents=True)
            (pdir / "dataset" / "a.png").write_bytes(b"x")
            with mock.patch.object(training, "project_dir", lambda _n: pdir), \
                 mock.patch.object(training, "generate_captions", fake_generate), \
                 mock.patch.object(training, "ensure_engine", lambda *a: None), \
                 mock.patch.object(training, "ensure_models", lambda *a: None), \
                 mock.patch.object(training, "gpu_info", lambda: {}):
                for redo in (False, True):
                    job = _FakeJob()
                    with self.assertRaises(JobFailed):
                        training.run_training(job, {"project": "p", "model": "flux-klein",
                                                    "mode": "style-rush", "trigger": "zz",
                                                    "overrides": {"redo_captions": redo}})
        self.assertEqual(calls, [("style-rush", "zz", False), ("style-rush", "zz", True)])


class TestBeginnerScenarios(TestRunTrainingLora):
    """A beginner uploads images, picks a model and clicks TREINAR."""

    def _strip_captions(self, pdir):
        for t in (pdir / "dataset").glob("*.txt"):
            t.unlink()

    def _run_uncaptioned(self, td, model_key, env, caption_impl=None, params_extra=None):
        from trainero import training

        order = []
        real_make = _make_dataset

        def make_without_captions(pdir, n=6):
            base = real_make(pdir, n)
            for t in base.glob("*.txt"):
                t.unlink()
            return base

        def fake_caption(job, dirs, **kw):
            order.append(("captions", kw))
            if caption_impl:
                return caption_impl(dirs)
            for d in dirs:
                for img in d.glob("*.png"):
                    img.with_suffix(".txt").write_text(f"{kw.get('trigger')}, a caption")

        with mock.patch.dict("os.environ", env, clear=False), \
             mock.patch(__name__ + "._make_dataset", make_without_captions), \
             mock.patch.object(training, "caption_datasets", fake_caption):
            pdir, job, _ = self._run(td, model_key, params_extra)
        return pdir, job, order

    def test_missing_captions_are_written_by_the_job_before_any_download(self):
        with tempfile.TemporaryDirectory() as td:
            pdir, job, order = self._run_uncaptioned(
                Path(td), "flux-klein", {"OPENROUTER_API_KEY": "sk"},
                params_extra={"kind": "style"})
            self.assertEqual(order[0][0], "captions")
            self.assertEqual(order[0][1]["kind"], "style")
            self.assertEqual(job.phases[:2], ["Captions", "Engine"])
            self.assertTrue(all(t.read_text().startswith("anime loven")
                                for t in (pdir / "dataset").glob("*.txt")))

    def test_missing_captions_without_a_key_fail_in_the_first_second(self):
        from trainero.jobs import JobFailed

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(JobFailed) as ctx:
                self._run_uncaptioned(Path(td), "flux-klein", {"OPENROUTER_API_KEY": ""})
            self.assertIn("OPENROUTER_API_KEY", str(ctx.exception))
            self.assertIn(".txt", str(ctx.exception))

    def test_a_trigger_profile_without_a_trigger_fails_before_downloading(self):
        from trainero.jobs import JobFailed

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(JobFailed) as ctx:
                self._run_uncaptioned(Path(td), "flux-klein", {"OPENROUTER_API_KEY": "sk"},
                                      params_extra={"kind": "style", "trigger": ""})
            self.assertIn("trigger", str(ctx.exception))

    def test_a_concept_needs_no_trigger(self):
        with tempfile.TemporaryDirectory() as td:
            _, _, order = self._run_uncaptioned(
                Path(td), "flux-klein", {"OPENROUTER_API_KEY": "sk"},
                params_extra={"kind": "concept", "trigger": ""})
            self.assertEqual(order[0][0], "captions")

    def test_everything_refused_ends_with_a_clear_message(self):
        from trainero.jobs import JobFailed

        def refuse_all(dirs):
            for d in dirs:
                for img in d.glob("*.png"):
                    img.unlink()

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(JobFailed) as ctx:
                self._run_uncaptioned(Path(td), "flux-klein", {"OPENROUTER_API_KEY": "sk"},
                                      caption_impl=refuse_all, params_extra={"kind": "concept"})
            self.assertIn("nenhum item sobrou", str(ctx.exception))

    def test_a_retrain_starts_with_an_empty_output_and_keeps_the_old_one(self):
        """output/ used to be shared: a retrain re-uploaded old checkpoints and
        the gallery mixed old samples above new ones."""
        with tempfile.TemporaryDirectory() as td:
            pdir, _, _ = self._run(Path(td), "qwen-image")
            (pdir / "output" / "sample").mkdir()
            (pdir / "output" / "sample" / "old_e000009_00_x.png").write_bytes(b"x")
            (pdir / "output" / "p-000001.safetensors").write_bytes(b"OLD")
            pdir, job, _ = self._run(Path(td), "qwen-image")
            self.assertEqual((pdir / "output" / "p-000001.safetensors").read_bytes(), b"ckpt")
            self.assertFalse((pdir / "output" / "sample").exists())
            archived = list((pdir / "runs").glob("*/output/p-000001.safetensors"))
            self.assertEqual([a.read_bytes() for a in archived], [b"OLD"])
            self.assertEqual(job.extra["outputs"], ["p-000001.safetensors"])


if __name__ == "__main__":
    unittest.main()
