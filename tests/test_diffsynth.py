"""Qwen Image 2.1 through DiffSynth-Studio: datasets, config, and the wrapper.

The end-to-end class runs trainero/tools/diffsynth_train.py with DiffSynth's
real parser, dataset, runner and logger on CPU — only the 33 GB model is a
stand-in (tests/fixtures/fake_diffsynth_train.py). Point DIFFSYNTH_PYTHON at a
venv with DiffSynth installed and DIFFSYNTH_REPO at its checkout to run it.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from trainero import dataset as ds
from trainero import diffsynth as dsyn
from trainero import engines
from trainero.jobs import Job, JobFailed, SyncJob
from trainero.presets import MODELS, style_rush_models, suggest_schedule
from trainero.training import _cli_args

KEY = "qwen-image-21"
REPO = Path(__file__).resolve().parent.parent
DIFFSYNTH_PYTHON = os.environ.get("DIFFSYNTH_PYTHON", "")
DIFFSYNTH_REPO = os.environ.get("DIFFSYNTH_REPO", "")


def _img(path: Path, size=(96, 64), color=(200, 40, 40)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _triplets(root: Path, n: int = 3) -> Path:
    """n results, each with two references and an instruction."""
    d = root / "dataset"
    for i in range(n):
        _img(d / f"item{i}.png", (128, 96), (i * 60, 10, 10))
        _img(d / "control" / f"item{i}_0.png", (64, 64), (10, i * 60, 10))
        _img(d / "control" / f"item{i}_1.png", (80, 48), (10, 10, i * 60))
        (d / f"item{i}.txt").write_text(f"put the two people of the references together {i}")
    return d


class TestDatasetShapes(unittest.TestCase):
    def test_triplets_pair_by_name_in_reference_order(self):
        with tempfile.TemporaryDirectory() as td:
            d = _triplets(Path(td))
            controls = dsyn.control_images(d)
            self.assertEqual([p.name for p in controls["item1"]], ["item1_0.png", "item1_1.png"])
            self.assertTrue(dsyn.preflight(d, ds.inspect(d)))

    def test_a_plain_control_name_still_pairs(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "dataset"
            _img(d / "a.png")
            _img(d / "control" / "a.png")
            (d / "a.txt").write_text("x")
            self.assertEqual([p.name for p in dsyn.control_images(d)["a"]], ["a.png"])

    def test_a_target_named_like_a_suffix_keeps_its_own_control(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "dataset"
            _img(d / "img.png")
            _img(d / "img_01.png")
            _img(d / "control" / "img_01.png")
            controls = dsyn.control_images(d)
            self.assertEqual(list(controls), ["img_01"])

    def test_no_control_folder_is_text_to_image(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "dataset"
            _img(d / "a.png")
            (d / "a.txt").write_text("x")
            self.assertFalse(dsyn.preflight(d, ds.inspect(d)))

    def test_half_paired_is_refused_before_anything_downloads(self):
        with tempfile.TemporaryDirectory() as td:
            d = _triplets(Path(td))
            _img(d / "lonely.png")
            (d / "lonely.txt").write_text("x")
            with self.assertRaises(JobFailed) as ctx:
                dsyn.preflight(d, ds.inspect(d))
            self.assertIn("lonely.png", str(ctx.exception))

    def test_edit_needs_the_instruction_from_the_owner(self):
        with tempfile.TemporaryDirectory() as td:
            d = _triplets(Path(td))
            (d / "item0.txt").unlink()
            with self.assertRaises(JobFailed) as ctx:
                dsyn.preflight(d, ds.inspect(d))
            self.assertIn("instrução", str(ctx.exception))

    def test_metadata_carries_both_references(self):
        with tempfile.TemporaryDirectory() as td:
            d = _triplets(Path(td))
            meta = json.loads(dsyn.write_metadata(d, Path(td) / "m.json", edit=True).read_text())
            self.assertEqual(len(meta), 3)
            self.assertEqual([Path(p).name for p in meta[0]["edit_image"]],
                             ["item0_0.png", "item0_1.png"])
            self.assertIn("together 0", meta[0]["prompt"])

    def test_zip_with_numbered_control_folders_becomes_triplets(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "staging" / "src" / "meu_dataset"
            for i in range(2):
                _img(src / "target" / f"p{i}.png")
                (src / "target" / f"p{i}.txt").write_text("junte")
                _img(src / "control_0" / f"p{i}.png")
                _img(src / "control_1" / f"p{i}.png")
            dest = Path(td) / "dataset"
            ds.normalize_into(Path(td) / "staging", dest, SyncJob())
            self.assertEqual(sorted(p.name for p in (dest / "control").iterdir()),
                             ["p0_0.png", "p0_1.png", "p1_0.png", "p1_1.png"])
            self.assertTrue(dsyn.preflight(dest, ds.inspect(dest)))


class TestConfig(unittest.TestCase):
    def _cfg(self, vram, edit=True, overrides=None):
        with tempfile.TemporaryDirectory() as td:
            return dsyn.build_config(KEY, overrides or {}, suggest_schedule(KEY, 20), vram,
                                     Path(td), Path(td) / "m.json", Path(td) / "out",
                                     "proj", edit, None)

    def test_rtx_5090_runs_fp8_with_activation_offload_for_edit(self):
        cfg = self._cfg(31.8)  # what nvidia-smi reports for a 5090
        self.assertEqual(cfg["fp8_models"].count("Qwen/Qwen-Image-2.1:"), 2)
        self.assertTrue(cfg["use_gradient_checkpointing_offload"])
        self.assertNotIn("enable_model_cpu_offload", cfg)
        self.assertEqual(cfg["data_file_keys"], "image,edit_image")
        self.assertEqual(cfg["extra_inputs"], "edit_image")

    def test_an_80gb_card_trains_in_bf16(self):
        cfg = self._cfg(80, edit=False)
        self.assertNotIn("fp8_models", cfg)
        self.assertNotIn("extra_inputs", cfg)
        self.assertNotIn("use_gradient_checkpointing_offload", cfg)

    def test_the_fp8_names_are_the_model_strings_diffsynth_matches(self):
        cfg = self._cfg(32)
        loaded = set(cfg["model_id_with_origin_paths"].split(","))
        self.assertTrue(set(cfg["fp8_models"].split(",")) <= loaded)

    def test_official_recipe_defaults(self):
        cfg = self._cfg(80)
        self.assertEqual((cfg["lora_rank"], cfg["learning_rate"]), (32, 1e-4))
        self.assertEqual(cfg["lora_target_modules"], "")
        self.assertEqual(cfg["max_pixels"], 1024 * 1024)

    def test_schedule_keeps_checkpoints_few_and_work_on_target(self):
        for n in (5, 20, 60, 300):
            s = suggest_schedule(KEY, n)
            self.assertTrue(4 <= s["epochs"] <= 10, (n, s))
            total = n * s["num_repeats"] * s["epochs"]
            self.assertTrue(1500 <= total <= 4000, (n, s, total))

    def test_not_offered_for_style_rush(self):
        self.assertNotIn(KEY, style_rush_models())

    def test_env_never_lets_diffsynth_download(self):
        env = dsyn.env_for(KEY)
        self.assertEqual(env["DIFFSYNTH_SKIP_DOWNLOAD"], "true")
        self.assertTrue(env["DIFFSYNTH_MODEL_BASE_PATH"].endswith(KEY))

    def test_downloads_land_where_the_patterns_look(self):
        locals_ = {local for _r, _f, local in MODELS[KEY]["downloads"]}
        for pattern in MODELS[KEY]["model_args"].values():
            model_id, file_pattern = pattern.rsplit(":", 1)
            folder = file_pattern.split("/")[0]
            self.assertIn(f"{model_id}/{folder}/", locals_)
        self.assertIn(MODELS[KEY]["processor"] + "/", locals_)


class TestTorchForTheDriver(unittest.TestCase):
    """PyPI torch 2.14 is CUDA 13.0 (driver >= 580): on an older driver it
    installs and then sees no GPU."""

    def test_index_by_driver(self):
        self.assertTrue(engines.torch_index_for((13, 0)).endswith("cu130"))
        self.assertTrue(engines.torch_index_for((12, 9)).endswith("cu129"))
        self.assertTrue(engines.torch_index_for((12, 8)).endswith("cu128"))  # 5090 on 570.x
        self.assertTrue(engines.torch_index_for((12, 4)).endswith("cu126"))
        self.assertIsNone(engines.torch_index_for(None))

    def test_diffsynth_is_pinned(self):
        from trainero.presets import ENGINES

        self.assertRegex(ENGINES["diffsynth"]["commit"], r"^[0-9a-f]{40}$")


@unittest.skipUnless(DIFFSYNTH_PYTHON and DIFFSYNTH_REPO,
                     "defina DIFFSYNTH_PYTHON e DIFFSYNTH_REPO para rodar o wrapper de verdade")
class TestWrapperEndToEnd(unittest.TestCase):
    def _run(self, edit: bool, sample: bool = True, prompt: str = "zz, a cat"):
        td = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(td, ignore_errors=True))
        if edit:
            d = _triplets(td)
        else:
            d = td / "dataset"
            for i in range(3):
                _img(d / f"i{i}.png", (128, 96))
                (d / f"i{i}.txt").write_text(f"zz, caption {i}")
        meta = dsyn.write_metadata(d, td / dsyn.METADATA, edit)
        schedule = {"num_repeats": 2, "epochs": 2, "save_every_n_epochs": 1}
        out = td / "output"
        smp = None
        if sample:
            if edit:
                refs, prompt, (w, h) = dsyn.edit_sample(d, 1024 * 1024)
                smp = {"prompt": prompt, "edit_image": ",".join(map(str, refs)), "width": w, "height": h}
            else:
                smp = {"prompt": prompt, "width": 1344, "height": 768}
            smp.update({"seed": 42, "steps": 40, "dir": str(out / "sample")})
        cfg = dsyn.build_config(KEY, {}, schedule, 32, d, meta, out, "proj", edit, smp)
        cfg["dataset_num_workers"] = 0
        record = td / "record.jsonl"
        env = dict(os.environ, FAKE_RECORD=str(record),
                   DIFFSYNTH_REAL_TRAIN=str(Path(DIFFSYNTH_REPO) / MODELS[KEY]["train_script"]),
                   **dsyn.env_for(KEY))
        cmd = [DIFFSYNTH_PYTHON, str(dsyn.WRAPPER),
               "--diffsynth_script", str(REPO / "tests" / "fixtures" / "fake_diffsynth_train.py")]
        cmd += _cli_args(cfg)
        job = Job("t", "t", td / "job.log")
        code = job.run(cmd, env=env, parse_progress=True, check=False)
        log = job.log_tail(200000)
        self.assertEqual(code, 0, log[-3000:])
        events = [json.loads(line) for line in record.read_text().splitlines()]
        return out, events, job, log

    def test_triplet_edit_run(self):
        out, events, job, log = self._run(edit=True)
        steps = [e for e in events if e["event"] == "step"]
        self.assertEqual(len(steps), 3 * 2 * 2)            # items x repeats x epochs
        self.assertTrue(all(s["refs"] == 2 for s in steps))  # both references reach the model
        self.assertTrue(all(s["mode"] == "RGBA" for s in steps))
        self.assertEqual(sorted(p.name for p in out.glob("*.safetensors")),
                         ["proj-000001.safetensors", "proj-000002.safetensors"])
        samples = [e for e in events if e["event"] == "sample"]
        self.assertEqual(len(samples), 3)                   # before step 1 + each epoch
        self.assertTrue(all(s["refs"] == 2 for s in samples))
        # the shared scheduler goes back to training timesteps after each sample
        sets = [e for e in events if e["event"] == "set_timesteps"]
        self.assertEqual([(e["n"], e["training"]) for e in sets[-2:]], [(40, False), (1000, True)])
        names = sorted(p.name for p in (out / "sample").iterdir())
        self.assertEqual(len(names), 3)
        from server import parse_sample_name
        self.assertEqual(sorted(parse_sample_name(n)[0] for n in names), [0, 1, 2])
        # what the UI reads
        self.assertEqual((job.progress["step"], job.progress["total_steps"]), (12, 12))
        self.assertEqual((job.progress["epoch"], job.progress["total_epochs"]), (2, 2))
        self.assertIn("loss", job.progress)

    def test_text_to_image_run_and_the_flags_diffsynth_accepts(self):
        out, events, job, _ = self._run(edit=False)
        init = next(e for e in events if e["event"] == "init")
        self.assertEqual(init["lora_rank"], 32)
        self.assertEqual(init["fp8_models"].count(":"), 2)
        self.assertIsNone(init["extra_inputs"])
        steps = [e for e in events if e["event"] == "step"]
        self.assertTrue(all(s["refs"] is None for s in steps))
        self.assertEqual(len(list(out.glob("*.safetensors"))), 2)

    def test_sampling_off_still_trains(self):
        out, events, job, log = self._run(edit=True, sample=False)
        self.assertFalse([e for e in events if e["event"] == "sample"])
        self.assertEqual(len(list(out.glob("*.safetensors"))), 2)

    def test_a_failing_sample_does_not_fail_the_training(self):
        """An OOM while rendering the sample costs the sample, not the run —
        and training gets its 1000 timesteps back."""
        out, events, job, log = self._run(edit=False, prompt="FAIL")
        self.assertEqual(len(list(out.glob("*.safetensors"))), 2)
        self.assertIn("o treino continua", log)
        sets = [e for e in events if e["event"] == "set_timesteps"]
        self.assertEqual((sets[-1]["n"], sets[-1]["training"]), (1000, True))


if __name__ == "__main__":
    unittest.main()


class TestRunTrainingRoute(unittest.TestCase):
    """run_training sends Qwen 2.1 down the DiffSynth path: no caches, a
    metadata file, the wrapper command — and refuses bad datasets first."""

    def _run(self, td: Path, edit: bool):
        from unittest import mock

        from trainero import training

        pdir = td / "projects" / "p"
        if edit:
            _triplets(pdir)
        else:
            d = pdir / "dataset"
            for i in range(4):
                _img(d / f"i{i}.png")
                (d / f"i{i}.txt").write_text("zz, a caption")
        ran = {}

        def fake_run(self, cmd, cwd=None, env=None, parse_progress=False, **kw):
            ran["cmd"], ran["env"] = [str(c) for c in cmd], env
            out = Path(ran["cmd"][ran["cmd"].index("--output_path") + 1])
            (out / "p-000001.safetensors").write_bytes(b"ckpt")
            return 0

        job = Job("train", "t", td / "log.txt")
        with mock.patch.object(training, "PROJECTS_DIR", td / "projects"), \
             mock.patch.object(training, "ensure_engine"), \
             mock.patch.object(training, "ensure_models"), \
             mock.patch.object(training, "run_caches", side_effect=AssertionError("no caches")), \
             mock.patch.object(training, "hf_username", return_value=None), \
             mock.patch.object(training, "gpu_info", return_value={"name": "RTX 5090", "vram_mb": 32607}), \
             mock.patch.object(Job, "run", fake_run):
            training.run_training(job, {"project": "p", "model": KEY, "mode": "lora",
                                        "trigger": "zz", "overrides": {"hf_upload": False}})
        return pdir, job, ran

    def test_triplet_edit_end_to_end_wiring(self):
        with tempfile.TemporaryDirectory() as td:
            pdir, job, ran = self._run(Path(td), edit=True)
            cmd = ran["cmd"]
            self.assertIn(str(dsyn.WRAPPER), cmd)
            self.assertEqual(cmd[cmd.index("--extra_inputs") + 1], "edit_image")
            self.assertIn("--use_gradient_checkpointing_offload", cmd)
            self.assertEqual(cmd[cmd.index("--sample_edit_image") + 1].count(","), 1)
            self.assertEqual(ran["env"]["DIFFSYNTH_SKIP_DOWNLOAD"], "true")
            meta = json.loads((pdir / dsyn.METADATA).read_text())
            self.assertTrue(all(len(r["edit_image"]) == 2 for r in meta))
            self.assertNotIn("Cache de latents", [p["name"] for p in job.phases])
            self.assertEqual(job.extra["config_summary"]["task"], "edit")
            self.assertEqual(job.extra["outputs"], ["p-000001.safetensors"])

    def test_text_to_image_sample_carries_the_trigger(self):
        with tempfile.TemporaryDirectory() as td:
            _, job, ran = self._run(Path(td), edit=False)
            cmd = ran["cmd"]
            self.assertTrue(cmd[cmd.index("--sample_prompt") + 1].startswith("zz, "))
            self.assertNotIn("--extra_inputs", cmd)
            self.assertEqual(job.extra["config_summary"]["task"], "t2i")
