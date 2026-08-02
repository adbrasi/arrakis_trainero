"""Unit tests for the pure logic: presets, schedules, source detection, TOMLs."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainero.dataset import archive_ext, detect_source
from trainero.presets import (MODEL_ORDER, MODELS, net_types_for,
                              public_presets, suggest_schedule, vram_tier)
from trainero.training import (_cli_args, build_train_config, slugify,
                               write_dataset_toml)


class TestPresets(unittest.TestCase):
    def test_all_models_complete(self):
        for key in MODEL_ORDER:
            m = MODELS[key]
            self.assertIn(m["engine"], ("musubi", "musubi-ltx", "sd-scripts"), key)
            self.assertTrue(m["downloads"], key)
            self.assertTrue(m["model_args"], key)
            self.assertIn("network_dim", m["train"], key)
            self.assertTrue(m["vram_tiers"], key)
            self.assertEqual(m["vram_tiers"][-1]["min_gb"], 0, f"{key}: last tier must catch all")

    def test_vram_tiers_descending(self):
        for key in MODEL_ORDER:
            mins = [t["min_gb"] for t in MODELS[key]["vram_tiers"]]
            self.assertEqual(mins, sorted(mins, reverse=True), key)
            self.assertEqual(vram_tier(key, 9999), MODELS[key]["vram_tiers"][0])
            self.assertEqual(vram_tier(key, 1), MODELS[key]["vram_tiers"][-1])

    def test_schedules_land_in_sweet_spot(self):
        for key in MODEL_ORDER:
            for n in (5, 30, 200, 1500):
                s = suggest_schedule(key, n)
                self.assertGreaterEqual(s["epochs"], 4, (key, n))
                self.assertGreaterEqual(s["num_repeats"], 1)
                self.assertGreaterEqual(s["save_every_n_epochs"], 1)
                if "target_steps" in MODELS[key]:
                    total = n * s["num_repeats"] * s["epochs"]
                    target = MODELS[key]["target_steps"]
                    self.assertGreater(total, target * 0.4, (key, n, total))
                    # datasets enormes estouram o alvo pelo piso de 4 epochs —
                    # intencional: cada imagem precisa ser vista o suficiente
                    self.assertLess(total, max(target * 2.5, n * 4 + 1), (key, n, total))

    def test_net_types(self):
        self.assertEqual(net_types_for("anima"), ["lora"])
        self.assertIn("lokr", net_types_for("wan-22"))
        self.assertEqual(net_types_for("ideogram"), ["lora"])

    def test_public_presets_json_safe(self):
        import json

        json.dumps(public_presets())


class TestTrainConfig(unittest.TestCase):
    def _cfg(self, key, overrides=None, n=30):
        stats = {"items": n, "images": n, "videos": 0}
        sched = suggest_schedule(key, n)
        return build_train_config(key, overrides or {}, sched, stats, 48,
                                  Path("/tmp/ds.toml"), Path("/tmp/out"), "test", 1)

    def test_every_model_builds(self):
        for key in MODEL_ORDER:
            cfg = self._cfg(key)
            self.assertIn("network_module", cfg, key)
            self.assertIn("max_train_epochs", cfg, key)
            self.assertNotIn("max_train_steps", cfg, key)  # epochs, never steps
            self.assertTrue(cfg["output_name"], key)

    def test_lokr_switch(self):
        cfg = self._cfg("wan-22", {"net_type": "lokr"})
        self.assertEqual(cfg["network_module"], "networks.lokr")
        self.assertEqual(cfg["network_alpha"], cfg["network_dim"] // 2)

    def test_loraplus(self):
        cfg = self._cfg("flux-klein", {"loraplus": True})
        self.assertIn("loraplus_lr_ratio=16", cfg["network_args"])

    def test_invalid_net_type_rejected(self):
        from trainero.jobs import JobFailed

        with self.assertRaises(JobFailed):
            self._cfg("anima", {"net_type": "lokr"})

    def test_overrides_apply(self):
        cfg = self._cfg("qwen-image", {"network_dim": 16, "learning_rate": "1e-4"})
        self.assertEqual(cfg["network_dim"], 16)
        self.assertEqual(cfg["learning_rate"], "1e-4")

    def test_cli_args(self):
        args = _cli_args({"a": 1, "flag": True, "off": False, "lst": ["x", "y"], "skip": None})
        self.assertEqual(args, ["--a", "1", "--flag", "--lst", "x", "y"])


class TestDatasetToml(unittest.TestCase):
    def test_musubi_image(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            pdir = Path(td)
            toml = write_dataset_toml("qwen-image", pdir, pdir / "dataset", pdir / "cache",
                                      {"num_repeats": 3, "epochs": 10, "save_every_n_epochs": 1},
                                      [1024, 1024], 1, {"images": 10, "videos": 0})
            text = toml.read_text()
            self.assertIn("image_directory", text)
            self.assertIn("num_repeats = 3", text)
            self.assertIn("enable_bucket = true", text)
            self.assertNotIn("control_directory", text)

    def test_musubi_edit_has_control(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            pdir = Path(td)
            toml = write_dataset_toml("qwen-image-edit", pdir, pdir / "dataset", pdir / "cache",
                                      {"num_repeats": 1, "epochs": 10, "save_every_n_epochs": 1},
                                      [1024, 1024], 1, {"images": 10, "videos": 0})
            text = toml.read_text()
            self.assertIn("control_directory", text)
            self.assertIn("control_resolution = [1024, 1024]", text)

    def test_ltx_video(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            pdir = Path(td)
            toml = write_dataset_toml("ltx-23", pdir, pdir / "dataset", pdir / "cache",
                                      {"num_repeats": 1, "epochs": 10, "save_every_n_epochs": 1},
                                      [768, 512], 1, {"images": 0, "videos": 8},
                                      {"resolution": "768x512x81", "fps": 25.0})
            text = toml.read_text()
            self.assertIn("video_directory", text)
            self.assertIn("target_frames = [81]", text)
            self.assertIn("target_fps = 25.0", text)

    def test_sdscripts_anima(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            pdir = Path(td)
            toml = write_dataset_toml("anima", pdir, pdir / "dataset", pdir / "cache",
                                      {"num_repeats": 8, "epochs": 80, "save_every_n_epochs": 8},
                                      [1024, 1024], 8, {"images": 30, "videos": 0})
            text = toml.read_text()
            self.assertIn("[[datasets.subsets]]", text)
            self.assertIn("bucket_reso_steps = 64", text)
            self.assertIn("num_repeats = 8", text)


class TestDetectSource(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(detect_source("https://mega.nz/folder/abc#key"), "mega")
        self.assertEqual(detect_source("https://huggingface.co/datasets/u/r/resolve/main/d.zip"), "hf-file")
        self.assertEqual(detect_source("https://example.com/data.zip"), "url")
        self.assertEqual(detect_source("AdwolfCzar/meu_dataset"), "hf-repo")
        self.assertEqual(detect_source("/tmp"), "local-dir")
        self.assertEqual(detect_source(""), "empty")
        self.assertEqual(detect_source("???"), "unknown")

    def test_archive_ext(self):
        self.assertEqual(archive_ext("a.tar.gz"), ".tar.gz")
        self.assertEqual(archive_ext("a.ZIP"), ".zip")
        self.assertIsNone(archive_ext("a.safetensors"))


class TestSlug(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(slugify("Makima v1!"), "makima_v1")
        self.assertEqual(slugify("  "), "projeto")


if __name__ == "__main__":
    unittest.main()
