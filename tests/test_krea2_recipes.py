"""Krea 2 automatic recipes: kind -> rank, steps, repeats/epochs, mixed subsets."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainero.presets import (CHARACTER_STEP_CAP, MIXED_STYLE_SHARE, character_target_steps,
                              krea2_recipe, mixed_repeats, mixed_target_steps,
                              style_target_steps, suggest_kind)


def _total(recipe):
    return recipe["est_steps"]


class TestCharacterRecipe(unittest.TestCase):
    def test_anchor_steps_from_the_research(self):
        self.assertEqual(character_target_steps(1), 1000)
        self.assertEqual(character_target_steps(3), 1000)
        self.assertEqual(character_target_steps(8), 1450)
        self.assertEqual(character_target_steps(15), 2100)
        self.assertEqual(character_target_steps(30), 2500)

    def test_interpolates_between_anchors(self):
        self.assertTrue(1000 < character_target_steps(5) < 1450)
        self.assertTrue(1450 < character_target_steps(10) < 2100)
        self.assertTrue(2100 < character_target_steps(20) < 2500)

    def test_never_past_the_collapse_cap(self):
        for n in (31, 40, 80, 500):
            self.assertLessEqual(character_target_steps(n), CHARACTER_STEP_CAP)
            self.assertLessEqual(_total(krea2_recipe("character", n)), CHARACTER_STEP_CAP)

    def test_rank_16_up_to_five_images_then_32(self):
        for n in (1, 2, 3, 5):
            r = krea2_recipe("character", n)
            self.assertEqual((r["network_dim"], r["network_alpha"]), (16, 16), n)
        for n in (6, 10, 30):
            r = krea2_recipe("character", n)
            self.assertEqual((r["network_dim"], r["network_alpha"]), (32, 32), n)

    def test_total_lands_near_the_target_with_about_eight_checkpoints(self):
        for n in (1, 3, 5, 8, 10, 15, 20, 30):
            r = krea2_recipe("character", n)
            target = character_target_steps(n)
            # epochs x repeats is the schedule; the total is a consequence of it
            self.assertEqual(r["est_steps"], n * r["num_repeats"] * r["epochs"], n)
            self.assertLess(abs(r["est_steps"] - target) / target, 0.08, (n, r))
            self.assertTrue(7 <= r["checkpoints"] <= 10, (n, r))
            self.assertEqual(r["epochs"] % r["save_every_n_epochs"], 0)
            self.assertEqual(r["batch_size"], 1)

    def test_single_image_is_a_thousand_steps(self):
        r = krea2_recipe("character", 1)
        self.assertEqual(r["est_steps"], 1000)
        self.assertEqual(r["checkpoints"], 8)

    def test_line_says_what_will_happen(self):
        r = krea2_recipe("character", 15)
        self.assertTrue(r["line"].startswith("Receita: personagem · 15 imgs · dim 32 · ~"), r["line"])
        self.assertIn("checkpoints", r["line"])


class TestStyleRecipe(unittest.TestCase):
    def test_steps_formula(self):
        self.assertEqual(style_target_steps(10), 2000)        # floor
        self.assertEqual(style_target_steps(100), 2700)       # 1500 + 12n
        self.assertEqual(style_target_steps(1000), 4000)      # ceiling

    def test_rank_grows_past_300_images(self):
        self.assertEqual(krea2_recipe("style", 120)["network_dim"], 32)
        self.assertEqual(krea2_recipe("style", 301)["network_dim"], 64)
        self.assertEqual(krea2_recipe("style", 301)["network_alpha"], 64)

    def test_style_lands_near_target(self):
        for n in (50, 120, 200):
            r = krea2_recipe("style", n)
            target = style_target_steps(n)
            self.assertLess(abs(r["est_steps"] - target) / target, 0.12, (n, r))

    def test_huge_style_never_runs_absurdly_long(self):
        r = krea2_recipe("style", 3000)
        self.assertEqual(r["num_repeats"], 1)
        self.assertLessEqual(r["est_steps"], 6000)
        self.assertGreaterEqual(r["epochs"], 1)


class TestMixedRecipe(unittest.TestCase):
    def _concepts(self, chars, style):
        out = [{"name": f"char{i}", "items": n, "style": False} for i, n in enumerate(chars)]
        if style:
            out.append({"name": "style_x", "items": style, "style": True})
        return out

    def test_every_character_gets_the_same_exposure(self):
        subsets = mixed_repeats(self._concepts([10, 20, 40], 0))
        seen = [c["items"] * c["num_repeats"] for c in subsets]
        self.assertEqual(seen, [40, 40, 40])

    def test_style_takes_about_45_percent(self):
        subsets = mixed_repeats(self._concepts([30] * 10, 120))
        style = sum(c["items"] * c["num_repeats"] for c in subsets if c["style"])
        total = sum(c["items"] * c["num_repeats"] for c in subsets)
        self.assertAlmostEqual(style / total, MIXED_STYLE_SHARE, delta=0.06)

    def test_owner_scenario_ten_characters_plus_style(self):
        r = krea2_recipe("mixed", 0, self._concepts([30] * 10, 120), batch_size=2)
        self.assertEqual(r["items"], 420)
        self.assertEqual(r["network_dim"], 32)
        self.assertEqual(r["batch_size"], 2)
        self.assertTrue(5000 * 0.9 <= r["est_steps"] <= 6000 * 1.05, r)
        self.assertTrue(0.40 <= r["style_share"] <= 0.50, r)
        self.assertEqual(r["characters"], 10)
        self.assertIn("misto", r["line"])
        self.assertIn("batch 2", r["line"])

    def test_big_mixed_gets_rank_64(self):
        r = krea2_recipe("mixed", 0, self._concepts([60] * 10, 120))
        self.assertEqual(r["network_dim"], 64)

    def test_huge_style_pool_does_not_overtrain_small_characters(self):
        subsets = mixed_repeats(self._concepts([10, 40], 800))
        small = next(c for c in subsets if c["name"] == "char0")
        # at most 3x the largest character folder, never ~49 repeats
        self.assertLessEqual(small["items"] * small["num_repeats"], 3 * 40)

    def test_style_only_and_characters_only(self):
        only_style = mixed_repeats(self._concepts([], 100))
        self.assertEqual(only_style[0]["num_repeats"], 1)
        only_chars = mixed_repeats(self._concepts([5, 10], 0))
        self.assertEqual([c["num_repeats"] for c in only_chars], [2, 1])

    def test_target_steps_band(self):
        self.assertEqual(mixed_target_steps(100), 5000)
        self.assertEqual(mixed_target_steps(450), 5750)
        self.assertEqual(mixed_target_steps(5000), 6000)

    def test_mixed_needs_concepts(self):
        with self.assertRaises(ValueError):
            krea2_recipe("mixed", 100, [])

    def test_unknown_kind(self):
        with self.assertRaises(ValueError):
            krea2_recipe("sticker", 10)


class TestSuggestKind(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(suggest_kind(1), "character")
        self.assertEqual(suggest_kind(30), "character")
        self.assertEqual(suggest_kind(120), "style")
        self.assertEqual(suggest_kind(0, n_concepts=3), "mixed")


# ---------------------------------------------------------------------------
# dataset side: concept folders, import, mixed dataset.toml, run_training
# ---------------------------------------------------------------------------

def _img(path: Path, caption: str | None = "x, a girl"):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(path.name)) % 1000)
    Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(path)
    if caption is not None:
        path.with_suffix(".txt").write_text(caption)


class TestConcepts(unittest.TestCase):
    def test_folders_become_concepts(self):
        from trainero import dataset as ds

        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "dataset"
            for i in range(3):
                _img(d / "makima__anime_woman" / f"a{i}.png")
            for i in range(5):
                _img(d / "style_ghibli" / f"s{i}.png", None)
            _img(d / "rem" / "r0.png")
            (d / "control").mkdir()
            (d / "empty").mkdir()
            got = {c["name"]: c for c in ds.concepts(d, style_folders=["rem"])}
            self.assertEqual(set(got), {"makima__anime_woman", "style_ghibli", "rem"})
            self.assertEqual(got["makima__anime_woman"]["trigger"], "makima")
            self.assertEqual(got["makima__anime_woman"]["character_class"], "anime woman")
            self.assertFalse(got["makima__anime_woman"]["style"])
            self.assertEqual(got["style_ghibli"]["trigger"], "ghibli")
            self.assertTrue(got["style_ghibli"]["style"])
            self.assertEqual(got["style_ghibli"]["missing_captions"], 5)
            self.assertTrue(got["rem"]["style"])  # marked by the owner
            caps = ds.all_captions_map(d)
            self.assertIn("makima__anime_woman/a0.png", caps)

    def test_import_keeps_concept_folders_only_when_asked(self):
        from trainero import dataset as ds
        from trainero.jobs import SyncJob

        for keep in (True, False):
            with tempfile.TemporaryDirectory() as td:
                staging = Path(td) / "_staging"
                src = staging / "src" / "my_zip"
                _img(src / "makima" / "a.png")
                _img(src / "makima" / "nested" / "b.png")
                _img(src / "style_x" / "c.png")
                dest = Path(td) / "dataset"
                ds.normalize_into(staging, dest, SyncJob(), keep_concepts=keep)
                if keep:
                    self.assertTrue((dest / "makima" / "a.png").exists())
                    self.assertTrue((dest / "makima" / "b.png").exists())  # flattened inside
                    self.assertTrue((dest / "makima" / "a.txt").exists())
                    self.assertTrue((dest / "style_x" / "c.png").exists())
                    self.assertEqual(ds.inspect(dest)["items"], 0)
                else:
                    self.assertEqual(ds.inspect(dest)["items"], 3)
                    self.assertEqual(ds.concepts(dest), [])

    def test_single_concept_folder_is_not_unwrapped(self):
        from trainero import dataset as ds
        from trainero.jobs import SyncJob

        with tempfile.TemporaryDirectory() as td:
            staging = Path(td) / "_staging"
            _img(staging / "src" / "makima" / "a.png")
            dest = Path(td) / "dataset"
            ds.normalize_into(staging, dest, SyncJob(), keep_concepts=True)
            self.assertTrue((dest / "makima" / "a.png").exists())


class TestMixedDatasetToml(unittest.TestCase):
    def test_one_block_per_concept_with_its_repeats(self):
        from trainero.training import image_subset, subset_dirs, write_dataset_toml

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            subsets = [image_subset(td / "dataset" / "makima", td / "cache" / "makima", 3),
                       image_subset(td / "dataset" / "style_x", td / "cache" / "style_x", 2)]
            path = write_dataset_toml("krea2", td / "dataset.toml", subsets, [1024, 1024], 2)
            text = path.read_text()
            self.assertEqual(text.count("[[datasets]]"), 2)
            self.assertIn("num_repeats = 3", text)
            self.assertIn("num_repeats = 2", text)
            self.assertIn("batch_size = 2", text)
            self.assertNotIn("control_directory", text)
            pairs = subset_dirs(path)
            self.assertEqual([p[0].name for p in pairs], ["makima", "style_x"])


class _FakeJob:
    def __init__(self):
        self.lines, self.extra = [], {}

    def log(self, msg):
        self.lines.append(msg)

    def check_cancel(self):
        pass

    def set_phases(self, names):
        pass

    def start_phase(self, name):
        pass

    def end_phase(self, name, ok=True):
        pass


class TestRunTrainingKrea2(unittest.TestCase):
    def _run(self, td: Path, build, params_extra=None, vram_mb=32768):
        from trainero import training

        pdir = td / "projects" / "p"
        build(pdir / "dataset")
        seen = {}

        def fake_launch(mk, cfg, project_dir, job, toml_name="train.toml"):
            seen["cfg"] = cfg
            (Path(cfg["output_dir"]) / "p-000001.safetensors").write_bytes(b"ckpt")

        job = _FakeJob()
        params = {"project": "p", "model": "krea2", "mode": "lora", "trigger": "mk",
                  "overrides": {"hf_upload": False}}
        params.update(params_extra or {})
        with mock.patch.object(training, "PROJECTS_DIR", td / "projects"), \
             mock.patch.object(training, "ensure_engine"), \
             mock.patch.object(training, "ensure_models"), \
             mock.patch.object(training, "run_caches"), \
             mock.patch.object(training, "resolve", lambda mk, rel: Path("/models") / rel), \
             mock.patch.object(training, "supports_sampling", return_value=True), \
             mock.patch.object(training, "gpu_info",
                               return_value={"name": "T", "vram_mb": vram_mb}), \
             mock.patch.object(training, "launch_training", fake_launch):
            training.run_training(job, params)
        return pdir, job, seen["cfg"]

    def test_character_recipe_reaches_the_config(self):
        def build(d):
            for i in range(15):
                _img(d / f"i{i}.png")

        with tempfile.TemporaryDirectory() as td:
            pdir, job, cfg = self._run(Path(td), build)
            recipe = krea2_recipe("character", 15)
            self.assertEqual(job.extra["recipe"]["kind"], "character")
            self.assertEqual(cfg["network_dim"], 32)
            self.assertEqual(cfg["max_train_epochs"], recipe["epochs"])
            self.assertIn(f"num_repeats = {recipe['num_repeats']}",
                          (pdir / "dataset.toml").read_text())
            self.assertIn(recipe["line"], job.lines)

    def test_small_set_gets_rank_16_and_overrides_still_win(self):
        def build(d):
            for i in range(3):
                _img(d / f"i{i}.png")

        with tempfile.TemporaryDirectory() as td:
            _, _, cfg = self._run(Path(td), build)
            self.assertEqual((cfg["network_dim"], cfg["network_alpha"]), (16, 16))
        with tempfile.TemporaryDirectory() as td:
            _, _, cfg = self._run(Path(td), build, {"overrides": {
                "hf_upload": False, "network_dim": 48, "epochs": 3}})
            self.assertEqual(cfg["network_dim"], 48)
            self.assertEqual(cfg["max_train_epochs"], 3)

    def test_style_kind_is_honoured(self):
        def build(d):
            for i in range(20):
                _img(d / f"i{i}.png")

        with tempfile.TemporaryDirectory() as td:
            _, job, _ = self._run(Path(td), build, {"lora_kind": "style"})
            self.assertEqual(job.extra["recipe"]["kind"], "style")
            self.assertEqual(job.extra["recipe"]["target_steps"], 2000)

    def test_mixed_writes_one_subset_per_folder(self):
        def build(d):
            for c, n in (("makima", 6), ("rem", 3)):
                for i in range(n):
                    _img(d / c / f"{c}{i}.png", f"{c}, a girl")
            for i in range(8):
                _img(d / "style_x" / f"s{i}.png", "x style, a street")

        with tempfile.TemporaryDirectory() as td:
            pdir, job, cfg = self._run(Path(td), build, {"lora_kind": "mixed"})
            text = (pdir / "dataset.toml").read_text()
            self.assertEqual(text.count("[[datasets]]"), 3)
            self.assertIn("batch_size = 2", text)          # 32 GB tier
            recipe = job.extra["recipe"]
            reps = {c["name"]: c["num_repeats"] for c in recipe["subsets"]}
            self.assertEqual(reps["makima"] * 6, reps["rem"] * 3)
            # each concept trains its own captions: no project trigger prefixed
            prompt = (pdir / "sample_prompts.txt").read_text()
            self.assertTrue(prompt.startswith("makima, a girl"), prompt[:40])
            self.assertEqual(cfg["sample_every_n_epochs"], recipe["save_every_n_epochs"])

    def test_mixed_refuses_loose_images_and_missing_captions(self):
        from trainero.jobs import JobFailed

        def loose(d):
            _img(d / "makima" / "a.png")
            _img(d / "b.png")

        def uncaptioned(d):
            _img(d / "makima" / "a.png", None)

        for build, needle in ((loose, "soltas"), (uncaptioned, "sem caption")):
            with tempfile.TemporaryDirectory() as td, self.assertRaises(JobFailed) as ctx:
                self._run(Path(td), build, {"lora_kind": "mixed"})
            self.assertIn(needle, str(ctx.exception))

    def test_folders_without_mixed_say_so(self):
        from trainero.jobs import JobFailed

        def build(d):
            _img(d / "makima" / "a.png")

        with tempfile.TemporaryDirectory() as td, self.assertRaises(JobFailed) as ctx:
            self._run(Path(td), build, {"lora_kind": "character"})
        self.assertIn("Misto", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
