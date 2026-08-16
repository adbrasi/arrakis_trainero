"""Unit tests for the Style Rush restoration dataset (no GPU, no network, no API)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainero.degrade import apply_ai_grit_tiling_texture, degrade_file
from trainero.jobs import Cancelled, JobFailed
from trainero.style_rush import (RESTORE_CAPTION, RESTORE_COUNT, RESTORE_MANIFEST_NAME,
                                 RESTORE_SEED, build_restore_dataset, convert_sources,
                                 plan_restore)


class _FakeJob:
    def __init__(self):
        self.lines = []

    def log(self, msg):
        self.lines.append(msg)

    def check_cancel(self):
        pass


SOURCE_SIZE = (64, 48)


def _make_dataset(root: Path, n: int) -> Path:
    """A dataset of distinguishable images — flat grey would degrade to almost
    nothing and hide a broken degradation behind a passing assertion."""
    rng = np.random.default_rng(7)
    base = root / "dataset"
    base.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        ext = ".png" if i % 2 == 0 else ".jpg"
        pixels = rng.integers(40, 210, (SOURCE_SIZE[1], SOURCE_SIZE[0], 3), dtype=np.uint8)
        Image.fromarray(pixels).save(base / f"img_{i:03d}{ext}")
        (base / f"img_{i:03d}.txt").write_text("makima, a girl")
    return base


def _paths(base: Path) -> list[Path]:
    return sorted(p for p in base.iterdir() if p.suffix.lower() in {".png", ".jpg"})


class TestDegrade(unittest.TestCase):
    def test_degradation_changes_the_image(self):
        rng = np.random.default_rng(3)
        image = rng.integers(40, 210, (48, 64, 3), dtype=np.uint8)
        out, params = apply_ai_grit_tiling_texture(image, severity=0.6, seed=42)
        self.assertEqual(out.shape, image.shape)
        self.assertGreater(np.abs(out.astype(int) - image.astype(int)).mean(), 1.0,
                           "a degradação não alterou a imagem")
        self.assertEqual(params["seed"], 42)

    def test_same_seed_is_reproducible_and_different_seed_is_not(self):
        rng = np.random.default_rng(3)
        image = rng.integers(40, 210, (48, 64, 3), dtype=np.uint8)
        a, _ = apply_ai_grit_tiling_texture(image, seed=11)
        b, _ = apply_ai_grit_tiling_texture(image, seed=11)
        c, _ = apply_ai_grit_tiling_texture(image, seed=12)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c), "seeds diferentes deram a mesma textura")

    def test_dimensions_are_preserved(self):
        """The pairing guarantee: musubi buckets control and target from their
        own aspect ratios, so a degradation that resized would split the pair
        across two buckets and the model would never see them together."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "src.png"
            rng = np.random.default_rng(5)
            Image.fromarray(rng.integers(0, 255, (81, 137, 3), dtype=np.uint8)).save(src)
            degrade_file(src, root / "out.png", seed=1)
            with Image.open(src) as a, Image.open(root / "out.png") as b:
                self.assertEqual(a.size, b.size)

    def test_rejects_out_of_range_severity(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            apply_ai_grit_tiling_texture(image, severity=1.5)


class TestPlanRestore(unittest.TestCase):
    def test_prefers_images_the_conversion_did_not_use(self):
        with tempfile.TemporaryDirectory() as td:
            base = _make_dataset(Path(td), 160)
            images = _paths(base)
            used = {str(p) for p in images[:50]}
            slots = plan_restore(images, used, count=RESTORE_COUNT)
            self.assertEqual(len(slots), RESTORE_COUNT)
            picked = {s["source"] for s in slots}
            self.assertEqual(picked & used, set(),
                             "reaproveitou imagens da conversão havendo 110 livres")

    def test_reuses_only_when_there_are_not_enough_fresh_images(self):
        with tempfile.TemporaryDirectory() as td:
            base = _make_dataset(Path(td), 60)
            images = _paths(base)
            used = {str(p) for p in images[:50]}  # only 10 fresh, 100 slots needed
            slots = plan_restore(images, used, count=RESTORE_COUNT)
            self.assertEqual(len(slots), RESTORE_COUNT, "faltou preencher slots")
            fresh = {str(p) for p in images[50:]}
            first_ten = [s["source"] for s in slots[:10]]
            self.assertEqual(set(first_ten), fresh,
                             "as imagens livres têm de vir antes das reaproveitadas")

    def test_is_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            base = _make_dataset(Path(td), 120)
            images = _paths(base)
            self.assertEqual(plan_restore(images, set()), plan_restore(images, set()))

    def test_every_slot_gets_its_own_seed(self):
        """One shared seed would stamp the identical tile pattern on all 100
        images — the LoRA would learn to subtract that watermark, not to restore."""
        with tempfile.TemporaryDirectory() as td:
            base = _make_dataset(Path(td), 120)
            slots = plan_restore(_paths(base), set())
            seeds = [s["seed"] for s in slots]
            self.assertEqual(len(set(seeds)), len(seeds))
            self.assertEqual(seeds[0], RESTORE_SEED)

    def test_empty_dataset_raises(self):
        with self.assertRaises(ValueError):
            plan_restore([], set())


class TestConvertSources(unittest.TestCase):
    def test_reads_only_the_slots_that_succeeded(self):
        with tempfile.TemporaryDirectory() as td:
            convert = Path(td) / "dataset_convert"
            convert.mkdir()
            (convert / ".style_rush.json").write_text(json.dumps({"slots": {
                "slot_00": {"status": "ok", "source": "/a.png"},
                "slot_01": {"status": "refused", "source": "/b.png"},
                "slot_02": {"status": "failed", "source": "/c.png"},
                "slot_03": {"status": "ok", "source": "/d.png"},
            }}))
            self.assertEqual(convert_sources(convert), {"/a.png", "/d.png"})

    def test_missing_manifest_is_an_empty_set(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(convert_sources(Path(td)), set())


class TestBuildRestoreDataset(unittest.TestCase):
    def test_happy_path_writes_the_full_triples(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 160)
            restore = root / "dataset_restore"

            result = build_restore_dataset(base, restore, _FakeJob(), used=set(), workers=2)

            self.assertEqual(result["pairs"], RESTORE_COUNT)
            self.assertEqual(result["failed"], 0)
            targets = sorted(p.name for p in restore.glob("restore_*.png"))
            controls = sorted(p.name for p in (restore / "control").glob("restore_*.png"))
            self.assertEqual(len(targets), RESTORE_COUNT)
            self.assertEqual(targets, controls)
            self.assertEqual((restore / "restore_000.txt").read_text(), RESTORE_CAPTION)

    def test_control_is_damaged_and_target_is_the_untouched_original(self):
        """Swapping these two teaches the LoRA to *add* the grit. There is no
        symptom until inference, which is why this is asserted pixel-wise."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 120)
            restore = root / "dataset_restore"
            build_restore_dataset(base, restore, _FakeJob(), used=set(), count=4, workers=2)

            manifest = json.loads((restore / RESTORE_MANIFEST_NAME).read_text())
            source = Path(manifest["slots"]["restore_000"]["source"])
            original = np.asarray(Image.open(source).convert("RGB"))
            target = np.asarray(Image.open(restore / "restore_000.png").convert("RGB"))
            control = np.asarray(Image.open(restore / "control" / "restore_000.png"))

            np.testing.assert_array_equal(
                target, original, "o target tem de ser a original intocada")
            self.assertGreater(np.abs(control.astype(int) - original.astype(int)).mean(), 1.0,
                               "o control tem de ser a versão degradada")

    def test_control_and_target_keep_the_same_shape(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 120)
            restore = root / "dataset_restore"
            build_restore_dataset(base, restore, _FakeJob(), used=set(), count=6, workers=2)
            for target in restore.glob("restore_*.png"):
                with Image.open(target) as t, Image.open(restore / "control" / target.name) as c:
                    self.assertEqual(t.size, c.size, f"{target.name} desalinhou o par")

    def test_resume_skips_slots_already_done(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 120)
            restore = root / "dataset_restore"
            build_restore_dataset(base, restore, _FakeJob(), used=set(), count=8, workers=2)

            with mock.patch("trainero.style_rush.degrade.degrade_file") as never:
                result = build_restore_dataset(base, restore, _FakeJob(), used=set(),
                                               count=8, workers=2)
            never.assert_not_called()
            self.assertEqual(result["pairs"], 8)

    def test_one_unreadable_image_costs_its_slot_not_the_phase(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 120)
            restore = root / "dataset_restore"
            real = degrade_file
            calls = {"n": 0}

            def flaky(source, dest, **kw):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise OSError("cannot identify image file")
                return real(source, dest, **kw)

            with mock.patch("trainero.style_rush.degrade.degrade_file", flaky):
                job = _FakeJob()
                result = build_restore_dataset(base, restore, job, used=set(),
                                               count=5, workers=1)

            self.assertEqual(result["failed"], 1)
            self.assertEqual(result["pairs"], 4)
            self.assertTrue(any("⚠" in line for line in job.lines))

    def test_cancel_propagates(self):
        class _CancellingJob(_FakeJob):
            def check_cancel(self):
                raise Cancelled()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 120)
            with self.assertRaises(Cancelled):
                build_restore_dataset(base, root / "dataset_restore", _CancellingJob(),
                                      used=set(), count=4, workers=1)

    def test_empty_dataset_fails_the_job(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "dataset").mkdir()
            with self.assertRaises(JobFailed):
                build_restore_dataset(root / "dataset", root / "dataset_restore",
                                      _FakeJob(), used=set())

    def test_caption_is_rewritten_when_the_constant_changes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 120)
            restore = root / "dataset_restore"
            build_restore_dataset(base, restore, _FakeJob(), used=set(), count=3, workers=1)
            (restore / "restore_000.txt").write_text("caption velha")
            build_restore_dataset(base, restore, _FakeJob(), used=set(), count=3, workers=1)
            self.assertEqual((restore / "restore_000.txt").read_text(), RESTORE_CAPTION)


class TestStyleRushPipeline(unittest.TestCase):
    """Drives the real run_style_rush_training with only the expensive steps
    stubbed. Asserting on manually-built subsets is not enough: it leaves the
    call site free to stop wiring the restoration dataset with every test green."""

    def _run(self, td: Path):
        from trainero import training

        pdir = td / "projects" / "p"
        # 160 images so the 50 the conversion takes still leave 110 free for the
        # 100 restoration slots — the no-overlap assertion below needs the slack
        base = _make_dataset(pdir, 160)

        def fake_convert(base_dir, convert_dir, trigger, job, **kw):
            (convert_dir / "control").mkdir(parents=True, exist_ok=True)
            (convert_dir / ".style_rush.json").write_text(json.dumps({"slots": {
                f"slot_{i:02d}": {"status": "ok", "source": str(_paths(base_dir)[i])}
                for i in range(50)}}))
            return {"pairs": 50, "refused": 0, "failed": 0, "cost": 0.42}

        def fake_launch(model_key, cfg, project_dir, job, toml_name="train.toml"):
            (Path(cfg["output_dir"]) / "p-000001.safetensors").write_bytes(b"ckpt")

        job = _FakeJob()
        job.set_phases = lambda names: None
        job.start_phase = lambda name: None
        job.end_phase = lambda name, ok=True: None
        job.extra = {}

        with mock.patch.object(training, "PROJECTS_DIR", td / "projects"), \
             mock.patch.object(training, "ensure_engine"), \
             mock.patch.object(training, "ensure_models"), \
             mock.patch.object(training, "run_caches"), \
             mock.patch.object(training, "hf_username", return_value=""), \
             mock.patch.object(training, "gpu_info",
                               return_value={"name": "T", "vram_mb": 32768}), \
             mock.patch.object(training.sr, "build_convert_dataset", fake_convert), \
             mock.patch.object(training, "launch_training", fake_launch):
            training.run_style_rush_training(job, {"project": "p", "model": "flux-klein",
                                                   "trigger": "makima", "overrides": {}})
        return pdir, job

    def test_the_restoration_dataset_reaches_dataset_toml(self):
        with tempfile.TemporaryDirectory() as td:
            pdir, job = self._run(Path(td))

            text = (pdir / "dataset.toml").read_text()
            self.assertEqual(text.count("[[datasets]]"), 3,
                             "o dataset de restauração não chegou ao dataset.toml")
            self.assertIn(str(pdir / "dataset_restore"), text)
            self.assertIn(str(pdir / "dataset_restore" / "control"), text)
            self.assertEqual(job.extra["restore"]["pairs"], RESTORE_COUNT)
            self.assertEqual(job.extra["config_summary"]["restore"], RESTORE_COUNT)

    def test_the_pairs_are_actually_written_and_do_not_reuse_the_conversion(self):
        with tempfile.TemporaryDirectory() as td:
            pdir, _ = self._run(Path(td))

            restore = pdir / "dataset_restore"
            self.assertEqual(len(list(restore.glob("restore_*.png"))), RESTORE_COUNT)
            self.assertEqual(len(list((restore / "control").glob("restore_*.png"))),
                             RESTORE_COUNT)
            used = {str(p) for p in _paths(pdir / "dataset")[:50]}
            manifest = json.loads((restore / RESTORE_MANIFEST_NAME).read_text())
            picked = {e["source"] for e in manifest["slots"].values()}
            self.assertEqual(picked & used, set(),
                             "pegou imagens que a conversão já tinha usado")


class TestTrainingWiring(unittest.TestCase):
    def test_dataset_toml_gets_a_third_block_with_its_control_dir(self):
        """The restoration subset only trains if it reaches dataset.toml as its
        own [[datasets]] block carrying control_directory."""
        from trainero.training import image_subset, write_dataset_toml

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subsets = [
                image_subset(root / "dataset", root / "cache" / "images", 1),
                image_subset(root / "dataset_convert", root / "cache" / "convert", 1,
                             control_dir=root / "dataset_convert" / "control",
                             control_resolution=[1024, 1024]),
                image_subset(root / "dataset_restore", root / "cache" / "restore", 1,
                             control_dir=root / "dataset_restore" / "control",
                             control_resolution=[1024, 1024]),
            ]
            text = write_dataset_toml("qwen-image-edit", root / "dataset.toml", subsets,
                                      [1024, 1024], 1).read_text()

            self.assertEqual(text.count("[[datasets]]"), 3)
            self.assertIn("dataset_restore", text)
            self.assertIn(str(root / "dataset_restore" / "control"), text)
            self.assertEqual(text.count("control_directory"), 2)


if __name__ == "__main__":
    unittest.main()
