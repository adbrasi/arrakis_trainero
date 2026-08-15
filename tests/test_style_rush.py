"""Unit tests for the Style Rush synthetic dataset pipeline (no GPU, no network)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainero.imagegen import RefusedError, RetriableError
from trainero.style_rush import (SLOT_COUNT, CAPTION_TEMPLATE, MANIFEST_NAME,
                                 build_convert_dataset, load_style_prompts, plan_slots)


class _FakeJob:
    """Minimal stand-in for trainero.jobs.Job — collects log lines."""

    def __init__(self):
        self.lines = []

    def log(self, msg):
        self.lines.append(msg)

    def check_cancel(self):
        pass


def _png_bytes(color=b"\x00"):
    """A 1x1 PNG, enough for the pipeline to copy bytes around."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    return buf.getvalue()


def _make_dataset(root: Path, n: int) -> Path:
    from PIL import Image

    base = root / "dataset"
    base.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (64, 48)).save(base / f"img_{i:03d}.png")
        (base / f"img_{i:03d}.txt").write_text("makima, a girl")
    return base


class TestStylePrompts(unittest.TestCase):
    def test_fifty_distinct_prompts(self):
        prompts = load_style_prompts()
        self.assertEqual(len(prompts), SLOT_COUNT)
        self.assertEqual(len(set(prompts)), SLOT_COUNT, "prompts must be distinct")
        for p in prompts:
            self.assertTrue(p.strip(), "no blank prompt")

    def test_caption_template(self):
        self.assertEqual(
            CAPTION_TEMPLATE.format(trigger="makima"),
            "convert the style of this image to the makima style",
        )


class TestPlanSlots(unittest.TestCase):
    def _imgs(self, n):
        return [Path(f"/ds/img_{i:03d}.png") for i in range(n)]

    def test_always_fifty_slots(self):
        prompts = load_style_prompts()
        for n in (1, 2, 7, 50, 200):
            slots = plan_slots(self._imgs(n), prompts)
            self.assertEqual(len(slots), SLOT_COUNT, n)
            self.assertEqual([s["slot"] for s in slots],
                             [f"slot_{i:02d}" for i in range(SLOT_COUNT)], n)

    def test_each_slot_gets_a_distinct_prompt(self):
        prompts = load_style_prompts()
        slots = plan_slots(self._imgs(10), prompts)
        used = [s["prompt"] for s in slots]
        self.assertEqual(len(set(used)), SLOT_COUNT)
        self.assertEqual(set(used), set(prompts))

    def test_large_dataset_uses_distinct_images(self):
        slots = plan_slots(self._imgs(200), load_style_prompts())
        primaries = [s["sources"][0] for s in slots]
        self.assertEqual(len(set(primaries)), SLOT_COUNT)

    def test_small_dataset_wraps_around(self):
        slots = plan_slots(self._imgs(10), load_style_prompts())
        primaries = [s["sources"][0] for s in slots]
        self.assertEqual(len(set(primaries)), 10)
        # each image is reused 5 times, always with a different prompt
        self.assertEqual(len(set(s["prompt"] for s in slots)), SLOT_COUNT)

    def test_fallback_differs_from_primary(self):
        slots = plan_slots(self._imgs(10), load_style_prompts())
        for s in slots:
            self.assertEqual(len(s["sources"]), 2, s["slot"])
            self.assertNotEqual(s["sources"][0], s["sources"][1], s["slot"])

    def test_single_image_has_no_fallback(self):
        slots = plan_slots(self._imgs(1), load_style_prompts())
        for s in slots:
            self.assertEqual(len(s["sources"]), 1)

    def test_deterministic(self):
        prompts = load_style_prompts()
        a = plan_slots(self._imgs(37), prompts)
        b = plan_slots(self._imgs(37), prompts)
        self.assertEqual(a, b)

    def test_empty_dataset_raises(self):
        with self.assertRaises(ValueError):
            plan_slots([], load_style_prompts())


class TestBuildConvertDataset(unittest.TestCase):
    def test_happy_path_writes_fifty_pairs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 60)
            convert = root / "dataset_convert"
            calls = []

            def fake_generate(prompt, image_path, timeout=300.0):
                calls.append((prompt, str(image_path)))
                return _png_bytes(), 0.011

            result = build_convert_dataset(base, convert, "makima", _FakeJob(),
                                           generate=fake_generate, workers=2)

            self.assertEqual(result["pairs"], SLOT_COUNT)
            self.assertEqual(result["refused"], 0)
            self.assertAlmostEqual(result["cost"], 0.011 * SLOT_COUNT, places=4)
            self.assertEqual(len(calls), SLOT_COUNT)
            targets = sorted(p.name for p in convert.glob("slot_*.png"))
            controls = sorted(p.name for p in (convert / "control").glob("slot_*.png"))
            self.assertEqual(len(targets), SLOT_COUNT)
            self.assertEqual(targets, controls)
            caption = (convert / "slot_00.txt").read_text()
            self.assertEqual(caption, "convert the style of this image to the makima style")

    def test_refusal_retries_with_a_different_image(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 10)
            convert = root / "dataset_convert"
            seen = []

            def fake_generate(prompt, image_path, timeout=300.0):
                seen.append(str(image_path))
                # the very first attempt is refused; the fallback image works
                if len(seen) == 1:
                    raise RefusedError("moderação recusou a imagem")
                return _png_bytes(), 0.011

            result = build_convert_dataset(base, convert, "makima", _FakeJob(),
                                           generate=fake_generate, workers=1)
            self.assertEqual(result["pairs"], SLOT_COUNT)
            self.assertEqual(result["refused"], 1)
            self.assertNotEqual(seen[0], seen[1], "retry must use a different image")

    def test_two_refusals_drop_the_slot(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 10)
            convert = root / "dataset_convert"

            def fake_generate(prompt, image_path, timeout=300.0):
                raise RefusedError("moderação recusou a imagem")

            from trainero.jobs import JobFailed

            with self.assertRaises(JobFailed):
                build_convert_dataset(base, convert, "makima", _FakeJob(),
                                      generate=fake_generate, workers=1)

    def test_retriable_error_retries_the_same_image(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 10)
            convert = root / "dataset_convert"
            attempts = {}

            def fake_generate(prompt, image_path, timeout=300.0):
                key = str(image_path)
                attempts[key] = attempts.get(key, 0) + 1
                if attempts[key] == 1:
                    raise RetriableError("HTTP 503")
                return _png_bytes(), 0.011

            with mock.patch("trainero.style_rush.time.sleep"):
                result = build_convert_dataset(base, convert, "makima", _FakeJob(),
                                               generate=fake_generate, workers=1)
            self.assertEqual(result["pairs"], SLOT_COUNT)
            self.assertEqual(result["refused"], 0)

    def test_resume_does_not_regenerate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 60)
            convert = root / "dataset_convert"

            def fake_generate(prompt, image_path, timeout=300.0):
                return _png_bytes(), 0.011

            build_convert_dataset(base, convert, "makima", _FakeJob(),
                                  generate=fake_generate, workers=2)

            second_calls = []

            def counting_generate(prompt, image_path, timeout=300.0):
                second_calls.append(prompt)
                return _png_bytes(), 0.011

            result = build_convert_dataset(base, convert, "makima", _FakeJob(),
                                           generate=counting_generate, workers=2)
            self.assertEqual(second_calls, [])
            self.assertEqual(result["pairs"], SLOT_COUNT)

    def test_manifest_records_each_slot(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 60)
            convert = root / "dataset_convert"

            def fake_generate(prompt, image_path, timeout=300.0):
                return _png_bytes(), 0.011

            build_convert_dataset(base, convert, "makima", _FakeJob(),
                                  generate=fake_generate, workers=2)
            manifest = json.loads((convert / MANIFEST_NAME).read_text())
            self.assertEqual(len(manifest["slots"]), SLOT_COUNT)
            entry = manifest["slots"]["slot_00"]
            self.assertEqual(entry["status"], "ok")
            self.assertTrue(entry["prompt"])
            self.assertTrue(entry["source"])

    def test_empty_base_dataset_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = root / "dataset"
            base.mkdir()
            from trainero.jobs import JobFailed

            with self.assertRaises(JobFailed):
                build_convert_dataset(base, root / "dataset_convert", "makima", _FakeJob(),
                                      generate=lambda *a, **k: (_png_bytes(), 0.0))


class TestStyleRushGuards(unittest.TestCase):
    def _job(self):
        job = _FakeJob()
        job.set_phases = lambda names: None
        job.start_phase = lambda name: None
        job.end_phase = lambda name, ok=True: None
        job.extra = {}
        return job

    def test_unsupported_model_is_rejected(self):
        from trainero.jobs import JobFailed
        from trainero.training import run_style_rush_training

        with self.assertRaises(JobFailed) as ctx:
            run_style_rush_training(self._job(), {"project": "p", "model": "wan-22",
                                                  "trigger": "t", "overrides": {}})
        self.assertIn("control", str(ctx.exception).lower())

    def test_empty_trigger_is_rejected(self):
        from trainero.jobs import JobFailed
        from trainero.training import run_style_rush_training

        with self.assertRaises(JobFailed) as ctx:
            run_style_rush_training(self._job(), {"project": "p", "model": "flux-klein",
                                                  "trigger": "  ", "overrides": {}})
        self.assertIn("trigger", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
