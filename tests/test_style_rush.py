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


#: originals are 64x48, the stub the fake generator returns is 8x8 — the size is
#: what tells control (generated) from target (original) apart on disk.
SOURCE_SIZE = (64, 48)
GENERATED_SIZE = (8, 8)


def _make_dataset(root: Path, n: int) -> Path:
    from PIL import Image

    base = root / "dataset"
    base.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        # mixed extensions on purpose: real datasets are not all PNG
        ext = ".png" if i % 2 == 0 else ".jpg"
        Image.new("RGB", SOURCE_SIZE).save(base / f"img_{i:03d}{ext}")
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

    def test_control_is_the_generated_image_and_target_is_the_original(self):
        """Swapping these two teaches the LoRA the inverse conversion — 50 API
        calls and a whole training run wasted, with no symptom until inference."""
        from PIL import Image

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 60)
            convert = root / "dataset_convert"

            def fake_generate(prompt, image_path, timeout=300.0):
                return _png_bytes(), 0.011

            build_convert_dataset(base, convert, "makima", _FakeJob(),
                                  generate=fake_generate, workers=2)
            with Image.open(convert / "control" / "slot_00.png") as im:
                self.assertEqual(im.size, GENERATED_SIZE, "control = saída do GPT Image")
            with Image.open(convert / "slot_00.png") as im:
                self.assertEqual(im.size, SOURCE_SIZE, "target = imagem original do dono")

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
            attempts, calls, failed_at = {}, [], []

            def fake_generate(prompt, image_path, timeout=300.0):
                key = str(image_path)
                calls.append((prompt, key))
                attempts[key] = attempts.get(key, 0) + 1
                if attempts[key] == 1:  # every image fails once, then works
                    failed_at.append(len(calls) - 1)
                    raise RetriableError("HTTP 503")
                return _png_bytes(), 0.011

            with mock.patch("trainero.style_rush.time.sleep"):
                result = build_convert_dataset(base, convert, "makima", _FakeJob(),
                                               generate=fake_generate, workers=1)
            self.assertEqual(result["pairs"], SLOT_COUNT)
            self.assertEqual(result["refused"], 0)
            self.assertEqual(len(failed_at), 10)  # one per distinct image
            # a transient error retries the SAME image; falling through to the
            # moderation fallback would be a different (wrong) recovery
            for i in failed_at:
                self.assertEqual(calls[i + 1], calls[i], f"retry {i} trocou de imagem")

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

    def test_cancelling_keeps_what_was_already_paid_for(self):
        """The owner cancelling mid-phase must not throw away the manifest —
        without it the next run regenerates all 50 slots and pays again."""
        from trainero.jobs import Cancelled

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 60)
            convert = root / "dataset_convert"
            made = []

            class _CancelJob(_FakeJob):
                def check_cancel(self):
                    if len(made) >= 10:
                        raise Cancelled()

            def fake_generate(prompt, image_path, timeout=300.0):
                made.append(prompt)
                return _png_bytes(), 0.011

            with self.assertRaises(Cancelled):
                build_convert_dataset(base, convert, "makima", _CancelJob(),
                                      generate=fake_generate, workers=1)

            manifest = json.loads((convert / MANIFEST_NAME).read_text())
            self.assertEqual(len(manifest["slots"]), 10)

            second = []

            def counting_generate(prompt, image_path, timeout=300.0):
                second.append(prompt)
                return _png_bytes(), 0.011

            result = build_convert_dataset(base, convert, "makima", _FakeJob(),
                                           generate=counting_generate, workers=1)
            self.assertEqual(len(second), SLOT_COUNT - 10, "regerou o que já estava pago")
            self.assertEqual(result["pairs"], SLOT_COUNT)

    def test_one_broken_image_costs_its_slot_not_the_phase(self):
        """Pillow cannot open every extension in IMAGE_EXTS (.avif). A single
        unreadable file must not abort the other 49 slots."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 60)
            convert = root / "dataset_convert"
            first = sorted(p for p in base.iterdir() if p.suffix in (".png", ".jpg"))[0]

            def fake_generate(prompt, image_path, timeout=300.0):
                if Path(image_path) == first:
                    raise OSError("cannot identify image file")
                return _png_bytes(), 0.011

            result = build_convert_dataset(base, convert, "makima", _FakeJob(),
                                           generate=fake_generate, workers=2)
            self.assertEqual(result["failed"], 1)
            self.assertEqual(result["pairs"], SLOT_COUNT - 1)
            manifest = json.loads((convert / MANIFEST_NAME).read_text())
            bad = [e for e in manifest["slots"].values() if e["status"] == "failed"]
            self.assertEqual(len(bad), 1)
            self.assertIn("OSError", bad[0]["error"])

    def test_changing_the_trigger_rewrites_the_captions(self):
        """Slots are not regenerated on resume, but the caption carries the
        trigger word — leaving it stale would train the wrong token."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 60)
            convert = root / "dataset_convert"

            def fake_generate(prompt, image_path, timeout=300.0):
                return _png_bytes(), 0.011

            build_convert_dataset(base, convert, "makima", _FakeJob(),
                                  generate=fake_generate, workers=2)
            calls = []

            def counting_generate(prompt, image_path, timeout=300.0):
                calls.append(prompt)
                return _png_bytes(), 0.011

            build_convert_dataset(base, convert, "outro_estilo", _FakeJob(),
                                  generate=counting_generate, workers=2)
            self.assertEqual(calls, [], "não deve pagar a API de novo")
            for slot in ("slot_00", "slot_49"):
                self.assertEqual((convert / f"{slot}.txt").read_text(),
                                 "convert the style of this image to the outro_estilo style")

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
