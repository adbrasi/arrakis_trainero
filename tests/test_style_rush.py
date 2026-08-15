"""Unit tests for the Style Rush synthetic dataset pipeline (no GPU, no network)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainero.style_rush import (SLOT_COUNT, CAPTION_TEMPLATE, load_style_prompts,
                                 plan_slots)


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


if __name__ == "__main__":
    unittest.main()
