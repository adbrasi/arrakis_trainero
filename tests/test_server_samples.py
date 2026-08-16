"""Unit tests for the sample-listing helpers in server.py (no HTTP)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server


class TestParseSampleName(unittest.TestCase):
    def test_musubi_pattern(self):
        # <output_name>_e{epoch:06d}_{idx:02d}_{timestamp}_{seed}.png
        self.assertEqual(server.parse_sample_name("makima_e000003_01_20260815120000_42.png"),
                         (3, 1))

    def test_step_based_pattern_without_epoch(self):
        self.assertEqual(server.parse_sample_name("makima_000500_00_20260815120000.png"),
                         (-1, 0))

    def test_unknown_name(self):
        self.assertEqual(server.parse_sample_name("whatever.png"), (-1, -1))

    def test_digits_in_the_project_slug_do_not_win(self):
        # "makima_202601_01_v2" is a legal slug and matches the pattern too
        self.assertEqual(
            server.parse_sample_name("makima_202601_01_v2_e000003_00_20260815120000_42.png"),
            (3, 0))


class TestListSamples(unittest.TestCase):
    def test_newest_first(self):
        with tempfile.TemporaryDirectory() as td:
            sample_dir = Path(td) / "sample"
            sample_dir.mkdir()
            for epoch in (1, 2, 3):
                (sample_dir / f"m_e{epoch:06d}_00_2026081512000{epoch}_42.png").write_bytes(b"x")
            names = [s["name"] for s in server.list_samples(sample_dir)]
            self.assertEqual(len(names), 3)
            self.assertIn("e000003", names[0])
            self.assertIn("e000001", names[-1])

    def test_missing_dir_is_empty(self):
        self.assertEqual(server.list_samples(Path("/nope/sample")), [])


class TestSafeSampleName(unittest.TestCase):
    def test_rejects_traversal(self):
        for bad in ("../secret.png", "a/b.png", "", "..", "x.txt"):
            self.assertFalse(server.safe_sample_name(bad), bad)

    def test_rejects_nul_byte(self):
        # open() answers a NUL with ValueError, which is not an OSError and
        # would escape the handler's except clause
        self.assertFalse(server.safe_sample_name("a\x00.png"))
        self.assertFalse(server.safe_sample_name("a.png\x00"))

    def test_accepts_plain_png(self):
        self.assertTrue(server.safe_sample_name("m_e000001_00_x_42.png"))


class TestDatasetThumbs(unittest.TestCase):
    def test_lists_images_and_total(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for i in range(30):
                Image.new("RGB", (64, 48)).save(d / f"img_{i:03d}.png")
            (d / "img_000.txt").write_text("caption")  # captions are not images
            names, total = server.dataset_thumbs(d, limit=24)
            self.assertEqual(total, 30)
            self.assertEqual(len(names), 24)
            self.assertTrue(all(n.endswith(".png") for n in names))

    def test_missing_dir_is_empty(self):
        self.assertEqual(server.dataset_thumbs(Path("/nope/ds")), ([], 0))

    def test_thumb_is_small(self):
        import io

        from PIL import Image

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "big.png"
            Image.new("RGB", (4000, 3000)).save(src)
            data = server.render_thumb(src, edge=220)
            with Image.open(io.BytesIO(data)) as im:
                self.assertLessEqual(max(im.size), 220)
                self.assertEqual(im.format, "JPEG")
            self.assertLess(len(data), src.stat().st_size)


if __name__ == "__main__":
    unittest.main()
