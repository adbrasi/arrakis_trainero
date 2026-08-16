"""Unit tests for the OpenRouter Images API client (payload shape only, no network)."""

import base64
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainero.imagegen import (MODEL, RefusedError, RetriableError, aspect_ratio_for,
                               build_payload, classify_http_error, decode_first_image)


class TestAspectRatio(unittest.TestCase):
    def test_square(self):
        self.assertEqual(aspect_ratio_for(1024, 1024), "1:1")
        self.assertEqual(aspect_ratio_for(900, 910), "1:1")

    def test_landscape_and_portrait(self):
        self.assertEqual(aspect_ratio_for(1536, 1024), "3:2")
        self.assertEqual(aspect_ratio_for(1024, 1536), "2:3")
        self.assertEqual(aspect_ratio_for(1920, 1080), "16:9")
        self.assertEqual(aspect_ratio_for(1080, 1920), "9:16")
        self.assertEqual(aspect_ratio_for(1024, 768), "4:3")
        self.assertEqual(aspect_ratio_for(768, 1024), "3:4")

    def test_ultrawide_uses_the_ratio_the_model_supports(self):
        # 21:9 is accepted by gpt-image-2 — snapping these to 16:9 would crop
        self.assertEqual(aspect_ratio_for(2560, 1080), "21:9")
        self.assertEqual(aspect_ratio_for(3440, 1440), "21:9")

    def test_beyond_the_widest_ratio_clamps_to_it(self):
        self.assertEqual(aspect_ratio_for(5000, 1000), "21:9")

    def test_zero_dimension_is_square(self):
        self.assertEqual(aspect_ratio_for(0, 0), "1:1")


class TestReferenceDownscale(unittest.TestCase):
    """The reference image is billed by pixel area, so its size is our lever."""

    def _img(self, td, size, name="a.png", fmt=None):
        from PIL import Image

        p = Path(td) / name
        Image.new("RGB", size).save(p, format=fmt)
        return p

    def test_oversized_reference_is_downscaled(self):
        import tempfile

        from trainero.imagegen import REFERENCE_MAX_EDGE, to_data_url

        with tempfile.TemporaryDirectory() as td:
            _url, w, h = to_data_url(self._img(td, (4096, 3072)))
            self.assertEqual(max(w, h), REFERENCE_MAX_EDGE)
            self.assertAlmostEqual(w / h, 4096 / 3072, places=2)  # ratio kept

    def test_small_reference_is_passed_through_untouched(self):
        import tempfile

        from trainero.imagegen import to_data_url

        with tempfile.TemporaryDirectory() as td:
            src = self._img(td, (640, 480), "a.jpg", fmt="JPEG")
            url, w, h = to_data_url(src)
            self.assertEqual((w, h), (640, 480))
            self.assertTrue(url.startswith("data:image/jpeg;base64,"))
            self.assertEqual(base64.b64decode(url.split(",", 1)[1]), src.read_bytes())

    def test_downscaled_payload_is_smaller_than_the_original(self):
        import tempfile

        from PIL import Image

        from trainero.imagegen import to_data_url

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "noise.png"
            Image.effect_noise((3000, 3000), 96).convert("RGB").save(p)
            url, _w, _h = to_data_url(p)
            sent = len(base64.b64decode(url.split(",", 1)[1]))
            self.assertLess(sent, p.stat().st_size)


class TestPayload(unittest.TestCase):
    def test_shape(self):
        p = build_payload("make it manga", "data:image/png;base64,AAAA", "3:2")
        self.assertEqual(p["model"], MODEL)
        self.assertEqual(p["prompt"], "make it manga")
        self.assertEqual(p["n"], 1)
        self.assertEqual(p["quality"], "low")
        self.assertEqual(p["moderation"], "low")
        self.assertEqual(p["aspect_ratio"], "3:2")
        self.assertEqual(p["input_references"],
                         [{"type": "image_url",
                           "image_url": {"url": "data:image/png;base64,AAAA"}}])

    def test_never_sends_size_or_resolution(self):
        p = build_payload("x", "data:image/png;base64,AAAA", "1:1")
        self.assertNotIn("size", p)
        self.assertNotIn("resolution", p)


class TestDecode(unittest.TestCase):
    def test_b64_json(self):
        raw = b"\x89PNG\r\n\x1a\nfake"
        body = {"data": [{"b64_json": base64.b64encode(raw).decode()}]}
        self.assertEqual(decode_first_image(body), raw)

    def test_data_url(self):
        raw = b"\x89PNG\r\n\x1a\nfake"
        url = "data:image/png;base64," + base64.b64encode(raw).decode()
        body = {"data": [{"url": url}]}
        self.assertEqual(decode_first_image(body), raw)

    def test_empty_data_raises(self):
        from trainero.imagegen import ImageGenError

        with self.assertRaises(ImageGenError):
            decode_first_image({"data": []})


class TestErrorClassification(unittest.TestCase):
    def test_moderation_is_refusal(self):
        exc = classify_http_error(400, '{"error":{"message":"rejected by the safety system"}}')
        self.assertIsInstance(exc, RefusedError)

    def test_moderation_variants(self):
        for msg in ("safety_violations detected", "Your request was flagged by moderation",
                    "content_policy_violation"):
            self.assertIsInstance(classify_http_error(400, msg), RefusedError, msg)

    def test_rate_limit_and_server_errors_are_retriable(self):
        for code in (429, 500, 502, 503, 504):
            self.assertIsInstance(classify_http_error(code, "boom"), RetriableError, code)

    def test_auth_error_is_fatal(self):
        exc = classify_http_error(401, "no key")
        self.assertNotIsInstance(exc, RetriableError)
        self.assertNotIsInstance(exc, RefusedError)


if __name__ == "__main__":
    unittest.main()
