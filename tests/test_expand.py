"""Expand mode: prompt building, judge parsing, payloads and the run loop (no network)."""

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainero import expand, imagegen
from trainero.jobs import JobFailed


class _Job:
    def __init__(self):
        self.lines, self.extra = [], {}

    def log(self, msg):
        self.lines.append(msg)

    def check_cancel(self):
        pass


def _png(seed: int, size=(64, 64)) -> bytes:
    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def _refs(d: Path, n: int):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (d / f"ref{i}.png").write_bytes(_png(1000 + i))


def _verdict(identity=9, clean=9, style=9, single=True):
    return {"same_identity": identity, "clean": clean, "same_style": style,
            "follows_prompt": 8, "single_subject": single, "issues": "none", "cost": 0.001}


class TestPrompts(unittest.TestCase):
    def test_about_thirty_variations_with_canonical_outfit_near_40_percent(self):
        self.assertGreaterEqual(len(expand.VARIATIONS), 28)
        canonical = sum(1 for v in expand.VARIATIONS if v["outfit"] is None)
        self.assertAlmostEqual(canonical / len(expand.VARIATIONS), 0.4, delta=0.07)
        texts = [v["text"] for v in expand.VARIATIONS]
        self.assertEqual(len(texts), len(set(texts)))
        for word in ("profile", "behind", "high-angle", "low-angle", "close-up", "full body",
                     "waist-up"):
            self.assertTrue(any(word in t for t in texts), word)

    def test_prompt_pins_identity_and_medium_and_changes_only_the_variation(self):
        v = {"text": "side profile facing left", "aspect": "1:1", "outfit": None}
        p = expand.build_prompt(v, "anime girl", 3)
        self.assertIn("exact same anime girl", p)
        self.assertIn("3 reference images", p)
        self.assertIn("side profile facing left", p)
        self.assertIn("keep exactly the outfit", p)
        self.assertIn("photograph stays a realistic photograph", p)
        self.assertIn("anime", p)
        p2 = expand.build_prompt({"text": "x", "aspect": "1:1", "outfit": "a red coat"}, "", 1)
        self.assertIn("the reference image", p2)
        self.assertIn("a red coat", p2)
        self.assertIn("same character", p2)

    def test_reference_window_rotates(self):
        refs = [Path(f"r{i}") for i in range(5)]
        self.assertEqual(expand.refs_for_attempt(refs, 0, 3), refs[:3])
        self.assertEqual(expand.refs_for_attempt(refs, 4, 3), [refs[4], refs[0], refs[1]])
        self.assertEqual(expand.refs_for_attempt(refs[:2], 7, 3), refs[:2])


class TestJudgeParsing(unittest.TestCase):
    def _body(self, content, cost=0.002):
        return {"choices": [{"message": {"content": content}}], "usage": {"cost": cost}}

    def test_plain_json(self):
        v = expand.parse_judge(self._body(json.dumps(
            {"same_identity": 9, "clean": 8, "same_style": 10, "follows_prompt": 7,
             "single_subject": True, "issues": "none"})))
        self.assertEqual((v["same_identity"], v["clean"], v["same_style"]), (9, 8, 10))
        self.assertTrue(v["single_subject"])
        self.assertAlmostEqual(v["cost"], 0.002)

    def test_fenced_json_and_out_of_range_scores(self):
        v = expand.parse_judge(self._body(
            '```json\n{"same_identity": 12, "clean": -3, "same_style": "7", '
            '"follows_prompt": 6.6, "single_subject": "false", "issues": "two people"}\n```'))
        self.assertEqual(v["same_identity"], 10)
        self.assertEqual(v["clean"], 0)
        self.assertEqual(v["same_style"], 7)
        self.assertEqual(v["follows_prompt"], 7)
        self.assertFalse(v["single_subject"])

    def test_garbage_raises(self):
        for body in ({}, self._body("no json here"), self._body('{"same_identity": "high"}')):
            with self.assertRaises(expand.JudgeError):
                expand.parse_judge(body)

    def test_accept_thresholds(self):
        self.assertEqual(expand.accept(_verdict()), (True, ""))
        self.assertFalse(expand.accept(_verdict(identity=6))[0])
        self.assertIn("identidade", expand.accept(_verdict(identity=6))[1])
        self.assertFalse(expand.accept(_verdict(style=3))[0])
        self.assertFalse(expand.accept(_verdict(clean=4))[0])
        self.assertFalse(expand.accept(_verdict(single=False))[0])

    def test_judge_payload_shape(self):
        p = expand.build_judge_payload(["data:a", "data:b"], "data:c", "close-up", model="m/x")
        self.assertEqual(p["model"], "m/x")
        parts = p["messages"][0]["content"]
        urls = [x["image_url"]["url"] for x in parts if x["type"] == "image_url"]
        self.assertEqual(urls, ["data:a", "data:b", "data:c"])   # candidate last
        self.assertIn("close-up", parts[0]["text"])
        self.assertTrue(p["response_format"]["json_schema"]["strict"])
        self.assertEqual(p["reasoning"], {"effort": "low"})

    def test_judge_over_mocked_http(self):
        body = self._body(json.dumps({"same_identity": 9, "clean": 9, "same_style": 9,
                                      "follows_prompt": 9, "single_subject": True,
                                      "issues": "none"}))
        sent = {}

        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=0):
            sent["url"] = req.full_url
            sent["payload"] = json.loads(req.data.decode())
            return _Resp(json.dumps(body).encode())

        with tempfile.TemporaryDirectory() as td:
            _refs(Path(td), 1)
            with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "k"}), \
                 mock.patch.object(expand.urllib.request, "urlopen", fake_urlopen):
                v = expand.judge([Path(td) / "ref0.png"], _png(5), "smile")
        self.assertEqual(v["same_identity"], 9)
        self.assertEqual(sent["url"], expand.CHAT_URL)
        self.assertEqual(len([x for x in sent["payload"]["messages"][0]["content"]
                              if x["type"] == "image_url"]), 2)


class TestReferencePayload(unittest.TestCase):
    NANO = {"resolution": {"type": "enum", "values": ["1K", "2K", "4K"]},
            "aspect_ratio": {"type": "enum", "values": ["1:1", "2:3", "3:2", "3:4", "4:3"]},
            "input_references": {"type": "range", "min": 0, "max": 2}}
    GPT = {"aspect_ratio": {"values": ["1:1", "2:3", "3:4"]},
           "quality": {"values": ["low", "medium", "high"]},
           "input_references": {"max": 16}}

    def test_declared_parameters_only(self):
        p = imagegen.build_reference_payload("google/gemini-nano-banana-2.1", "x",
                                             ["a", "b", "c"], "3:4", self.NANO, quality="medium")
        self.assertEqual(p["resolution"], "1K")
        self.assertNotIn("quality", p)
        self.assertNotIn("moderation", p)
        self.assertEqual(len(p["input_references"]), 2)   # model max

    def test_openai_gets_quality_and_moderation(self):
        p = imagegen.build_reference_payload("openai/gpt-image-2", "x", ["a"], "3:4",
                                             self.GPT, quality="medium")
        self.assertEqual(p["quality"], "medium")
        self.assertEqual(p["moderation"], "low")
        self.assertNotIn("resolution", p)
        # unknown parameters (listing unreachable): the verified OpenAI shape
        p = imagegen.build_reference_payload("openai/gpt-image-2", "x", ["a"], "3:4", None, "low")
        self.assertEqual(p["quality"], "low")

    def test_unsupported_ratio_snaps_to_nearest(self):
        self.assertEqual(imagegen.nearest_supported_ratio("4:3", ["1:1", "3:2", "2:3"]), "3:2")
        self.assertEqual(imagegen.nearest_supported_ratio("2:3", ["1:1", "2:3"]), "2:3")
        self.assertEqual(imagegen.nearest_supported_ratio("2:3", None), "2:3")


class TestExpandRun(unittest.TestCase):
    def test_reaches_target_and_filters(self):
        """Every third candidate fails identity; one is a copy of the reference."""
        calls = {"n": 0}

        def generate(prompt, refs, aspect):
            calls["n"] += 1
            if calls["n"] == 2:
                return refs[0].read_bytes(), 0.02   # the model returned the reference
            return _png(calls["n"]), 0.02

        def judge_fn(refs, png, change):
            return _verdict(identity=4 if calls["n"] % 3 == 0 else 9)

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _refs(d, 2)
            job = _Job()
            totals = expand.expand_dataset(d, job, target=5, generate=generate,
                                           judge_fn=judge_fn, workers=1)
            gens = expand.generated_images(d)
            self.assertEqual(len(gens), 5)
            self.assertEqual(totals["accepted"], 5)
            self.assertGreaterEqual(totals["rejected"], 1)
            self.assertEqual(totals["duplicates"], 1)
            self.assertTrue(all(g.name.startswith("gen_") for g in gens))
            self.assertGreater(totals["cost"], 0)
            manifest = json.loads((d / expand.MANIFEST_NAME).read_text())
            self.assertEqual(len(manifest["attempts"]), totals["attempts"])
            # references untouched, and generated images are not references
            self.assertEqual(len(expand.reference_images(d)), 2)

    def test_ceiling_stops_a_hopeless_run(self):
        def generate(prompt, refs, aspect):
            return _png(len(prompt) + np.random.randint(1e6)), 0.01

        with tempfile.TemporaryDirectory() as td, \
                mock.patch.dict("os.environ", {"EXPAND_MAX_ATTEMPTS": "7"}):
            d = Path(td)
            _refs(d, 1)
            job = _Job()
            totals = expand.expand_dataset(d, job, target=4, generate=generate,
                                           judge_fn=lambda *a: _verdict(identity=2), workers=3)
            self.assertEqual(totals["attempts"], 7)
            self.assertEqual(totals["accepted"], 0)
            self.assertTrue(any("meta não atingida" in l for l in job.lines))

    def test_resume_counts_existing_and_never_overshoots(self):
        seq = iter(range(100))

        def generate(prompt, refs, aspect):
            return _png(next(seq)), 0.01

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _refs(d, 1)
            expand.expand_dataset(d, _Job(), target=3, generate=generate,
                                  judge_fn=lambda *a: _verdict(), workers=8)
            self.assertEqual(len(expand.generated_images(d)), 3)
            totals = expand.expand_dataset(d, _Job(), target=5, generate=generate,
                                           judge_fn=lambda *a: _verdict(), workers=8)
            self.assertEqual(totals["existing"], 3)
            self.assertEqual(totals["attempts"], 2)
            self.assertEqual(len(expand.generated_images(d)), 5)
            names = [g.name for g in expand.generated_images(d)]
            self.assertEqual(len(names), len(set(names)))

    def test_dead_account_fails_fast(self):
        def generate(prompt, refs, aspect):
            raise imagegen.AccountError("sem crédito")

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            _refs(d, 1)
            with self.assertRaises(JobFailed) as ctx:
                expand.expand_dataset(d, _Job(), target=10, generate=generate,
                                      judge_fn=lambda *a: _verdict(), workers=2)
            self.assertIn("sem crédito", str(ctx.exception))

    def test_needs_references(self):
        with tempfile.TemporaryDirectory() as td, self.assertRaises(JobFailed):
            expand.expand_dataset(Path(td), _Job(), target=3,
                                  generate=lambda *a: (b"", 0), judge_fn=lambda *a: _verdict())

    def test_judge_failure_keeps_the_image_out(self):
        def judge_fn(*a):
            raise expand.JudgeError("bad")

        with tempfile.TemporaryDirectory() as td, \
                mock.patch.dict("os.environ", {"EXPAND_MAX_ATTEMPTS": "3"}):
            d = Path(td)
            _refs(d, 1)
            totals = expand.expand_dataset(d, _Job(), target=2,
                                           generate=lambda p, r, a: (_png(7), 0.01),
                                           judge_fn=judge_fn, workers=1)
            self.assertEqual(totals["accepted"], 0)
            self.assertEqual(expand.generated_images(d), [])

    def test_dhash_sees_copies_not_different_images(self):
        a, b = _png(1, (256, 256)), _png(2, (256, 256))
        img = Image.open(io.BytesIO(a))
        resized = io.BytesIO()
        img.resize((200, 200)).save(resized, "PNG")
        self.assertTrue(expand.is_duplicate(expand.dhash(resized.getvalue()), [expand.dhash(a)]))
        self.assertFalse(expand.is_duplicate(expand.dhash(b), [expand.dhash(a)]))


if __name__ == "__main__":
    unittest.main()
