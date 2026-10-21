"""What a beginner actually does to the server, one scenario per test.

Every test drives the real handler over HTTP (see LiveServer). The scenarios
come from an audit of what went wrong silently: wrong captions after a second
upload, a dropped connection instead of an error, another project's gallery
under a running job, one click captioning only half a slider.
"""

import json
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server
from test_server_guards import LiveServer
from trainero import config, jobs


def _png_bytes(color=(10, 20, 30)) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


class TestUploadNeverOverwrites(LiveServer):
    def upload(self, name, data):
        return self._call(f"/api/dataset/file?side=pos&name={name}", raw=data)

    def test_a_second_image_with_the_same_name_is_renamed_not_replaced(self):
        """Phones name every photo image.jpg; a second batch reuses 1.png. The
        old caption used to stay beside the new picture."""
        self.upload("1.png", _png_bytes((1, 1, 1)))
        self.upload("1.txt", b"a red car")
        code, body = self.upload("1.png", _png_bytes((2, 2, 2)))
        self.assertEqual(code, 200)
        self.assertEqual(body["name"], "1_2.png")
        ds = server.dataset_dir("pos")
        self.assertEqual((ds / "1.txt").read_text(), "a red car")
        self.assertEqual((ds / "1.png").read_bytes(), _png_bytes((1, 1, 1)))
        self.assertFalse((ds / "1_2.txt").exists())

    def test_the_same_file_twice_is_a_no_op(self):
        self.upload("a.png", _png_bytes())
        code, body = self.upload("a.png", _png_bytes())
        self.assertEqual(code, 200)
        self.assertTrue(body.get("duplicate"))
        self.assertEqual(body["stats"]["items"], 1)

    def test_names_that_escape_or_hide_are_refused(self):
        for bad in ("..", ".", ".hidden.png", "%2e%2e", "a%2Fb.png"):
            code, body = self.upload(bad, b"x")
            self.assertEqual(code, 400, bad)
            self.assertIn("inválido", body.get("error", ""), bad)


class TestErrorsAreAnswersNotDroppedConnections(LiveServer):
    def test_a_json_body_that_is_not_an_object(self):
        code, body = self._call("/api/project", raw=b"[1, 2]")
        self.assertEqual(code, 400)
        self.assertIn("nome", body["error"])

    def test_garbage_json(self):
        code, _ = self._call("/api/train", raw=b"{nope")
        self.assertEqual(code, 400)

    def test_an_unknown_mode_is_refused(self):
        code, body = self._call("/api/train", {"model": "flux-klein", "mode": "turbo"})
        self.assertEqual(code, 400)
        self.assertIn("modo", body["error"])

    def test_an_unexpected_exception_becomes_a_500_json(self):
        with mock.patch.object(server.Handler, "_status", side_effect=RuntimeError("boom")):
            import urllib.error
            import urllib.request

            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/status", timeout=5)
            self.assertEqual(ctx.exception.code, 500)
            self.assertIn("boom", json.loads(ctx.exception.read())["error"])


class TestCaptionsRoute(LiveServer):
    def _dataset(self, side, names, captioned=()):
        d = server.dataset_dir(side)
        d.mkdir(parents=True, exist_ok=True)
        for n in names:
            (d / n).write_bytes(b"x")
        for n in captioned:
            (d / n).with_suffix(".txt").write_text("c")
        return d

    def _run(self, payload):
        seen = {}
        done = threading.Event()

        def fake(job, dirs, **kw):
            seen.update(dirs=[d.name for d in dirs], **kw)
            done.set()

        with mock.patch.object(server, "caption_datasets", fake):
            code, body = self._call("/api/captions", payload)
            if code == 202:
                done.wait(5)
        return code, body, seen

    def test_one_click_captions_both_slider_sides(self):
        self._dataset("pos", ["a.png"])
        self._dataset("neg", ["b.png"])
        code, _, seen = self._run({"sides": ["pos", "neg"]})
        self.assertEqual(code, 202)
        self.assertEqual(seen["dirs"], ["dataset", "dataset_neg"])

    def test_a_side_with_nothing_missing_is_left_out(self):
        self._dataset("pos", ["a.png"], captioned=["a.png"])
        self._dataset("neg", ["b.png"])
        _, _, seen = self._run({"sides": ["pos", "neg"]})
        self.assertEqual(seen["dirs"], ["dataset_neg"])

    def test_nothing_to_do_is_an_error_not_an_empty_job(self):
        self._dataset("pos", ["a.png"], captioned=["a.png"])
        code, body, _ = self._run({"sides": ["pos"]})
        self.assertEqual(code, 400)
        self.assertIsNone(jobs.current())

    def test_the_kind_and_trigger_of_the_project_are_used(self):
        self._dataset("pos", ["a.png"])
        self._call("/api/project", {"name": "guardtest", "trigger": "zz", "kind": "character"})
        _, _, seen = self._run({"sides": ["pos"], "model": "anima"})
        self.assertEqual((seen["kind"], seen["trigger"], seen["model_key"]), ("character", "zz", "anima"))

    def test_an_unknown_profile_is_refused(self):
        self._dataset("pos", ["a.png"])
        code, _, _ = self._run({"profile": "nope"})
        self.assertEqual(code, 400)


class TestKindAndCaptionPlan(LiveServer):
    def test_the_plan_follows_the_kind_and_the_model(self):
        d = server.dataset_dir("pos")
        (d / "a.png").write_bytes(b"x")
        self._call("/api/project", {"name": "guardtest", "kind": "style"})
        s = json.loads(self._get("/api/status"))
        self.assertEqual(s["kind"], "style")
        self.assertEqual(s["caption_plan"]["anima"], "anima-style")
        self.assertEqual(s["caption_plan"]["flux-klein"], "generic-style")

    def test_an_explicit_profile_wins(self):
        (server.dataset_dir("pos") / "a.png").write_bytes(b"x")
        self._call("/api/project", {"name": "guardtest", "kind": "style", "caption_profile": "default"})
        s = json.loads(self._get("/api/status"))
        self.assertEqual(set(s["caption_plan"].values()), {"default"})

    def test_an_unknown_kind_is_ignored(self):
        self._call("/api/project", {"name": "guardtest", "kind": "banana"})
        self.assertEqual(json.loads(self._get("/api/status"))["kind"], "")


class TestTheGalleryFollowsTheJob(LiveServer):
    def test_samples_come_from_the_running_jobs_project(self):
        """Typing another project name mid-training used to swap the gallery
        under the progress card for the other project's (or nothing)."""
        sample = config.PROJECTS_DIR / "guardtest" / "output" / "sample"
        sample.mkdir(parents=True)
        (sample / "guardtest_e000001_00_20260101.png").write_bytes(_png_bytes())
        gate = threading.Event()
        jobs.start("train", "t", config.PROJECTS_DIR / "guardtest" / "logs" / "t.log",
                   lambda _j: gate.wait(20), project="guardtest")
        self.addCleanup(gate.set)
        self._call("/api/project", {"name": "outro"})
        samples = json.loads(self._get("/api/samples"))["samples"]
        self.assertEqual([s["epoch"] for s in samples], [1])


if __name__ == "__main__":
    unittest.main()


class TestEditPairsInStatus(LiveServer):
    def test_status_says_how_the_triplets_pair(self):
        d = server.dataset_dir("pos")
        for i in range(2):
            (d / f"r{i}.png").write_bytes(_png_bytes())
            (d / "control").mkdir(exist_ok=True)
            (d / "control" / f"r{i}_0.png").write_bytes(_png_bytes())
            (d / "control" / f"r{i}_1.png").write_bytes(_png_bytes())
        (d / "solo.png").write_bytes(_png_bytes((9, 9, 9)))
        ep = json.loads(self._get("/api/status"))["edit_pairs"]
        self.assertEqual((ep["paired"], ep["unpaired"], ep["refs"]), (2, 1, [2]))
        self.assertEqual(ep["unpaired_sample"], ["solo.png"])

    def test_a_control_upload_replaces_instead_of_renaming(self):
        """A control is found by its name; renaming it would unpair it."""
        self._call("/api/dataset/file?side=pos&name=a.png", raw=_png_bytes())
        self._call("/api/dataset/file?side=pos&name=a_0.png&dir=control", raw=_png_bytes((1, 1, 1)))
        code, body = self._call("/api/dataset/file?side=pos&name=a_0.png&dir=control",
                                raw=_png_bytes((2, 2, 2)))
        self.assertEqual((code, body["name"]), (200, "a_0.png"))
        self.assertEqual((server.dataset_dir("pos") / "control" / "a_0.png").read_bytes(),
                         _png_bytes((2, 2, 2)))
