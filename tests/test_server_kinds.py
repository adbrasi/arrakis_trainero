"""Server side of the Krea 2 kinds: project fields, recipe in status, captions
per concept folder, the expand route and concept thumbnails."""

import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server
from trainero import captioner, config, jobs
from test_server_guards import LiveServer


def _img(path: Path, caption=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(len(str(path)))
    Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(path)
    if caption:
        path.with_suffix(".txt").write_text(caption)


class TestKindStatus(LiveServer):
    def _status(self):
        return json.loads(self._get("/api/status"))

    def _ds(self):
        return config.PROJECTS_DIR / "guardtest" / "dataset"

    def test_small_dataset_defaults_to_character_recipe(self):
        for i in range(4):
            _img(self._ds() / f"a{i}.png", "x")
        s = self._status()
        self.assertEqual(s["lora_kind"], "")
        self.assertEqual(s["kind"], "character")
        self.assertEqual(s["recipe"]["network_dim"], 16)
        self.assertTrue(s["recipe"]["line"].startswith("Receita: personagem · 4 imgs"))
        self.assertEqual(s["schedules"]["krea2"]["epochs"], s["recipe"]["epochs"])
        self.assertEqual(s["expand_refs"], 4)

    def test_kind_and_class_are_saved_with_the_project(self):
        for i in range(4):
            _img(self._ds() / f"a{i}.png", "x")
        code, _ = self.post("/api/project", {"name": "guardtest", "lora_kind": "style",
                                              "character_class": "anime girl"})
        self.assertEqual(code, 200)
        s = self._status()
        self.assertEqual((s["lora_kind"], s["kind"]), ("style", "style"))
        self.assertEqual(s["character_class"], "anime girl")
        self.assertTrue(s["recipe"]["line"].startswith("Receita: estilo"))
        code, body = self.post("/api/project", {"name": "guardtest", "lora_kind": "banana"})
        self.assertEqual(code, 400)

    def test_concept_folders_switch_to_mixed(self):
        for c in ("makima", "style_x"):
            for i in range(3):
                _img(self._ds() / c / f"{c}{i}.png")
        s = self._status()
        self.assertEqual(s["kind"], "mixed")
        self.assertEqual({c["name"] for c in s["concepts"]}, {"makima", "style_x"})
        self.assertIn("misto", s["recipe"]["line"])
        # wrong kind: the recipe explains instead of crashing the poll
        self.post("/api/project", {"name": "guardtest", "lora_kind": "character"})
        self.assertIn("Misto", self._status()["recipe"]["error"])

    def test_concept_thumbnails_are_served(self):
        _img(self._ds() / "makima" / "a.png")
        names = json.loads(self._get("/api/dataset/thumbs?side=pos"))["names"]
        self.assertEqual(names, ["makima/a.png"])
        data = urllib_get_bytes(self, "/api/dataset/thumb?side=pos&name=makima%2Fa.png")
        self.assertEqual(Image.open(io.BytesIO(data)).format, "JPEG")

    def test_thumb_names_cannot_escape(self):
        for bad in ("../x.png", "a/../../x.png", "a/b/c.png", ".hidden/x.png", "", "a//b.png"):
            self.assertFalse(server.safe_thumb_name(bad), bad)
        self.assertTrue(server.safe_thumb_name("x.png"))
        self.assertTrue(server.safe_thumb_name("makima/x.png"))


def _wait_job(timeout=10.0):
    import time

    end = time.time() + timeout
    while jobs.current() and jobs.current().status == "running" and time.time() < end:
        time.sleep(0.02)
    job = jobs.current()
    assert job and job.status == "done", (job and job.status, job and job.error)


def urllib_get_bytes(test, path):
    import urllib.request

    with urllib.request.urlopen(f"http://127.0.0.1:{test.port}{path}", timeout=10) as res:
        return res.read()


class TestCaptionsPerKind(LiveServer):
    def _ds(self):
        return config.PROJECTS_DIR / "guardtest" / "dataset"

    def test_mixed_captions_each_folder_with_its_profile(self):
        _img(self._ds() / "makima__anime_woman" / "a.png")
        _img(self._ds() / "style_ghibli" / "b.png")
        _img(self._ds() / "rem" / "c.png", "already captioned")
        calls = []

        def fake_generate(folder, media, profile, prompt_vars, job, mode="lora"):
            calls.append((Path(folder).name, profile, dict(prompt_vars)))

        with mock.patch.object(server, "generate_captions", fake_generate):
            code, body = self.post("/api/captions", {"kind": "mixed", "side": "pos"})
            self.assertEqual(code, 202, body)
            _wait_job()
        self.assertEqual(sorted(calls), [
            ("makima__anime_woman", "krea2-character",
             {"character_name": "makima", "character_class": "anime woman"}),
            ("style_ghibli", "generic-style", {"style_name": "ghibli"}),
        ])

    def test_character_profile_gets_the_project_class(self):
        _img(self._ds() / "a.png")
        self.post("/api/project", {"name": "guardtest", "character_class": "woman"})
        calls = []

        def fake_generate(folder, media, profile, prompt_vars, job, mode="lora"):
            calls.append((profile, dict(prompt_vars)))

        with mock.patch.object(server, "generate_captions", fake_generate):
            code, _ = self.post("/api/captions", {"profile": "krea2-character",
                                                  "var_name": "character_name",
                                                  "trigger": "jane_k2", "side": "pos"})
            self.assertEqual(code, 202)
            _wait_job()
        self.assertEqual(calls, [("krea2-character",
                                  {"character_name": "jane_k2", "character_class": "woman"})])

    def test_plan_skips_captioned_folders_unless_redo(self):
        _img(self._ds() / "rem" / "c.png", "done")
        self.assertEqual(captioner.concept_caption_plan(self._ds()), [])
        self.assertEqual(len(captioner.concept_caption_plan(self._ds(), redo=True)), 1)
        plan = captioner.concept_caption_plan(self._ds(), style_folders=["rem"], redo=True)
        self.assertEqual(plan[0]["profile"], "generic-style")


class TestExpandRoute(LiveServer):
    def test_needs_key_and_references(self):
        with mock.patch.dict("os.environ", {}, clear=False) as env:
            env.pop("OPENROUTER_API_KEY", None)
            code, body = self.post("/api/expand", {"n": 10})
            self.assertEqual(code, 400)
            self.assertIn("OPENROUTER_API_KEY", body["error"])
        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "k"}):
            code, body = self.post("/api/expand", {"n": 10})
            self.assertEqual(code, 400)
            self.assertIn("imagens", body["error"])
            code, body = self.post("/api/expand", {"n": 999})
            self.assertEqual(code, 400)

    def test_starts_a_job_with_the_class(self):
        _img(config.PROJECTS_DIR / "guardtest" / "dataset" / "ref.png")
        seen = {}

        def fake_expand(d, job, target=20, char_class=""):
            seen.update(target=target, char_class=char_class)

        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "k"}), \
             mock.patch.object(server.ex, "expand_dataset", fake_expand):
            code, body = self.post("/api/expand", {"n": 12, "character_class": "anime girl"})
            self.assertEqual(code, 202, body)
            _wait_job()
        self.assertEqual(seen, {"target": 12, "char_class": "anime girl"})


if __name__ == "__main__":
    unittest.main()
