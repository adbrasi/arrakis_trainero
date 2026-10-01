"""Captioning: the one phase that spends money per image and can refuse.

These run the real `captioner` (captionkreator) as a subprocess against a fake
OpenRouter on localhost, so what is tested is the actual contract between the
two repos: the CLI flags, the profile format, captions.json, the exit codes and
the progress lines — not a stand-in of them.

Point CAPTIONER_BIN at a captionkreator install to run them; by default they
look for a sibling checkout (../captionkreator/.venv/bin/captioner) and skip
when there is none.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainero import captioner
from trainero.captioner import (FLAGGED_FILE, QUARANTINE_DIR, caption_datasets,
                                clear_captions, flagged_names, generate_captions)
from trainero.jobs import Job, JobFailed

REPO = Path(__file__).resolve().parent.parent
CAPTIONER_BIN = Path(os.environ.get(
    "CAPTIONER_BIN", REPO.parent / "captionkreator" / ".venv" / "bin" / "captioner"))

PRIMARY, SECOND, THIRD = "p/primary", "f/second", "f/third"


class FakeOpenRouter:
    """/models, /chat/completions and /key, with per-model refusals.

    `refuse[model]` holds file names that model refuses (a provider refusal,
    the way OpenRouter reports a content filter). `credit` False makes every
    completion a 402 and /key report an empty balance.
    """

    def __init__(self):
        self.refuse: dict[str, set[str]] = {}
        self.credit = True
        self.calls: list[tuple[str, str]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, payload):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.endswith("/models"):
                    models = [{"id": m, "architecture": {"input_modalities": ["text", "image", "video"]},
                               "supported_parameters": ["response_format", "reasoning"]}
                              for m in (PRIMARY, SECOND, THIRD)]
                    return self._send(200, {"data": models})
                if self.path.endswith("/key"):
                    return self._send(200, {"data": {"limit_remaining": 10 if outer.credit else 0}})
                self._send(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                model = body["model"]
                text = json.dumps(body["messages"])
                name = re.search(r"file=([\w.\-]+)", text).group(1)
                trigger = (re.search(r"trigger=([\w\-]*)", text) or [None, ""])[1]
                outer.calls.append((model, name))
                if not outer.credit:
                    return self._send(402, {"error": {"message": "Insufficient credits"}})
                usage = {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001}
                if name in outer.refuse.get(model, set()):
                    return self._send(200, {"model": model, "usage": usage, "choices": [
                        {"message": {"content": None, "refusal": "content policy"},
                         "finish_reason": "stop"}]})
                caption = f"{trigger} a caption of {name} by {model}".strip()
                return self._send(200, {"model": model, "usage": usage, "choices": [
                    {"message": {"content": json.dumps({"caption": caption})},
                     "finish_reason": "stop"}]})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/api/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _write_profile(root: Path, media: str, key: str, trigger: bool) -> None:
    d = root / media / key
    d.mkdir(parents=True)
    (d / "system.md").write_text('Output {"caption": "..."}.\n')
    user = "file={{filename}}" + (" trigger={{trigger}}" if trigger else "")
    (d / "user.md").write_text(user + "\n")
    lines = [f"name: {key}", f"media: {media}", "concurrency: 4"]
    if trigger:
        lines += ["vars:", "  - name: trigger", "    required: true"]
    lines += [f"model: {PRIMARY}", "fallback:", f"  - {SECOND}", f"  - {THIRD}",
              "extract_json: caption", "retries: 0", "transport_retries: 0",
              "validate:", "  - not_empty"]
    if trigger:
        lines += ["  - type: contains", '    value: "{{trigger}}"']
    (d / "profile.yaml").write_text("\n".join(lines) + "\n")


def _png(path: Path, color=(200, 30, 30)) -> None:
    from PIL import Image

    Image.new("RGB", (32, 32), color).save(path)


@unittest.skipUnless(CAPTIONER_BIN.exists(), f"captionkreator não instalado em {CAPTIONER_BIN}")
class CaptionerEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.api = FakeOpenRouter()
        self.profiles = self.tmp / "profiles"
        # the real keys: the presets' trigger metadata applies to them
        _write_profile(self.profiles, "image", "generic-style", trigger=True)
        _write_profile(self.profiles, "image", "default", trigger=False)
        _write_profile(self.profiles, "video", "nsfw_caption_video", trigger=False)
        self.project = self.tmp / "proj"
        self.ds = self.project / "dataset"
        self.ds.mkdir(parents=True)
        self.job = Job("captions", "t", self.tmp / "log.txt")
        patches = [
            mock.patch.object(captioner, "PROFILES_DIR", self.profiles),
            mock.patch.object(captioner, "ensure_engine", lambda *_a: None),
            mock.patch.object(captioner, "venv_bin", lambda *_a: CAPTIONER_BIN),
            mock.patch.object(captioner, "engine_dir", lambda *_a: self.tmp),
            mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test",
                                         "OPENROUTER_BASE_URL": self.api.url,
                                         "HOME": str(self.tmp)}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.api.close)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def images(self, *names):
        for i, n in enumerate(names):
            _png(self.ds / n, (i * 40 % 255, 80, 120))

    def caption(self, name):
        return (self.ds / name).with_suffix(".txt").read_text().strip()

    # -- the normal path ---------------------------------------------------
    def test_every_image_gets_a_caption_with_the_trigger(self):
        self.images("a.png", "b.png", "c.png")
        generate_captions(self.ds, "generic-style", self.job, trigger="zzstyle")
        for n in ("a.png", "b.png", "c.png"):
            self.assertTrue(self.caption(n).startswith("zzstyle "), self.caption(n))
        self.assertEqual({m for m, _ in self.api.calls}, {PRIMARY})
        self.assertFalse(flagged_names(self.ds))

    def test_progress_reaches_the_job_log(self):
        self.images("a.png", "b.png")
        generate_captions(self.ds, "default", self.job)
        self.assertRegex(self.job.log_tail(), r"progress: 2/2")
        self.assertGreater(self.job.extra.get("caption_cost", 0), 0)

    def test_control_images_are_not_captioned(self):
        self.images("a.png")
        (self.ds / "control").mkdir()
        _png(self.ds / "control" / "a.png")
        generate_captions(self.ds, "default", self.job)
        self.assertFalse((self.ds / "control" / "a.txt").exists())

    # -- the owner's own captions --------------------------------------------
    def test_captions_the_owner_sent_are_never_rewritten_or_paid_again(self):
        self.images("mine.png", "new.png")
        (self.ds / "mine.txt").write_text("my hand written caption\n")
        generate_captions(self.ds, "default", self.job)
        self.assertEqual(self.caption("mine.png"), "my hand written caption")
        self.assertEqual([n for _, n in self.api.calls], ["new.png"])

    def test_nothing_missing_spends_nothing(self):
        self.images("a.png")
        (self.ds / "a.txt").write_text("ok\n")
        generate_captions(self.ds, "default", self.job)
        self.assertEqual(self.api.calls, [])

    def test_a_second_run_after_an_interruption_only_does_the_rest(self):
        self.images("a.png", "b.png")
        generate_captions(self.ds, "default", self.job)
        self.api.calls.clear()
        (self.ds / "b.txt").unlink()  # "deleted by hand to redo it"
        _png(self.ds / "c.png", (1, 2, 3))
        generate_captions(self.ds, "default", self.job)
        # b comes back from captionkreator's state for free; only c is new work
        self.assertEqual([n for _, n in self.api.calls], ["c.png"])
        self.assertTrue((self.ds / "b.txt").exists())

    def test_redo_rewrites_everything_including_the_owners(self):
        self.images("a.png")
        (self.ds / "a.txt").write_text("old\n")
        generate_captions(self.ds, "default", self.job, redo=True)
        self.assertIn("a caption of a.png", self.caption("a.png"))

    # -- refusals ------------------------------------------------------------
    def test_what_the_primary_refuses_a_fallback_captions_and_it_is_flagged(self):
        self.images("ok.png", "spicy.png")
        self.api.refuse[PRIMARY] = {"spicy.png"}
        generate_captions(self.ds, "default", self.job)
        self.assertIn(SECOND, self.caption("spicy.png"))
        self.assertEqual(flagged_names(self.ds), {"spicy.png"})

    def test_the_third_model_runs_when_the_second_also_refuses(self):
        self.images("x.png")
        self.api.refuse[PRIMARY] = self.api.refuse[SECOND] = {"x.png"}
        generate_captions(self.ds, "default", self.job)
        self.assertIn(THIRD, self.caption("x.png"))

    def test_what_no_model_captions_is_moved_out_not_deleted(self):
        self.images("ok.png", "never.png")
        for m in (PRIMARY, SECOND, THIRD):
            self.api.refuse[m] = {"never.png"}
        generate_captions(self.ds, "default", self.job)
        self.assertFalse((self.ds / "never.png").exists())
        self.assertTrue((self.project / QUARANTINE_DIR / "never.png").exists())
        self.assertTrue((self.ds / "ok.txt").exists())

    def test_the_flag_list_survives_a_later_run(self):
        self.images("a.png", "b.png")
        self.api.refuse[PRIMARY] = {"a.png"}
        generate_captions(self.ds, "default", self.job)
        self.api.refuse[PRIMARY] = {"c.png"}
        _png(self.ds / "c.png", (9, 9, 9))
        generate_captions(self.ds, "default", self.job)
        self.assertEqual(flagged_names(self.ds), {"a.png", "c.png"})

    # -- infrastructure failures -----------------------------------------------
    def test_no_credit_keeps_every_image_and_says_why(self):
        self.images("a.png", "b.png")
        self.api.credit = False
        with self.assertRaises(JobFailed) as ctx:
            generate_captions(self.ds, "default", self.job)
        self.assertIn("sem crédito", str(ctx.exception))
        self.assertTrue((self.ds / "a.png").exists() and (self.ds / "b.png").exists())
        self.assertFalse((self.project / QUARANTINE_DIR).exists())

    def test_no_key_fails_before_spending(self):
        self.images("a.png")
        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
            with self.assertRaises(JobFailed):
                generate_captions(self.ds, "default", self.job)
        self.assertEqual(self.api.calls, [])

    def test_a_trigger_profile_without_a_trigger_fails_before_spending(self):
        self.images("a.png")
        with self.assertRaises(JobFailed) as ctx:
            generate_captions(self.ds, "generic-style", self.job, trigger="")
        self.assertIn("trigger", str(ctx.exception))
        self.assertEqual(self.api.calls, [])

    # -- mixed datasets ---------------------------------------------------------
    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg ausente")
    def test_images_and_videos_each_get_their_own_profile(self):
        self.images("still.png")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
                        "testsrc=size=64x64:rate=8", "-t", "1", "-pix_fmt", "yuv420p",
                        str(self.ds / "clip.mp4")], check=True)
        caption_datasets(self.job, [self.ds], model_key=None, kind="concept")
        self.assertTrue((self.ds / "still.txt").exists())
        self.assertTrue((self.ds / "clip.txt").exists())

    def test_both_slider_sides_in_one_call(self):
        neg = self.project / "dataset_neg"
        neg.mkdir()
        self.images("p.png")
        _png(neg / "n.png")
        caption_datasets(self.job, [self.ds, neg], model_key="flux-klein", kind="concept")
        self.assertTrue((self.ds / "p.txt").exists() and (neg / "n.txt").exists())


class ClearCaptions(unittest.TestCase):
    def test_clear_takes_captions_state_and_flags_but_not_the_quarantine(self):
        with tempfile.TemporaryDirectory() as t:
            ds = Path(t) / "dataset"
            ds.mkdir()
            (ds / "a.png").write_bytes(b"x")
            (ds / "a.txt").write_text("c")
            for name in (".captioner.sqlite", "captions.json", FLAGGED_FILE, ".tagger_log.json"):
                (ds / name).write_text("{}")
            q = Path(t) / QUARANTINE_DIR
            q.mkdir()
            (q / "gone.png").write_bytes(b"x")
            self.assertEqual(clear_captions(ds), 1)
            self.assertEqual(sorted(p.name for p in ds.iterdir()), ["a.png"])
            self.assertTrue((q / "gone.png").exists())

    def test_command_fills_gaps_only_and_never_recurses_into_control(self):
        cmd = captioner.captioner_cmd(Path("/d"), "default", "")
        self.assertIn("--skip-captioned", cmd)
        self.assertIn("--no-recursive", cmd)
        self.assertNotIn("--force", cmd)
        self.assertNotIn("--var", cmd)
        self.assertEqual(cmd[cmd.index("--prompts-dir") + 1], str(captioner.PROFILES_DIR))


class ShippedProfiles(unittest.TestCase):
    """The profiles this repo ships are loadable by captionkreator and agree
    with the trigger metadata the UI and the preflight use."""

    def test_trigger_metadata_matches_each_profile_yaml(self):
        from trainero.presets import CAPTION_PROFILES, STYLE_RUSH_CAPTION_PROFILE

        entries = [(m, p) for m, ps in CAPTION_PROFILES.items() for p in ps]
        entries.append(("image", STYLE_RUSH_CAPTION_PROFILE))
        for media, p in entries:
            yaml_text = (captioner.PROFILES_DIR / media / p["key"] / "profile.yaml").read_text()
            self.assertEqual("name: trigger" in yaml_text, p["trigger"], p["key"])
            self.assertIn(f"media: {media}", yaml_text, p["key"])

    def test_every_placeholder_is_one_captionkreator_knows(self):
        known = {"trigger", "booru_tags", "existing_caption", "filename"}
        for md in captioner.PROFILES_DIR.rglob("*.md"):
            names = set(re.findall(r"\{\{\s*(\w+)", md.read_text()))
            self.assertLessEqual(names, known, md)
            # a single-brace leftover from the old engine would reach the model raw
            self.assertFalse(re.findall(r"(?<!\{)\{(tags|trigger_\w+|style_name|character_name)\}",
                                        md.read_text()), md)

    @unittest.skipUnless(CAPTIONER_BIN.exists(), "captionkreator ausente")
    def test_captionkreator_loads_every_profile(self):
        py = CAPTIONER_BIN.parent / "python"
        code = ("import sys; from pathlib import Path;"
                "from captioner.config import list_profiles, load_profile, find_profile;"
                "from captioner.prompts import preflight_prompts;"
                "root=[Path(sys.argv[1])];"
                "[preflight_prompts(load_profile(find_profile(r['name'], root)), {'trigger': 't'})"
                " for r in list_profiles(root)];"
                "print(len(list_profiles(root)))")
        out = subprocess.run([str(py), "-c", code, str(captioner.PROFILES_DIR)],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertGreaterEqual(int(out.stdout.strip()), 12)


if __name__ == "__main__":
    unittest.main()
