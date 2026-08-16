"""Captioning is the one phase that spends money per image on every retry."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainero import captioner
from trainero.captioner import TAGGER_LOG, generate_captions, prune_stale_log


class _FakeJob:
    def __init__(self):
        self.lines = []
        self.commands = []

    def log(self, msg):
        self.lines.append(msg)

    def run(self, cmd, cwd=None, env=None, parse_progress=False):
        self.commands.append([str(c) for c in cmd])
        return 0


def _dataset(root: Path, captioned: int, uncaptioned: int) -> Path:
    ds = root / "dataset"
    ds.mkdir(parents=True, exist_ok=True)
    logged = {}
    for i in range(captioned):
        img = ds / f"ok_{i:03d}.jpg"
        img.write_bytes(b"x")
        img.with_suffix(".txt").write_text("uma caption")
        logged[str(img)] = {"taggers": ["grok", "pixai"], "timestamp": "2026-08-16T15:00:00"}
    for i in range(uncaptioned):
        (ds / f"falta_{i:03d}.jpg").write_bytes(b"x")
    (ds / TAGGER_LOG).write_text(json.dumps({"processed": logged}))
    return ds


class TestNoReprocessing(unittest.TestCase):
    def test_the_command_does_not_reprocess_what_is_already_captioned(self):
        """--force made the tagger redo every file in its log. A run that died
        at image 190 of 282 then charged OpenRouter for all 282 on the retry."""
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=190, uncaptioned=92)
            job = _FakeJob()
            with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "k"}), \
                 mock.patch.object(captioner, "ensure_engine"), \
                 mock.patch.object(captioner, "venv_python", return_value=Path("py")), \
                 mock.patch.object(captioner, "engine_dir", return_value=Path("/eng")):
                generate_captions(ds, "image", "generic-style", {"style_name": "t"}, job)

            self.assertEqual(len(job.commands), 1)
            self.assertNotIn("--force", job.commands[0])

    def test_the_log_keeps_the_files_that_still_have_their_caption(self):
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=5, uncaptioned=3)
            self.assertEqual(prune_stale_log(ds), 0)
            kept = json.loads((ds / TAGGER_LOG).read_text())["processed"]
            self.assertEqual(len(kept), 5)


class TestPruneStaleLog(unittest.TestCase):
    def test_a_caption_deleted_by_hand_goes_back_into_the_queue(self):
        """Without --force the tagger trusts its log. A log entry whose .txt is
        gone would be skipped forever, leaving the item uncaptioned and the
        training blocked with no way out."""
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=4, uncaptioned=0)
            (ds / "ok_001.txt").unlink()
            job = _FakeJob()

            removed = prune_stale_log(ds, job)

            self.assertEqual(removed, 1)
            left = json.loads((ds / TAGGER_LOG).read_text())["processed"]
            self.assertEqual(len(left), 3)
            self.assertNotIn(str(ds / "ok_001.jpg"), left)
            self.assertTrue(any("voltaram para a fila" in ln for ln in job.lines))

    def test_a_missing_log_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "dataset"
            ds.mkdir()
            self.assertEqual(prune_stale_log(ds), 0)

    def test_a_corrupt_log_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "dataset"
            ds.mkdir()
            (ds / TAGGER_LOG).write_text("{nao é json")
            self.assertEqual(prune_stale_log(ds), 0)

    def test_no_openrouter_key_fails_before_spending_anything(self):
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=1, uncaptioned=1)
            with mock.patch.dict("os.environ", {}, clear=True):
                with self.assertRaises(Exception):
                    generate_captions(ds, "image", "default", {}, _FakeJob())


if __name__ == "__main__":
    unittest.main()
