"""Captioning: the one phase that spends money per image and can refuse.

A caption is not optional — one item without one blocks the training — so the
pipeline is cheap-model, then fallback model, then drop what neither will take.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainero import captioner
from trainero.captioner import (DEFAULT_CAPTION_MODEL, FALLBACK_CAPTION_MODEL,
                                TAGGER_LOG, discard_uncaptionable,
                                generate_captions, prune_stale_log)


class _FakeJob:
    """Stands in for Job, and plays the tagger: `writes` names which models
    manage to caption, so a refusal is simply a model that writes nothing."""

    def __init__(self, writes: dict[str, set[str]] | None = None):
        self.lines = []
        self.commands = []
        self.writes = writes or {}

    def log(self, msg):
        self.lines.append(msg)

    def run(self, cmd, cwd=None, env=None, parse_progress=False):
        cmd = [str(c) for c in cmd]
        self.commands.append(cmd)
        model = cmd[cmd.index("--grok_model") + 1]
        target = Path(cmd[2])
        # the real tagger skips anything already in its processing log unless
        # --force; without that here, dropping the prune would look harmless
        skip = self._logged(target) if "--force" not in cmd else set()
        for name in self.writes.get(model, set()):
            item = target / name
            if item.exists() and str(item) not in skip:
                item.with_suffix(".txt").write_text("uma caption")
                self._mark(target, item)
        return 0

    @staticmethod
    def _logged(target: Path) -> set[str]:
        try:
            return set(json.loads((target / TAGGER_LOG).read_text())["processed"])
        except (OSError, json.JSONDecodeError, KeyError):
            return set()

    @staticmethod
    def _mark(target: Path, item: Path):
        log = target / TAGGER_LOG
        try:
            data = json.loads(log.read_text())
        except (OSError, json.JSONDecodeError):
            data = {"processed": {}}
        data["processed"][str(item)] = {"taggers": ["grok"], "timestamp": "now"}
        log.write_text(json.dumps(data))

    def models_used(self):
        return [c[c.index("--grok_model") + 1] for c in self.commands]


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


def _run(ds: Path, job: _FakeJob):
    with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "k"}), \
         mock.patch.object(captioner, "ensure_engine"), \
         mock.patch.object(captioner, "venv_python", return_value=Path("py")), \
         mock.patch.object(captioner, "engine_dir", return_value=Path("/eng")):
        generate_captions(ds, "image", "generic-style", {"style_name": "t"}, job)


class TestNoReprocessing(unittest.TestCase):
    def test_the_command_does_not_reprocess_what_is_already_captioned(self):
        """--force made the tagger redo every file in its log. A run that died
        at image 190 of 282 then charged OpenRouter for all 282 on the retry."""
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=190, uncaptioned=92)
            job = _FakeJob({DEFAULT_CAPTION_MODEL: {f"falta_{i:03d}.jpg" for i in range(92)}})
            _run(ds, job)
            self.assertNotIn("--force", job.commands[0])

    def test_the_cheap_model_alone_is_enough_when_nothing_is_refused(self):
        """The fallback is a second paid pass over the dataset — it must not run
        when the first model captioned everything."""
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=10, uncaptioned=5)
            job = _FakeJob({DEFAULT_CAPTION_MODEL: {f"falta_{i:03d}.jpg" for i in range(5)}})
            _run(ds, job)
            self.assertEqual(job.models_used(), [DEFAULT_CAPTION_MODEL])


class TestFallback(unittest.TestCase):
    def test_what_gemini_refuses_goes_to_grok(self):
        """Gemini answers PROHIBITED_CONTENT on material Grok captions fine."""
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=5, uncaptioned=3)
            job = _FakeJob({
                DEFAULT_CAPTION_MODEL: {"falta_000.jpg", "falta_001.jpg"},
                FALLBACK_CAPTION_MODEL: {"falta_002.jpg"},
            })
            _run(ds, job)

            self.assertEqual(job.models_used(),
                             [DEFAULT_CAPTION_MODEL, FALLBACK_CAPTION_MODEL])
            self.assertTrue((ds / "falta_002.txt").exists(),
                            "o fallback tinha de ter escrito a caption")
            self.assertTrue((ds / "falta_002.jpg").exists(), "nada devia ser removido")

    def test_the_refused_item_is_not_skipped_by_the_tagger_log(self):
        """A refusal can leave the item in the tagger log with no caption; the
        fallback pass would then skip the one file it exists to rescue."""
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=2, uncaptioned=1)
            log = json.loads((ds / TAGGER_LOG).read_text())
            log["processed"][str(ds / "falta_000.jpg")] = {"taggers": ["grok"]}
            (ds / TAGGER_LOG).write_text(json.dumps(log))

            job = _FakeJob({FALLBACK_CAPTION_MODEL: {"falta_000.jpg"}})
            _run(ds, job)

            self.assertTrue((ds / "falta_000.txt").exists())
            self.assertIn(str(ds / "falta_000.jpg"),
                          json.loads((ds / TAGGER_LOG).read_text())["processed"])


class TestDiscard(unittest.TestCase):
    def test_what_neither_model_captions_leaves_the_dataset(self):
        """One uncaptioned item fails the whole training, and this one has been
        refused twice on content grounds — retrying is not going to help."""
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=4, uncaptioned=2)
            job = _FakeJob({FALLBACK_CAPTION_MODEL: {"falta_000.jpg"}})
            _run(ds, job)

            self.assertTrue((ds / "falta_000.jpg").exists(), "o fallback salvou esta")
            self.assertFalse((ds / "falta_001.jpg").exists(), "esta tinha de sair")
            self.assertEqual(len(list(ds.glob("ok_*.jpg"))), 4,
                             "não pode encostar nas que já tinham caption")
            self.assertTrue(any("falta_001.jpg" in ln for ln in job.lines),
                            "o nome removido tem de ir para o log")

    def test_the_discarded_item_leaves_the_processing_log_too(self):
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=1, uncaptioned=1)
            item = ds / "falta_000.jpg"
            job = _FakeJob()
            discard_uncaptionable(ds, [item], job)

            self.assertFalse(item.exists())
            left = json.loads((ds / TAGGER_LOG).read_text())["processed"]
            self.assertNotIn(str(item), left)
            self.assertEqual(len(left), 1)

    def test_the_dataset_is_captioned_end_to_end_after_the_three_passes(self):
        with tempfile.TemporaryDirectory() as td:
            from trainero.dataset import inspect
            ds = _dataset(Path(td), captioned=3, uncaptioned=4)
            job = _FakeJob({
                DEFAULT_CAPTION_MODEL: {"falta_000.jpg", "falta_001.jpg"},
                FALLBACK_CAPTION_MODEL: {"falta_002.jpg"},
            })
            _run(ds, job)
            self.assertEqual(inspect(ds)["missing_captions"], 0,
                             "o treino é bloqueado enquanto sobrar um sem caption")


class TestPruneStaleLog(unittest.TestCase):
    def test_a_caption_deleted_by_hand_goes_back_into_the_queue(self):
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=4, uncaptioned=0)
            (ds / "ok_001.txt").unlink()
            job = _FakeJob()

            removed = prune_stale_log(ds, job)

            self.assertEqual(removed, 1)
            left = json.loads((ds / TAGGER_LOG).read_text())["processed"]
            self.assertEqual(len(left), 3)
            self.assertNotIn(str(ds / "ok_001.jpg"), left)

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
