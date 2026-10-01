"""Scenarios around importing, installing and cancelling — each one a way the
trainer used to fail late or silently."""

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainero import dataset as ds
from trainero import engines, jobs, style_rush
from trainero.jobs import Job, JobFailed, SyncJob


class TestMacZips(unittest.TestCase):
    def test_finder_resource_forks_are_not_dataset_items(self):
        """Every zip made in Finder carries __MACOSX/._001.png: an "image"
        without pixels or caption that blocked the training."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "staging" / "src"
            (src / "set").mkdir(parents=True)
            (src / "__MACOSX" / "set").mkdir(parents=True)
            (src / "set" / "001.png").write_bytes(b"png")
            (src / "set" / "001.txt").write_text("cap")
            (src / "__MACOSX" / "set" / "._001.png").write_bytes(b"fork")
            (src / "set" / ".DS_Store").write_bytes(b"x")
            dest = root / "dataset"
            ds.normalize_into(root / "staging", dest, SyncJob())
            self.assertEqual(sorted(p.name for p in dest.iterdir()), ["001.png", "001.txt"])
            self.assertEqual(ds.inspect(dest)["items"], 1)

    def test_a_dotfile_already_in_the_dataset_is_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "a.png").write_bytes(b"x")
            (d / "a.txt").write_text("c")
            (d / "._a.png").write_bytes(b"x")
            st = ds.inspect(d)
            self.assertEqual((st["items"], st["missing_captions"]), (1, 0))
            self.assertEqual(ds.uncaptioned(d), [])


class TestLocalImport(unittest.TestCase):
    def test_importing_the_projects_own_folder_is_refused(self):
        """_staging lives inside the project: copying the project into it
        recursed until the disk filled."""
        with tempfile.TemporaryDirectory() as td:
            pdir = Path(td) / "proj"
            (pdir / "dataset").mkdir(parents=True)
            for source in (pdir, pdir / "dataset", Path(td)):
                with self.assertRaises(JobFailed, msg=source):
                    ds.import_source(str(source), pdir / "dataset", SyncJob())

    def test_only_media_and_captions_are_copied(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src"
            (src / "sub").mkdir(parents=True)
            (src / "a.png").write_bytes(b"x")
            (src / "a.txt").write_text("c")
            (src / "sub" / "huge.safetensors").write_bytes(b"0" * 10)
            dest = Path(td) / "proj" / "dataset"
            ds.import_source(str(src), dest, SyncJob())
            self.assertEqual(sorted(p.name for p in dest.iterdir()), ["a.png", "a.txt"])

    def test_a_failed_import_leaves_no_staging_behind(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "empty"
            src.mkdir()
            pdir = Path(td) / "proj"
            with self.assertRaises(JobFailed):
                ds.import_source(str(src), pdir / "dataset", SyncJob())
            self.assertFalse((pdir / "_staging").exists())


class TestHalfInstalledEngine(unittest.TestCase):
    def test_a_venv_without_the_marker_is_not_installed(self):
        """uv creates .venv/bin/python before torch and the requirements; one
        failed pip used to mark the engine ready forever."""
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "musubi-tuner"
            (dest / ".venv" / "bin").mkdir(parents=True)
            (dest / ".venv" / "bin" / "python").touch()
            with mock.patch.object(engines, "engine_dir", return_value=dest):
                self.assertFalse(engines.is_installed("musubi"))
                (dest / ".venv" / engines.INSTALL_MARKER).write_text(engines._spec_stamp("musubi"))
                self.assertTrue(engines.is_installed("musubi"))
                # a changed engine spec (new branch, new extras) reinstalls
                (dest / ".venv" / engines.INSTALL_MARKER).write_text("old")
                self.assertFalse(engines.is_installed("musubi"))

    def test_the_marker_is_written_only_after_every_step(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "captionkreator"
            (dest / ".git").mkdir(parents=True)

            class FailingJob:
                def log(self, *_): pass

                def run(self, cmd, cwd=None, **_kw):
                    if "-e" in [str(c) for c in cmd]:
                        raise JobFailed("pip falhou")
                    if "venv" in [str(c) for c in cmd]:
                        (dest / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
                        (dest / ".venv" / "bin" / "python").touch()

            with mock.patch.object(engines, "engine_dir", return_value=dest):
                with self.assertRaises(JobFailed):
                    engines.ensure_engine("captioner", FailingJob())
                self.assertFalse(engines.is_installed("captioner"))

    def test_the_captioner_engine_installs_no_torch(self):
        from trainero.presets import ENGINES

        self.assertIs(ENGINES["captioner"]["torch"], False)
        self.assertIn("captionkreator", ENGINES["captioner"]["repo"])


class TestCancel(unittest.TestCase):
    def test_a_cancel_before_the_child_exists_still_stops_it(self):
        with tempfile.TemporaryDirectory() as td:
            job = Job("t", "t", Path(td) / "log")
            job._cancel.set()
            orig = job.check_cancel
            job.check_cancel = lambda: None  # the race: check passed, then cancel
            t0 = time.monotonic()
            with self.assertRaises(jobs.Cancelled):
                job.run([sys.executable, "-c", "import time; time.sleep(30)"])
            self.assertLess(time.monotonic() - t0, 10)
            job.check_cancel = orig

    def test_a_grandchild_holding_the_pipe_is_killed(self):
        """The leader exits on SIGTERM; a stuck grandchild kept stdout open and
        the job sat in "running" forever."""
        with tempfile.TemporaryDirectory() as td:
            job = Job("t", "t", Path(td) / "log")
            script = ("import subprocess,sys,signal,time;"
                      "subprocess.Popen([sys.executable,'-c',"
                      "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)']);"
                      "time.sleep(60)")
            result = {}

            def go():
                try:
                    job.run([sys.executable, "-c", script])
                except BaseException as exc:  # noqa: BLE001
                    result["exc"] = exc

            th = threading.Thread(target=go)
            th.start()
            time.sleep(1.0)
            with mock.patch.object(jobs, "KILL_GRACE", 1.0):
                job.cancel()
                th.join(15)
            self.assertFalse(th.is_alive(), "o job ficou preso no pipe do neto")
            self.assertIsInstance(result.get("exc"), jobs.Cancelled)


class TestStyleRushManifest(unittest.TestCase):
    def test_a_save_never_leaves_a_half_written_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            style_rush._save_manifest(d, {"slots": {"slot_000": {"status": "done"}}})
            self.assertEqual([p.name for p in d.iterdir()], [style_rush.MANIFEST_NAME])
            self.assertIn("slot_000", style_rush._load_manifest(d)["slots"])

    def test_an_empty_restore_manifest_still_has_slots(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / style_rush.RESTORE_MANIFEST_NAME).write_text("{}")
            self.assertEqual(style_rush._load_restore_manifest(d), {"slots": {}})


if __name__ == "__main__":
    unittest.main()
