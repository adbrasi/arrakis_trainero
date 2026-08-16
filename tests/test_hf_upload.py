"""UploadWatcher: what reaches HuggingFace while a run is in flight."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainero.hf_upload import UploadWatcher


class _FakeJob:
    def __init__(self):
        self.lines = []
        self.extra = {}

    def log(self, msg):
        self.lines.append(msg)


class TestUploadWatcher(unittest.TestCase):
    def _watcher(self, out: Path, job=None, uploaded=None):
        w = UploadWatcher("dono/p", out, job or _FakeJob())
        w._upload = lambda path: (uploaded.append(path.name) if uploaded is not None else None) or True
        return w

    def test_a_second_run_uploads_its_checkpoints_again(self):
        """Checkpoint names repeat across runs (project-000001.safetensors). A
        dedupe log left behind by the previous run made the retrain upload
        nothing at all, silently — the repo kept the old weights."""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / "p-000001.safetensors").write_bytes(b"primeiro treino")

            first = []
            w = self._watcher(out, uploaded=first)
            w._sweep(wait_stable=False)
            self.assertEqual(first, ["p-000001.safetensors"])

            # the retrain overwrites the file with different weights
            (out / "p-000001.safetensors").write_bytes(b"segundo treino")
            second = []
            w2 = self._watcher(out, uploaded=second)
            with mock.patch.object(w2, "_thread"):
                w2.start()
            w2._sweep(wait_stable=False)
            self.assertEqual(second, ["p-000001.safetensors"],
                             "o segundo treino não enviou o checkpoint novo")

    def test_within_one_run_a_checkpoint_is_not_sent_twice(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / "p-000001.safetensors").write_bytes(b"x")
            sent = []
            w = self._watcher(out, uploaded=sent)
            w._sweep(wait_stable=False)
            w._sweep(wait_stable=False)
            self.assertEqual(sent, ["p-000001.safetensors"])

    def test_an_auth_failure_stops_after_the_first_checkpoint(self):
        """Retrying a 403 for every checkpoint just fills the log with the same
        error, so the first one disables the rest of the run."""
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            for i in (1, 2, 3):
                (out / f"p-00000{i}.safetensors").write_bytes(b"x")
            job = _FakeJob()
            w = UploadWatcher("dono/p", out, job)

            with mock.patch("huggingface_hub.HfApi") as api:
                api.return_value.upload_file.side_effect = RuntimeError("403 Forbidden")
                w._sweep(wait_stable=False)

            self.assertTrue(w._disabled, "403 tem de desligar os uploads do run")
            self.assertEqual(api.return_value.upload_file.call_count, 1)


if __name__ == "__main__":
    unittest.main()
