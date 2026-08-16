"""The download progress reporter.

huggingface_hub's own bar goes through tqdm, which disables itself when the
stream is not a terminal — which is always, here. Injecting a tqdm_class is the
only reason a 17 GB transfer shows anything at all between "Baixando…" and "ok".
"""

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainero.hf_fetch import Progress, with_progress


def drive(total, steps, interval=0.0):
    bar = Progress(desc="modelo.safetensors", total=total, initial=0)
    bar.interval = interval  # emit every update instead of once a second
    out = io.StringIO()
    with redirect_stdout(out):
        for s in steps:
            bar.update(s)
        bar.close()
    return out.getvalue()


class TestProgressOutput(unittest.TestCase):
    def test_frames_end_in_carriage_return_so_they_overwrite(self):
        text = drive(1000, [100, 100])
        # every frame but the last closes with \r; the last one closes the line
        self.assertGreaterEqual(text.count("\r"), 2)
        self.assertTrue(text.endswith("\n"))

    def test_a_frame_carries_size_and_percentage(self):
        frame = drive(200 * 1024 * 1024, [50 * 1024 * 1024]).split("\r")[0]
        self.assertIn("50.0MB", frame)
        self.assertIn("200.0MB", frame)
        self.assertIn("25%", frame)
        self.assertIn("modelo.safetensors", frame)

    def test_no_percentage_is_invented_when_the_size_is_unknown(self):
        frame = drive(None, [1024 * 1024]).split("\r")[0]
        self.assertIn("1.0MB", frame)
        self.assertNotIn("%", frame)

    def test_the_throttle_holds_frames_back(self):
        """A 17 GB Xet transfer updates far faster than anyone can read."""
        bar = Progress(desc="x", total=100)
        out = io.StringIO()
        with redirect_stdout(out):
            for _ in range(50):
                bar.update(1)
        self.assertLessEqual(out.getvalue().count("\r"), 1)

    def test_resume_offset_is_reported_when_the_hub_supplies_one(self):
        bar = Progress(desc="x", total=1000, initial=400)
        bar.interval = 0.0
        out = io.StringIO()
        with redirect_stdout(out):
            bar.update(100)
        self.assertIn("retomado", out.getvalue())
        self.assertIn("50%", out.getvalue())  # 500 of 1000, not 100 of 1000


class TestCompatFallback(unittest.TestCase):
    """Losing the bar beats losing Xet if a hub release changes the argument."""

    def test_a_hub_that_rejects_tqdm_class_still_downloads(self):
        calls = []

        def picky(**kwargs):
            if "tqdm_class" in kwargs:
                raise TypeError("unexpected keyword argument 'tqdm_class'")
            calls.append(kwargs)
            return "/path/to/file"

        out = io.StringIO()
        with redirect_stdout(out):
            got = with_progress(picky, repo_id="r/r", filename="f")
        self.assertEqual(got, "/path/to/file")
        self.assertEqual(len(calls), 1)
        self.assertIn("progresso indisponível", out.getvalue())

    def test_a_real_failure_is_not_swallowed(self):
        def broken(**_kw):
            raise OSError("disco cheio")

        with self.assertRaises(OSError):
            with_progress(broken, repo_id="r/r")


if __name__ == "__main__":
    unittest.main()
