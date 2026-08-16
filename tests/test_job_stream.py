"""Live output: a progress bar has to reach the log while it is still running.

Every download and every training step reports through tqdm, which redraws with
a bare carriage return and emits no newline until it finishes. A reader built on
readline() shows nothing at all until then — a 20 GB download looks hung.
"""

import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainero.jobs import Job


def child(body: str) -> list[str]:
    return [sys.executable, "-u", "-c", textwrap.dedent(body)]


class TestCarriageReturnProgress(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.job = Job("test", "t", Path(self.tmp.name) / "log.txt")

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_bar_that_never_prints_a_newline_still_reaches_the_log(self):
        """The child draws 5 frames, then sleeps. The log must have the frames
        long before the child exits."""
        done = threading.Event()

        def run_and_flag():
            try:
                self.job.run(child("""
                    import sys, time
                    for i in range(5):
                        sys.stderr.write(f"baixando: {i * 20}%\\r")
                        sys.stderr.flush()
                        time.sleep(0.05)
                    time.sleep(2.5)
                    sys.stderr.write("pronto\\n")
                """))
            except Exception:  # noqa: BLE001 — the cancel below lands here
                pass
            done.set()

        proc = threading.Thread(target=run_and_flag, daemon=True)
        proc.start()

        deadline = time.time() + 2.0
        seen = ""
        while time.time() < deadline and "80%" not in seen:
            seen = self.job.log_tail()
            time.sleep(0.05)

        self.assertIn("80%", seen, "as frames só apareceram depois que o filho terminou")
        self.assertFalse(done.is_set(), "o filho já tinha saído — isto não provou nada")
        self.job.cancel()
        proc.join(timeout=10)

    def test_frames_overwrite_instead_of_piling_up(self):
        """A long download redraws thousands of times; the log must hold one
        live line, not thousands of near-identical ones."""
        self.job.run(child("""
            import sys
            for i in range(400):
                sys.stderr.write(f"{i}/400 [{i * 100 // 400}%]\\r")
            sys.stderr.write("\\n")
            print("terminou")
        """))
        # skip the "$ <cmd>" echo and the source it carries
        lines = [x for x in self.job.log_tail().splitlines()
                 if x.strip() and "sys.stderr" not in x and not x.startswith("$ ")]
        bars = [x for x in lines if "/400" in x]
        self.assertEqual(len(bars), 1, f"{len(bars)} linhas de barra no log: {bars[:3]}")
        self.assertIn("399/400", bars[0], "a linha viva não é a última frame")
        self.assertIn("terminou", lines[-1], "a saída normal não sobreviveu à barra")

    def test_normal_lines_are_never_overwritten(self):
        self.job.run(child("""
            import sys
            print("primeira")
            sys.stderr.write("bar 1\\rbar 2\\r")
            print("segunda")
            print("terceira")
        """))
        text = self.job.log_tail()
        for expected in ("primeira", "segunda", "terceira"):
            self.assertIn(expected, text)

    def test_progress_is_parsed_from_a_bar_that_has_no_newline(self):
        """The training bar is the same shape: without \\r splitting, step and
        epoch never update in the UI until the run ends."""
        self.job.run(child("""
            import sys
            sys.stderr.write("steps:  4%|      | 120/3000 [01:00<23:00, avr_loss=0.052]\\r")
            sys.stderr.write("epoch 2/5\\n")
        """), parse_progress=True)
        self.assertEqual(self.job.progress.get("step"), 120)
        self.assertEqual(self.job.progress.get("total_steps"), 3000)
        self.assertAlmostEqual(self.job.progress.get("loss"), 0.052)
        self.assertEqual(self.job.progress.get("epoch"), 2)


if __name__ == "__main__":
    unittest.main()
