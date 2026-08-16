"""Download layer: transport choice, cancellation, and completeness.

The network test at the bottom is the one that catches a preset pointing at a
file that does not exist — the failure mode there is a run that dies AFTER
downloading tens of GB.
"""

import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainero import models_download as md
from trainero.jobs import Cancelled, JobFailed
from trainero.presets import MODELS, MODEL_ORDER


class FakeJob:
    """Records commands instead of running them."""

    def __init__(self, fail=(), cancel_on=None):
        self.raw = []               # argv lists, as given
        self.cmds = []              # the same, space-joined, for `used()`
        self.lines = []
        self.fail = fail            # substrings whose command should fail
        self.cancel_on = cancel_on  # substring whose command raises Cancelled

    def log(self, msg):
        self.lines.append(str(msg))

    def check_cancel(self):
        pass

    def run(self, cmd, cwd=None, env=None, parse_progress=False):
        parts = [str(c) for c in cmd]
        joined = " ".join(parts)
        self.raw.append(parts)
        self.cmds.append(joined)
        if self.cancel_on and self.cancel_on in joined:
            raise Cancelled()
        if any(f in joined for f in self.fail):
            raise JobFailed("boom")
        if parts[0] == "aria2c":  # write the output file real aria2c would write
            argfile = Path(parts[parts.index("-i") + 1])
            opts = dict(line.strip().split("=", 1)
                        for line in argfile.read_text().splitlines()[1:] if "=" in line)
            (Path(opts["dir"]) / opts["out"]).write_bytes(b"x" * 4096)
        return 0

    def used(self, needle):
        return any(needle in c for c in self.cmds)


class TestTransportOrder(unittest.TestCase):
    """Xet is the fast path and must be tried first whenever it is usable."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self.tmp.name) / "model.safetensors"
        self._ready, self._prefers, self._which = md.xet_ready, md.prefers_xet, md.shutil.which
        md.shutil.which = lambda _n: "/usr/bin/aria2c"

    def tearDown(self):
        md.xet_ready, md.prefers_xet, md.shutil.which = self._ready, self._prefers, self._which
        self.tmp.cleanup()

    def test_xet_backed_file_goes_to_huggingface_hub_first(self):
        md.prefers_xet = lambda *a: True
        job = FakeJob()
        md._single_file("some/repo", "f.safetensors", self.dest, None, job)
        self.assertIn("huggingface_hub", job.cmds[0])
        self.assertNotIn("aria2c", job.cmds[0])

    def test_non_xet_file_prefers_aria2_over_single_stream_http(self):
        md.prefers_xet = lambda *a: False
        job = FakeJob()
        md._single_file("some/repo", "f.safetensors", self.dest, None, job)
        self.assertIn("aria2c", job.cmds[0])

    def test_xet_failure_falls_back_to_aria2(self):
        md.prefers_xet = lambda *a: True
        job = FakeJob(fail=("huggingface_hub",))
        md._single_file("some/repo", "f.safetensors", self.dest, None, job)
        self.assertTrue(job.used("aria2c"))

    def test_token_never_reaches_argv(self):
        md.prefers_xet = lambda *a: True
        job = FakeJob()
        md._single_file("some/repo", "f.safetensors", self.dest, "hf_SECRET", job)
        self.assertNotIn("hf_SECRET", " ".join(job.cmds))


class TestXetPreference(unittest.TestCase):
    """A failed probe is not evidence that a file is off Xet."""

    def setUp(self):
        self._ready = md.xet_ready

    def tearDown(self):
        md.xet_ready = self._ready

    def _probe(self, raise_it=False, is_xet=True):
        import huggingface_hub

        class Meta:
            xet_file_data = object() if is_xet else None

        def fake(*_a, **_kw):
            if raise_it:
                raise RuntimeError("401 gated / rate limited / offline")
            return Meta()

        self._real = huggingface_hub.get_hf_file_metadata
        huggingface_hub.get_hf_file_metadata = fake
        self.addCleanup(setattr, huggingface_hub, "get_hf_file_metadata", self._real)

    def test_gated_repo_whose_probe_fails_still_goes_over_xet(self):
        md.xet_ready = lambda: True
        self._probe(raise_it=True)
        self.assertTrue(md.prefers_xet("black-forest-labs/FLUX.2-dev", "ae.safetensors", None))

    def test_a_file_known_not_to_be_on_xet_does_not(self):
        md.xet_ready = lambda: True
        self._probe(is_xet=False)
        self.assertFalse(md.prefers_xet("some/repo", "f.safetensors", None))

    def test_without_hf_xet_nothing_goes_over_xet(self):
        md.xet_ready = lambda: False
        self._probe(is_xet=True)
        self.assertFalse(md.prefers_xet("some/repo", "f.safetensors", None))


class TestCancellation(unittest.TestCase):
    """Cancelling must stop the job, not silently move to the next transport."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self.tmp.name) / "model.safetensors"
        self._ready, self._prefers, self._which = md.xet_ready, md.prefers_xet, md.shutil.which
        md.shutil.which = lambda _n: "/usr/bin/aria2c"

    def tearDown(self):
        md.xet_ready, md.prefers_xet, md.shutil.which = self._ready, self._prefers, self._which
        self.tmp.cleanup()

    def test_cancel_during_aria2_does_not_start_the_hub_fallback(self):
        md.prefers_xet = lambda *a: False  # aria2 goes first
        job = FakeJob(cancel_on="aria2c")
        with self.assertRaises(Cancelled):
            md._single_file("some/repo", "f.safetensors", self.dest, None, job)
        self.assertFalse(job.used("huggingface_hub"))

    def test_cancel_during_hub_does_not_start_the_aria2_fallback(self):
        md.prefers_xet = lambda *a: True
        job = FakeJob(cancel_on="huggingface_hub")
        with self.assertRaises(Cancelled):
            md._single_file("some/repo", "f.safetensors", self.dest, None, job)
        self.assertFalse(job.used("aria2c"))


class TestCompleteness(unittest.TestCase):
    def test_an_interrupted_snapshot_is_never_mistaken_for_a_finished_one(self):
        """The child moves the staging dir into place only when it finished, so
        a dir that stopped after its first shard is still a .hfpart."""
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "text_encoder"
            stage = Path(td) / "text_encoder.hfpart"
            stage.mkdir()
            (stage / "model-00001-of-00004.safetensors").write_bytes(b"x" * 4096)
            self.assertFalse(md._present(dest))
            stage.rename(dest)
            self.assertTrue(md._present(dest))

    def test_snapshot_downloads_to_staging_not_straight_into_the_destination(self):
        """Writing into dest means a snapshot killed after its first shard looks
        finished forever — the next run trains against half a text encoder."""
        import json

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "text_encoder"
            job = FakeJob()
            md._snapshot("r/r", "text_encoder", dest, None, job)
            spec = json.loads(job.raw[0][-1])
            self.assertEqual(spec["dest"], str(dest))
            self.assertNotEqual(spec["stage"], str(dest))

    def test_the_child_only_moves_after_the_download_returns(self):
        src = md._FETCH_CHILD
        self.assertLess(src.index("snapshot_download("), src.index("os.replace(inner, dest)"))
        self.assertLess(src.index("hf_hub_download("), src.index("os.replace(got, dest)"))

    def test_staging_names_do_not_collide_between_transports(self):
        """aria2 stages a FILE at <name>.part; the hub child stages a DIRECTORY.
        Sharing the path would make the fallback trip over the first attempt."""
        import json

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "model.safetensors"
            job = FakeJob()
            md._hf_fetch("r/r", "f.safetensors", "", dest, None, job)
            spec = json.loads(job.raw[0][-1])
            aria2_part = str(dest.with_suffix(dest.suffix + ".part"))
            self.assertNotEqual(spec["stage"], aria2_part)
            self.assertEqual(spec["dest"], str(dest))


class TestDownloadUrlsResolve(unittest.TestCase):
    """Every configured asset must exist on the Hub.

    200 = public, 401/403 = gated (needs HF_TOKEN, still correct), 404 = the
    preset is wrong and the run would die after downloading everything else.
    """

    @staticmethod
    def _status(url: str) -> int:
        req = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=25) as res:
                return res.status
        except urllib.error.HTTPError as exc:
            return exc.code

    def test_no_asset_is_404(self):
        try:
            self._status("https://huggingface.co/api/models/Comfy-Org/Qwen-Image_ComfyUI")
        except Exception as exc:  # noqa: BLE001 — offline box, not a code failure
            self.skipTest(f"sem rede: {exc}")
        broken = []
        for key in MODEL_ORDER:
            for repo, remote, _local in MODELS[key]["downloads"]:
                url = (f"https://huggingface.co/api/models/{repo}" if remote.endswith("/")
                       else f"https://huggingface.co/{repo}/resolve/main/{remote}")
                if self._status(url) == 404:
                    broken.append(f"{key}: {repo}/{remote}")
        self.assertEqual(broken, [], "assets inexistentes no HuggingFace")


if __name__ == "__main__":
    unittest.main()
