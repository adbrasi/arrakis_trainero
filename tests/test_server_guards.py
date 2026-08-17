"""A job reads the dataset; nothing else may rewrite it meanwhile.

These drive the real HTTP handler on a throwaway port with a throwaway
workspace. No engine, no model, no training — only the server's own routes.
"""

import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server
from trainero import config, jobs


class LiveServer(unittest.TestCase):
    """One server, one temp workspace, torn down after each test."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self._saved = (config.PROJECTS_DIR, config.STATE_FILE, config.DATA_DIR,
                       server.project_dir)
        config.PROJECTS_DIR = root / "projects"
        # DATA_DIR too: save_state writes its temp file there and os.replace
        # cannot cross filesystems
        config.DATA_DIR = root / "data"
        config.STATE_FILE = config.DATA_DIR / "state.json"
        config.PROJECTS_DIR.mkdir(parents=True)
        config.DATA_DIR.mkdir(parents=True)
        server.project_dir = lambda name: config.PROJECTS_DIR / server.slugify(name)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.post("/api/project", {"name": "guardtest"})

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        jobs._current = None
        (config.PROJECTS_DIR, config.STATE_FILE, config.DATA_DIR,
         server.project_dir) = self._saved
        self.tmp.cleanup()

    # -- tiny client -------------------------------------------------------
    def _call(self, path, data=None, raw=None):
        url = f"http://127.0.0.1:{self.port}{path}"
        body = raw if raw is not None else json.dumps(data or {}).encode()
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as res:
                return res.status, json.loads(res.read() or b"{}")
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"{}")

    post = _call

    def _get(self, path):
        url = f"http://127.0.0.1:{self.port}{path}"
        with urllib.request.urlopen(url, timeout=10) as res:
            return res.read().decode()

    def start_fake_job(self):
        """Occupy the job slot with something that just waits."""
        gate = threading.Event()
        job = jobs.start("train", "Treinando fake",
                         config.PROJECTS_DIR / "guardtest" / "logs" / "t.log",
                         lambda _j: gate.wait(20))
        self.addCleanup(gate.set)
        return job


class TestDatasetIsFrozenDuringAJob(LiveServer):
    def test_upload_is_refused_while_a_job_runs(self):
        self.start_fake_job()
        code, body = self._call("/api/dataset/file?side=pos&name=a.png", raw=b"\x89PNG____")
        self.assertEqual(code, 409)
        self.assertIn("rodando", body.get("error", ""))

    def test_clear_is_refused_while_a_job_runs(self):
        ds = server.dataset_dir("pos")
        ds.mkdir(parents=True, exist_ok=True)
        (ds / "keep.png").write_bytes(b"x" * 2048)
        self.start_fake_job()
        code, _ = self._call("/api/dataset/clear?side=pos")
        self.assertEqual(code, 409)
        self.assertTrue((ds / "keep.png").exists(), "o dataset foi apagado sob o job")

    def test_upload_and_clear_work_when_idle(self):
        code, _ = self._call("/api/dataset/file?side=pos&name=a.png", raw=b"x" * 2048)
        self.assertEqual(code, 200)
        code, _ = self._call("/api/dataset/clear?side=pos")
        self.assertEqual(code, 200)


class TestClearWaitsForUploads(LiveServer):
    def test_clear_is_refused_while_an_upload_is_in_flight(self):
        """rmtree under a file that is still being received deletes it mid-write."""
        ds = server.dataset_dir("pos")
        ds.mkdir(parents=True, exist_ok=True)
        (ds / "keep.png").write_bytes(b"x" * 2048)
        with server._UploadInFlight():
            code, body = self._call("/api/dataset/clear?side=pos")
        self.assertEqual(code, 409)
        self.assertIn("upload", body.get("error", ""))
        self.assertTrue((ds / "keep.png").exists(), "o dataset foi apagado sob o upload")


class TestUploadIsAtomic(LiveServer):
    def test_a_successful_upload_leaves_no_staging_file(self):
        code, _ = self._call("/api/dataset/file?side=pos&name=a.png", raw=b"x" * 2048)
        self.assertEqual(code, 200)
        ds = server.dataset_dir("pos")
        self.assertTrue((ds / "a.png").exists())
        self.assertEqual([], [p.name for p in ds.iterdir() if p.name.endswith(".uploading")])


class TestTrainWaitsForUploads(LiveServer):
    def test_train_is_refused_while_an_upload_is_in_flight(self):
        with server._UploadInFlight():
            code, body = self._call("/api/train", {"model": "flux-klein", "mode": "lora"})
        self.assertEqual(code, 409)
        self.assertIn("upload", body.get("error", ""))
        self.assertIsNone(jobs.current(), "o treino começou com upload no meio")

    def test_the_counter_returns_to_zero(self):
        self.assertEqual(server.uploads_in_flight(), 0)
        with server._UploadInFlight():
            self.assertEqual(server.uploads_in_flight(), 1)
        self.assertEqual(server.uploads_in_flight(), 0)


if __name__ == "__main__":
    unittest.main()
