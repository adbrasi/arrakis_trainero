"""UploadWatcher: what reaches HuggingFace while a run is in flight."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainero import hf_upload
from trainero.hf_upload import UploadWatcher, model_card, upload_run_files


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


class TestModelCard(unittest.TestCase):
    """The repo page has to say which model the LoRA was trained on — that is
    the one fact a downloaded .safetensors cannot carry by itself."""

    def test_the_card_names_the_model_and_nothing_else(self):
        card = model_card("Qwen Image", "Comfy-Org/Qwen-Image_ComfyUI")
        self.assertIn("base_model: Comfy-Org/Qwen-Image_ComfyUI", card)
        self.assertIn("# Qwen Image", card)
        # no prose: everything else about the run is already in the repo as data
        body = card.split("---")[-1].strip().splitlines()
        self.assertEqual(body, ["# Qwen Image"])

    def test_a_preset_without_a_base_model_still_produces_valid_frontmatter(self):
        card = model_card("Krea 2", "")
        self.assertNotIn("base_model:", card)
        self.assertTrue(card.startswith("---\n"))
        self.assertIn("# Krea 2", card)

    def test_every_preset_carries_the_repo_its_weights_come_from(self):
        """Reading downloads[0] instead would be inferring the fact from an
        ordering that nothing guarantees."""
        from trainero.presets import MODELS

        for key, model in MODELS.items():
            base = model.get("base_model", "")
            self.assertTrue(base, f"{key} sem base_model")
            self.assertEqual(base, model["downloads"][0][0],
                             f"{key}: base_model não é de onde os pesos vêm")

    def test_upload_run_files_really_writes_the_readme(self):
        """model_card can be perfect while nothing calls it."""
        sent = {}
        job = _FakeJob()
        with mock.patch.object(hf_upload, "upload_text",
                               lambda repo, path, content, j: sent.__setitem__(path, content)):
            upload_run_files("dono/p", job, captions={},
                             info={"model": "qwen-image", "model_label": "Qwen Image",
                                   "base_model": "Comfy-Org/Qwen-Image_ComfyUI"})
        self.assertIn("README.md", sent)
        self.assertIn("base_model: Comfy-Org/Qwen-Image_ComfyUI", sent["README.md"])
        self.assertIn("trainero_config.json", sent)


if __name__ == "__main__":
    unittest.main()
