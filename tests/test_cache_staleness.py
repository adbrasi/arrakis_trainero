"""State derived from the dataset has to die when the dataset changes.

Both cases here are silent: nothing fails, nothing is logged, and the run
finishes reporting numbers that are not what was trained.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainero.dataset import captions_digest
from trainero.training import (CAPTION_STAMP, drop_stale_te_cache, image_subset,
                               subset_dirs, write_dataset_toml)


class _FakeJob:
    def __init__(self):
        self.lines = []

    def log(self, msg):
        self.lines.append(msg)


def _media(ds: Path, names_to_captions: dict[str, str]) -> None:
    ds.mkdir(parents=True, exist_ok=True)
    for name, caption in names_to_captions.items():
        (ds / name).write_bytes(b"x")
        (ds / name).with_suffix(".txt").write_text(caption)


def _te_cache(cache: Path, basenames) -> list[Path]:
    cache.mkdir(parents=True, exist_ok=True)
    made = []
    for b in basenames:
        f = cache / f"{b}_qwen_image_te.safetensors"
        f.write_bytes(b"embeddings")
        made.append(f)
    return made


class TestCaptionsDigest(unittest.TestCase):
    def test_editing_a_caption_changes_the_digest(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "dataset"
            _media(ds, {"a.png": "arkstyle, a girl", "b.png": "arkstyle, a cat"})
            before = captions_digest(ds)
            (ds / "a.txt").write_text("novastyle, a girl")
            self.assertNotEqual(before, captions_digest(ds))

    def test_the_digest_is_stable_when_nothing_changed(self):
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "dataset"
            _media(ds, {"a.png": "arkstyle, a girl"})
            self.assertEqual(captions_digest(ds), captions_digest(ds))


class TestStaleTextEncoderCache(unittest.TestCase):
    def test_a_changed_trigger_word_invalidates_the_cached_embeddings(self):
        """musubi names the cache after the media file only, so --skip_existing
        happily reuses embeddings of the old text. The run then trains on the
        previous trigger while the samples use the new one."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ds, cache = root / "dataset", root / "cache" / "images"
            _media(ds, {"a.png": "arkstyle, a girl", "b.png": "arkstyle, a cat"})
            cache.mkdir(parents=True)
            drop_stale_te_cache([(ds, cache)], _FakeJob())  # primeiro run: grava o selo
            files = _te_cache(cache, ["a", "b"])

            (ds / "a.txt").write_text("novastyle, a girl")
            job = _FakeJob()
            drop_stale_te_cache([(ds, cache)], job)

            self.assertFalse(any(f.exists() for f in files),
                             "treinaria com os embeddings da trigger antiga")
            self.assertTrue(any("captions" in ln for ln in job.lines))

    def test_an_unchanged_dataset_keeps_its_cache(self):
        """Rebuilding a 300-image text encoder cache for nothing is minutes of
        GPU on every resumed run."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ds, cache = root / "dataset", root / "cache" / "images"
            _media(ds, {"a.png": "arkstyle, a girl"})
            cache.mkdir(parents=True)
            drop_stale_te_cache([(ds, cache)], _FakeJob())
            files = _te_cache(cache, ["a"])

            drop_stale_te_cache([(ds, cache)], _FakeJob())

            self.assertTrue(all(f.exists() for f in files))

    def test_the_stamp_is_written_on_the_first_run_without_deleting_anything(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ds, cache = root / "dataset", root / "cache" / "images"
            _media(ds, {"a.png": "arkstyle, a girl"})
            files = _te_cache(cache, ["a"])
            drop_stale_te_cache([(ds, cache)], _FakeJob())
            self.assertTrue(all(f.exists() for f in files))
            self.assertEqual((cache / CAPTION_STAMP).read_text(), captions_digest(ds))


class TestCachePhaseWiring(unittest.TestCase):
    """Asserting on drop_stale_te_cache alone leaves run_caches free to stop
    calling it with every test still green — that is how the restore subset
    nearly shipped unwired."""

    def test_run_caches_drops_the_stale_cache_before_building(self):
        from unittest import mock

        from trainero import training

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ds, cache = root / "dataset", root / "cache" / "images"
            _media(ds, {"a.png": "arkstyle, a girl"})
            cache.mkdir(parents=True)
            (cache / CAPTION_STAMP).write_text("um sha de outra caption")
            files = _te_cache(cache, ["a"])

            toml = write_dataset_toml("qwen-image", root / "dataset.toml",
                                      [image_subset(ds, cache, 1)], [1024, 1024], 1)
            job = _FakeJob()
            job.run = lambda *a, **kw: 0
            job.start_phase = lambda name: None
            job.end_phase = lambda name, ok=True: None

            with mock.patch.object(training, "venv_python", return_value=Path("py")), \
                 mock.patch.object(training, "engine_dir", return_value=root), \
                 mock.patch.object(training, "resolve",
                                   side_effect=lambda mk, rel: Path("/m") / rel):
                training.run_caches("qwen-image", toml, 32.0, job)

            self.assertFalse(any(f.exists() for f in files),
                             "run_caches não invalidou o cache velho")


class TestSampleFrame(unittest.TestCase):
    def test_an_image_sample_is_rendered_wide(self):
        """Testing sample_resolution alone leaves write_sample_prompts free to
        ignore it."""
        from trainero.training import write_sample_prompts

        with tempfile.TemporaryDirectory() as td:
            path = write_sample_prompts(Path(td) / "sample_prompts.txt",
                                        "uma garota", "makima", [1024, 1024])
            line = path.read_text()
            self.assertIn("--w 1360", line)
            self.assertIn("--h 768", line)
            self.assertNotIn("--w 1024", line)

    def test_a_video_sample_keeps_the_shape_it_trains_on(self):
        from trainero.training import write_sample_prompts

        with tempfile.TemporaryDirectory() as td:
            line = write_sample_prompts(Path(td) / "s.txt", "um gato", "trg",
                                        [768, 512], frames=81).read_text()
            self.assertIn("--w 768", line)
            self.assertIn("--h 512", line)
            self.assertIn("--f 81", line)

    def test_the_default_prompt_has_no_golden_hour_and_meets_the_viewer(self):
        from trainero.presets import SAMPLE_PROMPT

        low = SAMPLE_PROMPT.lower()
        self.assertNotIn("golden hour", low)
        self.assertNotIn("amber", low)
        self.assertIn("viewer", low)


class TestSubsetDirs(unittest.TestCase):
    def test_every_block_of_the_toml_is_paired_with_its_cache(self):
        """Style Rush writes three; missing one leaves that subset stale."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            subsets = [
                image_subset(root / "dataset", root / "cache" / "images", 1),
                image_subset(root / "dataset_convert", root / "cache" / "convert", 1,
                             control_dir=root / "dataset_convert" / "control",
                             control_resolution=[1024, 1024]),
                image_subset(root / "dataset_restore", root / "cache" / "restore", 1,
                             control_dir=root / "dataset_restore" / "control",
                             control_resolution=[1024, 1024]),
            ]
            toml = write_dataset_toml("qwen-image", root / "dataset.toml", subsets,
                                      [1024, 1024], 1)
            pairs = subset_dirs(toml)
            self.assertEqual(len(pairs), 3)
            self.assertEqual(pairs[0], (root / "dataset", root / "cache" / "images"))
            self.assertEqual(pairs[2],
                             (root / "dataset_restore", root / "cache" / "restore"))


class TestClearDatasetDropsDerived(unittest.TestCase):
    def _project(self, root: Path) -> Path:
        pdir = root / "projeto"
        for name in ("dataset", "dataset_neg", "dataset_convert", "dataset_restore",
                     "cache"):
            (pdir / name).mkdir(parents=True)
        (pdir / "dataset" / "a.png").write_bytes(b"x")
        (pdir / "dataset_convert" / ".style_rush.json").write_text(
            json.dumps({"slots": {"slot_00": {"status": "ok", "source": "/velha.png"}}}))
        (pdir / "dataset_restore" / "restore_000.png").write_bytes(b"x")
        (pdir / "cache" / "images").mkdir()
        return pdir

    def test_replacing_the_dataset_invalidates_convert_restore_and_cache(self):
        """Their manifests only check that their own files exist, so a leftover
        dataset_convert reports "já completo (50 pares)" and trains pairs built
        from images that are not on disk any more — with nothing failing."""
        from trainero.dataset import clear_dataset

        with tempfile.TemporaryDirectory() as td:
            pdir = self._project(Path(td))
            clear_dataset(pdir, "pos")

            self.assertTrue((pdir / "dataset").exists())
            self.assertEqual(list((pdir / "dataset").iterdir()), [])
            for derived in ("dataset_convert", "dataset_restore", "cache"):
                self.assertFalse((pdir / derived).exists(), derived)

    def test_clearing_the_negative_side_leaves_the_rest_alone(self):
        """The (−) dataset of a slider feeds nothing synthetic."""
        from trainero.dataset import clear_dataset

        with tempfile.TemporaryDirectory() as td:
            pdir = self._project(Path(td))
            clear_dataset(pdir, "neg")

            self.assertTrue((pdir / "dataset" / "a.png").exists())
            for derived in ("dataset_convert", "dataset_restore", "cache"):
                self.assertTrue((pdir / derived).exists(), derived)

    def test_the_route_delegates_instead_of_keeping_its_own_copy(self):
        """A second implementation inside the handler is what let this drift in
        the first place."""
        import inspect as py_inspect

        import server

        source = py_inspect.getsource(server.Handler._clear_dataset)
        self.assertIn("clear_dataset(", source)
        self.assertNotIn("rmtree", source)


if __name__ == "__main__":
    unittest.main()
