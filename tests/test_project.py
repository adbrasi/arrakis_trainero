"""A dataset costs money and curation; a second model must not cost it twice.

Everything here is about the copy staying a copy: the new project owns its
files, and the trigger word it inherited keeps describing what is actually
written inside the .txt files on disk.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server
from trainero import config
from trainero import project as pj

from test_server_guards import LiveServer


def _dataset(pdir: Path, names, *, caption="arkstyle, uma garota") -> None:
    (pdir / "dataset").mkdir(parents=True, exist_ok=True)
    for n in names:
        (pdir / "dataset" / f"{n}.png").write_bytes(b"\x89PNG" + b"x" * 64)
        (pdir / "dataset" / f"{n}.txt").write_text(caption)


class TestForkCopies(unittest.TestCase):
    def _src(self, root: Path) -> Path:
        src = root / "origem"
        _dataset(src, ["a", "b", "c"])
        (src / "dataset_convert").mkdir(parents=True)
        (src / "dataset_convert" / "slot_00.png").write_bytes(b"\x89PNGp")
        (src / "dataset_convert" / ".style_rush.json").write_text(
            json.dumps({"slots": {"slot_00": {"status": "ok"}}}))
        (src / "dataset_restore").mkdir(parents=True)
        (src / "dataset_restore" / "restore_000.png").write_bytes(b"\x89PNGr")
        (src / "cache").mkdir()
        (src / "cache" / "a_qwen.safetensors").write_bytes(b"latents")
        (src / "output").mkdir()
        (src / "output" / "lora.safetensors").write_bytes(b"weights")
        pj.save_meta(src, name="origem", trigger="arkstyle")
        return src

    def test_the_dataset_and_the_paid_pairs_come_along(self):
        """The conversion pairs are dollars already spent and the restore pairs
        are derived from the same images — neither depends on the model."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src, dst = self._src(root), root / "destino"
            copied = pj.fork_dataset(src, dst)

            self.assertEqual(copied["dataset"], 3)
            self.assertEqual(copied["dataset_convert"], 1)
            self.assertEqual(copied["dataset_restore"], 1)
            self.assertTrue((dst / "dataset" / "a.txt").exists())
            self.assertTrue((dst / "dataset_convert" / ".style_rush.json").exists())

    def test_the_cache_and_the_checkpoints_stay_behind(self):
        """They are functions of the model, which is the entire reason for the
        copy: carrying them over would train the new model on the old one's
        latents and upload the old one's weights."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src, dst = self._src(root), root / "destino"
            pj.fork_dataset(src, dst)
            self.assertFalse((dst / "cache").exists())
            self.assertFalse((dst / "output").exists())

    def test_the_files_are_independent_of_the_original(self):
        """A symlink or a hardlink would make "Limpar dataset" on the copy empty
        the original, and editing one caption edit both."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src, dst = self._src(root), root / "destino"
            pj.fork_dataset(src, dst)

            copy = dst / "dataset" / "a.txt"
            self.assertFalse(copy.is_symlink())
            self.assertNotEqual(copy.stat().st_ino, (src / "dataset" / "a.txt").stat().st_ino)
            copy.write_text("novastyle, outra coisa")
            self.assertEqual((src / "dataset" / "a.txt").read_text(), "arkstyle, uma garota")

    def test_nothing_is_left_staged_after_a_copy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src, dst = self._src(root), root / "destino"
            pj.fork_dataset(src, dst)
            leftovers = [p.name for p in dst.iterdir() if p.name.startswith(".")
                         and p.name.endswith(".incoming")]
            self.assertEqual(leftovers, [])


class TestForkRefuses(unittest.TestCase):
    def test_a_destination_that_already_has_images(self):
        """Copying on top would mix two datasets into one silently."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src, dst = root / "origem", root / "destino"
            _dataset(src, ["a"])
            _dataset(dst, ["ja_estava_aqui"])
            with self.assertRaises(pj.ForkError):
                pj.fork_dataset(src, dst)
            self.assertTrue((dst / "dataset" / "ja_estava_aqui.png").exists())

    def test_a_source_with_no_images(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "vazio" / "dataset").mkdir(parents=True)
            with self.assertRaises(pj.ForkError):
                pj.fork_dataset(root / "vazio", root / "destino")

    def test_a_source_that_does_not_exist(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(pj.ForkError):
                pj.fork_dataset(root / "nao_existe", root / "destino")

    def test_copying_a_project_onto_itself(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _dataset(root / "p", ["a"])
            with self.assertRaises(pj.ForkError):
                pj.fork_dataset(root / "p", root / "p")


class TestInheritedTrigger(unittest.TestCase):
    def test_the_copy_carries_the_trigger_and_locks_it(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src, dst = root / "origem", root / "destino"
            _dataset(src, ["a"])
            pj.save_meta(src, trigger="arkstyle")
            pj.fork_dataset(src, dst)

            self.assertEqual(pj.load_meta(dst)["trigger"], "arkstyle")
            self.assertEqual(pj.load_meta(dst)["origin"]["project"], "origem")
            self.assertTrue(pj.trigger_locked(dst))

    def test_clearing_the_dataset_releases_the_lock(self):
        """With the inherited images and captions gone there is nothing left
        for the inherited word to describe."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src, dst = root / "origem", root / "destino"
            _dataset(src, ["a"])
            pj.save_meta(src, trigger="arkstyle")
            pj.fork_dataset(src, dst)

            pj.clear_origin(dst)
            self.assertFalse(pj.trigger_locked(dst))
            self.assertEqual(pj.load_meta(dst)["trigger"], "arkstyle",
                             "a palavra some da tela junto com o cadeado")
            self.assertNotIn("origin", pj.load_meta(dst))


class TestProjectRoutes(LiveServer):
    """The helpers above can all be right while nothing calls them."""

    def _make_source(self, name="fonte", trigger="arkstyle"):
        src = config.PROJECTS_DIR / name
        _dataset(src, ["a", "b"])
        pj.save_meta(src, name=name, trigger=trigger)
        return src

    def test_the_listing_offers_other_projects_but_not_the_current_one(self):
        self._make_source()
        body = json.loads(self._get("/api/projects"))
        slugs = [p["slug"] for p in body["projects"]]
        self.assertIn("fonte", slugs)
        self.assertEqual(body["current"], "guardtest")
        offered = [p for p in body["projects"] if p["items"] > 0 and p["slug"] != body["current"]]
        self.assertEqual([p["slug"] for p in offered], ["fonte"])

    def test_the_fork_route_fills_the_current_project(self):
        self._make_source()
        code, body = self._call("/api/project/fork", {"source": "fonte"})
        self.assertEqual(code, 200, body)
        self.assertEqual(body["stats"]["items"], 2)
        self.assertTrue((server.dataset_dir("pos") / "a.txt").exists())

    def test_the_fork_route_is_refused_while_a_job_runs(self):
        self._make_source()
        self.start_fake_job()
        code, _ = self._call("/api/project/fork", {"source": "fonte"})
        self.assertEqual(code, 409)

    def test_the_status_reports_the_inherited_trigger_as_locked(self):
        self._make_source()
        self._call("/api/project/fork", {"source": "fonte"})
        s = json.loads(self._get("/api/status"))
        self.assertEqual(s["trigger"], "arkstyle")
        self.assertTrue(s["trigger_locked"])
        self.assertEqual(s["origin"], "fonte")

    def test_the_trigger_of_a_copy_cannot_be_overwritten(self):
        """Every caption on disk starts with the inherited word; accepting
        another one here trains one thing and samples another."""
        self._make_source()
        self._call("/api/project/fork", {"source": "fonte"})
        self._call("/api/project", {"name": "guardtest", "trigger": "novastyle"})
        s = json.loads(self._get("/api/status"))
        self.assertEqual(s["trigger"], "arkstyle")

    def test_clearing_the_dataset_lets_the_trigger_be_set_again(self):
        self._make_source()
        self._call("/api/project/fork", {"source": "fonte"})
        self._call("/api/dataset/clear?side=pos")
        self._call("/api/project", {"name": "guardtest", "trigger": "novastyle"})
        s = json.loads(self._get("/api/status"))
        self.assertEqual(s["trigger"], "novastyle")
        self.assertFalse(s["trigger_locked"])

    def test_the_trigger_belongs_to_the_project_and_not_to_the_session(self):
        """Switching projects used to keep the previous project's trigger: the
        run then trained `arkstyle` captions while sampling the other word."""
        self._call("/api/project", {"name": "guardtest", "trigger": "primeira"})
        self._call("/api/project", {"name": "outra", "trigger": "segunda"})
        self.assertEqual(json.loads(self._get("/api/status"))["trigger"], "segunda")
        self._call("/api/project", {"name": "guardtest"})
        self.assertEqual(json.loads(self._get("/api/status"))["trigger"], "primeira")


class TestGlobalTriggerMigration(unittest.TestCase):
    def test_the_stored_trigger_moves_into_its_project_and_leaves_state(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            saved = (config.PROJECTS_DIR, config.DATA_DIR, config.STATE_FILE,
                     server.project_dir)
            config.PROJECTS_DIR = root / "projects"
            config.DATA_DIR = root / "data"
            config.STATE_FILE = config.DATA_DIR / "state.json"
            config.PROJECTS_DIR.mkdir(parents=True)
            config.DATA_DIR.mkdir(parents=True)
            server.project_dir = lambda n: config.PROJECTS_DIR / server.slugify(n)
            try:
                (config.PROJECTS_DIR / "velho").mkdir()
                config.save_state({"project": "velho", "trigger": "arkstyle"})

                server.migrate_global_trigger()

                self.assertEqual(pj.load_meta(config.PROJECTS_DIR / "velho")["trigger"],
                                 "arkstyle")
                self.assertNotIn("trigger", config.load_state(),
                                 "dois canais para a mesma verdade")
            finally:
                (config.PROJECTS_DIR, config.DATA_DIR, config.STATE_FILE,
                 server.project_dir) = saved


if __name__ == "__main__":
    unittest.main()
