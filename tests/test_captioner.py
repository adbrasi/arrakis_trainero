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
from trainero.captioner import (QUARANTINE_DIR, TAGGER_LOG, caption_models,
                                generate_captions, prune_stale_log,
                                quarantine_uncaptionable)

# Os testes deste arquivo exercem a cascata do Style Rush, que é a ordem em que
# o primário é o Gemini — é essa ordem que alimenta record_flagged.
PRIMARY, SECOND, THIRD = caption_models("style-rush")


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


def _run(ds: Path, job: _FakeJob, problem: str = "", mode: str = "style-rush"):
    """`problem` is what the OpenRouter health check reports — empty means the
    account works, so an item still missing really was refused by both models."""
    with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "k"}), \
         mock.patch.object(captioner, "ensure_engine"), \
         mock.patch.object(captioner, "venv_python", return_value=Path("py")), \
         mock.patch.object(captioner, "engine_dir", return_value=Path("/eng")), \
         mock.patch.object(captioner, "openrouter_problem", return_value=problem):
        generate_captions(ds, "image", "generic-style", {"style_name": "t"}, job, mode=mode)


class TestNoReprocessing(unittest.TestCase):
    def test_the_command_does_not_reprocess_what_is_already_captioned(self):
        """--force made the tagger redo every file in its log. A run that died
        at image 190 of 282 then charged OpenRouter for all 282 on the retry."""
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=190, uncaptioned=92)
            job = _FakeJob({PRIMARY: {f"falta_{i:03d}.jpg" for i in range(92)}})
            _run(ds, job)
            self.assertNotIn("--force", job.commands[0])

    def test_the_cheap_model_alone_is_enough_when_nothing_is_refused(self):
        """The fallback is a second paid pass over the dataset — it must not run
        when the first model captioned everything."""
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=10, uncaptioned=5)
            job = _FakeJob({PRIMARY: {f"falta_{i:03d}.jpg" for i in range(5)}})
            _run(ds, job)
            self.assertEqual(job.models_used(), [PRIMARY])


class TestFallback(unittest.TestCase):
    def test_what_gemini_refuses_goes_to_grok(self):
        """Gemini answers PROHIBITED_CONTENT on material Grok captions fine."""
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=5, uncaptioned=3)
            job = _FakeJob({
                PRIMARY: {"falta_000.jpg", "falta_001.jpg"},
                SECOND: {"falta_002.jpg"},
            })
            _run(ds, job)

            self.assertEqual(job.models_used(),
                             [PRIMARY, SECOND])
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

            job = _FakeJob({SECOND: {"falta_000.jpg"}})
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
            job = _FakeJob({SECOND: {"falta_000.jpg"}})
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
            quarantine_uncaptionable(ds, [item], job)

            self.assertFalse(item.exists())
            left = json.loads((ds / TAGGER_LOG).read_text())["processed"]
            self.assertNotIn(str(item), left)
            self.assertEqual(len(left), 1)

    def test_the_dataset_is_captioned_end_to_end_after_the_three_passes(self):
        with tempfile.TemporaryDirectory() as td:
            from trainero.dataset import inspect
            ds = _dataset(Path(td), captioned=3, uncaptioned=4)
            job = _FakeJob({
                PRIMARY: {"falta_000.jpg", "falta_001.jpg"},
                SECOND: {"falta_002.jpg"},
            })
            _run(ds, job)
            self.assertEqual(inspect(ds)["missing_captions"], 0,
                             "o treino é bloqueado enquanto sobrar um sem caption")


class TestInfraFailureIsNotARefusal(unittest.TestCase):
    """Out of credit leaves every item uncaptioned, which on disk is identical
    to every model refusing them. One is recoverable and the other is not."""

    def test_no_credit_does_not_delete_the_owners_images(self):
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=190, uncaptioned=92)
            job = _FakeJob()  # nenhum modelo escreve nada: OpenRouter fora do ar

            with self.assertRaises(Exception) as ctx:
                _run(ds, job, problem="OpenRouter sem crédito")

            self.assertIn("crédito", str(ctx.exception))
            self.assertEqual(len(list(ds.glob("falta_*.jpg"))), 92,
                             "apagou o dataset por causa de um problema de billing")
            self.assertEqual(len(list(ds.glob("ok_*.jpg"))), 190)

    def test_a_healthy_account_still_discards_what_was_refused(self):
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=2, uncaptioned=1)
            _run(ds, _FakeJob(), problem="")
            self.assertFalse((ds / "falta_000.jpg").exists())

    def test_the_discarded_image_is_moved_and_not_destroyed(self):
        """The account check cannot catch every infra failure — a retired model
        id answers nothing while the key and the balance are perfect. Moving
        keeps that mistake recoverable; unlink on the owner's only copy does not."""
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=1, uncaptioned=1)
            _run(ds, _FakeJob(), problem="")
            self.assertFalse((ds / "falta_000.jpg").exists())
            self.assertTrue((ds.parent / QUARANTINE_DIR / "falta_000.jpg").exists(),
                            "a imagem tem de continuar existindo fora do dataset")

    def test_the_quarantine_is_outside_the_dataset_the_trainer_scans(self):
        from trainero.dataset import inspect

        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=2, uncaptioned=1)
            _run(ds, _FakeJob(), problem="")
            self.assertEqual(inspect(ds)["missing_captions"], 0)
            self.assertEqual(inspect(ds)["items"], 2, "a descartada ainda conta no dataset")

    def test_an_infra_failure_does_not_flag_images_for_the_conversion(self):
        """Flagging on an outage would exclude the whole dataset from the paid
        conversion phase permanently, blaming moderation for a billing problem."""
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=190, uncaptioned=92)
            with self.assertRaises(Exception):
                _run(ds, _FakeJob(), problem="OpenRouter sem crédito")
            self.assertEqual(captioner.flagged_names(ds), set())


class TestFlagged(unittest.TestCase):
    def test_what_the_strict_model_refused_is_recorded_for_the_conversion(self):
        """gpt-image-2 objects to the same images Gemini does, and there a
        refusal costs a paid slot."""
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=2, uncaptioned=2)
            job = _FakeJob({SECOND: {"falta_000.jpg", "falta_001.jpg"}})
            _run(ds, job)

            self.assertEqual(captioner.flagged_names(ds),
                             {"falta_000.jpg", "falta_001.jpg"})

    def test_nothing_is_flagged_when_the_first_model_takes_everything(self):
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=2, uncaptioned=2)
            job = _FakeJob({PRIMARY: {"falta_000.jpg", "falta_001.jpg"}})
            _run(ds, job)
            self.assertEqual(captioner.flagged_names(ds), set())

    def test_the_flag_survives_a_second_caption_run(self):
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=1, uncaptioned=1)
            _run(ds, _FakeJob({SECOND: {"falta_000.jpg"}}))
            _run(ds, _FakeJob())  # nada a fazer, tudo já tem caption
            self.assertEqual(captioner.flagged_names(ds), {"falta_000.jpg"})


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


class TestCascadeOrder(unittest.TestCase):
    """A ordem depende do que o modo faz com a recusa do primário: só o Style
    Rush lê o .caption_refused.json, e lá ele vale dinheiro."""

    def test_lora_starts_with_muse_spark(self):
        self.assertEqual(caption_models("lora")[0], "meta/muse-spark-1.2-contributor")

    def test_style_rush_starts_with_gemini(self):
        self.assertEqual(caption_models("style-rush")[0], "google/gemini-3.7-flash")

    def test_both_modes_end_with_grok(self):
        for mode in ("lora", "style-rush"):
            self.assertEqual(caption_models(mode)[-1], "x-ai/grok-4.20", mode)

    def test_both_modes_hold_the_same_three_models(self):
        self.assertEqual(set(caption_models("lora")), set(caption_models("style-rush")))

    def test_an_unknown_mode_falls_back_to_the_lora_order(self):
        self.assertEqual(caption_models("qualquer-coisa"), caption_models("lora"))

    def test_the_env_var_overrides_every_mode(self):
        with mock.patch.dict("os.environ", {"CAPTION_MODELS": "a/one, b/two"}, clear=False):
            self.assertEqual(caption_models("lora"), ["a/one", "b/two"])
            self.assertEqual(caption_models("style-rush"), ["a/one", "b/two"])

    def test_the_third_model_runs_when_the_second_also_refuses(self):
        """Dois passes eram o teto antigo. Um item que o Gemini e o Muse recusam
        tem de chegar no Grok em vez de sair do dataset."""
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=2, uncaptioned=1)
            job = _FakeJob({THIRD: {"falta_000.jpg"}})
            _run(ds, job)

            self.assertEqual(job.models_used(), [PRIMARY, SECOND, THIRD])
            self.assertTrue((ds / "falta_000.txt").exists(), "o terceiro tinha de salvar")
            self.assertTrue((ds / "falta_000.jpg").exists(), "nada devia sair do dataset")

    def test_the_cascade_stops_as_soon_as_nothing_is_missing(self):
        """Cada passe extra é uma varredura paga sobre o dataset inteiro."""
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=2, uncaptioned=2)
            job = _FakeJob({SECOND: {"falta_000.jpg", "falta_001.jpg"}})
            _run(ds, job)
            self.assertEqual(job.models_used(), [PRIMARY, SECOND],
                             "o terceiro modelo não tinha trabalho")

    def test_the_flag_records_the_primary_that_actually_ran(self):
        """O arquivo diz de quem é a recusa. Gravar um id fixo enquanto a ordem
        muda com o modo transformaria o registro em mentira."""
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=1, uncaptioned=1)
            second_of_lora = caption_models("lora")[1]
            _run(ds, _FakeJob({second_of_lora: {"falta_000.jpg"}}), mode="lora")
            written = json.loads((ds / captioner.FLAGGED_FILE).read_text())
            self.assertEqual(written["model"], caption_models("lora")[0])


class TestConcurrency(unittest.TestCase):
    def test_the_command_asks_for_many_parallel_calls(self):
        """Mil imagens em fila serial é o que faz a fase de caption ser a mais
        lenta do treino."""
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=0, uncaptioned=1)
            job = _FakeJob({PRIMARY: {"falta_000.jpg"}})
            _run(ds, job)
            cmd = job.commands[0]
            self.assertIn("--grok_concurrency", cmd)
            self.assertEqual(int(cmd[cmd.index("--grok_concurrency") + 1]),
                             captioner.CAPTION_CONCURRENCY)
            self.assertGreaterEqual(captioner.CAPTION_CONCURRENCY, 32)
