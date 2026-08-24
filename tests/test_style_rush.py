"""Unit tests for the Style Rush synthetic dataset pipeline (no GPU, no network)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainero.imagegen import RefusedError, RetriableError
from trainero.style_rush import (ATTEMPT_MULTIPLIER, CAPTION_TEMPLATE,
                                 DEFAULT_CONVERT_TARGET, MANIFEST_NAME,
                                 build_convert_dataset, load_style_prompts,
                                 plan_attempts)


class _FakeJob:
    """Minimal stand-in for trainero.jobs.Job — collects log lines."""

    def __init__(self):
        self.lines = []

    def log(self, msg):
        self.lines.append(msg)

    def check_cancel(self):
        pass


def _png_bytes(color=b"\x00"):
    """A 1x1 PNG, enough for the pipeline to copy bytes around."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    return buf.getvalue()


#: originals are 64x48, the stub the fake generator returns is 8x8 — the size is
#: what tells control (generated) from target (original) apart on disk.
SOURCE_SIZE = (64, 48)
GENERATED_SIZE = (8, 8)


def _make_dataset(root: Path, n: int) -> Path:
    from PIL import Image

    base = root / "dataset"
    base.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        # mixed extensions on purpose: real datasets are not all PNG
        ext = ".png" if i % 2 == 0 else ".jpg"
        Image.new("RGB", SOURCE_SIZE).save(base / f"img_{i:03d}{ext}")
        (base / f"img_{i:03d}.txt").write_text("makima, a girl")
    return base


class TestStylePrompts(unittest.TestCase):
    def test_a_hundred_distinct_prompts(self):
        prompts = load_style_prompts()
        self.assertEqual(len(prompts), 100)
        self.assertEqual(len(set(prompts)), 100, "prompts must be distinct")

    def test_every_prompt_names_a_medium_or_technique(self):
        """Um prompt de uma palavra ("make it artistic") não converte estilo
        nenhum: o gpt-image-2 precisa de descritores que ele consiga renderizar."""
        for prompt in load_style_prompts():
            self.assertGreaterEqual(len(prompt.split()), 8, prompt)


class TestPlanAttempts(unittest.TestCase):
    @staticmethod
    def _imgs(n):
        return [Path(f"/ds/img_{i:03d}.png") for i in range(n)]

    def test_the_queue_is_long_enough_to_absorb_refusals(self):
        """Uma fila do tamanho da meta morre no primeiro item recusado."""
        attempts = plan_attempts(self._imgs(10), load_style_prompts(), 20)
        self.assertEqual(len(attempts), 20 * ATTEMPT_MULTIPLIER)

    def test_prompts_cycle_so_any_target_works(self):
        prompts = ["p0", "p1", "p2"]
        attempts = plan_attempts(self._imgs(10), prompts, 4)
        self.assertEqual([a["prompt"] for a in attempts[:6]],
                         ["p0", "p1", "p2", "p0", "p1", "p2"])

    def test_a_wrapped_prompt_gets_a_different_image(self):
        """Repetir o par (prompt, imagem) gasta uma tentativa que nao pode dar
        um resultado novo."""
        prompts = ["p0", "p1"]
        attempts = plan_attempts(self._imgs(5), prompts, 4)
        pairs = [(a["prompt"], a["source"]) for a in attempts[:8]]
        self.assertEqual(len(set(pairs)), 8)

    def test_refused_images_never_enter_the_queue(self):
        images = self._imgs(6)
        avoid = {images[0].name, images[1].name}
        attempts = plan_attempts(images, load_style_prompts(), 10, avoid=avoid)
        used = {Path(a["source"]).name for a in attempts}
        self.assertEqual(used & avoid, set())

    def test_it_is_deterministic(self):
        a = plan_attempts(self._imgs(9), load_style_prompts(), 7)
        b = plan_attempts(self._imgs(9), load_style_prompts(), 7)
        self.assertEqual(a, b, "um run retomado tem de reconstruir a mesma fila")

    def test_an_empty_dataset_is_refused(self):
        with self.assertRaises(ValueError):
            plan_attempts([], load_style_prompts(), 10)

    def test_a_fully_flagged_dataset_is_refused(self):
        images = self._imgs(3)
        with self.assertRaises(ValueError):
            plan_attempts(images, load_style_prompts(), 10,
                          avoid={p.name for p in images})


class TestBuildConvertDataset(unittest.TestCase):
    TARGET = 12  # pequeno de proposito: o teste nao pode depender do default

    def test_the_happy_path_hits_the_target_exactly(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 20)
            convert = root / "dataset_convert"
            calls = []

            def fake_generate(prompt, image_path, timeout=300.0):
                calls.append((prompt, str(image_path)))
                return _png_bytes(), 0.011

            result = build_convert_dataset(base, convert, "makima", _FakeJob(),
                                           generate=fake_generate, workers=2,
                                           target=self.TARGET)

            self.assertEqual(result["pairs"], self.TARGET)
            self.assertEqual(result["refused"], 0)
            self.assertEqual(len(calls), self.TARGET)
            targets = sorted(p.name for p in convert.glob("slot_*.png"))
            controls = sorted(p.name for p in (convert / "control").glob("slot_*.png"))
            self.assertEqual(len(targets), self.TARGET)
            self.assertEqual(targets, controls)
            caption = next(convert.glob("slot_*.txt")).read_text()
            self.assertEqual(caption, "convert the style of this image to the makima style")

    def test_the_target_is_never_exceeded_with_many_workers(self):
        """Sem reserva de orcamento sob lock, N workers compram ate N-1 imagens
        depois da meta. Com 16 workers isso e dinheiro real."""
        import threading as _t

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 40)
            convert = root / "dataset_convert"
            calls = []
            lock = _t.Lock()

            def fake_generate(prompt, image_path, timeout=300.0):
                with lock:
                    calls.append(str(image_path))
                return _png_bytes(), 0.011

            result = build_convert_dataset(base, convert, "makima", _FakeJob(),
                                           generate=fake_generate, workers=16,
                                           target=self.TARGET)

            self.assertEqual(result["pairs"], self.TARGET)
            self.assertEqual(len(calls), self.TARGET, "pagou por imagem alem da meta")

    def test_a_refusal_keeps_going_until_the_target_is_met(self):
        """O comportamento que este design existe para consertar: antes, uma
        recusa custava um slot e o dataset saia curto em silencio."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 20)
            convert = root / "dataset_convert"
            refused_names = {"img_000.png", "img_001.jpg", "img_002.png"}

            def fake_generate(prompt, image_path, timeout=300.0):
                if Path(image_path).name in refused_names:
                    raise RefusedError("moderacao recusou a imagem")
                return _png_bytes(), 0.011

            result = build_convert_dataset(base, convert, "makima", _FakeJob(),
                                           generate=fake_generate, workers=1,
                                           target=self.TARGET)

            self.assertEqual(result["pairs"], self.TARGET, "a meta tem de ser atingida")
            self.assertGreater(result["refused"], 0)
            manifest = json.loads((convert / MANIFEST_NAME).read_text())
            self.assertTrue(set(manifest["refused_images"]) <= refused_names)

    def test_a_refused_image_is_paid_for_only_once(self):
        """Com poucas imagens e muitas tentativas, a mesma foto reaparece na
        fila dezenas de vezes. Redescobrir a recusa custa por vez."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 4)
            convert = root / "dataset_convert"
            seen = []

            def fake_generate(prompt, image_path, timeout=300.0):
                seen.append(Path(image_path).name)
                if Path(image_path).name == "img_000.png":
                    raise RefusedError("moderacao recusou a imagem")
                return _png_bytes(), 0.011

            build_convert_dataset(base, convert, "makima", _FakeJob(),
                                  generate=fake_generate, workers=1,
                                  target=self.TARGET)

            self.assertEqual(seen.count("img_000.png"), 1)

    def test_the_attempt_ceiling_ends_a_hopeless_run(self):
        """Tudo recusado tem de terminar, nao girar ate a conta secar."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 6)
            convert = root / "dataset_convert"

            def fake_generate(prompt, image_path, timeout=300.0):
                raise RefusedError("moderacao recusou a imagem")

            from trainero.jobs import JobFailed

            with self.assertRaises(JobFailed):
                build_convert_dataset(base, convert, "makima", _FakeJob(),
                                      generate=fake_generate, workers=1,
                                      target=self.TARGET)

    def test_a_short_run_says_so_out_loud(self):
        """Truncar em silencio e o defeito original. Se a meta nao bate, o log
        tem de dizer quanto faltou."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 4)
            convert = root / "dataset_convert"
            job = _FakeJob()

            def fake_generate(prompt, image_path, timeout=300.0):
                if Path(image_path).name != "img_000.png":
                    raise RefusedError("moderacao recusou a imagem")
                return _png_bytes(), 0.011

            result = build_convert_dataset(base, convert, "makima", job,
                                           generate=fake_generate, workers=1,
                                           target=self.TARGET)

            self.assertLess(result["pairs"], self.TARGET)
            self.assertTrue(any("meta" in ln.lower() for ln in job.lines),
                            "o log tem de dizer que o dataset saiu curto")

    def test_control_is_the_generated_image_and_target_is_the_original(self):
        """Trocar os dois ensina o LoRA a conversao inversa — um treino inteiro
        perdido, sem sintoma nenhum ate a inferencia."""
        from PIL import Image

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 20)
            convert = root / "dataset_convert"

            def fake_generate(prompt, image_path, timeout=300.0):
                return _png_bytes(), 0.011

            build_convert_dataset(base, convert, "makima", _FakeJob(),
                                  generate=fake_generate, workers=2, target=self.TARGET)

            pair = sorted(convert.glob("slot_*.png"))[0]
            with Image.open(convert / "control" / pair.name) as im:
                self.assertEqual(max(im.size), max(GENERATED_SIZE),
                                 "control = saida do GPT Image")
            with Image.open(pair) as im:
                self.assertEqual(im.size, SOURCE_SIZE, "target = imagem original do dono")

    def test_retriable_error_retries_the_same_image(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 20)
            convert = root / "dataset_convert"
            attempts, calls, failed_at = {}, [], []

            def fake_generate(prompt, image_path, timeout=300.0):
                key = str(image_path)
                calls.append((prompt, key))
                attempts[key] = attempts.get(key, 0) + 1
                if attempts[key] == 1:  # cada imagem falha uma vez, depois funciona
                    failed_at.append(len(calls) - 1)
                    raise RetriableError("HTTP 503")
                return _png_bytes(), 0.011

            with mock.patch("trainero.style_rush.time.sleep"):
                result = build_convert_dataset(base, convert, "makima", _FakeJob(),
                                               generate=fake_generate, workers=1,
                                               target=self.TARGET)
            self.assertEqual(result["pairs"], self.TARGET)
            self.assertEqual(result["refused"], 0)
            # erro transitorio repete a MESMA imagem; cair para a proxima
            # entrada da fila seria a recuperacao errada
            for i in failed_at:
                self.assertEqual(calls[i + 1], calls[i], f"retry {i} trocou de imagem")

    def test_resume_does_not_regenerate(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 20)
            convert = root / "dataset_convert"

            def fake_generate(prompt, image_path, timeout=300.0):
                return _png_bytes(), 0.011

            build_convert_dataset(base, convert, "makima", _FakeJob(),
                                  generate=fake_generate, workers=2, target=self.TARGET)

            second_calls = []

            def counting_generate(prompt, image_path, timeout=300.0):
                second_calls.append(prompt)
                return _png_bytes(), 0.011

            result = build_convert_dataset(base, convert, "makima", _FakeJob(),
                                           generate=counting_generate, workers=2,
                                           target=self.TARGET)
            self.assertEqual(second_calls, [])
            self.assertEqual(result["pairs"], self.TARGET)

    def test_a_raised_target_only_buys_the_difference(self):
        """Subir a meta de 12 para 18 tem de comprar 6, nao 18."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 30)
            convert = root / "dataset_convert"

            def fake_generate(prompt, image_path, timeout=300.0):
                return _png_bytes(), 0.011

            build_convert_dataset(base, convert, "makima", _FakeJob(),
                                  generate=fake_generate, workers=2, target=self.TARGET)

            bought = []

            def counting_generate(prompt, image_path, timeout=300.0):
                bought.append(prompt)
                return _png_bytes(), 0.011

            result = build_convert_dataset(base, convert, "makima", _FakeJob(),
                                           generate=counting_generate, workers=2,
                                           target=self.TARGET + 6)
            self.assertEqual(len(bought), 6)
            self.assertEqual(result["pairs"], self.TARGET + 6)

    def test_the_default_target_is_a_hundred(self):
        self.assertEqual(DEFAULT_CONVERT_TARGET, 100)
