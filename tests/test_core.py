"""Unit tests for the pure logic: presets, schedules, source detection, TOMLs."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainero.dataset import archive_ext, detect_source
from trainero.presets import (MODEL_ORDER, MODELS, net_types_for,
                              public_presets, suggest_schedule, vram_tier)
from trainero.training import (_cli_args, build_train_config, image_subset, slugify,
                               write_dataset_toml)


class TestPresets(unittest.TestCase):
    def test_all_models_complete(self):
        for key in MODEL_ORDER:
            m = MODELS[key]
            self.assertIn(m["engine"], ("musubi", "musubi-ltx", "sd-scripts"), key)
            self.assertTrue(m["downloads"], key)
            self.assertTrue(m["model_args"], key)
            self.assertIn("network_dim", m["train"], key)
            self.assertTrue(m["vram_tiers"], key)
            self.assertEqual(m["vram_tiers"][-1]["min_gb"], 0, f"{key}: last tier must catch all")

    def test_every_model_can_render_a_sample(self):
        """--sample_prompts makes the trainer load the VAE and the text encoder
        from the very args model_args writes into the config (trainer_base
        _prepare_sampling). A preset missing them only crashes after the engine
        install, the 40 GB download and the caching — so it is caught here."""
        te_keys = ("text_encoder", "t5", "qwen3", "gemma_root")
        for key in MODEL_ORDER:
            m = MODELS[key]
            if m["engine"] == "musubi-ltx":
                continue  # ltx2_checkpoint carries the VAE, gemma_root is the TE
            args = m["model_args"]
            self.assertIn("vae", args, key)
            self.assertTrue(set(te_keys) & set(args), key)

    def test_vram_tiers_descending(self):
        for key in MODEL_ORDER:
            mins = [t["min_gb"] for t in MODELS[key]["vram_tiers"]]
            self.assertEqual(mins, sorted(mins, reverse=True), key)
            self.assertEqual(vram_tier(key, 9999), MODELS[key]["vram_tiers"][0])
            self.assertEqual(vram_tier(key, 1), MODELS[key]["vram_tiers"][-1])

    def test_schedules_land_in_sweet_spot(self):
        for key in MODEL_ORDER:
            for n in (5, 30, 200, 1500):
                s = suggest_schedule(key, n)
                self.assertGreaterEqual(s["epochs"], 4, (key, n))
                self.assertGreaterEqual(s["num_repeats"], 1)
                self.assertGreaterEqual(s["save_every_n_epochs"], 1)
                if "target_steps" in MODELS[key]:
                    total = n * s["num_repeats"] * s["epochs"]
                    target = MODELS[key]["target_steps"]
                    self.assertGreater(total, target * 0.4, (key, n, total))
                    # datasets enormes estouram o alvo pelo piso de 4 epochs —
                    # intencional: cada imagem precisa ser vista o suficiente
                    self.assertLess(total, max(target * 2.5, n * 4 + 1), (key, n, total))

    def test_net_types(self):
        self.assertEqual(net_types_for("anima"), ["lora"])
        self.assertIn("lokr", net_types_for("wan-22"))
        self.assertEqual(net_types_for("ideogram"), ["lora"])

    def test_public_presets_json_safe(self):
        import json

        json.dumps(public_presets())

    def test_style_rush_models(self):
        from trainero.presets import style_rush_models

        keys = style_rush_models()
        self.assertIn("flux-klein", keys)
        self.assertIn("qwen-image-edit", keys)
        self.assertNotIn("wan-22", keys)
        self.assertNotIn("anima", keys)

    def test_style_rush_schedule_is_fixed(self):
        from trainero.presets import STYLE_RUSH_SCHEDULE

        self.assertEqual(STYLE_RUSH_SCHEDULE,
                         {"num_repeats": 2, "epochs": 5, "save_every_n_epochs": 1})

    def test_sample_prompt_is_prose(self):
        from trainero.presets import SAMPLE_PROMPT

        self.assertGreater(len(SAMPLE_PROMPT.split()), 60)
        self.assertNotIn("\n", SAMPLE_PROMPT)
        self.assertNotIn("--", SAMPLE_PROMPT)  # flags are added by write_sample_prompts

    def test_comfy_convert_is_data(self):
        self.assertEqual(MODELS["anima"]["comfy_convert"],
                         {"script": "networks/convert_anima_lora_to_comfy.py"})
        for key in MODEL_ORDER:
            cc = MODELS[key].get("comfy_convert")
            if cc is not None:
                self.assertIsInstance(cc, dict, key)
                self.assertTrue({"script", "convert_lora"} & set(cc), key)

    def test_public_presets_expose_control(self):
        pub = public_presets()
        self.assertTrue(pub["flux-klein"]["supports_control"])
        self.assertFalse(pub["krea2"]["supports_control"])


class TestTrainConfig(unittest.TestCase):
    def _cfg(self, key, overrides=None, n=30):
        stats = {"items": n, "images": n, "videos": 0}
        sched = suggest_schedule(key, n)
        return build_train_config(key, overrides or {}, sched, stats, 48,
                                  Path("/tmp/ds.toml"), Path("/tmp/out"), "test", 1)

    def test_every_model_builds(self):
        for key in MODEL_ORDER:
            cfg = self._cfg(key)
            self.assertIn("network_module", cfg, key)
            self.assertIn("max_train_epochs", cfg, key)
            self.assertNotIn("max_train_steps", cfg, key)  # epochs, never steps
            self.assertTrue(cfg["output_name"], key)

    def test_lokr_switch(self):
        cfg = self._cfg("wan-22", {"net_type": "lokr"})
        self.assertEqual(cfg["network_module"], "networks.lokr")
        self.assertEqual(cfg["network_alpha"], cfg["network_dim"] // 2)

    def test_loraplus(self):
        cfg = self._cfg("flux-klein", {"loraplus": True})
        self.assertIn("loraplus_lr_ratio=16", cfg["network_args"])

    def test_invalid_net_type_rejected(self):
        from trainero.jobs import JobFailed

        with self.assertRaises(JobFailed):
            self._cfg("anima", {"net_type": "lokr"})

    def test_overrides_apply(self):
        cfg = self._cfg("qwen-image", {"network_dim": "16", "learning_rate": "1e-4"})
        self.assertEqual(cfg["network_dim"], 16)
        # str viraria TOML com aspas e o trainer crasharia no optimizer
        self.assertIsInstance(cfg["learning_rate"], float)
        self.assertEqual(cfg["learning_rate"], 1e-4)

    def test_bad_lr_rejected(self):
        from trainero.jobs import JobFailed

        with self.assertRaises(JobFailed):
            self._cfg("qwen-image", {"learning_rate": "abc"})

    def test_cli_args(self):
        args = _cli_args({"a": 1, "flag": True, "off": False, "lst": ["x", "y"], "skip": None})
        self.assertEqual(args, ["--a", "1", "--flag", "--lst", "x", "y"])


class TestDatasetToml(unittest.TestCase):
    def _write(self, key, subsets, resolution=(1024, 1024), batch_size=1, ltx_cfg=None):
        import tempfile

        td = Path(tempfile.mkdtemp())
        # the writer really creates the cache dirs, so they must live in the
        # temp dir; the dataset dirs stay fake because nothing touches them.
        for i, sub in enumerate(subsets):
            sub["cache"] = td / "cache" / f"s{i}"
        return write_dataset_toml(key, td / "dataset.toml", subsets, list(resolution),
                                  batch_size, ltx_cfg)

    def test_musubi_image(self):
        toml = self._write("qwen-image", [
            image_subset(Path("/ds"), Path("/cache"), 3),
        ])
        text = toml.read_text()
        self.assertIn("image_directory", text)
        self.assertIn("num_repeats = 3", text)
        self.assertIn("enable_bucket = true", text)
        self.assertNotIn("control_directory", text)

    def test_musubi_edit_has_control(self):
        toml = self._write("qwen-image-edit", [
            image_subset(Path("/ds"), Path("/cache"), 1,
                         control_dir=Path("/ds/control"), control_resolution=[1024, 1024]),
        ])
        text = toml.read_text()
        self.assertIn("control_directory", text)
        self.assertIn("control_resolution = [1024, 1024]", text)

    def test_two_subsets_only_one_has_control(self):
        toml = self._write("flux-klein", [
            image_subset(Path("/ds"), Path("/cache/images"), 1),
            image_subset(Path("/conv"), Path("/cache/convert"), 1,
                         control_dir=Path("/conv/control"), control_resolution=[1024, 1024]),
        ])
        text = toml.read_text()
        self.assertEqual(text.count("[[datasets]]"), 2)
        self.assertEqual(text.count("control_directory"), 1)
        self.assertEqual(text.count("control_resolution"), 1)
        self.assertIn('image_directory = "/ds"', text)
        self.assertIn('image_directory = "/conv"', text)

    def test_ltx_video(self):
        toml = self._write("ltx-23", [
            {"dir": Path("/ds"), "cache": Path("/cache"), "num_repeats": 1,
             "media": "video", "control_dir": None, "control_resolution": None},
        ], resolution=(768, 512),
            ltx_cfg={"resolution": "768x512x81", "fps": 25.0})
        text = toml.read_text()
        self.assertIn("video_directory", text)
        self.assertIn("target_frames = [81]", text)
        self.assertIn("target_fps = 25.0", text)

    def test_sdscripts_anima(self):
        toml = self._write("anima", [
            image_subset(Path("/ds"), Path("/cache"), 8),
        ], batch_size=8)
        text = toml.read_text()
        self.assertIn("[[datasets.subsets]]", text)
        self.assertIn("bucket_reso_steps = 64", text)
        self.assertIn("num_repeats = 8", text)


class TestSamplePrompts(unittest.TestCase):
    def test_line_has_trigger_and_flags(self):
        from trainero.presets import SAMPLE_PROMPT
        from trainero.training import sample_prompt_line

        line = sample_prompt_line(SAMPLE_PROMPT, "makima", [1024, 1024])
        self.assertTrue(line.startswith("makima, A young woman"))
        self.assertIn("--w 1024", line)
        self.assertIn("--h 1024", line)
        self.assertIn("--d 42", line)
        self.assertIn("--s 28", line)
        self.assertIn("--g 4.0", line)
        self.assertNotIn("\n", line)

    def test_no_trigger_means_no_prefix(self):
        from trainero.training import sample_prompt_line

        line = sample_prompt_line("a cat", "", [1024, 1024])
        self.assertTrue(line.startswith("a cat --w"))

    def test_video_gets_frame_count(self):
        from trainero.training import sample_prompt_line

        line = sample_prompt_line("a cat", "trg", [768, 512], frames=81)
        self.assertIn("--f 81", line)

    def test_krea2_sample_line_turns_cfg_on(self):
        """musubi only enables CFG when the line carries --n, and the scale is
        --l (--g is embedded guidance, which K2 ignores). Without both, the K2
        RAW model samples blurry by design — docs/krea2.md."""
        from trainero.presets import MODELS
        from trainero.training import sample_prompt_line

        line = sample_prompt_line("a cat", "trg", [1024, 1024],
                                  extra=MODELS["krea2"]["sample_args"])
        self.assertIn("--l 5.5", line)
        self.assertIn(" --n ", line)
        # the parser hands --n everything after it, so it must close the line
        self.assertGreater(line.index("--n"), line.index("--l"))
        self.assertGreater(line.index("--n"), line.index("--s"))

    def test_only_krea2_needs_sample_args(self):
        """The other musubi archs inject their own default negative prompt
        (qwen/flux/wan) or run their own CFG scheme (ideogram4)."""
        from trainero.presets import MODEL_ORDER, MODELS

        carriers = [k for k in MODEL_ORDER if "sample_args" in MODELS[k]]
        self.assertEqual(carriers, ["krea2"])

    def test_write_creates_single_line_file(self):
        import tempfile
        from trainero.training import write_sample_prompts

        with tempfile.TemporaryDirectory() as td:
            path = write_sample_prompts(Path(td) / "sample_prompts.txt",
                                        "a cat", "trg", [1024, 1024])
            lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
            self.assertEqual(len(lines), 1)


class TestSamplingInConfig(unittest.TestCase):
    def test_sampling_args_present(self):
        stats = {"items": 30, "images": 30, "videos": 0}
        sched = suggest_schedule("flux-klein", 30)
        cfg = build_train_config("flux-klein", {}, sched, stats, 48,
                                 Path("/tmp/ds.toml"), Path("/tmp/out"), "test", 1,
                                 sample_prompts=Path("/tmp/sp.txt"))
        self.assertEqual(cfg["sample_prompts"], "/tmp/sp.txt")
        self.assertEqual(cfg["sample_every_n_epochs"], 1)
        self.assertTrue(cfg["sample_at_first"])

    def test_no_sampling_when_not_requested(self):
        stats = {"items": 30, "images": 30, "videos": 0}
        sched = suggest_schedule("flux-klein", 30)
        cfg = build_train_config("flux-klein", {}, sched, stats, 48,
                                 Path("/tmp/ds.toml"), Path("/tmp/out"), "test", 1)
        self.assertNotIn("sample_prompts", cfg)


class TestComfyConvert(unittest.TestCase):
    def test_anima_uses_its_script(self):
        from trainero.training import comfy_convert_command

        cmd = comfy_convert_command("anima", Path("/out/a.safetensors"),
                                    Path("/out/a_comfy.safetensors"))
        self.assertIsNotNone(cmd)
        self.assertTrue(any("convert_anima_lora_to_comfy.py" in str(c) for c in cmd))
        self.assertIn("/out/a.safetensors", [str(c) for c in cmd])

    def test_models_without_the_key_convert_nothing(self):
        from trainero.training import comfy_convert_command

        for key in ("flux-klein", "qwen-image", "krea2", "wan-22"):
            self.assertIsNone(
                comfy_convert_command(key, Path("/o/a.safetensors"),
                                      Path("/o/a_comfy.safetensors")), key)

    def test_conversion_is_decided_by_the_preset_alone(self):
        """No UI switch, no override key: the preset is the single source of
        truth for whether a model needs converting."""
        import inspect

        from trainero import training

        self.assertNotIn("forced", inspect.signature(training.comfy_convert_command).parameters)
        self.assertNotIn("forced", inspect.signature(training.comfy_converter).parameters)
        src = Path(training.__file__).read_text()
        self.assertNotIn('overrides.get("comfy_convert")', src)


class TestCaptionModel(unittest.TestCase):
    """Os flags do captioner se chamam --grok_* por razões históricas; o modelo
    atrás deles é uma escolha, e ela muda com o modo."""

    def test_the_flag_default_is_still_gemini(self):
        from trainero.captioner import DEFAULT_CAPTION_MODEL

        self.assertEqual(DEFAULT_CAPTION_MODEL, "google/gemini-3.7-flash")

    def test_the_command_passes_the_mode_primary_to_openrouter(self):
        import os
        from pathlib import Path as P

        from trainero import captioner

        cmds = []

        class FakeJob:
            def log(self, *_): pass

            def run(self, cmd, cwd=None, **_kw):
                cmds.append([str(c) for c in cmd])

        saved = (captioner.ensure_engine, captioner.venv_python, captioner.engine_dir,
                 os.environ.get("OPENROUTER_API_KEY"))
        captioner.ensure_engine = lambda *_a: None
        captioner.venv_python = lambda _e: P("/v/python")
        captioner.engine_dir = lambda _e: P("/e/captioner")
        os.environ["OPENROUTER_API_KEY"] = "sk-test"
        try:
            for mode in ("lora", "style-rush"):
                cmds.clear()
                captioner.generate_captions(P("/ds"), "image", "generic-style",
                                            {"style_name": "makima"}, FakeJob(), mode=mode)
                cmd = cmds[0]
                self.assertIn("--grok_model", cmd)
                self.assertEqual(cmd[cmd.index("--grok_model") + 1],
                                 captioner.caption_models(mode)[0], mode)
                self.assertEqual(cmd[cmd.index("--grok_provider") + 1], "openrouter")
                self.assertIn("style_name=makima", cmd)
        finally:
            captioner.ensure_engine, captioner.venv_python, captioner.engine_dir = saved[:3]
            if saved[3] is None:
                os.environ.pop("OPENROUTER_API_KEY", None)
            else:
                os.environ["OPENROUTER_API_KEY"] = saved[3]


class TestDetectSource(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(detect_source("https://mega.nz/folder/abc#key"), "mega")
        self.assertEqual(detect_source("https://huggingface.co/datasets/u/r/resolve/main/d.zip"), "hf-file")
        self.assertEqual(detect_source("https://example.com/data.zip"), "url")
        self.assertEqual(detect_source("AdwolfCzar/meu_dataset"), "hf-repo")
        self.assertEqual(detect_source("/tmp"), "local-dir")
        self.assertEqual(detect_source(""), "empty")
        self.assertEqual(detect_source("???"), "unknown")

    def test_archive_ext(self):
        self.assertEqual(archive_ext("a.tar.gz"), ".tar.gz")
        self.assertEqual(archive_ext("a.ZIP"), ".zip")
        self.assertIsNone(archive_ext("a.safetensors"))


class TestSlug(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(slugify("Makima v1!"), "makima_v1")
        self.assertEqual(slugify("  "), "projeto")


if __name__ == "__main__":
    unittest.main()


class TestEnginePull(unittest.TestCase):
    """O clone do captioner é o único cujo conteúdo (os prompts) muda entre
    runs enquanto as dependências não. Pull no musubi arriscaria um venv bom."""

    def test_only_the_captioner_declares_pull(self):
        from trainero.presets import ENGINES

        pulling = {k for k, v in ENGINES.items() if v.get("pull")}
        self.assertEqual(pulling, {"captioner"})

    def test_an_existing_clone_is_fast_forwarded(self):
        import tempfile
        from pathlib import Path as P
        from unittest import mock

        from trainero import engines

        with tempfile.TemporaryDirectory() as td:
            dest = P(td) / "data_araknideo"
            (dest / ".git").mkdir(parents=True)
            ran = []

            class FakeJob:
                def log(self, *_): pass

                def run(self, cmd, cwd=None, **_kw):
                    ran.append(([str(c) for c in cmd], str(cwd) if cwd else None))

            with mock.patch.object(engines, "engine_dir", return_value=dest), \
                 mock.patch.object(engines, "is_installed", return_value=True):
                engines.ensure_engine("captioner", FakeJob())

            self.assertEqual(ran, [(["git", "pull", "--ff-only"], str(dest))])

    def test_a_training_engine_is_left_alone(self):
        import tempfile
        from pathlib import Path as P
        from unittest import mock

        from trainero import engines

        with tempfile.TemporaryDirectory() as td:
            dest = P(td) / "musubi-tuner"
            (dest / ".git").mkdir(parents=True)
            ran = []

            class FakeJob:
                def log(self, *_): pass

                def run(self, cmd, cwd=None, **_kw):
                    ran.append([str(c) for c in cmd])

            with mock.patch.object(engines, "engine_dir", return_value=dest), \
                 mock.patch.object(engines, "is_installed", return_value=True):
                engines.ensure_engine("musubi", FakeJob())

            self.assertEqual(ran, [], "pull no musubi pode quebrar um venv que funciona")

    def test_a_failed_pull_does_not_stop_the_job(self):
        """Sem rede, ou com commit local no clone, o pull falha. O prompt velho
        é ruim; não captionar nada é pior."""
        import tempfile
        from pathlib import Path as P
        from unittest import mock

        from trainero import engines

        with tempfile.TemporaryDirectory() as td:
            dest = P(td) / "data_araknideo"
            (dest / ".git").mkdir(parents=True)
            lines = []

            class FakeJob:
                def log(self, msg): lines.append(msg)

                def run(self, cmd, cwd=None, **_kw):
                    raise RuntimeError("fatal: not possible to fast-forward")

            with mock.patch.object(engines, "engine_dir", return_value=dest), \
                 mock.patch.object(engines, "is_installed", return_value=True):
                engines.ensure_engine("captioner", FakeJob())  # não pode levantar

            self.assertTrue(any("pull" in ln.lower() for ln in lines),
                            "o dono tem de ver que o clone ficou velho")


class TestSampleCaption(unittest.TestCase):
    """O sample só serve porque é o mesmo prompt a cada época. Uma escolha que
    mudasse entre resumes jogaria fora a comparação que ele existe para dar."""

    def _ds(self, root, captions):
        ds = root / "dataset"
        ds.mkdir(parents=True, exist_ok=True)
        for i, cap in enumerate(captions):
            (ds / f"img_{i:03d}.png").write_bytes(b"x")
            (ds / f"img_{i:03d}.txt").write_text(cap, encoding="utf-8")
        return ds

    def test_it_returns_a_caption_from_the_dataset(self):
        import tempfile
        from pathlib import Path as P

        from trainero.dataset import sample_caption

        with tempfile.TemporaryDirectory() as td:
            ds = self._ds(P(td), ["mkstyle, uma", "mkstyle, duas", "mkstyle, tres"])
            self.assertIn(sample_caption(ds),
                          {"mkstyle, uma", "mkstyle, duas", "mkstyle, tres"})

    def test_it_is_deterministic(self):
        import tempfile
        from pathlib import Path as P

        from trainero.dataset import sample_caption

        with tempfile.TemporaryDirectory() as td:
            ds = self._ds(P(td), [f"cap {i}" for i in range(20)])
            self.assertEqual(sample_caption(ds), sample_caption(ds))

    def test_an_uncaptioned_dataset_gives_an_empty_string(self):
        import tempfile
        from pathlib import Path as P

        from trainero.dataset import sample_caption

        with tempfile.TemporaryDirectory() as td:
            ds = P(td) / "dataset"
            ds.mkdir(parents=True)
            (ds / "img_000.png").write_bytes(b"x")
            self.assertEqual(sample_caption(ds), "")

    def test_the_training_falls_back_to_it_only_without_a_trigger(self):
        from pathlib import Path as P

        src = (P(__file__).resolve().parent.parent / "trainero" / "training.py").read_text()
        self.assertIn("ds.sample_caption(dataset_dir)", src)
        self.assertIn("not trigger.strip()", src)
