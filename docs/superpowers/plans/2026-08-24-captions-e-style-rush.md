# Captions e Style Rush — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** trocar a cascata de caption para uma lista ordenada por modo, instalar o prompt novo do `generic-style`, e fazer o Style Rush insistir até bater uma meta de sucessos em vez de entregar dataset curto em silêncio.

**Architecture:** `trainero/captioner.py` deixa de ter duas constantes de modelo e passa a resolver uma lista ordenada a partir do modo (`lora` começa pelo Muse Spark, `style-rush` pelo Gemini, porque só o Style Rush consome a lista de flagradas). `trainero/style_rush.py` troca o plano fixo de 50 slots por uma fila de tentativas com reserva de orçamento sob lock, o que torna a meta exata mesmo com muitos workers. O prompt em si vive no repo `adbrasi/data_araknideo` e o `ensure_engine` ganha um campo declarado `pull` para o clone não envelhecer.

**Tech Stack:** Python 3.11, stdlib only no servidor (`urllib`, `threading`, `concurrent.futures`), `unittest` para os testes, HTML/CSS/JS sem framework na UI.

**Spec:** `docs/superpowers/specs/2026-08-24-captions-e-style-rush-design.md`

## Global Constraints

- Rodar testes com `python3 -m pytest tests/<arquivo> -q` a partir de `/home/adolfocesar/projects/arrakis_trainero`. `python3 -m unittest` também funciona; use pytest.
- Respostas ao dono e mensagens de commit em **pt-BR**; código, identificadores e comentários em **inglês**.
- Commit depois de cada tarefa. Push só quando o dono mandar — **exceto** o repo `data_araknideo` na Tarefa 3, onde o dono já autorizou o push explicitamente.
- Nenhuma dependência nova. O venv do servidor fica sem pacote de rede de propósito.
- Ids de modelo, verbatim do spec:
  - `meta/muse-spark-1.2-contributor`
  - `google/gemini-3.7-flash`
  - `x-ai/grok-4.20`
- Defaults, verbatim do spec: `DEFAULT_CONVERT_TARGET = 100`, `CONVERT_WORKERS = 8`, `CAPTION_CONCURRENCY = 64`, `ATTEMPT_MULTIPLIER = 3`, `STYLE_RUSH_SCHEDULE["num_repeats"] = 2`.
- O segundo repositório é `/home/adolfocesar/projects/data_araknideo` (remote `adbrasi/data_araknideo`, branch `main`).

---

### Task 1: Cascata de caption por modo, e concorrência

Hoje `captioner.py` tem `DEFAULT_CAPTION_MODEL` e `FALLBACK_CAPTION_MODEL` soltas e `generate_captions` chama `_pass` uma vez para cada. Vira lista ordenada escolhida pelo modo. O `--grok_concurrency` nunca foi passado, então a captionagem roda no default 32 do tagger — baixo para datasets de mil imagens.

**Files:**
- Modify: `trainero/captioner.py:22-32` (constantes), `:63-89` (`_tagger_cmd`), `:129-147` (`record_flagged`), `:196-235` (`generate_captions`)
- Test: `tests/test_captioner.py`, `tests/test_core.py:333-376`

**Interfaces:**
- Consumes: nada de tarefas anteriores.
- Produces:
  - `caption_models(mode: str = "lora") -> list[str]`
  - `DEFAULT_CAPTION_MODEL: str` (continua existindo, vale `"google/gemini-3.7-flash"`)
  - `CAPTION_CONCURRENCY: int`
  - `record_flagged(dataset_dir: Path, items: list[Path], model: str) -> None` — **ganhou o terceiro parâmetro**
  - `generate_captions(dataset_dir, media: str, profile: str, prompt_vars: dict[str, str], job: Job, mode: str = "lora") -> None` — **ganhou `mode`**

- [ ] **Step 1: Escrever os testes que falham**

Em `tests/test_captioner.py`, troque o bloco de import no topo (linhas 17-19) por:

```python
from trainero import captioner
from trainero.captioner import (QUARANTINE_DIR, TAGGER_LOG, caption_models,
                                generate_captions, prune_stale_log,
                                quarantine_uncaptionable)

# Os testes deste arquivo exercem a cascata do Style Rush, que é a ordem em que
# o primário é o Gemini — é essa ordem que alimenta record_flagged.
PRIMARY, SECOND, THIRD = caption_models("style-rush")
```

Depois, no mesmo arquivo, substitua **todas** as ocorrências de `DEFAULT_CAPTION_MODEL` por `PRIMARY` e de `FALLBACK_CAPTION_MODEL` por `SECOND`. São estas linhas: 102, 111, 113, 122, 123, 128, 142, 156, 183, 184, 250, 259, 266.

E troque o helper `_run` (linha ~85) para passar o modo:

```python
def _run(ds: Path, job: _FakeJob, problem: str = "", mode: str = "style-rush"):
    """`problem` é o que o health check do OpenRouter reporta — vazio significa
    que a conta funciona, então um item ainda sem caption foi mesmo recusado."""
    with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "k"}, clear=False), \
         mock.patch.object(captioner, "ensure_engine"), \
         mock.patch.object(captioner, "venv_python", return_value=Path("py")), \
         mock.patch.object(captioner, "engine_dir", return_value=Path("/eng")), \
         mock.patch.object(captioner, "openrouter_problem", return_value=problem):
        generate_captions(ds, "image", "generic-style", {"style_name": "t"}, job, mode=mode)
```

Agora acrescente esta classe no fim de `tests/test_captioner.py`:

```python
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
            _run(ds, _FakeJob({SECOND: {"falta_000.jpg"}}), mode="lora")
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
```

E em `tests/test_core.py`, substitua a classe `TestCaptionModel` inteira (linhas 333-376) por:

```python
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
```

- [ ] **Step 2: Rodar e ver falhar**

```bash
python3 -m pytest tests/test_captioner.py tests/test_core.py -q
```

Esperado: FAIL com `ImportError: cannot import name 'caption_models'`.

- [ ] **Step 3: Implementar**

Em `trainero/captioner.py`, substitua o bloco de constantes (linhas 26-32) por:

```python
# The captioner's CLI names this stage "grok" for historical reasons; the flag
# takes any OpenRouter model id with vision.
MUSE_SPARK = "meta/muse-spark-1.2-contributor"
GEMINI_FLASH = "google/gemini-3.7-flash"
GROK = "x-ai/grok-4.20"

# The order depends on what the mode does with the primary's refusal. The only
# consumer of .caption_refused.json is Style Rush's content_flagged, which uses
# it to avoid paying for a slot gpt-image-2 would refuse — so there the primary
# has to be the model whose filter most resembles OpenAI's, which is Gemini. A
# plain LoRA has no paid phase and nothing reads the list, so its order is pure
# cost and quality: Muse Spark captions the same explicit material for a tenth
# of the price.
DEFAULT_CAPTION_MODELS = {
    "lora": [MUSE_SPARK, GEMINI_FLASH, GROK],
    "style-rush": [GEMINI_FLASH, MUSE_SPARK, GROK],
}

# What record_flagged means by "the strict model". Kept as a name because the
# flag file records it and the Style Rush half reasons about it.
DEFAULT_CAPTION_MODEL = DEFAULT_CAPTION_MODELS["style-rush"][0]

# The tagger's own default is 32, which is what a thousand-image dataset waits
# on. One knob, not one per model.
CAPTION_CONCURRENCY = int(os.environ.get("CAPTION_CONCURRENCY", "64"))


def caption_models(mode: str = "lora") -> list[str]:
    """The cascade for this mode, cheapest-useful first.

    CAPTION_MODELS overrides every mode at once: a comma-separated list is one
    knob for a cascade of any length, where numbered env vars would silently
    pin it to the length they were written for.
    """
    override = os.environ.get("CAPTION_MODELS", "").strip()
    if override:
        return [m.strip() for m in override.split(",") if m.strip()]
    return list(DEFAULT_CAPTION_MODELS.get(mode) or DEFAULT_CAPTION_MODELS["lora"])
```

Em `_tagger_cmd`, acrescente o flag de concorrência logo depois de `"--grok_model", model,`:

```python
        "--grok_model", model,
        "--grok_concurrency", str(CAPTION_CONCURRENCY),
```

Troque a assinatura e o corpo de `record_flagged`:

```python
def record_flagged(dataset_dir: Path, items: list[Path], model: str) -> None:
    """Remember which items the primary model would not caption.

    They are the ones a content filter objects to, which is the same objection
    gpt-image-2 raises in the Style Rush conversion phase — and there a refusal
    costs a paid slot. Written next to the dataset so that phase can read it
    without re-running any model. `model` is recorded because the primary is
    not the same in every mode, and a list is only as meaningful as the filter
    that produced it.
    """
    path = dataset_dir / FLAGGED_FILE
    try:
        known = set(json.loads(path.read_text()).get("refused_by_primary", []))
    except (OSError, json.JSONDecodeError, AttributeError):
        known = set()
    known |= {i.name for i in items}
    path.write_text(json.dumps(
        {"model": model, "refused_by_primary": sorted(known)},
        indent=2, ensure_ascii=False))
```

E substitua o corpo de `generate_captions` (da linha `refused_by_primary = _pass(...)` até `record_flagged(...)`) por:

```python
def generate_captions(dataset_dir, media: str, profile: str, prompt_vars: dict[str, str],
                      job: Job, mode: str = "lora") -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise JobFailed("defina OPENROUTER_API_KEY para gerar captions com LLM")
    dataset_dir = Path(dataset_dir)
    ensure_engine("captioner", job)

    models = caption_models(mode)
    job.log(f"Cascata de caption: {' → '.join(models)}")

    missing = _pass(dataset_dir, media, profile, prompt_vars, job, models[0])
    refused_by_primary = missing
    for model in models[1:]:
        if not missing:
            break
        job.log(f"{len(missing)} sem caption — tentando {model}: "
                f"{', '.join(p.name for p in missing[:5])}"
                f"{'…' if len(missing) > 5 else ''}")
        missing = _pass(dataset_dir, media, profile, prompt_vars, job, model)

    # Flag only what a later model rescued. "Still uncaptioned after pass 1" has
    # two causes that look identical on disk — the model refused it, or the API
    # was down — and only one of them justifies excluding the image from the
    # paid conversion phase forever. A second model producing a caption is the
    # proof that the first one refused on content, not that the account died.
    rescued = [p for p in refused_by_primary if p not in set(missing)]
    if rescued:
        record_flagged(dataset_dir, rescued, models[0])
```

O resto de `generate_captions` (o bloco `if missing:` com `openrouter_problem` e a quarentena, e o `job.log("✔ Captions geradas.")`) fica exatamente como está.

- [ ] **Step 4: Rodar e ver passar**

```bash
python3 -m pytest tests/test_captioner.py tests/test_core.py -q
```

Esperado: PASS.

- [ ] **Step 5: Commit**

```bash
git add trainero/captioner.py tests/test_captioner.py tests/test_core.py
git commit -m "feat(captions): cascata de N modelos com ordem por modo e mais concorrencia"
```

---

### Task 2: `clear_captions`

O dono precisa poder refazer todas as captions de um dataset que já chegou com `.txt`. Apagar só os `.txt` não basta: o tagger pula tudo que está no `.tagger_log.json`, então o botão não faria nada.

**Files:**
- Modify: `trainero/captioner.py` (função nova, depois de `prune_stale_log`)
- Test: `tests/test_captioner.py`

**Interfaces:**
- Consumes: `TAGGER_LOG`, `FLAGGED_FILE`, `QUARANTINE_DIR` de `trainero/captioner.py` (Tarefa 1).
- Produces: `clear_captions(dataset_dir: Path, job: Job | None = None) -> int` — devolve quantos `.txt` foram apagados.

- [ ] **Step 1: Escrever os testes que falham**

Acrescente em `tests/test_captioner.py`:

```python
class TestClearCaptions(unittest.TestCase):
    def test_every_caption_is_deleted_and_counted(self):
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=6, uncaptioned=2)
            removed = captioner.clear_captions(ds, _FakeJob())
            self.assertEqual(removed, 6)
            self.assertEqual(list(ds.glob("*.txt")), [])
            self.assertEqual(len(list(ds.glob("*.jpg"))), 8, "nenhuma imagem pode sumir")

    def test_the_tagger_log_goes_too(self):
        """O tagger pula o que está no log. Sem apagá-lo, refazer não refaz nada."""
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=3, uncaptioned=0)
            captioner.clear_captions(ds, _FakeJob())
            self.assertFalse((ds / TAGGER_LOG).exists())

    def test_the_flag_file_goes_too(self):
        """A lista de flagradas é uma conclusão dos modelos antigos sobre estas
        imagens. Refazer as captions é justamente refazer essa conclusão."""
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=1, uncaptioned=0)
            captioner.record_flagged(ds, [ds / "ok_000.jpg"], "algum/modelo")
            captioner.clear_captions(ds, _FakeJob())
            self.assertEqual(captioner.flagged_names(ds), set())

    def test_the_quarantine_is_not_touched(self):
        """descartadas/ já saiu do dataset e é a única cópia do que foi movido."""
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=2, uncaptioned=0)
            quarantine = ds.parent / QUARANTINE_DIR
            quarantine.mkdir(parents=True, exist_ok=True)
            (quarantine / "fora.jpg").write_bytes(b"x")
            (quarantine / "fora.txt").write_text("uma caption")

            captioner.clear_captions(ds, _FakeJob())

            self.assertTrue((quarantine / "fora.jpg").exists())
            self.assertTrue((quarantine / "fora.txt").exists())

    def test_an_already_clean_dataset_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=0, uncaptioned=3)
            (ds / TAGGER_LOG).unlink()
            self.assertEqual(captioner.clear_captions(ds, _FakeJob()), 0)

    def test_the_caption_is_written_again_after_a_clear(self):
        """O teste que prova o ponto todo: com o log apagado, o passe seguinte
        reescreve o .txt de um item que já tinha caption."""
        with tempfile.TemporaryDirectory() as td:
            ds = _dataset(Path(td), captioned=2, uncaptioned=0)
            captioner.clear_captions(ds, _FakeJob())
            _run(ds, _FakeJob({PRIMARY: {"ok_000.jpg", "ok_001.jpg"}}))
            self.assertTrue((ds / "ok_000.txt").exists())
            self.assertTrue((ds / "ok_001.txt").exists())
```

- [ ] **Step 2: Rodar e ver falhar**

```bash
python3 -m pytest tests/test_captioner.py -k ClearCaptions -q
```

Esperado: FAIL com `AttributeError: module 'trainero.captioner' has no attribute 'clear_captions'`.

- [ ] **Step 3: Implementar**

Em `trainero/captioner.py`, logo depois de `prune_stale_log`:

```python
def clear_captions(dataset_dir, job: Job | None = None) -> int:
    """Delete every caption in the dataset so the next run writes them all again.

    Deleting only the .txt would do nothing: the tagger skips anything already
    in its processing log, which is exactly the property that makes an
    interrupted run resume for free. The log and the flag file are conclusions
    the old models drew about these images, and redoing the captions is redoing
    those conclusions — so all three go together or none of them do.

    QUARANTINE_DIR is not touched. It sits beside the dataset, it already left,
    and it is the owner's only copy of what was moved out.
    """
    dataset_dir = Path(dataset_dir)
    removed = 0
    for item in dataset_dir.iterdir():
        if not item.is_file() or item.suffix.lower() not in MEDIA_EXTS:
            continue
        txt = item.with_suffix(".txt")
        try:
            txt.unlink()
        except OSError:
            continue
        removed += 1
    for bookkeeping in (dataset_dir / TAGGER_LOG, dataset_dir / FLAGGED_FILE):
        bookkeeping.unlink(missing_ok=True)
    if job:
        job.log(f"{removed} captions apagadas — todas serão reescritas.")
    return removed
```

E acrescente o import de extensões no topo do arquivo, junto dos outros:

```python
from .config import IMAGE_EXTS, VIDEO_EXTS
```

com, logo abaixo das constantes de arquivo:

```python
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS
```

- [ ] **Step 4: Rodar e ver passar**

```bash
python3 -m pytest tests/test_captioner.py -q
```

Esperado: PASS.

- [ ] **Step 5: Commit**

```bash
git add trainero/captioner.py tests/test_captioner.py
git commit -m "feat(captions): clear_captions apaga txt, log do tagger e lista de flagradas"
```

---

### Task 3: Prompt novo do `generic-style` (repo `data_araknideo`)

Este é o **outro repositório**. O prompt novo, literal, diz *"You receive an image and nothing else. The image is the only source of truth"* — mas o pipeline roda o pixai antes e injeta `{tags}`. Colar sem resolver deixa o modelo com duas ordens opostas.

**Files (todos em `/home/adolfocesar/projects/data_araknideo`):**
- Modify: `prompts/image/generic-style/system_prompt.md` (substituição integral)
- Modify: `prompts/image/generic-style/user_prompt.md` (substituição integral)
- Não mexer: `prompts/image/generic-style/profile.json` — a variável `style_name` continua sendo a trigger.

**Interfaces:**
- Consumes: `/home/adolfocesar/projects/arrakis_trainero/new_system_prompt.md` como fonte do texto.
- Produces: o perfil `generic-style`, consumido por `trainero/captioner.py` via `--prompt_profile generic-style`.

- [ ] **Step 1: Escrever o system prompt**

Copie `new_system_prompt.md` para `prompts/image/generic-style/system_prompt.md` com **três** mudanças, e só elas:

1. Troque a **Regra 1** de `**The image is the only source of truth.** Describe what is actually visible.` por:

```markdown
1. **The image is the source of truth.** Describe what is actually visible. Booru tags are
   provided alongside it as supporting evidence only: use them to confirm a detail you can
   see, and to recover a character name, series item, or object you would not recognise on
   your own. A tag that contradicts the image is wrong — the image wins, every time. Never
   copy tag syntax into the caption: no underscores, no `(series)` suffixes, no keyword
   lists. Never mention a tag you cannot see in the image.
```

2. Apague a seção `## Length` inteira (o cabeçalho, as duas linhas de bullet e a linha
   "Compact, dense, visual...") e ponha no lugar:

```markdown
## Density

There is no word limit. Longer is better when every clause carries visual information.
Write until the frame is described. No sentence that could be deleted without losing
information.
```

3. No cabeçalho, troque a primeira frase `You receive **an image and nothing else**.` por
   `You receive **an image**, and a list of booru tags describing it.`

Tudo o mais entra literal: política de conteúdo explícito, a ordem do camera report (seções 1 a 8), o que NÃO descrever, o template, os três exemplos.

- [ ] **Step 2: Escrever o user prompt**

Substitua `prompts/image/generic-style/user_prompt.md` inteiro por:

```markdown
Analyze the provided image. Produce the JSON caption output.

The caption must begin with `{style_name},` and then continue as natural language.

**Input image:** [The attached image]
**Booru Tags (supporting evidence, not ground truth):**
{tags}
---
Look at the image carefully. The image is the source of truth. Use the tags to confirm
details you can already see and to recover names you would not recognise on your own; ignore
any tag the image contradicts, and never write a tag you cannot see. Produce the JSON output
exactly as required by the system instructions.
```

- [ ] **Step 3: Conferir que os placeholders sobreviveram**

```bash
cd /home/adolfocesar/projects/data_araknideo
grep -c "{style_name}" prompts/image/generic-style/system_prompt.md
grep -c "{tags}" prompts/image/generic-style/user_prompt.md
grep -ci "length" prompts/image/generic-style/system_prompt.md
```

Esperado: o primeiro `>= 5`, o segundo `1`, o terceiro `0`. Se `{style_name}` sumir, toda caption sai sem trigger word e o LoRA não aprende nada — este grep é o teste desta tarefa.

- [ ] **Step 4: Commit e push**

O dono autorizou o push deste repositório explicitamente.

```bash
cd /home/adolfocesar/projects/data_araknideo
git add prompts/image/generic-style/system_prompt.md prompts/image/generic-style/user_prompt.md
git commit -m "feat(prompts): generic-style vira vision-first com as tags como apoio"
git push origin main
```

---

### Task 4: `ensure_engine` atualiza o clone do captioner

`ensure_engine` clona uma vez e nunca dá pull: se `.git` existe, loga "já clonado" e sai. Um pod que já roda há dias captionaria o dataset inteiro com o prompt velho da Tarefa 3, e o dono só descobriria depois do treino.

**Files:**
- Modify: `trainero/presets.py:42-48` (spec do engine `captioner`)
- Modify: `trainero/engines.py:41-56` (`ensure_engine`)
- Test: `tests/test_core.py`

**Interfaces:**
- Consumes: `ENGINES` de `trainero/presets.py`.
- Produces: campo `"pull": True` no spec do engine `captioner`; `ensure_engine` inalterado na assinatura.

- [ ] **Step 1: Escrever os testes que falham**

Acrescente em `tests/test_core.py`:

```python
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
```

- [ ] **Step 2: Rodar e ver falhar**

```bash
python3 -m pytest tests/test_core.py -k EnginePull -q
```

Esperado: FAIL — `pulling` sai vazio e `ran` sai vazio.

- [ ] **Step 3: Implementar**

Em `trainero/presets.py`, no spec do engine `captioner`, acrescente o campo:

```python
    "captioner": {
        "repo": "https://github.com/adbrasi/data_araknideo.git",
        "branch": "main",
        "dir": "data_araknideo",
        "install": "requirements",
        "script_prefix": "",
        # The only engine whose *content* changes between runs while its
        # dependencies do not: the caption prompts live here. Without this a
        # long-lived pod captions a whole dataset with a stale prompt and the
        # owner only finds out after the training.
        "pull": True,
    },
```

Em `trainero/engines.py`, dentro de `ensure_engine`, troque o `else` do clone:

```python
    if not (dest / ".git").exists():
        job.log(f"Clonando {spec['repo']} ({spec['branch']})...")
        job.run(["git", "clone", "--depth", "1", "--single-branch",
                 "--branch", spec["branch"], spec["repo"], str(dest)])
    elif spec.get("pull"):
        # A stale prompt is bad; a phase that refuses to run is worse. Never
        # let a network hiccup or a local commit in the clone stop the job.
        try:
            job.run(["git", "pull", "--ff-only"], cwd=dest)
        except Exception as exc:
            job.log(f"⚠ git pull falhou em {name} ({exc}) — seguindo com o clone atual.")
    else:
        job.log(f"Engine {name} já clonado.")
```

- [ ] **Step 4: Rodar e ver passar**

```bash
python3 -m pytest tests/test_core.py -q
```

Esperado: PASS.

- [ ] **Step 5: Commit**

```bash
git add trainero/presets.py trainero/engines.py tests/test_core.py
git commit -m "fix(engines): clone do captioner e atualizado a cada run"
```

---

### Task 5: mais 50 prompts de estilo

`data/style_prompts.txt` tem 50 linhas. A meta de conversão passa a ser 100, e mais variedade de "estilo ruim de entrada" é o que faz o LoRA generalizar a conversão.

**Files:**
- Modify: `data/style_prompts.txt` (50 → 100 linhas)
- Test: `tests/test_style_rush.py:60-64`

**Interfaces:**
- Consumes: `/tmp/claude-1000/-home-adolfocesar-projects-arrakis-trainero/d0300d44-7f24-4af5-95fa-8ec95f0fde99/scratchpad/new_style_prompts.txt` — 50 linhas já escritas e verificadas (sem duplicata interna, sem sobreposição com as atuais, sem iluminação nem cor).
- Produces: `data/style_prompts.txt` com 100 linhas distintas.

- [ ] **Step 1: Escrever o teste que falha**

Em `tests/test_style_rush.py`, substitua a classe `TestStylePrompts` por:

```python
class TestStylePrompts(unittest.TestCase):
    def test_a_hundred_distinct_prompts(self):
        prompts = load_style_prompts()
        self.assertEqual(len(prompts), 100)
        self.assertEqual(len(set(prompts)), 100, "prompts must be distinct")

    def test_no_prompt_is_about_light_or_color_grading(self):
        """Luz e cor não são conversão de estilo: o controle tem de chegar num
        traço diferente, não na mesma arte com outro filtro."""
        banned = ("lighting", "rim light", "color grade", "color grading",
                  "warmer palette", "cooler palette")
        for prompt in load_style_prompts():
            low = prompt.lower()
            for word in banned:
                self.assertNotIn(word, low, prompt)
```

- [ ] **Step 2: Rodar e ver falhar**

```bash
python3 -m pytest tests/test_style_rush.py -k StylePrompts -q
```

Esperado: FAIL com `AssertionError: 50 != 100`.

- [ ] **Step 3: Anexar as 50 linhas**

```bash
cd /home/adolfocesar/projects/arrakis_trainero
SCRATCH=/tmp/claude-1000/-home-adolfocesar-projects-arrakis-trainero/d0300d44-7f24-4af5-95fa-8ec95f0fde99/scratchpad/new_style_prompts.txt
tail -c1 data/style_prompts.txt | read -r _ || echo >> data/style_prompts.txt
cat "$SCRATCH" >> data/style_prompts.txt
grep -c . data/style_prompts.txt          # esperado: 100
sort data/style_prompts.txt | uniq -d     # esperado: nenhuma saída
```

- [ ] **Step 4: Rodar e ver passar**

```bash
python3 -m pytest tests/test_style_rush.py -k StylePrompts -q
```

Esperado: PASS.

- [ ] **Step 5: Commit**

```bash
git add data/style_prompts.txt tests/test_style_rush.py
git commit -m "feat(style-rush): 50 prompts novos de estilo, so midia e tecnica"
```

---

### Task 6: Style Rush com meta de sucessos

O coração do plano. Hoje `plan_slots` monta 50 slots fixos, slot `i` amarrado ao prompt `i`, com imagem primária e uma de fallback; recusa nas duas descarta o slot. O dataset sai curto e o treino roda em cima disso sem um único erro.

**Files:**
- Modify: `trainero/style_rush.py:31-32` (constantes), `:50-95` (`load_style_prompts`, `plan_slots`), `:97-98` (`RETRIABLE_ATTEMPTS`), `:170-296` (`build_convert_dataset`)
- Test: `tests/test_style_rush.py`

**Interfaces:**
- Consumes: `load_style_prompts()` da Tarefa 5; `imagegen.generate`, `imagegen.RefusedError`, `imagegen.RetriableError`, `imagegen.AccountError`.
- Produces:
  - `DEFAULT_CONVERT_TARGET: int` (= 100)
  - `CONVERT_WORKERS: int` (= 8)
  - `ATTEMPT_MULTIPLIER: int` (= 3)
  - `plan_attempts(images: list[Path], prompts: list[str], target: int, avoid: set[str] | None = None) -> list[dict]` — cada item tem as chaves `attempt`, `prompt`, `source`
  - `build_convert_dataset(base_dir, convert_dir, trigger, job, generate=imagegen.generate, workers=CONVERT_WORKERS, target=DEFAULT_CONVERT_TARGET) -> dict`
  - `SLOT_COUNT` e `plan_slots` **deixam de existir**

- [ ] **Step 1: Escrever os testes que falham**

Em `tests/test_style_rush.py`, troque o import (linhas 12-13) por:

```python
from trainero.style_rush import (ATTEMPT_MULTIPLIER, CAPTION_TEMPLATE,
                                 DEFAULT_CONVERT_TARGET, MANIFEST_NAME,
                                 build_convert_dataset, load_style_prompts,
                                 plan_attempts)
```

Apague as classes `TestPlanSlots` e `TestBuildConvertDataset` inteiras (linhas ~66 até o fim de `test_resume_does_not_regenerate`) e ponha no lugar:

```python
class TestPlanAttempts(unittest.TestCase):
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
        """Repetir o par (prompt, imagem) gasta uma tentativa que não pode dar
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

    @staticmethod
    def _imgs(n):
        return [Path(f"/ds/img_{i:03d}.png") for i in range(n)]


class TestBuildConvertDataset(unittest.TestCase):
    TARGET = 12  # pequeno de propósito: o teste não pode depender do default

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
        """Sem reserva de orçamento sob lock, N workers compram até N-1 imagens
        depois da meta. Com 16 workers isso é dinheiro real."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 40)
            convert = root / "dataset_convert"
            calls = []
            lock = __import__("threading").Lock()

            def fake_generate(prompt, image_path, timeout=300.0):
                with lock:
                    calls.append(str(image_path))
                return _png_bytes(), 0.011

            result = build_convert_dataset(base, convert, "makima", _FakeJob(),
                                           generate=fake_generate, workers=16,
                                           target=self.TARGET)

            self.assertEqual(result["pairs"], self.TARGET)
            self.assertEqual(len(calls), self.TARGET, "pagou por imagem além da meta")

    def test_a_refusal_keeps_going_until_the_target_is_met(self):
        """O comportamento que este design existe para consertar: antes, uma
        recusa custava um slot e o dataset saía curto em silêncio."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 20)
            convert = root / "dataset_convert"
            refused_names = {"img_000.png", "img_001.jpg", "img_002.png"}

            def fake_generate(prompt, image_path, timeout=300.0):
                if Path(image_path).name in refused_names:
                    raise RefusedError("moderação recusou a imagem")
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
                    raise RefusedError("moderação recusou a imagem")
                return _png_bytes(), 0.011

            build_convert_dataset(base, convert, "makima", _FakeJob(),
                                  generate=fake_generate, workers=1,
                                  target=self.TARGET)

            self.assertEqual(seen.count("img_000.png"), 1)

    def test_the_attempt_ceiling_ends_a_hopeless_run(self):
        """Tudo recusado tem de terminar, não girar até a conta secar."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 6)
            convert = root / "dataset_convert"

            def fake_generate(prompt, image_path, timeout=300.0):
                raise RefusedError("moderação recusou a imagem")

            from trainero.jobs import JobFailed

            with self.assertRaises(JobFailed):
                build_convert_dataset(base, convert, "makima", _FakeJob(),
                                      generate=fake_generate, workers=1,
                                      target=self.TARGET)

    def test_a_short_run_says_so_out_loud(self):
        """Truncar em silêncio é o defeito original. Se a meta não bate, o log
        tem de dizer quanto faltou."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = _make_dataset(root, 4)
            convert = root / "dataset_convert"
            job = _FakeJob()

            def fake_generate(prompt, image_path, timeout=300.0):
                if Path(image_path).name != "img_000.png":
                    raise RefusedError("moderação recusou a imagem")
                return _png_bytes(), 0.011

            result = build_convert_dataset(base, convert, "makima", job,
                                           generate=fake_generate, workers=1,
                                           target=self.TARGET)

            self.assertLess(result["pairs"], self.TARGET)
            self.assertTrue(any("meta" in ln.lower() for ln in job.lines),
                            "o log tem de dizer que o dataset saiu curto")

    def test_control_is_the_generated_image_and_target_is_the_original(self):
        """Trocar os dois ensina o LoRA a conversão inversa — um treino inteiro
        perdido, sem sintoma nenhum até a inferência."""
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
                                 "control = saída do GPT Image")
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
            # erro transitório repete a MESMA imagem; cair para a próxima
            # entrada da fila seria a recuperação errada
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
        """Subir a meta de 12 para 18 tem de comprar 6, não 18."""
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
```

- [ ] **Step 2: Rodar e ver falhar**

```bash
python3 -m pytest tests/test_style_rush.py -q
```

Esperado: FAIL com `ImportError: cannot import name 'plan_attempts'`.

- [ ] **Step 3: Implementar**

Em `trainero/style_rush.py`, troque `SLOT_COUNT = 50` (linha 31) por:

```python
# The conversion half stops when it has this many *successes*, not when it has
# tried this many times. A refusal used to cost a pair; now it costs an attempt.
DEFAULT_CONVERT_TARGET = 100
# Enough queue to absorb refusals, and a hard end for a dataset where every
# image is refused — otherwise the loop runs until the account is empty.
ATTEMPT_MULTIPLIER = 3
# gpt-image-2 calls are slow and independent. The budget reservation below is
# what makes raising this safe.
CONVERT_WORKERS = int(os.environ.get("CONVERT_WORKERS", "8"))
```

e acrescente `import os` no topo do arquivo, junto dos outros imports da stdlib.

Substitua `load_style_prompts` por:

```python
def load_style_prompts(path: Path | None = None) -> list[str]:
    """The style prompts, one per line. Blank lines and '#' comments ignored.

    No minimum count any more: the attempt queue cycles the list, so any number
    of prompts serves any target.
    """
    src = path or PROMPTS_FILE
    try:
        lines = src.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FileNotFoundError(f"style prompts file missing: {src}") from exc
    prompts = [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]
    if not prompts:
        raise ValueError(f"{src} não tem nenhum prompt de estilo")
    return prompts
```

Substitua `plan_slots` inteira por:

```python
def plan_attempts(images: list[Path], prompts: list[str], target: int,
                  avoid: set[str] | None = None) -> list[dict]:
    """An ordered queue of (prompt, source) attempts, longer than the target.

    A queue the length of the target dies on its first refusal, which is the
    bug this replaces. ATTEMPT_MULTIPLIER gives the run room to route around
    refused images and, just as importantly, a place to stop: a dataset where
    every image is refused ends here instead of spending the whole balance.

    The image index carries an extra `i // len(prompts)` so that when the prompt
    list wraps, the pairing shifts by one image. Without it the queue would
    repeat the exact (prompt, source) pair it already spent an attempt on.

    Selection is deterministic so a resumed run rebuilds the same queue.

    `avoid` holds file names a content filter has already objected to — the
    primary caption model refused them, or gpt-image-2 itself did on an earlier
    run. Feeding those back in buys another refusal at full price.
    """
    if not images:
        raise ValueError("dataset base vazio — nada para converter")
    avoid = avoid or set()
    usable = [p for p in images if p.name not in avoid]
    if not usable:
        raise ValueError(
            f"todas as {len(images)} imagens do dataset já foram recusadas por "
            f"filtro de conteúdo — não há o que mandar para o gpt-image-2")

    order = sorted(str(p) for p in usable)
    rng = random.Random(PLAN_SEED)
    rng.shuffle(order)

    return [{"attempt": f"slot_{i:03d}",
             "prompt": prompts[i % len(prompts)],
             "source": order[(i + i // len(prompts)) % len(order)]}
            for i in range(target * ATTEMPT_MULTIPLIER)]
```

Substitua `build_convert_dataset` inteira por:

```python
def build_convert_dataset(base_dir: Path, convert_dir: Path, trigger: str, job,
                          generate=imagegen.generate, workers: int = CONVERT_WORKERS,
                          target: int = DEFAULT_CONVERT_TARGET) -> dict:
    """Produce `target` (control, target, caption) triples under convert_dir.

    control/<slot>.png  the restyled image, cropped to the target's aspect ratio
    <slot>.png          an untouched copy of the source image (the target)
    <slot>.txt          the same caption for every pair

    The pair is named after the attempt that produced it, so the numbering has
    gaps wherever an attempt was refused. Nothing downstream cares: musubi walks
    the directory. Numbering by success count instead would mean two numbering
    schemes living side by side, one for work and one for output.

    Pairs already recorded as ok in the manifest count toward the target, so a
    cancelled run resumes without paying for them again.
    """
    images = base_images(base_dir)
    if not images:
        raise JobFailed("dataset base vazio — importe as imagens antes de treinar")

    avoid = content_flagged(base_dir, convert_dir)
    if avoid:
        job.log(f"{len(avoid)} imagens fora da conversão por recusa de filtro de "
                f"conteúdo (o gpt-image-2 recusaria de novo, cobrando a tentativa).")
    try:
        attempts = plan_attempts(images, load_style_prompts(), target, avoid=avoid)
    except (ValueError, FileNotFoundError) as exc:
        raise JobFailed(str(exc))

    control_dir = convert_dir / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(convert_dir)
    caption = CAPTION_TEMPLATE.format(trigger=trigger)

    lock = threading.Lock()
    totals = {"pairs": 0, "refused": 0, "failed": 0, "cost": 0.0}

    def done(name: str) -> bool:
        entry = manifest["slots"].get(name)
        return bool(entry and entry.get("status") == "ok"
                    and (convert_dir / f"{name}.png").exists()
                    and (control_dir / f"{name}.png").exists())

    pending = [a for a in attempts if not done(a["attempt"])]
    totals["pairs"] = len(attempts) - len(pending)
    # what is still to buy. Guarded by `lock` from here on.
    budget = target - totals["pairs"]

    # the caption carries the trigger word, so a run with a changed trigger has
    # to rewrite it even for pairs it will not regenerate
    for existing in convert_dir.glob("slot_*.png"):
        existing.with_suffix(".txt").write_text(caption, encoding="utf-8")

    if budget > 0:
        job.log(f"Gerando {budget} imagens de conversão com {imagegen.MODEL} "
                f"(quality low, {workers} em paralelo) — "
                f"~${imagegen.COST_PER_IMAGE * budget:.2f}")
    else:
        job.log(f"Dataset de conversão já completo ({totals['pairs']} pares).")

    # Refusals live per image, not per attempt: with a target far above the
    # dataset size the same picture is the source many times over, so without
    # this every worker pays to rediscover the same no.
    refused_now: set[str] = set(manifest.get("refused_images", []))

    def record(name: str, entry: dict, **deltas) -> None:
        """Persist one attempt's outcome. The manifest is written on every
        attempt, not once at the end: a cancelled run (job.check_cancel raises)
        or a dead pod would otherwise lose the record of images already paid."""
        with lock:
            for k, v in deltas.items():
                totals[k] += v
            manifest["slots"][name] = entry
            _save_manifest(convert_dir, manifest)

    def note_refused(source: str) -> None:
        with lock:
            refused_now.add(Path(source).name)
            manifest["refused_images"] = sorted(refused_now)
            _save_manifest(convert_dir, manifest)

    def claim() -> bool:
        """Take one slot of budget before spending money on it.

        This is what makes the target exact. Without it, N workers in flight
        when the last pair lands each go on to buy one more image."""
        nonlocal budget
        with lock:
            if budget <= 0:
                return False
            budget -= 1
            return True

    def release() -> None:
        """Give the budget back — the attempt produced no pair."""
        nonlocal budget
        with lock:
            budget += 1

    # ThreadPoolExecutor.map keeps dispatching after a task raises — the error
    # only surfaces when the results are consumed. Without this the whole queue
    # still runs, and a dead account is asked hundreds more times.
    account_dead: list[Exception] = []

    def work(attempt: dict) -> None:
        if account_dead:
            return
        job.check_cancel()
        name, source = attempt["attempt"], attempt["source"]
        with lock:
            if Path(source).name in refused_now:
                return
        if not claim():
            return                      # the target is already met
        try:
            png, cost = _generate_with_retries(generate, attempt["prompt"], Path(source))
        except imagegen.AccountError as exc:
            release()
            with lock:                  # not this attempt's problem — it ends the phase
                account_dead.append(exc)
            return
        except imagegen.RefusedError as exc:
            release()
            note_refused(source)
            record(name, {"status": "refused", "prompt": attempt["prompt"],
                          "source": source, "error": str(exc)}, refused=1)
            return
        except Cancelled:
            release()
            raise
        except Exception as exc:
            # failed before the call was billed: one unreadable file (an .avif
            # this Pillow cannot open) costs its own attempt, never the phase
            release()
            record(name, {"status": "failed", "prompt": attempt["prompt"], "source": source,
                          "error": f"{type(exc).__name__}: {exc}"}, failed=1)
            job.log(f"⚠ {name}: {type(exc).__name__}: {exc}")
            return

        # billed from here on, so the cost is counted whatever happens next
        try:
            png = fit_control_to_target(png, Path(source))
            (control_dir / f"{name}.png").write_bytes(png)
            shutil.copyfile(source, convert_dir / f"{name}.png")
            (convert_dir / f"{name}.txt").write_text(caption, encoding="utf-8")
        except Cancelled:
            release()
            raise
        except Exception as exc:
            release()                   # paid, but there is no pair to show for it
            record(name, {"status": "failed", "prompt": attempt["prompt"], "source": source,
                          "paid": True, "cost": cost,
                          "error": f"{type(exc).__name__}: {exc}"},
                   failed=1, cost=cost)
            job.log(f"⚠ {name}: imagem paga mas não gravada "
                    f"({type(exc).__name__}: {exc}) — ${cost:.4f} perdidos")
            return

        record(name, {"status": "ok", "prompt": attempt["prompt"], "source": source,
                      "cost": cost}, pairs=1, cost=cost)

    if budget > 0:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(work, pending))
        if account_dead:
            _save_manifest(convert_dir, manifest)
            raise JobFailed(
                f"{account_dead[0]} — {totals['pairs']} de {target} pares prontos. "
                f"Resolva e rode de novo: o que já foi pago não é refeito.")

    if totals["pairs"] == 0:
        raise JobFailed(
            "o dataset de conversão ficou vazio — todas as imagens foram recusadas ou "
            "falharam. Sem ele o LoRA não aprende a converter estilo.")

    if totals["pairs"] < target:
        job.log(f"⚠ meta de {target} pares não atingida: {totals['pairs']} prontos depois "
                f"de {len(pending)} tentativas. {len(refused_now)} imagens do dataset "
                f"foram recusadas por moderação.")

    job.log(f"✔ Dataset de conversão: {totals['pairs']} pares, "
            f"{totals['refused']} recusas, custo ~${totals['cost']:.2f}")
    return totals
```

- [ ] **Step 4: Rodar e ver passar**

```bash
python3 -m pytest tests/test_style_rush.py -q
```

Esperado: PASS.

- [ ] **Step 5: Commit**

```bash
git add trainero/style_rush.py tests/test_style_rush.py
git commit -m "feat(style-rush): conversao insiste ate bater a meta de pares"
```

---

### Task 7: `num_repeats` do Style Rush vai para 2

**Files:**
- Modify: `trainero/presets.py:411`
- Test: `tests/test_core.py:84-88`

**Interfaces:**
- Consumes: nada.
- Produces: `STYLE_RUSH_SCHEDULE == {"num_repeats": 2, "epochs": 5, "save_every_n_epochs": 1}`.

- [ ] **Step 1: Ajustar o teste para o valor novo**

Em `tests/test_core.py`, substitua `test_style_rush_schedule_is_fixed` por:

```python
    def test_style_rush_schedule_is_fixed(self):
        from trainero.presets import STYLE_RUSH_SCHEDULE

        self.assertEqual(STYLE_RUSH_SCHEDULE,
                         {"num_repeats": 2, "epochs": 5, "save_every_n_epochs": 1})
```

- [ ] **Step 2: Rodar e ver falhar**

```bash
python3 -m pytest tests/test_core.py -k style_rush_schedule -q
```

Esperado: FAIL — `num_repeats` ainda é 1.

- [ ] **Step 3: Implementar**

Em `trainero/presets.py`, linha 411:

```python
STYLE_RUSH_SCHEDULE = {"num_repeats": 2, "epochs": 5, "save_every_n_epochs": 1}
```

- [ ] **Step 4: Rodar e ver passar**

```bash
python3 -m pytest tests/test_core.py -q
```

Esperado: PASS.

- [ ] **Step 5: Commit**

```bash
git add trainero/presets.py tests/test_core.py
git commit -m "feat(style-rush): num_repeats 2"
```

---

### Task 8: Ligar meta, redo e modo no pipeline de treino

`run_style_rush_training` chama `build_convert_dataset` sem `target` e `generate_captions` sem `mode`. E o LoRA normal (`run_training`) **não** gera caption nenhuma — recusa começar se faltar alguma (`training.py:754`), então o redo dele mora só no botão manual da Tarefa 9.

**Files:**
- Modify: `trainero/training.py:23` (import), `:616-629` (fase de captions e conversão)
- Test: `tests/test_training_pipeline.py`

**Interfaces:**
- Consumes: `clear_captions` (Tarefa 2), `generate_captions(..., mode=...)` (Tarefa 1), `sr.build_convert_dataset(..., target=...)` e `sr.DEFAULT_CONVERT_TARGET` (Tarefa 6).
- Produces: `overrides["convert_target"]` e `overrides["redo_captions"]` passam a ter efeito no modo `style-rush`.

- [ ] **Step 1: Escrever os testes que falham**

Acrescente em `tests/test_training_pipeline.py`:

```python
class TestStyleRushOverrides(unittest.TestCase):
    """Estes overrides existem para o dono mexer neles pela UI; um deles não
    ligado ao pipeline é um campo que mente sobre o que vai acontecer."""

    def _source(self):
        from pathlib import Path as P
        return (P(__file__).resolve().parent.parent / "trainero" / "training.py").read_text()

    def test_the_convert_target_reaches_the_builder(self):
        src = self._source()
        self.assertIn('overrides.get("convert_target")', src)
        self.assertIn("target=convert_target", src)

    def test_the_default_target_comes_from_style_rush(self):
        src = self._source()
        self.assertIn("sr.DEFAULT_CONVERT_TARGET", src)

    def test_redo_captions_clears_before_the_caption_phase(self):
        src = self._source()
        self.assertIn('overrides.get("redo_captions")', src)
        self.assertIn("clear_captions(", src)

    def test_style_rush_captions_use_the_style_rush_cascade(self):
        """No Style Rush o primário tem de ser o Gemini: é a recusa dele que
        alimenta o content_flagged da fase paga."""
        src = self._source()
        self.assertIn('mode="style-rush"', src)
```

- [ ] **Step 2: Rodar e ver falhar**

```bash
python3 -m pytest tests/test_training_pipeline.py -k StyleRushOverrides -q
```

Esperado: FAIL nas quatro.

- [ ] **Step 3: Implementar**

Em `trainero/training.py`, linha 23, troque o import:

```python
from .captioner import clear_captions, generate_captions
```

Substitua o bloco da fase "Captions" (linhas 616-626) por:

```python
    job.start_phase("Captions")
    if overrides.get("redo_captions"):
        clear_captions(dataset_dir, job)
        stats = ds.inspect(dataset_dir)
    if stats["missing_captions"]:
        job.log(f"{stats['missing_captions']} itens sem caption — gerando com "
                f"generic-style e trigger {trigger}")
        generate_captions(dataset_dir, "image", "generic-style", {"style_name": trigger},
                          job, mode="style-rush")
        stats = ds.inspect(dataset_dir)
        if stats["missing_captions"]:
            raise JobFailed(f"{stats['missing_captions']} itens continuam sem caption")
    else:
        job.log("Todas as imagens já têm caption.")
    job.end_phase("Captions")
```

E o bloco da fase "Dataset de conversão" (linhas 628-631) por:

```python
    job.start_phase("Dataset de conversão")
    convert_target = int(overrides.get("convert_target") or sr.DEFAULT_CONVERT_TARGET)
    convert_stats = sr.build_convert_dataset(dataset_dir, convert_dir, trigger, job,
                                             target=convert_target)
    job.extra["style_rush"] = convert_stats
    job.end_phase("Dataset de conversão")
```

- [ ] **Step 4: Rodar e ver passar**

```bash
python3 -m pytest tests/ -q
```

Esperado: PASS na suíte inteira. Este é o portão completo do plano — a partir daqui só falta UI.

- [ ] **Step 5: Commit**

```bash
git add trainero/training.py tests/test_training_pipeline.py
git commit -m "feat(style-rush): meta de conversao, redo de captions e cascata por modo no pipeline"
```

---

### Task 9: UI e endpoint

Dois controles novos, um por entrada. Não é duplicação: as captions são escritas em dois momentos diferentes e cada modo só vê um deles. No LoRA normal a única entrada é o botão manual (o treino recusa começar sem caption); no Style Rush a única entrada é o treino (o card de caption fica escondido nesse modo).

**Files:**
- Modify: `server.py:484-499` (`_captions`)
- Modify: `web/index.html:143-152` (card de caption), `:188-201` (painel avançado)
- Modify: `web/app.js:385-405` (`renderCaptionCard`), `:420-442` (click do botão), `:479-508` (`fillAdvanced`), `:519-529` (`collectOverrides`)
- Test: `tests/test_server_guards.py`

**Interfaces:**
- Consumes: `clear_captions` (Tarefa 2), `generate_captions(..., mode=...)` (Tarefa 1), `sr.DEFAULT_CONVERT_TARGET` (Tarefa 6).
- Produces: `POST /api/captions` aceita `redo: bool`; `overrides.convert_target: int` e `overrides.redo_captions: bool` saem da UI.

- [ ] **Step 1: Escrever o teste que falha**

Acrescente em `tests/test_server_guards.py`:

```python
class TestCaptionRedo(unittest.TestCase):
    def _source(self):
        from pathlib import Path as P
        return (P(__file__).resolve().parent.parent / "server.py").read_text()

    def test_the_endpoint_reads_the_redo_flag(self):
        self.assertIn('body.get("redo")', self._source())

    def test_the_endpoint_clears_before_captioning(self):
        src = self._source()
        self.assertIn("clear_captions", src)

    def test_the_manual_button_uses_the_lora_cascade(self):
        """O botão manual é o caminho do LoRA normal, onde o primário é o mais
        barato — nada lê a lista de flagradas nesse modo."""
        self.assertIn('mode="lora"', self._source())


class TestStyleRushUiControls(unittest.TestCase):
    def _html(self):
        from pathlib import Path as P
        return (P(__file__).resolve().parent.parent / "web" / "index.html").read_text()

    def _js(self):
        from pathlib import Path as P
        return (P(__file__).resolve().parent.parent / "web" / "app.js").read_text()

    def test_the_convert_target_input_exists(self):
        self.assertIn('id="adv-convert-target"', self._html())

    def test_the_convert_target_reaches_the_overrides(self):
        self.assertIn("o.convert_target", self._js())

    def test_the_redo_checkbox_exists_in_both_entry_points(self):
        html = self._html()
        self.assertIn('id="caption-redo"', html)
        self.assertIn('id="adv-redo-captions"', html)

    def test_both_redo_controls_are_off_by_default(self):
        html = self._html()
        for line in html.splitlines():
            if 'id="caption-redo"' in line or 'id="adv-redo-captions"' in line:
                self.assertNotIn("checked", line, line.strip())
```

- [ ] **Step 2: Rodar e ver falhar**

```bash
python3 -m pytest tests/test_server_guards.py -q
```

Esperado: FAIL nas oito.

- [ ] **Step 3: Implementar o servidor**

Em `server.py`, substitua `_captions` por:

```python
    def _captions(self):
        project = self._require_project()
        if not project:
            return
        body = self._body_json()
        profile = body.get("profile") or "default"
        side = body.get("side", "pos")
        redo = bool(body.get("redo"))
        media = "video" if ds.inspect(dataset_dir(side)).get("videos") else "image"
        prompt_vars = {}
        var = body.get("var_name")
        if var and body.get("trigger"):
            prompt_vars[var] = body["trigger"]
        pdir = project_dir(project)
        target = dataset_dir(side)

        def run(j):
            # Deleting the .txt is not enough on its own: the tagger skips
            # whatever is in its processing log, so the clear has to happen
            # inside the job, right before the pass that rewrites them.
            if redo:
                clear_captions(target, j)
            generate_captions(target, media, profile, prompt_vars, j, mode="lora")

        job = jobs.start("captions", "Gerando captions com LLM",
                         pdir / "logs" / "captions.log", run)
        self._json({"ok": True, "job": job.kind}, 202)
```

E no import de `generate_captions` no topo de `server.py`, acrescente `clear_captions`:

```python
from trainero.captioner import clear_captions, generate_captions
```

(Confira o nome exato do import existente com `grep -n "generate_captions" server.py` e edite a mesma linha.)

- [ ] **Step 4: Implementar o HTML**

Em `web/index.html`, no card de caption, acrescente a checkbox depois do `<div class="row">`:

```html
        <label class="check" id="caption-redo-wrap">
          <input type="checkbox" id="caption-redo">
          Refazer as que já têm caption
          <span class="help">apaga todo .txt do dataset antes de escrever</span>
        </label>
```

No painel avançado, dentro de `<div class="adv-grid">`, depois do campo `adv-ltx-res-wrap`:

```html
          <label class="field" id="adv-convert-target-wrap" hidden><span class="label">Imagens para conversão</span><input type="number" id="adv-convert-target" min="1"></label>
```

e junto das outras checkboxes:

```html
          <label class="check" id="adv-redo-captions-wrap" hidden><input type="checkbox" id="adv-redo-captions"> Refazer todas as captions</label>
```

- [ ] **Step 5: Implementar o JS**

Em `web/app.js`, na lista `ADV_FIELDS` (linha 473), acrescente o campo novo:

```javascript
const ADV_FIELDS = ["adv-net", "adv-dim", "adv-alpha", "adv-lr", "adv-epochs", "adv-repeats", "adv-save", "adv-ltx-res", "adv-convert-target"];
```

No fim de `fillAdvanced`, depois do bloco `isLtx`:

```javascript
  // Only Style Rush pays gpt-image-2, and only Style Rush writes captions from
  // inside the training job — in every other mode both controls would describe
  // a run that never happens.
  const rush = styleRush();
  $("#adv-convert-target-wrap").hidden = !rush;
  $("#adv-redo-captions-wrap").hidden = !rush;
  if (rush && !state.advTouched.has("adv-convert-target")) {
    $("#adv-convert-target").value = state.presets?.default_convert_target ?? 100;
  }
```

Em `collectOverrides`, junto dos outros `touched.has(...)`:

```javascript
  if (touched.has("adv-convert-target")) o.convert_target = parseInt($("#adv-convert-target").value, 10);
  o.redo_captions = $("#adv-redo-captions").checked;
```

Em `renderCaptionCard`, troque o cálculo de `show` e as duas linhas seguintes por:

```javascript
  const items = (s.dataset?.items || 0) + (s.dataset_neg?.items || 0);
  const redo = $("#caption-redo").checked;
  // The card used to appear only while something was missing. Redoing captions
  // on a dataset that arrived with .txt files needs it visible when nothing is.
  const show = !sliderIsNative() && !styleRush()
               && (missing > 0 || (state.mode === "lora" && items > 0));
  $("#caption-card").hidden = !show;
  if (!show) return;
  $("#caption-redo-wrap").hidden = state.mode !== "lora";
  $("#caption-msg").textContent = missing > 0
    ? `${missing} itens sem caption`
    : `${items} itens, todos com caption`;
  $("#caption-key-hint").hidden = !!s.openrouter;
  $("#btn-captions").disabled = !s.openrouter || (missing === 0 && !redo);
  $("#btn-captions").textContent = redo ? "Refazer captions" : "Escrever captions";
```

Acrescente o listener que redesenha o card quando a checkbox muda, junto dos outros no fim
da seção de captions:

```javascript
$("#caption-redo").addEventListener("change", renderCaptionCard);
```

E no click de `#btn-captions`, troque o corpo do `post` por:

```javascript
    const redo = $("#caption-redo").checked;
    const sides = [];
    if (redo) {
      if (state.status?.dataset?.items) sides.push("pos");
    } else {
      if (state.status?.dataset?.missing_captions) sides.push("pos");
      if (state.status?.dataset_neg?.missing_captions) sides.push("neg");
    }
    await post("/api/captions", {
      profile: sel.value,
      var_name: opt?.dataset.var || null,
      trigger,
      side: sides[0] || "pos",
      redo,
    });
```

Por fim, em `server.py`, no dicionário de `/api/status` que já expõe `style_rush_schedule`
(linha ~300), acrescente:

```python
            "default_convert_target": sr.DEFAULT_CONVERT_TARGET,
```

importando `from trainero import style_rush as sr` no topo se ainda não estiver lá (confira
com `grep -n "style_rush" server.py`).

- [ ] **Step 6: Rodar a suíte inteira**

```bash
python3 -m pytest tests/ -q
```

Esperado: PASS.

- [ ] **Step 7: Subir o servidor e conferir os controles na tela**

```bash
python3 server.py
```

Abra `http://localhost:8090` e confira, nesta ordem:

1. Modo **LoRA** com um dataset todo captionado → o card de caption aparece dizendo "N itens, todos com caption", com o botão desabilitado e a checkbox "Refazer as que já têm caption" visível e desmarcada.
2. Marque a checkbox → o botão habilita e o texto vira "Refazer captions".
3. Modo **Style Rush** → o card de caption some; abra "Ajustes finos" e o campo "Imagens para conversão" mostra 100, com "Refazer todas as captions" desmarcada logo abaixo.
4. Modo **LoRA** de novo → os dois controles do painel avançado somem.

- [ ] **Step 8: Commit**

```bash
git add server.py web/index.html web/app.js tests/test_server_guards.py
git commit -m "feat(ui): meta de conversao no Style Rush e opcao de refazer captions"
```

---

## Cobertura do spec

| Seção do spec | Tarefa |
|---|---|
| 1 — cascata de três modelos, ordem por modo | 1 |
| 1 — `record_flagged` grava o primário real | 1 |
| 1 — concorrência do captioner (`--grok_concurrency`) | 1 |
| 2 — `system_prompt.md` e `user_prompt.md` novos | 3 |
| 2 — sem teto de comprimento | 3 |
| 2 — `pull` no clone do captioner | 4 |
| 3 — fila de tentativas, meta, teto, reserva de orçamento | 6 |
| 3 — 50 prompts novos | 5 |
| 3 — concorrência da conversão (`CONVERT_WORKERS`) | 6 |
| 3 — input na UI | 9 |
| 4 — `clear_captions` | 2 |
| 4 — as duas entradas na UI e no endpoint | 9 |
| 5 — `num_repeats` 2 | 7 |
| ligação dos overrides no pipeline | 8 |

---

### Task 10: sample sem trigger word usa uma caption do dataset

Reportado pelo dono: importou um dataset que já vinha com `.txt`, treinou um Krea 2, e o sample saiu sem a trigger word. `sample_prompt_line:175` só prefixa a trigger quando ela existe, e no LoRA normal ela é opcional — um dataset que chega captionado nunca passa pela tela que a pede. O resultado é `SAMPLE_PROMPT` puro: uma cena genérica que renderiza o modelo base e ficaria idêntica sem LoRA nenhum.

Uma caption do próprio dataset é o único prompt garantidamente dentro da distribuição em treino: ela carrega, por construção, o vocabulário que o LoRA está aprendendo.

**Files:**
- Modify: `trainero/dataset.py` (função nova, depois de `captions_map`)
- Modify: `trainero/training.py:800-809` (bloco de sampling do modo normal)
- Test: `tests/test_core.py`

**Interfaces:**
- Consumes: `captions_map` de `trainero/dataset.py`.
- Produces: `sample_caption(dataset_dir: Path) -> str` — devolve `""` quando não há caption nenhuma.

- [ ] **Step 1: Escrever os testes que falham**

```python
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
```

- [ ] **Step 2: Rodar e ver falhar**

```bash
python3 -m pytest tests/test_core.py -k SampleCaption -q
```

Esperado: FAIL com `ImportError: cannot import name 'sample_caption'`.

- [ ] **Step 3: Implementar**

Em `trainero/dataset.py`, depois de `captions_map`:

```python
# Fixed so the sample prompt survives a resume. The sample is only worth
# looking at because it is the same prompt at every epoch.
SAMPLE_CAPTION_SEED = 8821


def sample_caption(dataset_dir: Path) -> str:
    """One caption from the dataset, to stand in as the sample prompt.

    A caption from the dataset is the only prompt guaranteed to sit inside the
    distribution being trained: it carries, by construction, whatever trigger
    and vocabulary the dataset uses. Empty string when nothing has a caption.
    """
    captions = sorted(set(captions_map(Path(dataset_dir)).values()))
    if not captions:
        return ""
    return random.Random(SAMPLE_CAPTION_SEED).choice(captions)
```

Acrescente `import random` no topo de `trainero/dataset.py` se ainda não estiver lá.

Em `trainero/training.py`, no bloco de sampling do modo normal, troque a chamada de
`write_sample_prompts` por:

```python
        # Without a trigger word the generic prompt exercises nothing: it renders
        # the base model, and the owner watches a sample that would look exactly
        # the same with no LoRA loaded. A dataset that arrived already captioned
        # never passes through the screen that asks for a trigger, so this is the
        # normal case, not the edge one.
        prompt_text = overrides.get("sample_prompt")
        if not prompt_text and not trigger.strip():
            prompt_text = ds.sample_caption(dataset_dir)
            if prompt_text:
                job.log(f"Sem trigger word — o sample usa uma caption do dataset: "
                        f"{prompt_text[:80]}")
        sample_path = write_sample_prompts(
            pdir / "sample_prompts.txt",
            prompt_text or SAMPLE_PROMPT,
            trigger, resolution, frames, extra=model.get("sample_args"))
        job.log(f"Samples a cada época: {sample_path}")
```

- [ ] **Step 4: Rodar e ver passar**

```bash
python3 -m pytest tests/ -q
```

Esperado: PASS.

- [ ] **Step 5: Commit**

```bash
git add trainero/dataset.py trainero/training.py tests/test_core.py
git commit -m "fix(sample): sem trigger word o sample usa uma caption do dataset"
```
