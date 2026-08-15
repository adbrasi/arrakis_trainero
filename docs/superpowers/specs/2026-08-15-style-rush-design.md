# Style Rush — Design

Data: 2026-08-15. Status: aprovado pelo dono (sampling em todo modo, prompt padrão único,
`control_resolution = [1024, 1024]`, balanceamento 70/30, moderação `low` com uma segunda
tentativa em outra imagem).

## O que é

Um terceiro modo de treino no Arrakis Trainero, ao lado de LoRA padrão e Concept Slider.
O dono entrega **um** dataset de imagens; o modo constrói sozinho um **segundo** dataset
sintético de conversão de estilo e treina os dois juntos num único LoRA.

O LoRA resultante sabe duas coisas:

1. **gerar** no estilo do dataset (t2i, disparado pela trigger word);
2. **converter** uma imagem de qualquer outro estilo para esse estilo (edição com control image).

O dataset sintético é o que ensina (2): para cada imagem do dataset, o GPT Image produz uma
versão dela em **outro** estilo; essa versão vira a *control image* e a imagem original vira
o *target*. O modelo aprende o caminho "estilo qualquer → meu estilo".

## Fluxo do dono

Nome do projeto → **trigger word** → arrasta o dataset → **TREINAR**. Nada mais. Tudo abaixo
acontece como fase do job, com log e progresso na mesma página.

## Modelos elegíveis

Só os que aceitam control image no musubi: **Flux Klein 9B** (`klein-base-9b`, default) e
**Qwen Image Edit**. A elegibilidade é derivada de uma chave `supports_control` no preset —
não existe lista paralela.

Verificado no musubi-tuner (`kohya-ss/musubi-tuner@main`):

- `DatasetGroup` é um `ConcatDataset` de **batches já montados** por `BucketBatchManager`;
  um batch nunca mistura datasets.
- Os buckets são separados por presença **e contagem** de `latents_control_*`
  (`image_video_dataset.py:534-546`).
- `flux_2_train_network.py:283` decide por batch: `if "latents_control_0" in batch:`.

Logo, um `[[datasets]]` com `control_directory` e outro sem, no mesmo TOML e no mesmo treino,
é suportado nativamente. Nenhum patch, flag ou caminho paralelo é necessário.

## Fases do job

### 1. Engine + modelos base

Inalterado.

### 2. Captions do dataset base

`data_araknideo`, profile **`generic-style`**, `--prompt_var style_name=<trigger>`. Esse
profile já obriga a caption a começar exatamente com a trigger word (system prompt:
"Every caption MUST start with exactly this trigger phrase"). Roda apenas se houver itens
sem `.txt`; itens com caption manual são preservados.

### 3. Dataset de conversão

Amostra **até 50** imagens do dataset base (todas, se houver menos). Sorteio determinístico
(seed fixa) para que uma retomada reproduza a mesma seleção.

Cada slot recebe **um prompt de estilo distinto** de `data/style_prompts.txt` (50 linhas,
expandido a partir das 9 do dono). Chamada à Images API do OpenRouter:

```
POST https://openrouter.ai/api/v1/images
{
  "model": "openai/gpt-image-2",
  "prompt": "<linha do style_prompts.txt>",
  "n": 1,
  "quality": "low",
  "aspect_ratio": "<1:1 | 3:2 | 2:3 | 4:3 | 3:4 | 16:9 | 9:16, o mais próximo do original>",
  "moderation": "low",
  "input_references": [{"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}]
}
```

Confirmado no catálogo `/api/v1/images/models`: `openai/gpt-image-2` aceita
`input_references` (0–16), `quality` (`auto|low|medium|high`), `aspect_ratio` e passthrough
de `moderation`. Não aceita `size`/`resolution` — a saída de `quality: low` já é 1K nativo,
que é o pedido do dono.

Custo (pricing do endpoint OpenAI no OpenRouter): `output_image` $3e-5/token,
`input_image` $8e-6/token → ~$0.011 por imagem, **~$0.55 pelas 50**.

**Recusa da moderação:** o slot tenta uma segunda vez com **outra imagem** do pool ainda não
usado. Duas tentativas por slot, sempre no mesmo modelo. Falhou duas vezes → o slot é
descartado e o dataset fecha com menos pares; o log diz quantos e por quê.

Saída, em `projects/<slug>/dataset_convert/`:

| Arquivo | Conteúdo |
|---|---|
| `<stem>.png` | cópia da imagem original (target) |
| `control/<stem>.png` | saída do GPT Image (control) |
| `<stem>.txt` | `convert the style of this image to the <trigger> style` — idêntico nos 50 |
| `.style_rush.json` | manifest: imagem de origem, prompt usado, status, custo |

O manifest torna a fase idempotente: cancelar e reapertar TREINAR não regera o que já existe.

Paralelismo: `ThreadPoolExecutor` com 4 workers (50 chamadas sequenciais levariam 15–25 min).

### 4. Configuração

Um `dataset.toml` com dois subsets:

```toml
[general]
resolution = [1024, 1024]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
image_directory = ".../dataset"
cache_directory = ".../cache/images"
num_repeats = 1

[[datasets]]
image_directory = ".../dataset_convert"
cache_directory = ".../cache/convert"
control_directory = ".../dataset_convert/control"
control_resolution = [1024, 1024]
num_repeats = 3
```

`control_resolution = [1024, 1024]` (não os `[2024, 2024]` do inference oficial do FLUX.2):
1024 dá 4096 tokens de referência, o que já dobra o comprimento da sequência dos batches de
conversão; 2024 daria 4x isso em VRAM e tempo por step sem ganho para este uso, já que a
saída do GPT Image é 1K nativo.

**Balanceamento 70/30.** As exposições — não os itens — ficam 70% no base e 30% na conversão:

```
repeats_base = clamp(round(250 / N_base), 1, 20)
repeats_conv = clamp(round(repeats_base * (30/70) * (N_base / N_conv)), 1, 40)
itens_por_epoch = N_base*repeats_base + N_conv*repeats_conv
epochs = clamp(round(target_steps * batch_size / itens_por_epoch), 4, 40)
save_every_n_epochs = max(1, round(epochs / 10))
```

Exemplo (200 base, 50 conv): repeats 1 e 2 → 200 vs 100 exposições (67/33), 7 epochs,
~2100 steps. Exemplo (30 base, 30 conv): repeats 8 e 3 → 240 vs 90 (73/27), 6 epochs.

`write_dataset_toml` passa a receber uma **lista de subsets**; o modo LoRA normal e o slider
passam uma lista de um elemento. Um único caminho de código escreve todos os TOMLs.

### 5. Cache de latents e text encoder

Inalterado: um comando cada, o `--dataset_config` cobre os dois subsets.

### 6. Treino com sampling

`--sample_prompts <pdir>/sample_prompts.txt --sample_every_n_epochs <save_every> --sample_at_first`.

Sampling é **ligado em todos os modos e modelos**, não só no Style Rush. A frequência é
amarrada ao `save_every_n_epochs`: cada sample corresponde ao checkpoint recém-salvo,
o que dá ~10 samples por treino sem desperdiçar tempo de GPU.

O suporte a `--sample_prompts` está confirmado no musubi upstream (`training/parser_common.py`,
`training/trainer_base.py`) e o `flux_2_train_network.py` implementa `do_inference` com
`control_image_path`. Para os outros dois engines — `musubi-ltx` (fork LTX 2.3) e `sd-scripts`
(Anima) — a existência do argumento é verificada na instalação do engine; se um deles não
tiver o argumento, o sampling fica desligado **para aquele modelo**, registrado no log, e o
resto do preset segue igual. Nada de emular sampling por fora do trainer.

**Um prompt padrão**, começando pela trigger word:

```
<trigger>, a girl with long hair sitting on her bed in a sunlit bedroom, holding a cup of
coffee, a cat curled up on her lap, looking at the viewer, detailed background
--w 1024 --h 1024 --d 42 --s 28 --g 4.0
```

Editável num campo do painel avançado. Fora do Style Rush a trigger word é opcional; sem
ela, o prompt vai sem prefixo. Para modelos de vídeo, o `--f` sai do preset.
Samples caem em `output/sample/<output_name>_e{epoch:06d}_{idx:02d}_{ts}_{seed}.png`
(padrão do `trainer_base.py:1001`).

### 7. Upload HF

Inalterado (watcher de checkpoints + model card).

## UI

- **Toggle do topo** ganha `Style Rush` ao lado de `LoRA` e `Concept Slider`. Selecionar
  Style Rush restringe a grade de modelos aos que têm `supports_control`.
- **Campo trigger word** promovido para o card 1 (Projeto). É a mesma trigger usada pelas
  captions e pelo prompt de sample. Obrigatório no Style Rush.
- **Galeria de samples** no card de progresso: grade das últimas imagens, mais recente
  primeiro, alimentada pelo polling que já existe.
- Painel avançado ganha: checkbox `Gerar samples` (ON) e o campo do prompt de sample.

API nova: `GET /api/samples` (lista `{name, epoch, idx}` de `output/sample/*.png`) e
`GET /api/sample?name=<basename>` (serve o PNG; valida basename, sem travessia).

## Código

| Arquivo | Papel |
|---|---|
| `trainero/imagegen.py` | cliente da Images API do OpenRouter: monta payload, faz POST, decodifica b64. stdlib `urllib` + Pillow. Sem estado. |
| `trainero/style_rush.py` | monta o dataset de conversão: amostragem, retry em outra imagem, manifest, escrita dos pares. |
| `data/style_prompts.txt` | os 50 prompts de estilo, um por linha, editável. |
| `trainero/presets.py` | `supports_control`, `style_rush_split = 0.7`, `sample_prompt` default. |
| `trainero/training.py` | `write_dataset_toml` por lista de subsets; `write_sample_prompts`; `run_style_rush_training`. |
| `server.py` | endpoints `/api/samples` e `/api/sample`; `side="convert"`. |
| `web/*` | modo, trigger no topo, galeria, dois campos no painel avançado. |

Dependência nova: **Pillow** (ler dimensões para o aspect ratio, converter e salvar).
HTTP fica em `urllib` da stdlib — nenhuma dependência de rede nova no venv do servidor.

## Erros

Mesma regra do resto do projeto: nada de gambiarra. Falha em qualquer fase para o job com a
causa real no log e na UI.

- Sem `OPENROUTER_API_KEY` → o modo Style Rush nem habilita o botão, com a mensagem dizendo
  qual variável falta.
- Todas as 50 recusadas → job falha dizendo que o dataset de conversão ficou vazio, em vez de
  treinar um LoRA que não aprende a converter.
- Falha parcial → segue com os pares que deram certo e o log informa o número final.
- Trigger word vazia no Style Rush → erro antes de gastar um centavo de API.

## Testes

Puros, sem GPU nem rede:

- balanceamento 70/30: as exposições batem em datasets de tamanhos variados (grande/pequeno,
  base menor que conv, conv vazio).
- `write_dataset_toml` com dois subsets produz TOML com exatamente um `control_directory`.
- `build_payload` do `imagegen`: aspect ratio escolhido para dimensões variadas, quality low,
  moderation low, referência como data URI.
- `style_rush`: retomada a partir de um manifest parcial não regera nada; recusa consome a
  segunda tentativa com outra imagem e para por aí.
- `write_sample_prompts`: trigger na frente, flags `--w/--h/--d/--s/--g` presentes.

Smoke: servidor sobe e `/api/samples` responde vazio sem projeto.
