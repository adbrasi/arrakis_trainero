# Style Rush — Design

Data: 2026-08-15. Status: aprovado pelo dono.

Decisões do dono, na ordem em que foram dadas: dataset de conversão **sempre com 50 pares**,
independente do tamanho do dataset enviado; **5 epochs no máximo** (ele cancela quando achar
que está bom); **checkpoint e sample a cada época**; sampling **em todo modo**, com **um**
prompt padrão de t2i começando pela trigger phrase; `control_resolution = [1024, 1024]`;
moderação `low` com uma segunda tentativa em outra imagem; conversão pro formato ComfyUI
onde for necessária.

## O que é

Um terceiro modo de treino no Arrakis Trainero, ao lado de LoRA padrão e Concept Slider.
O dono entrega **um** dataset de imagens; o modo constrói sozinho um **segundo** dataset
sintético de conversão de estilo e treina os dois juntos num único LoRA.

O LoRA resultante sabe duas coisas:

1. **gerar** no estilo do dataset (t2i, disparado pela trigger word);
2. **converter** uma imagem de qualquer outro estilo para esse estilo (edição com control image).

O dataset sintético é o que ensina (2): para cada imagem, o GPT Image produz uma versão dela
em **outro** estilo; essa versão vira a *control image* e a imagem original vira o *target*.
O modelo aprende o caminho "estilo qualquer → meu estilo".

## Fluxo do dono

Nome do projeto → **trigger word** → arrasta o dataset → **TREINAR**. Nada mais. Tudo abaixo
acontece como fase do job, com log, progresso e galeria de samples na mesma página.

## Modelos elegíveis

Só os que aceitam control image no musubi: **Flux Klein 9B** (`klein-base-9b`, default) e
**Qwen Image Edit**. A elegibilidade sai de uma chave `supports_control` no preset — não
existe lista paralela.

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
sem `.txt`; captions manuais são preservadas.

### 3. Dataset de conversão — sempre 50 pares

`data/style_prompts.txt` tem **50 prompts de estilo**, um por linha (expansão das 9 linhas do
dono). São 50 slots; slot `i` usa o prompt `i`.

A imagem de origem de cada slot vem do dataset base, percorrido em ordem embaralhada com
seed fixa e **circular**:

- dataset com ≥ 50 imagens → 50 imagens distintas;
- dataset com menos → as imagens se repetem, sempre com prompts diferentes (30 imagens → cada
  uma aparece 1 ou 2 vezes, com estilos distintos).

O tamanho do dataset enviado não muda nada além de quais imagens são sorteadas. Sem fórmula
de balanceamento: os dois datasets entram no treino com `num_repeats = 1`.

Chamada à Images API do OpenRouter, uma por slot:

```
POST https://openrouter.ai/api/v1/images
{
  "model": "openai/gpt-image-2",
  "prompt": "<linha do style_prompts.txt>",
  "n": 1,
  "quality": "low",
  "aspect_ratio": "<1:1|3:2|2:3|4:3|3:4|16:9|9:16, o mais próximo do original>",
  "moderation": "low",
  "input_references": [{"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}]
}
```

Confirmado no catálogo `/api/v1/images/models`: `openai/gpt-image-2` aceita `input_references`
(0–16), `quality` (`auto|low|medium|high`), `aspect_ratio` e passthrough de `moderation`.
Não aceita `size`/`resolution` — a saída de `quality: low` já é 1K nativo, que é o pedido.

Custo (pricing do endpoint OpenAI no OpenRouter): `output_image` $3e-5/token,
`input_image` $8e-6/token. Medido numa chamada real (referência 1024x1024, quality low):
1024 tokens de entrada + 23 do prompt + 196 de saída = **$0.0142 por imagem, ~$0.71 pelos 50
slots**. A referência é reduzida para 1024px antes do envio — acima disso o custo escala
com a área e a saída continua 1K.

**Recusa da moderação:** o slot tenta uma segunda vez com **outra imagem** do pool, mesmo
prompt e mesmo modelo. Duas tentativas por slot, no máximo. Falhou nas duas → o slot é
descartado e o log diz quantos caíram e por quê.

Saída, em `projects/<slug>/dataset_convert/`:

| Arquivo | Conteúdo |
|---|---|
| `<slot>.png` | cópia da imagem de origem (target) |
| `control/<slot>.png` | saída do GPT Image (control) |
| `<slot>.txt` | `convert the style of this image to the <trigger> style` — idêntico nos 50 |
| `.style_rush.json` | manifest: slot, imagem de origem, prompt, status, custo |

O nome do arquivo é o índice do slot (`slot_00`…`slot_49`), não o nome da imagem original —
é o que permite a mesma imagem aparecer em slots diferentes quando o dataset tem menos de 50.

O manifest torna a fase idempotente: cancelar e reapertar TREINAR não regera o que já existe.

Paralelismo: `ThreadPoolExecutor` com 4 workers (50 chamadas sequenciais levariam 15–25 min).

### 4. Configuração

Um `dataset.toml` com dois subsets, ambos `num_repeats = 1`:

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
num_repeats = 1
```

`control_resolution = [1024, 1024]`, não os `[2024, 2024]` do inference oficial do FLUX.2:
1024 dá 4096 tokens de referência, o que já dobra o comprimento da sequência dos batches de
conversão; 2024 custaria 4x isso em VRAM e tempo por step sem ganho aqui, já que a saída do
GPT Image é 1K nativo.

**Schedule do Style Rush:** `num_repeats = 1`, `max_train_epochs = 5`,
`save_every_n_epochs = 1`, `sample_every_n_epochs = 1`. Nada de `target_steps` ou cálculo de
epochs — o dono olha os samples e cancela quando achar bom. Os outros modos (LoRA padrão e
Concept Slider) mantêm o `suggest_schedule` que existe hoje.

`write_dataset_toml` passa a receber uma **lista de subsets**; o modo LoRA normal e o slider
passam uma lista de um elemento. Um único caminho de código escreve todos os TOMLs.

### 5. Cache de latents e text encoder

Inalterado: um comando cada, o `--dataset_config` cobre os dois subsets.

### 6. Treino com sampling

`--sample_prompts <pdir>/sample_prompts.txt --sample_every_n_epochs 1 --sample_at_first`.

Sampling é **ligado em todos os modos e modelos**, não só no Style Rush, a cada época.

Um único prompt padrão, o mesmo para **todos os modelos de imagem**, em prosa inglesa
descritiva, precedido pela trigger word:

```
<trigger>, A young woman sits alone at a tall window in a quiet apartment at golden hour,
one knee drawn up onto the cushioned sill, both hands wrapped around a chipped ceramic mug.
Late sunlight falls across her in long amber bars, warm on her cheek and throat, and fine
dust turns slowly in the air. A grey cat lies curled asleep against her hip, one paw over
its face. Her hair spills loose over one shoulder, her sweater slipping wide at the collar,
and she looks out through the glass with a soft, unhurried gaze while the city beyond
dissolves into hazy blue rooftops. --w 1024 --h 1024 --d 42 --s 28 --g 4.0
```

O texto é escolhido para exercitar de uma vez as coisas que revelam se o LoRA está aprendendo
ou queimando: rosto e olhar, mãos segurando um objeto, tecido, um animal, luz direcional
quente com sombra, e um fundo com profundidade.

Editável num campo do painel avançado. Fora do Style Rush a trigger é opcional; sem ela o
prompt vai sem prefixo. Para modelos de vídeo o `--f` sai do preset.

Suporte confirmado no musubi upstream (`training/parser_common.py:286-304`,
`training/trainer_base.py:853`); `docs/krea2.md:233` documenta o uso com `--text_encoder`.
Para `musubi-ltx` (fork LTX 2.3) e `sd-scripts` (Anima) a existência do argumento é verificada
na instalação do engine; faltando, o sampling fica desligado **para aquele modelo**, com aviso
no log, e o resto do preset segue igual. Nada de emular sampling por fora do trainer.

Samples caem em `output/sample/<output_name>_e{epoch:06d}_{idx:02d}_{ts}_{seed}.png`
(`trainer_base.py:1001`).

### 7. Upload HF + formato ComfyUI

O watcher de checkpoints já existente sobe cada `.safetensors` novo. O que muda: a conversão
pro formato ComfyUI, hoje hardcoded no Anima, vira uma chave de dados no preset.

```python
"comfy_convert": {"script": "networks/convert_anima_lora_to_comfy.py"}   # Anima (sd-scripts)
"comfy_convert": {"convert_lora": True}                                  # musubi: --target other
```

`{"convert_lora": True}` roda, no venv do engine:
`python convert_lora.py --input <ckpt> --output <ckpt>_comfy.safetensors --target other`.
Os dois arquivos vão pro HF.

**Quais modelos ligam isso hoje: só o Anima.** Evidência:

- `comfy/lora.py::model_lora_keys_unet` (ComfyUI master) abre com um bloco **genérico para
  toda arquitetura** que registra `lora_unet_<chave_achatada>` para cada chave do modelo —
  exatamente o formato que o musubi salva. Os blocos `isinstance(model, QwenImage/Krea2/
  Flux/…)` que vêm depois adicionam suporte a LoRAs em formato *diffusers*; não corrigem o
  formato do musubi.
- O musubi fornece script de conversão pro ComfyUI em exatamente duas arquiteturas
  (`networks/convert_hunyuan_video_1_5_lora_to_comfy.py`, `networks/convert_z_image_lora_to_comfy.py`)
  e nenhuma delas está no Trainero.
- `docs/wan.md:205`: "The trained LoRA weights are seemed to be compatible with ComfyUI".
- Anima é sd-scripts e tem script dedicado no próprio repo.

Como isso é inferência sobre o comportamento do ComfyUI e não teste em GPU, o painel avançado
ganha um checkbox **"Converter LoRA p/ formato ComfyUI"** (default: o que o preset declarar).
Se o Flux Klein precisar na prática, é marcar a caixa — ou uma linha no `presets.py`.

## UI

- **Toggle do topo** ganha `Style Rush` ao lado de `LoRA` e `Concept Slider`. Selecionar
  Style Rush restringe a grade de modelos aos que têm `supports_control`.
- **Campo trigger word** promovido para o card 1 (Projeto). É a mesma trigger usada pelas
  captions e pelo prompt de sample. Obrigatório no Style Rush.
- **Galeria de samples** no card de progresso: grade das últimas imagens, mais recente
  primeiro, alimentada pelo polling que já existe.
- Painel avançado ganha: checkbox `Gerar samples` (ON), campo do prompt de sample e checkbox
  `Converter LoRA p/ formato ComfyUI`.

API nova: `GET /api/samples` (lista `{name, epoch, idx}` de `output/sample/*.png`) e
`GET /api/sample?name=<basename>` (serve o PNG; valida basename, sem travessia).

## Código

| Arquivo | Papel |
|---|---|
| `trainero/imagegen.py` | cliente da Images API do OpenRouter: monta payload, faz POST, decodifica b64. stdlib `urllib` + Pillow. Sem estado. |
| `trainero/style_rush.py` | monta o dataset de conversão: 50 slots, retry em outra imagem, manifest, escrita dos pares. |
| `data/style_prompts.txt` | os 50 prompts de estilo, um por linha, editável. |
| `trainero/presets.py` | `supports_control`, `comfy_convert` como dado, `sample_prompt` default, schedule fixo do Style Rush. |
| `trainero/training.py` | `write_dataset_toml` por lista de subsets; `write_sample_prompts`; conversor ComfyUI genérico; `run_style_rush_training`. |
| `server.py` | endpoints `/api/samples` e `/api/sample`; `side="convert"`. |
| `web/*` | modo, trigger no topo, galeria, campos novos no painel avançado. |

Dependência nova: **Pillow** (ler dimensões para o aspect ratio, converter e salvar).
HTTP fica em `urllib` da stdlib — nenhuma dependência de rede nova no venv do servidor.

## Erros

Mesma regra do resto do projeto: nada de gambiarra. Falha em qualquer fase para o job com a
causa real no log e na UI.

- Sem `OPENROUTER_API_KEY` → o modo Style Rush nem habilita o botão, dizendo qual variável falta.
- Trigger word vazia no Style Rush → erro antes de gastar um centavo de API.
- Todos os 50 slots recusados → job falha dizendo que o dataset de conversão ficou vazio, em
  vez de treinar um LoRA que não aprende a converter.
- Falha parcial → segue com os slots que deram certo e o log informa o número final.

## Testes

Puros, sem GPU nem rede:

- seleção de slots: 50 slots sempre; ≥50 imagens → todas distintas; <50 → repetição circular
  com prompts distintos; dataset vazio → erro.
- `write_dataset_toml` com dois subsets produz TOML com exatamente um `control_directory`.
- `build_payload` do `imagegen`: aspect ratio escolhido para dimensões variadas, quality low,
  moderation low, referência como data URI.
- `style_rush`: retomada a partir de manifest parcial não regera nada; recusa consome a
  segunda tentativa com outra imagem e para por aí.
- `write_sample_prompts`: trigger na frente, flags `--w/--h/--d/--s/--g` presentes; sem
  trigger, prompt sem prefixo.
- schedule do Style Rush: repeats 1, epochs 5, save 1, sample 1, independente do dataset.

Smoke: servidor sobe e `/api/samples` responde vazio sem projeto.
