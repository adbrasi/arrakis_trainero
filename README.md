# ⚔ Arrakis Trainero

Trainer de LoRA 1-click para vast.ai/RunPod. Uma página, três passos: nome → dataset → modelo → **TREINAR**.

## Rodar no pod

```bash
export HF_TOKEN="hf_..."            # upload automático pro HuggingFace
export OPENROUTER_API_KEY="sk-..."  # captions via LLM (só se precisar)
curl -L https://raw.githubusercontent.com/adbrasi/arrakis_trainero/main/bootstrap.sh | bash
```

Abra a porta **8090** no painel do pod. Pronto.

Local (WSL): `bash start.sh`.

## O fluxo

1. **Nome do projeto** — vira o nome do LoRA e do repo HF.
2. **Dataset** — arraste imagens/vídeos + `.txt` (ou um zip), ou cole: link Mega,
   `user/repo` do HuggingFace, URL de zip, caminho de pasta. Se faltar caption,
   aparece o card de geração via LLM (presets do data_araknideo).
3. **Modelo** → **TREINAR**. O resto é automático: engine, download dos modelos
   base, cache, treino contado em **epochs**, upload de cada checkpoint pro HF.

## Modelos e presets (um por modelo, sempre LoRA padrão)

| Modelo | Backend | Preset |
|---|---|---|
| Krea 2 | musubi-tuner | **receita automática por tipo** (abaixo) · lr 1e-4 · krea2_shift |
| Anima | sd-scripts (`anima_train_network.py`) | dim 32/16 · lr 2e-5 · sigmoid · ~660 exposições/img |
| Flux Klein (base-9B) | musubi-tuner | dim 128/128 · lr 1e-4 · flux2_shift |
| Qwen Image | musubi-tuner | dim 64/64 · lr 5e-5 · shift 2.2 |
| Qwen Image Edit (2511) | musubi-tuner | idem + pasta `control/` no dataset |
| Ideogram 4 | musubi-tuner | dim 32/32 · lr 1e-4 · DiT já é FP8 |
| LTX 2.3 | musubi fork AkaneTendo25 (`ltx-2-dev`) | dim 256/256 · lr 1e-4 · shifted_logit_normal |
| Wan 2.2 (t2v-A14B) | musubi-tuner | dim 32/32 · lr 2e-4 · high+low noise |

fp8/blocks_to_swap são decididos sozinhos pela VRAM detectada.

No painel ⚙ dá pra trocar LoRA → **LoKr/LoHa**, ligar **LoRA+** (ratio 16), rank,
epochs. **DoRA não existe** nos backends escolhidos (musubi e sd-scripts) — LoKr
cobre o caso; se um dia o musubi ganhar DoRA, é uma linha no `presets.py`.

## Krea 2: receitas automáticas

No modo LoRA com Krea 2 aparece o **Tipo de LoRA** no passo 01. O tipo e o tamanho do
dataset decidem rank, total de steps e checkpoints. A linha "Receita: …" no passo 03
mostra o que vai rodar. Sem escolha explícita: subpastas viram Misto, mais de 40
imagens viram Estilo, o resto é Personagem. O que você digitar no painel ⚙ vence a receita.
Base: `docs/PESQUISA_RECEITAS_KREA2.md`.

| Tipo | Rank | Steps (batch 1) | Captions |
|---|---|---|---|
| Personagem | 16/16 até 5 imagens, 32/32 acima | 1→1000, 3→1000, 8→1450, 15→2100, 30→2500, interpolado, teto 2600 | `krea2-character` (trigger + classe; descreve só o que varia) |
| Estilo | 32/32, 64/64 acima de 300 | clamp(1500 + 12·n, 2000, 4000) | `generic-style` (trigger = nome do estilo) |
| Misto | 32/32, 64/64 acima de 600 | 5000–6000, batch pela VRAM (2 em 32 GB, 4 em 70 GB+) | cada subpasta com a própria trigger e o perfil do tipo |

Repeats e epochs são buscados juntos para cair no alvo, sempre com ~8 checkpoints e um
sample por checkpoint. Continua contado em epochs, nunca em max steps.

### Misto: como organizar

Mande um `.zip` (ou link) com uma subpasta por conceito. O projeto precisa estar em
**Misto** antes do import, para as subpastas serem preservadas.

```
dataset.zip
├── makima/              personagem, trigger "makima"
├── rem__anime girl/     personagem, trigger "rem", classe "anime girl"
└── style_ghibli/        estilo, trigger "ghibli" (ou marque "estilo" na lista da UI)
```

- Cada personagem recebe a mesma exposição por época, seja qual for o tamanho da pasta.
- O estilo fica com ~45% da época.
- Cada subpasta vira um `[[datasets]]` com os seus `num_repeats`.

### Poucas imagens → dataset (expansão)

Com Personagem e menos de 10 imagens, aparece o card **Gerar o resto do dataset**:

1. Você manda de 1 a 5 imagens do personagem e preenche a classe (woman, man, anime girl…).
2. Um modelo de imagem no OpenRouter desenha o mesmo personagem em 30 variações curadas:
   ângulo, enquadramento, expressão, roupa (40% mantêm a canônica), cenário, luz e ação.
   O prompt fixa a identidade e o meio: foto continua foto, anime continua anime.
3. Um juiz VLM compara cada candidata com as referências e dá notas de identidade,
   artefatos e estilo. Só entra quem passa dos limites; quase-duplicatas (dHash) saem.
4. Gera até a meta (padrão 20) ou até o teto de 2× gerações. As aceitas viram
   `dataset/gen_*.png`, o custo vai para o log e uma nova execução retoma de onde parou.
5. Depois é o fluxo normal: captions e TREINAR. A receita conta o total já expandido.

## Epochs, nunca max steps

No import calculamos repeats + epochs para o total de treino cair sempre na faixa
boa do modelo (~2000 steps imagem, ~3000 vídeo, ~660 exposições/img no Anima),
com ~10 checkpoints salvos. Qualquer tamanho de dataset.

No **Style Rush** isso não se aplica: são 5 epochs fixos, repeats 1, e o dono cancela quando
os samples ficam bons.

## Concept Sliders

Toggle "Concept Slider" no topo:

- **LTX 2.3**: slider nativo do fork — só pares de prompt (positivo/negativo), sem dataset.
- **Demais modelos**: dois datasets (conceito ALTO / BAIXO); treina dois LoRAs
  idênticos e monta o slider por concatenação de rank com sinal invertido
  (ΔW = ΔW⁺ − ΔW⁻, exato). Força positiva = mais conceito; negativa = menos.

## Style Rush

Toggle "Style Rush" no topo, para **Flux Klein** ou **Qwen Image Edit**. Você manda só o
dataset de estilo e preenche a trigger word; o trainer faz o resto:

1. captions do dataset via OpenRouter (profile `generic-style`, trigger na primeira palavra);
2. **50 pares de conversão** gerados com `openai/gpt-image-2` (quality low, no máximo
   ~$0.71 — medido; datasets não-quadrados saem mais baratos): a saída vira a control
   image **recortada para a proporção exata da sua imagem**, a sua imagem original vira o
   target, e a caption é a mesma nos 50 —
   `convert the style of this image to the <trigger> style`;
3. treino com os **dois datasets no mesmo `dataset.toml`** (só o de conversão tem
   `control_directory`) — o musubi mantém os batches separados sozinho;
4. 5 epochs, checkpoint e sample a cada época. Você olha as amostras e cancela quando quiser.

O dataset pode ter qualquer mistura de tamanhos e proporções: a referência é reduzida para
1024px antes do envio (o custo escala com a área) e a imagem gerada é recortada para a
proporção do target, de modo que os dois caem no **mesmo bucket** do musubi e são cortados
do mesmo jeito. Sem isso o modelo veria um control enquadrado diferente do target.

Se o dataset tiver menos de 50 imagens, elas se repetem entre os slots, sempre com estilos
diferentes. Imagem recusada pela moderação é tentada uma segunda vez com **outra** imagem;
falhou nas duas, o slot é descartado e o log diz quantos caíram.

Precisa de `OPENROUTER_API_KEY`.

## Samples durante o treino

Todo treino gera uma imagem de amostra por época (`--sample_prompts`), mostrada na galeria do
card de progresso e salva em `output/sample/`. O prompt padrão é o mesmo para todos os modelos
de imagem e começa pela trigger word; dá para trocar no painel ⚙.

## Layout no pod

```
/workspace/
├── arrakis_trainero/          este repo + venv leve do servidor
├── engines/                   musubi-tuner · musubi-ltx · sd-scripts · data_araknideo (1 venv cada)
├── models/<modelo>/           pesos base (baixados no primeiro treino)
└── trainero_projects/<slug>/  dataset/ · cache/ · output/ · logs/ · *.toml
```

## Chaves

| Env | Para quê |
|---|---|
| `HF_TOKEN` | repo + upload de checkpoints (privado por default), download de modelos gated (FLUX.2) |
| `OPENROUTER_API_KEY` | captions via LLM, Style Rush e expansão de dataset |

### Ajustes opcionais

| Env | Default | Para quê |
|---|---|---|
| `CAPTION_MODELS` | por modo | sobrescreve a cascata inteira, separada por vírgula |
| `CAPTION_CONCURRENCY` | 64 | chamadas paralelas na captionagem |
| `CAPTION_REASONING_EFFORT` | `minimal` | raciocínio do modelo de caption. Lido pelo tagger, não por este repo |
| `CONVERT_WORKERS` | 8 | chamadas paralelas ao gpt-image-2 no Style Rush |
| `EXPAND_IMAGE_MODEL` | `openai/gpt-image-2` | modelo de imagem da expansão; qualquer id de `/api/v1/images/models` (ex.: `google/gemini-nano-banana-2.1`) — o payload segue os parâmetros que o modelo declara |
| `EXPAND_IMAGE_QUALITY` | `medium` | só para modelos que aceitam `quality` |
| `EXPAND_JUDGE_MODEL` | `google/gemini-3.7-flash` | juiz VLM (chat/completions com json_schema) |
| `EXPAND_MAX_ATTEMPTS` | 2 × faltantes | teto de gerações pagas por execução |
| `EXPAND_MIN_IDENTITY` / `_CLEAN` / `_STYLE` | 8 / 7 / 7 | notas mínimas (0–10) para aceitar |
| `EXPAND_WORKERS` / `EXPAND_REFS_PER_CALL` | 4 / 3 | paralelismo e referências por chamada |

A cascata de caption é uma lista ordenada e a ordem muda com o modo. No LoRA
normal ela começa pelo modelo mais barato; no Style Rush começa pelo Gemini,
porque é a recusa dele que marca quais imagens o gpt-image-2 também vai recusar
— e lá uma recusa custa um slot pago.

`CAPTION_REASONING_EFFORT` fica em `minimal`, o menor tier que o OpenRouter
documenta. Os tiers são fração do `max_tokens` e o tagger não manda
`max_tokens` de propósito (um teto trunca a caption no meio), então se o gasto
continuar alto o próximo passo é `none`, que desliga o orçamento de raciocínio
em vez de encolhê-lo.
