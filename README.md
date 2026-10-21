# ⚔ Arrakis Trainero

Trainer de LoRA 1-click para vast.ai/RunPod. Uma página, três passos: nome → dataset → modelo → **TREINAR**.

## Rodar no pod

```bash
export HF_TOKEN="hf_..."            # upload automático pro HuggingFace
export OPENROUTER_API_KEY="sk-..."  # captions via LLM (só se precisar)
curl -L https://raw.githubusercontent.com/adbrasi/arrakis_trainero/main/bootstrap.sh | bash
```

Outra branch: `TRAINERO_BRANCH=<branch>` antes do `bash`, e o `bootstrap.sh` da mesma branch
na URL.

Abra a porta **8090** no painel do pod. Pronto. O servidor roda destacado do terminal
(fechar a aba não derruba um treino); log em `data/server.log`. Rodar o comando de novo
atualiza e reinicia — um treino em andamento é cancelado de forma limpa antes.

Local (WSL): `bash start.sh`.

## O fluxo

1. **Nome do projeto** — vira o nome do LoRA e do repo HF. Escolha **o que você está
   treinando** (estilo, personagem, conceito, roupa) e, se pedir, a **trigger word**.
2. **Dataset** — arraste imagens/vídeos (com ou sem `.txt`) ou um zip, ou cole: link
   Mega, `user/repo` do HuggingFace, URL de zip, caminho de pasta.
3. **Modelo** → **TREINAR**. O resto é automático: captions que faltam, engine,
   download dos modelos base, cache, treino contado em **epochs**, upload de cada
   checkpoint pro HF.

Não precisa saber o que é caption: se faltar, o próprio treino escreve antes de
baixar qualquer modelo. O botão "Escrever agora" existe só para revisar as captions
antes de treinar.

## Modelos e presets (um por modelo, sempre LoRA padrão)

| Modelo | Backend | Preset |
|---|---|---|
| Krea 2 | musubi-tuner | dim 32/32 · lr 1e-4 · krea2_shift |
| Anima | sd-scripts (`anima_train_network.py`) | dim 32/16 · lr 2e-5 · sigmoid · ~660 exposições/img |
| Flux Klein (base-9B) | musubi-tuner | dim 128/128 · lr 1e-4 · flux2_shift |
| Qwen Image | musubi-tuner | dim 64/64 · lr 5e-5 · shift 2.2 |
| Qwen Image 2.1 (T2I **e** edição) | DiffSynth-Studio (commit fixo) | dim 32 · lr 1e-4 · AdamW constante · até 1 MP |
| Ideogram 4 | musubi-tuner | dim 32/32 · lr 1e-4 · DiT já é FP8 |
| LTX 2.3 | musubi fork AkaneTendo25 (`ltx-2-dev`) | dim 256/256 · lr 1e-4 · shifted_logit_normal |

fp8/blocks_to_swap são decididos sozinhos pela VRAM detectada; um blocks_to_swap
acima do máximo do modelo é recusado antes de baixar qualquer coisa.

Só LoRA padrão (LoHa/LoKr saíram de propósito). No painel ⚙ dá pra ligar **LoRA+**
(ratio 16) e mexer em rank, lr, epochs.

Cada treino começa com `output/` vazio: o anterior vai para `runs/<data>/`, então
o repo HF e a galeria só mostram o treino atual.

## Qwen Image 2.1 — texto→imagem ou edição (pares e trios)

Treinado pelo [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (o musubi não
suporta o 2.1). Um checkpoint só faz as duas coisas; **o dataset decide**:

- **sem pasta `control/`** → texto→imagem, captions como em qualquer modelo.
- **com `control/`** → edição. Para cada imagem-resultado `foto1.png`:
  - par: `control/foto1.png`
  - trio (2 referências): `control/foto1_0.png` + `control/foto1_1.png`
  - `foto1.txt` = a **instrução** de edição (escrita por você — o LLM de caption não é usado
    em edição, ele descreveria a imagem em vez da instrução).

Jeitos de mandar um dataset de trios:

```
meu_dataset.zip
├── target/      foto1.png  foto1.txt  foto2.png  foto2.txt …
├── control_0/   foto1.png  foto2.png  …     (1ª referência, mesmo nome do resultado)
└── control_1/   foto1.png  foto2.png  …     (2ª referência)
```

ou arraste os resultados + `.txt` no dataset e as referências na área "Imagens de controle"
(nomes `foto1_0.png`, `foto1_1.png`). A tela mostra quantos itens ficaram pareados; misturar
itens com e sem controle é recusado antes de qualquer download (o DiffSynth não treina os dois
no mesmo run).

Por VRAM: ≥ 70 GB bf16 · 40–70 GB text encoder em FP8 · 30–40 GB (RTX 5090) DiT e text
encoder em FP8, e em edição as ativações vão para a RAM · < 30 GB offload (mais lento, sem
samples). Um sample por época é gerado dentro do próprio treino; em edição ele aplica a
instrução do primeiro item às referências dele. O LoRA sai no formato que o ComfyUI carrega
direto.

Pesos: `Qwen/Qwen-Image-2.1` do Hugging Face (~33 GB), baixados pelo trainer; o DiffSynth
nunca baixa do ModelScope.

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

Toggle "Style Rush" no topo, para **Flux Klein**. Você manda só o
dataset de estilo e preenche a trigger word; o trainer faz o resto:

1. captions do dataset via OpenRouter (perfil `style-rush`, trigger na primeira palavra);
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
├── engines/                   musubi-tuner · musubi-ltx · sd-scripts · captionkreator (1 venv cada)
├── models/<modelo>/           pesos base (baixados no primeiro treino)
└── trainero_projects/<slug>/  dataset/ · cache/ · output/ · runs/ · logs/ · *.toml
```

## Chaves

| Env | Para quê |
|---|---|
| `HF_TOKEN` | repo + upload de checkpoints (privado por default), download de modelos gated (FLUX.2) |
| `OPENROUTER_API_KEY` | captions via LLM |

## Captions

Feitas pelo [captionkreator](https://github.com/adbrasi/captionkreator) (`captioner run`),
instalado como engine próprio (sem torch). Os perfis são deste repo, em
`caption_profiles/<image|video>/<perfil>/` (`profile.yaml` + `system.md` + `user.md`):
tags WD14 locais como apoio + LLM no OpenRouter.

| O que você treina | Anima | Outros modelos de imagem |
|---|---|---|
| Estilo | `anima-style` | `generic-style` |
| Personagem | `anima-character` | `generic-character` |
| Conceito / outro | `anima-concept` | `default` |
| Roupa | `anima-outfit` | `default` |

Vídeo: `grok_looping_animation_template` no LTX 2.3. Dá para fixar outro perfil no card
de captions.

- **Só preenche o que falta.** `.txt` que você mandou nunca é sobrescrito nem cobrado.
- **Cascata:** `meta/muse-spark-1.3-contributor` → `google/gemini-3.8-flash` →
  `x-ai/grok-4.7` (vídeo: `z-ai/glm-5.3-flash`). No Style Rush o Gemini vem primeiro:
  o que ele recusa é o que o gpt-image-2 também recusaria.
- **Recusado por todos** sai do dataset para `descartadas/` (movido, não apagado) —
  mas só depois de confirmar que a conta do OpenRouter está viva. Sem crédito, o job
  para e nada sai do dataset.
- Retomada de graça: o state do captionkreator (`.captioner.sqlite`) fica no dataset.

### Ajustes opcionais

| Env | Default | Para quê |
|---|---|---|
| `CAPTION_CONCURRENCY` | 64 | chamadas paralelas na captionagem |
| `CONVERT_WORKERS` | 8 | chamadas paralelas ao gpt-image-2 no Style Rush |

Para trocar modelos ou prompts de caption, edite o `profile.yaml` do perfil — é a única
fonte da verdade.
