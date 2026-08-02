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
| Krea 2 | musubi-tuner | dim 32/32 · lr 1e-4 · krea2_shift |
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

## Epochs, nunca max steps

No import calculamos repeats + epochs para o total de treino cair sempre na faixa
boa do modelo (~2000 steps imagem, ~3000 vídeo, ~660 exposições/img no Anima),
com ~10 checkpoints salvos. Qualquer tamanho de dataset.

## Concept Sliders

Toggle "Concept Slider" no topo:

- **LTX 2.3**: slider nativo do fork — só pares de prompt (positivo/negativo), sem dataset.
- **Demais modelos**: dois datasets (conceito ALTO / BAIXO); treina dois LoRAs
  idênticos e monta o slider por concatenação de rank com sinal invertido
  (ΔW = ΔW⁺ − ΔW⁻, exato). Força positiva = mais conceito; negativa = menos.

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
| `OPENROUTER_API_KEY` | captions via LLM |
