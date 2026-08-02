# Arrakis Trainero — Design

Data: 2026-08-02. Status: aprovado para implementação (diretriz do dono: completar autônomo, sem perguntas).

## O que é

Trainer de LoRA "1-click" para pods vast.ai/RunPod. Uma página web, três passos:

1. Nome do projeto.
2. Dataset (upload de imagens/vídeos + .txt, zip, link Mega, repo/zip HuggingFace, pasta local).
3. Modelo → botão **TREINAR**.

Todo o resto é automático: instalação do engine, download dos modelos base, dataset.toml,
cache de latents/text-encoder, treino contado em **epochs**, upload contínuo pro HuggingFace.

Anti-objetivo explícito (transcrição do dono): NÃO é um trainer detalhado com presets por
rede/rank/época. **Todos os presets são o padrão LoRA por modelo.** Um painel colapsável
único permite trocar LoRA→LoKr/LoHa/DoRA (onde o backend suporta), LoRA+, rank, epochs.

## Modelos e backends

| Modelo | Backend | network_module | Preset (do upstream/projetos do dono) |
|---|---|---|---|
| Krea 2 | musubi-tuner (kohya-ss) | networks.lora_krea2 | dim/alpha 32, lr 1e-4, adamw8bit, `krea2_shift` |
| Anima | sd-scripts (ver relatório research_anima_train) | lora anima | modelo `circlestone-labs/Anima` |
| Flux Klein | musubi-tuner | networks.lora_flux_2 | `--model_version klein-base-9b`, dim 32, lr 1e-4, `flux2_shift` |
| Qwen Image | musubi-tuner | networks.lora_qwen_image | dim 16, lr 5e-5, `qwen_shift` |
| Qwen Image Edit | musubi-tuner | networks.lora_qwen_image | `--model_version edit-2511`, control em `control/`, control_resolution 1024 |
| Ideogram | musubi-tuner | networks.lora_ideogram4 | dim/alpha 16, lr 1e-4, `ideogram4_shift`, DiT já é fp8 |
| LTX 2.3 | musubi fork `AkaneTendo25/musubi-tuner@ltx-2-dev` | networks.lora_ltx2 | rank/alpha 256, lr 1e-4, `shifted_logit_normal`, caption_dropout 0.05, first_frame_p 1.0 (do script ltx23 do dono) |
| Wan 2.2 | musubi-tuner | networks.lora_wan | task t2v-A14B, dit low+high, dim 32, lr 2e-4, shift 3.0 |

Rede alternativa: musubi → LoRA/LoHa/LoKr (`networks.loha`/`networks.lokr`); LoRA+ via
`--network_args loraplus_lr_ratio=4`. DoRA só onde o backend tem (sd-scripts/Anima).

VRAM-adaptativo: `nvidia-smi` na hora do treino decide fp8_base/fp8_scaled/blocks_to_swap
por tabela por arquitetura.

## Epochs, nunca max steps

No import calculamos: `num_repeats = clamp(round(250/N), 1, 20)`;
`epochs = clamp(round(target_steps / (N*num_repeats)), 6, 20)` com target_steps 2000
(imagem) / 3000 (vídeo). Resultado: ~10 epochs, ~1 checkpoint por epoch
(`--save_every_n_epochs 1`), total de steps sempre numa faixa boa, qualquer tamanho de
dataset. Editável no painel.

## Dataset

Import (padrões portados do character_animatrem): detecção automática de fonte
(pasta local, arquivo local, URL http, Mega via megatools/megacmd, repo HF via
snapshot_download, URL de arquivo HF), extração zip/rar/7z/tar, descida de pasta-invólucro,
achatamento com par mídia+.txt preservado. Upload direto pelo browser (multipart, aceita
muitos arquivos ou um zip). Pasta `control/` dentro do dataset = control images (Qwen Edit).
Nunca mutamos a fonte: staging em `projects/<nome>/dataset/`.

Captions: se todo item já tem .txt → nada a fazer (fluxo normal do dono). Se faltar →
card de captions aparece: preset (prompts do data_araknideo: anima-character, anima-style,
anima-concept, anima-outfit, default/flux, generic-*; vídeo: nsfw_caption_video,
grok_looping_animation_template, live-wallpaper*), trigger word, botão Gerar. Execução:
clone do data_araknideo + `tag_images_by_wd14_tagger.py --taggers pixai,grok
--grok_provider openrouter --prompt_profile X --prompt_var ...` (OPENROUTER_API_KEY).

## Concept Sliders

Modo alternativo na mesma página: dois inputs de dataset (conceito ALTO / conceito BAIXO).
Treina dois LoRAs padrão com a mesma config e produz o slider por concatenação de rank com
sinal invertido (ΔW = ΔW_pos − ΔW_neg, exato para LoRA linear): para cada módulo,
down = concat(down_p, down_n), up = concat(up_p·s_p, −up_n·s_n), alpha = rank novo.
Roda no venv do engine (torch já instalado). Só LoRA padrão (sem LoKr) nesse modo.

## HuggingFace

Repo `{username}/{project}` criado no início do treino (toggle ON por default, privado por
default). Thread watcher: novo `.safetensors` no output → espera estabilizar → upload →
dedupe em `.hf_uploaded.log`. Model card + metadata gerados antes do treino (padrão
animatrem). Token: `HF_TOKEN` env ou cache do hf CLI.

## Arquitetura (padrão arrakis_start)

Python stdlib `ThreadingHTTPServer` + HTML/CSS/JS vanilla, zero build. Polling de
`/api/status` (pods expõem uma porta só; websocket não passa no proxy). Job único por vez
(slot com lock). Logs: arquivo por job + tail incremental via `/api/logs`. Estado:
`state.json` com escrita atômica (tmp+fsync+replace).

```
bootstrap.sh          curl|bash no pod → clona, venv leve, sobe server :8090
server.py             HTTP + API + estáticos
trainero/
  config.py           paths (/workspace ou cwd), state
  presets.py          registry dos 8 modelos (dados puros: downloads, args, vram tiers)
  engines.py          git clone + venv uv por engine (musubi, musubi-ltx, sd-scripts, captioner)
  models_download.py  aria2c → hf_hub_download fallback
  dataset.py          import/inspeção/contagens
  captioner.py        integração data_araknideo
  jobs.py             runner em thread, stream de log, parse de progresso, cancel
  training.py         dataset.toml + comandos cache/train por engine
  hf_upload.py        create_repo + watcher + model card
  sliders.py          orquestra pos/neg + merge
  tools/make_slider.py  roda no venv do engine (torch)
web/ index.html app.js styles.css   página única, dark, estética arrakis_start
```

API: `GET /api/status`, `GET /api/presets`, `POST /api/project`,
`POST /api/dataset/import`, `POST /api/dataset/upload` (multipart),
`POST /api/captions/generate`, `POST /api/train`, `POST /api/cancel`, `GET /api/logs`.

Pipeline de treino (fases visíveis na UI): Engine → Modelos base → Dataset.toml →
Cache latents → Cache text encoder → Treino (epochs) → Upload final.

## Erros

Sem gambiarras: falha em qualquer fase para o job, loga a causa real e mostra na UI com a
última página de log. Cancel = SIGTERM no process group → SIGKILL. Retry conservador do
cache de latents em SIGKILL/OOM (padrão do script ltx23: metade do batch/workers).

## Testes

`python -m compileall` + testes unitários dos puros: presets (todo modelo gera comando
válido), cálculo de epochs, detecção de fonte do dataset, montagem do dataset.toml,
merge de slider (matemática do concat verificada com torch no CI local se disponível).
Smoke: subir o server e bater nos endpoints principais sem GPU.
