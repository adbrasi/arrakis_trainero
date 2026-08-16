# Arrakis Trainero — handoff para a sessão na GPU

Escrito em 2026-08-16 por uma sessão que trabalhou **sem GPU**, em WSL2, com a
regra explícita do dono de não rodar o software localmente. Este documento
existe porque metade do produto nunca foi executada de verdade.

Leia inteiro antes de mexer. As seções "O que NUNCA foi executado" e "Regras do
dono" são as que evitam retrabalho.

---

## 1. O que é

Treinador de LoRA com UI web de página única, feito para pods vast.ai/RunPod.
Um servidor stdlib (`server.py`) serve a UI e uma API JSON; a UI faz polling.
Um único job por vez (import, captions, treino) roda numa thread daemon e
transmite o log para a UI.

Inspiração e referência do dono: `github.com/adbrasi/arrakis_start` (o
instalador/inferência dele, que funciona bem). **Vale ler
`hf_xet_worker.py`, `progress.py` e `downloader.py` de lá antes de mexer em
download ou progresso** — já resolveram problemas que este repo ainda tem.

- Repo: `github.com/adbrasi/arrakis_trainero` (público)
- Branch de trabalho: `main`
- Subir: `curl -L https://raw.githubusercontent.com/adbrasi/arrakis_trainero/main/bootstrap.sh | bash`
- Porta: 8090 (`WEB_PORT`)

### Três modos

| Modo | O que faz |
|---|---|
| `lora` | um dataset, um LoRA |
| `slider` | dois datasets (+/−); no LTX 2.3 é nativo, por pares de prompt, sem dataset |
| `style-rush` | dataset do dono + 50 pares sintéticos de conversão gerados no gpt-image-2, treinados juntos |

### Layout em disco (no pod)

```
/workspace/
  arrakis_trainero/          # este repo; .venv leve só do servidor
  engines/<engine>/          # musubi-tuner, musubi-ltx, sd-scripts, data_araknideo
    .venv/                   # cada engine tem o seu
  models/<model_key>/        # pesos base
  trainero_projects/<slug>/
    dataset/  dataset_neg/  dataset_convert/
    cache/  output/  output/sample/  logs/
```

### Módulos

| Arquivo | Responsabilidade |
|---|---|
| `server.py` | HTTP + rotas + thumbs + samples + shutdown |
| `trainero/jobs.py` | job único, log ao vivo, cancelamento por process group |
| `trainero/presets.py` | registro de modelos (a fonte da verdade) |
| `trainero/engines.py` | clone + venv + install de cada engine |
| `trainero/models_download.py` | escolha de transporte (Xet vs aria2c) |
| `trainero/hf_fetch.py` | processo filho que faz uma transferência |
| `trainero/dataset.py` | import (Mega/HF/URL/local), normalização, inspect |
| `trainero/captioner.py` | captions via data_araknideo + OpenRouter |
| `trainero/style_rush.py` | dataset sintético de conversão |
| `trainero/imagegen.py` | cliente OpenRouter Images (gpt-image-2) |
| `trainero/training.py` | dataset.toml, config, fases, execução |
| `trainero/hf_upload.py` | repo HF + upload contínuo de checkpoints |
| `web/` | index.html, styles.css, app.js |

---

## 2. Regras do dono — não negociáveis

Estão no `~/.claude/CLAUDE.md` dele, mas repito o que mais pesou aqui:

1. **Respostas e mensagens de commit em pt-BR.** Código, identificadores,
   comentários e docs técnicos em inglês.
2. **Push só quando ele autorizar explicitamente.** Ele testa antes.
   (Eu errei uma vez ao contrário: deixei uma correção crítica commitada e não
   pushada, e ele rodou um treino de 50 min com o log quebrado por causa disso.
   Se uma correção conserta algo que ele está sofrendo agora, avise e peça.)
3. **Nada de gambiarra.** Palavras dele: se você bate num muro, o muro é
   informação — o design está errado. Re-derive dos primeiros princípios.
   Flag, caso especial, shim de conversão, segundo canal, caminho paralelo,
   teste reescrito para desviar de regra quebrada: tudo isso é rejeitado 100%
   das vezes. Bloqueio reportado com honestidade é bom resultado; entrega
   "funcionando" em cima de gambiarra é o pior possível.
4. **Comunicação ADHD.** Primeira linha é uma ação executável. Passos
   numerados. Sem preâmbulo, sem recapitulação, sem "espero ter ajudado".
   Listas de no máximo 5 itens. Estimativas de tempo concretas.
5. Ele é autista e tem TDAH. Seja direto e factual, principalmente em erro.

### Como ele revisa

Ele roda um agente auditor em cima do seu trabalho e traz os achados. Vários
foram reais. **Confira cada um antes de aceitar** — um dos achados dele
("Krea 2 quebrado, `presets.py:94`") apontava a linha certa mas a causa errada,
e dizer isso claramente foi mais útil do que concordar.

---

## 3. Estado atual

`main` = `e2dd159`, pushado. 123 testes passando, 1 pulado (falta Torch).

### Verificado de verdade

- **Preços e API do gpt-image-2** no OpenRouter: chamadas reais. Custo medido
  $0,0142/imagem no teto (referência 1024×1024). Parâmetros aceitos:
  `aspect_ratio`, `quality`, `background`, `n`, `input_references`,
  `output_compression`. `size`/`resolution` são rejeitados.
- **Modelo de captions** `google/gemini-3.7-flash`: chamada real com o payload
  exato do captioner (json_schema estrito + `reasoning: low` + plugin
  `response-healing`). Devolve as 4 chaves do schema, ~$0,0007/imagem.
  Atenção: sem `reasoning: low` ele gasta o orçamento em raciocínio e devolve
  caption vazia.
- **Todos os 24 assets dos presets existem no HF** (teste de rede em
  `tests/test_downloads.py`). 20 públicos, 3 gated (Krea raw, FLUX klein base,
  FLUX ae — precisam de `HF_TOKEN`), 1 estava 404 e foi corrigido.
- **Xet**: transferência real de 242 MB em ~25s, com % correta no log.
- **Pareamento control/target do Style Rush**: portei o algoritmo de bucket do
  musubi para dentro de um teste e provei que sem `fit_control_to_target` os
  pares caem em buckets diferentes.
- **Frontend**: Playwright em `file://` com `fetch` stubado, dirigindo o
  `app.js` real. Ver seção 6.

### O que NUNCA foi executado — a razão de você estar aí

Nada abaixo jamais rodou. Trate tudo como não verificado:

1. **`bootstrap.sh` de ponta a ponta num pod limpo.**
2. **Instalação de qualquer engine** (`trainero/engines.py`): clone, venv, pip,
   `accelerate config`. Os 21 scripts esperados existem nos upstreams (checado
   por HTTP), mas nenhum `pip install` rodou.
3. **Download completo de um modelo base.**
4. **Qualquer treino, de qualquer modelo.** As flags geradas foram conferidas
   contra os parsers dos upstreams por leitura de código, não por execução.
5. **Cache de latents e de text encoder.**
6. **Sampling durante o treino** — a maior fonte de risco histórica aqui.
7. **Upload contínuo para o HuggingFace.**
8. **Conversão para formato ComfyUI** (só o Anima usa).
9. **Import de dataset** por Mega, repo HF, URL e caminho local.
10. **Geração de captions** de verdade (a chamada isolada ao Gemini funcionou;
    o `tag_images_by_wd14_tagger.py` nunca rodou).
11. **Style Rush completo** (as 50 gerações pagas nunca foram disparadas).

---

## 4. Problemas abertos

### 4.1 Retomada de download com Xet — NÃO RESOLVIDO

O dono pediu explicitamente: "ele continua de onde parou? precisa!"

O que eu medi (matando uma transferência real no meio e reiniciando):

- **O Xet não retoma por bytes.** A segunda tentativa baixa tudo de novo.
- O chunk-cache em `~/.cache/huggingface/xet` fica praticamente vazio
  (0,1 MB depois de 242 MB baixados), com e sem `HF_XET_HIGH_PERFORMANCE`.
  A docstring do `xet_get` do `huggingface_hub` promete reuso de chunks; na
  prática não observei.
- **Retomada por arquivo já funciona**: o que terminou é movido para o destino
  e pulado na próxima execução (`_present`).
- **O aria2c retoma** (`--continue=true` + `.part` estável), mas hoje ele só é
  a primeira escolha quando o arquivo comprovadamente não está no Xet.

Um caminho tentador e **errado**: mover o `.incomplete` do hub para o `.part` do
aria2. A reconstrução do Xet não escreve em ordem, então esse arquivo não é um
prefixo válido — daria corrupção silenciosa. Não faça.

Investigue no pod: talvez com banda de datacenter o cache se comporte diferente,
ou exista variável do `hf_xet` que eu não achei. O `downloader.py` do
`arrakis_start` tem watchdog de stall e um perfil de aquecimento documentado
que provavelmente ajuda.

### 4.2 Perfil de aquecimento do Xet

Medido: 1,6 MB em 3,5s → 12 MB em 11,7s → 242 MB em 24,5s. **Ele começa lento e
dispara.** Qualquer watchdog por taxa mata transferência saudável no aquecimento.
O `arrakis_start` documenta exatamente isso e usa "bytes entregues não crescerem
nada" como sinal de falha, não taxa. Se for colocar timeout, siga esse modelo.

### 4.3 Barra de progresso da UI durante o download

O log mostra a linha viva, mas `#bar-wrap` (a barra da UI) só é alimentada por
`step/total_steps` do treino. Bytes de download não entram lá. Decidi não
misturar as duas coisas; se for fazer, é campo novo, não sequestro do existente.

---

## 5. Armadilhas já pagas — não repita

1. **`readline()` congela qualquer barra.** O `tqdm` redesenha com `\r` e não
   emite `\n` enquanto anda. Corrigido em `jobs.py::_stream` com `read1()` +
   corte em `\r` e `\n`; frame terminada em `\r` é transiente e sobrescreve a
   anterior no arquivo. Custou um treino de 50 min do dono às cegas.
2. **`huggingface_hub` desabilita a própria barra fora de TTY.** A única saída é
   injetar `tqdm_class` (ver `trainero/hf_fetch.py::Progress`). Foi assim que o
   `arrakis_start` resolveu.
3. **`--sample_prompts` carrega VAE e text encoder** (`trainer_base.py`
   `_prepare_sampling`, só `if args.sample_prompts`). Se o preset não declarar
   `vae`/`text_encoder`/`t5` em `model_args`, o treino morre **depois** do
   download inteiro. Já mordeu wan-22 e ideogram. Existe
   `test_every_model_can_render_a_sample` para impedir a volta.
4. **`git checkout -- <arquivo>` apaga trabalho não commitado.** Eu perdi
   edições duas vezes usando isso em script de falsificação. **Commite antes de
   falsificar.**
5. **O atributo `hidden` perde para qualquer `display: flex/grid` do autor.**
   Todo vazamento entre modos da UI veio disso. Hoje há
   `[hidden] { display: none !important; }` no topo do `styles.css`.
6. **`os.replace` não cruza filesystem.** `save_state` grava o temporário em
   `DATA_DIR`; um teste que aponta `STATE_FILE` para `/tmp` quebra.
7. **Overrides são relativos a um preset.** Trocar de modelo ou de modo
   invalida todos (`resetOverrides` no `app.js`). Sem isso, rank do Krea
   chegava no Anima e repeats chegavam no Style Rush, que não tem repeats.
8. **O flag do captioner se chama `--grok_model` por herança**, mas aceita
   qualquer model id do OpenRouter. Não confunda nome de flag com modelo.

---

## 6. Como verificar sem GPU (o que eu montei)

Fica em `/tmp/.../scratchpad`, fora do repo. **Recrie se for útil:**

- `preview.py <modo> <empty|filled> <saida.png> [largura]` — copia `web/` para
  um diretório, injeta um `window.fetch` stubado alimentado pelos **presets
  reais** (`public_presets()`, `suggest_schedule`), e dispara o Playwright.
  Stub com valores inventados faz asserção passar por motivo errado: eu já tinha
  um que dava `dim 128` a todo modelo e o teste de vazamento de override passava
  sozinho.
- `assert_ui.cjs <file://.../index.html>` — dirige o `app.js` real no Chrome e
  afirma comportamento (vazamento de override, campo duplicado de trigger, ordem
  `/api/project` antes de `/api/dataset/import`, cor de erro, zero erro de JS).

Playwright já instalado em `~/node_modules/.pnpm/playwright@1.59.1/node_modules`
(passe via `NODE_PATH`). Use `reducedMotion: "reduce"` — screenshot por CLI do
Chrome pega transição no meio e a página sai lavada.

**Disciplina que o dono espera:** todo teste novo de correção crítica precisa ser
falsificado — reverta a correção e confirme que o teste quebra. Fiz isso com 11
correções. Um teste que nunca falhou não provou nada.

---

## 7. Plano sugerido para a sessão com GPU

Ordem pensada para falhar cedo e barato.

### Fase 1 — o caminho de instalação (~30 min, sem custo)

1. Pod limpo, `HF_TOKEN` e `OPENROUTER_API_KEY` exportados.
2. Rodar o `bootstrap.sh` e ver se a UI sobe na 8090.
3. `/api/status` deve reportar GPU, token HF e OpenRouter.
4. Instalar **um** engine (musubi) por um treino disparado e cancelado logo no
   início. Confirmar clone, venv, pip e `accelerate config`.

### Fase 2 — download (~15 min)

5. Modelo mais barato primeiro. Confirmar no log: linha de transporte
   (`via Xet.` / `via aria2c.`) e a **linha viva de progresso com % e
   velocidade**.
6. Cancelar no meio, reiniciar, e medir se retoma. É o item 4.1.
7. Confirmar que `<dest>.hfpart` some e o destino só aparece completo.

### Fase 3 — treino de verdade (~1 h)

8. LoRA simples, dataset pequeno (10–20 imagens), Flux Klein ou Qwen Image.
9. **Olhar o console durante o treino**: a barra `steps:` tem de atualizar ao
   vivo, e `epoch`/`loss`/barra da UI têm de andar — foi exatamente isso que
   quebrou para o dono.
10. Confirmar sample por época em `output/sample/` e a galeria da UI.
11. Confirmar checkpoint por época e upload para o HF, com
    `trainero_config.json` e `captions.json` no repo e **sem README**.

### Fase 4 — o resto

12. Captions de verdade (`tag_images_by_wd14_tagger.py` + Gemini).
13. Import por Mega, repo HF e URL.
14. Style Rush completo — **custa ~$0,71 em gpt-image-2**, peça autorização.
15. Anima (o único com conversão ComfyUI) e LTX 2.3 (engine e fork diferentes).

### Enquanto testa

- Cancelar tem de matar a árvore inteira de processos, sempre.
- Um job rodando tem de recusar upload/limpeza de dataset com 409.
- O botão Desligar tem de liberar a porta 8090 de verdade.

---

## 8. Ambiente

```bash
export HF_TOKEN="hf_..."             # necessário: 3 repos são gated
export OPENROUTER_API_KEY="sk-..."   # captions e Style Rush
export WEB_PORT=8090                 # opcional
export WORKSPACE_DIR=/workspace      # opcional
export CAPTION_MODEL="..."           # opcional; default google/gemini-3.7-flash
```

`HF_TOKEN` também é lido de `HUGGINGFACE_HUB_TOKEN` e `HUGGINGFACE_TOKEN`.

Rotas: `/api/status`, `/api/presets`, `/api/project`, `/api/dataset/{import,
file,clear,thumbs,thumb}`, `/api/captions`, `/api/train`, `/api/cancel`,
`/api/logs`, `/api/samples`, `/api/sample`, `/api/shutdown`.

Testes: `python3 -m pytest tests -q` na venv do servidor. Um teste de rede
(`test_no_asset_is_404`) pula sozinho sem internet.

---

## 9. Decisões de design que têm motivo

Não desfaça sem entender o porquê:

- **Um preset por modelo**, sem matriz de opções. O dono odeia
  over-engineering em trainer. Os únicos knobs ficam no painel "Ajustes finos" e
  entram como override por cima.
- **Epochs, não steps.** Preferência explícita dele.
- **Style Rush tem schedule fixo** (`STYLE_RUSH_SCHEDULE`, 5 epochs, repeats 1):
  ele cancela quando os samples ficam bons.
- **Sem checkbox de conversão ComfyUI.** LoRAs do musubi carregam direto no
  ComfyUI (`model_lora_keys_unet` mapeia `lora_unet_<chave achatada>` de forma
  genérica); só o Anima do sd-scripts precisa converter, e isso está declarado
  no preset. O preset é a única fonte de verdade.
- **Sem model card gerado.** Só `trainero_config.json` e `captions.json`.
- **Polling, não websocket.** Pods expõem uma porta só.
- **Nada parcial fica visível no destino.** Toda transferência aterrissa num
  `.part`/`.hfpart` e só é movida quando termina, então "o caminho existe" é o
  teste completo de prontidão.
- **Token nunca em argv.** `/proc/<pid>/cmdline` é legível por qualquer um; o
  aria2c recebe por arquivo 0600 e o filho do hub por `HF_TOKEN` no ambiente.

---

## 10. Primeira coisa a fazer na sessão nova

```bash
cd /workspace/arrakis_trainero && git log --oneline -5 && python3 -m pytest tests -q
```

Confirme que está em `e2dd159` ou depois e que os 123 testes passam. Depois
siga a Fase 1 da seção 7.
