# Captions e Style Rush — Design

Data: 2026-08-24. Status: aprovado pelo dono.

Decisões do dono, na ordem em que foram dadas: cascata de caption vira **três modelos**
(Gemini primeiro, Muse Spark depois, Grok por último), porque a recusa do Gemini é o sinal
que diz quais imagens o `gpt-image-2` vai recusar e esse sinal **tem que ser preservado**;
o `system_prompt.md` do perfil `generic-style` é substituído pelo prompt novo, **mantendo o
pixai** e rebaixando as tags a apoio; o Style Rush passa a insistir **até bater uma meta de
sucessos** em vez de morrer com o dataset curto, com a meta exposta como input na UI e
default **100**; nova opção de **refazer todas as captions**, desligada por default, nos dois
modos; `num_repeats` do Style Rush vai de 1 para **2**.

## Por que agora

Três problemas independentes, um mesmo caminho de código:

1. As captions do modo normal estão fracas. O `generic-style` atual é dominado por tags
   ("Tags are ground truth, never contradict a tag based on the image alone"), então produz
   prosa engessada com cara de lista de tag.
2. O Style Rush entrega dataset curto em silêncio. Slot recusado nas duas tentativas morre,
   e ninguém downstream nota que os 50 pares viraram 31.
3. Não há como refazer caption. Um dataset que chega com `.txt` pronto é aceito como está,
   mesmo quando o dono quer recaptionar tudo com o prompt novo.

---

## 1. Cascata de caption com três modelos

`trainero/captioner.py`. Hoje há duas constantes soltas — `DEFAULT_CAPTION_MODEL` e
`FALLBACK_CAPTION_MODEL` — e `generate_captions` chama `_pass` uma vez para cada. Vira
**uma lista ordenada**:

```python
DEFAULT_CAPTION_MODELS = [
    "google/gemini-3.7-flash",              # primário: barato e é o que flagra
    "meta/muse-spark-1.2-contributor",      # resgata o que o Gemini recusa
    "x-ai/grok-4.20",                       # última instância
]
```

Sobrescrita por `CAPTION_MODELS`, uma variável só, lista separada por vírgula. Três env vars
numeradas para uma cascata de tamanho fixo seria o mesmo erro que a lista existe para
consertar.

`DEFAULT_CAPTION_MODEL` continua existindo como `CAPTION_MODELS[0]`, porque `record_flagged`
grava esse id no `.caption_refused.json` e `tests/test_core.py:340` afirma o valor.

`generate_captions` roda a lista em ordem, cada passe recebendo só o que sobrou do anterior,
e para assim que `ds.uncaptioned` volta vazio — a cascata nunca paga por um modelo que não
tem trabalho.

### O que preserva a heurística de flag

`record_flagged` marca as imagens **recusadas pelo primário e resgatadas por qualquer modelo
posterior**. Essa é a prova de que a recusa foi de conteúdo e não de infraestrutura: uma conta
morta deixa tudo sem caption, um filtro de conteúdo não.

Manter o Gemini como primário é o que mantém essa inferência calibrada. O `content_flagged`
do Style Rush lê essa lista para não pagar por um slot que o `gpt-image-2` vai recusar, e
Gemini é o modelo cujo filtro mais se parece com o da OpenAI. Trocar o primário por um modelo
mais permissivo esvaziaria a lista; por um mais restrito, encheria de falso positivo e
encolheria o pool pago sem motivo. **Nenhuma mudança em `content_flagged`.**

### Evidência de que o Muse Spark faz o trabalho

O dono comparou cinco modelos com o prompt novo sobre uma imagem explícita
(`OpenRouter Chat Mon Aug 24 2026.json`). Resultado extraído do export:

| modelo | JSON válido | palavras | custo |
|---|---|---|---|
| `meta/muse-spark-1.2-contributor` | sim | 315 | **$0,00061** |
| `xiaomi/mimo-v2.5` | sim | 184 | $0,00070 |
| `minimax/minimax-m3` | sim | 193 | $0,00164 |
| `deepseek/deepseek-v4-flash-vision-exp` | sim | 220 | $0,00174 |
| `qwen/qwen3.8-27b` | sim | 189 | $0,00468 |

O Muse Spark captionou sem recusar, sem trace de raciocínio, e foi o mais barato dos cinco.
O `is_moderated: true` que o catálogo do OpenRouter reporta não bloqueou nada na prática.

Única ressalva: 315 palavras contra o teto de 220 que o prompt pede para cena densa. Ele
estoura o limite de comprimento em ~50%. Não é defeito de conteúdo e não justifica mexer no
prompt agora.

---

## 2. Prompt novo do `generic-style`

O perfil vive em **outro repositório**: `adbrasi/data_araknideo`, em
`prompts/image/generic-style/`. `PROMPTS_DIR` está fixo em
`os.path.dirname(tag_images_by_wd14_tagger.py)/prompts`, sem flag nem env var, então não há
como servir o prompt a partir do arrakis_trainero.

`system_prompt.md` é substituído pelo conteúdo de `new_system_prompt.md`, com duas
adaptações. Elas existem porque o prompt novo, literal, diz *"You receive an image and
nothing else. The image is the only source of truth"* — e o pipeline roda o pixai antes e
injeta `{tags}`. Colar o texto sem resolver deixaria o modelo com duas ordens opostas.

1. **Regra 1** passa a dizer: a imagem é a verdade; as booru tags do pixai são apoio, servem
   para confirmar detalhe e recuperar nome de personagem ou item que o modelo não reconhece,
   nunca para contradizer a imagem, e nunca entram como sintaxe de tag.
2. **`user_prompt.md`** hoje diz *"Tags are your primary source of truth"*. Inverte para a
   mesma hierarquia.

Todo o resto entra literal: política de conteúdo explícito, ordem do camera report, seção do
que não descrever, comprimento, os três exemplos.

`profile.json` não muda — a variável `style_name` continua sendo a trigger.

### Clone velho é a armadilha

`ensure_engine` clona o engine uma vez e **nunca dá pull**: se `.git` existe, loga
"já clonado" e sai. Pod novo pega o prompt novo sozinho; pod que já roda há dias captiona o
dataset inteiro com o prompt velho e o dono só descobre depois do treino.

Correção: campo declarado no spec do engine.

```python
"captioner": {..., "pull": True},
```

`ensure_engine` faz `git pull --ff-only` quando o campo está ligado. Ligado **só** no
captioner, e a razão é uma propriedade real dele: é o único engine cujo *conteúdo* muda entre
runs enquanto as dependências não. Dar pull no musubi arriscaria um venv que já funciona.

---

## 3. Style Rush: meta de sucessos

`trainero/style_rush.py`. Hoje `plan_slots` monta 50 slots fixos, slot `i` amarrado ao prompt
`i`, com imagem primária e uma imagem de fallback. Recusa nas duas → slot descartado. O
dataset sai curto e o treino roda em cima disso sem um único erro.

Vira **meta de sucessos**.

- **Fila de tentativas** determinística (`PLAN_SEED`), cada entrada um par
  `(prompts[i % len(prompts)], imagem)`. O ciclo por módulo é o que faz qualquer meta
  funcionar com qualquer quantidade de prompts.
- **Teto de `meta × 3` tentativas.** Sem isso, um dataset em que tudo é recusado gira até a
  conta secar. Estourou o teto, o job segue com o que conseguiu e loga quanto faltou —
  truncar em silêncio é o defeito que este design existe para consertar.
- **Reserva de orçamento sob lock.** O worker decrementa um contador antes de chamar a API e
  volta sem chamar se não sobrou nada. Isso é o que garante que a meta **nunca** é
  ultrapassada; `ThreadPoolExecutor` com 4 workers gastaria até 3 imagens a mais no fim.
- **Nome do par = índice da tentativa que deu certo** (`slot_007.png`). Buraco na numeração
  não importa: o musubi varre o diretório, nada depende de sequência. A alternativa —
  numerar por contagem de sucesso — exigiria dois esquemas de numeração convivendo.
- **Resume** por índice de tentativa: pula o que o manifest já registra como `ok` ou
  `refused`, continua até a contagem de `ok` bater a meta. O que já foi pago não é refeito.
- **Imagem recusada** entra em `refused_images` e é pulada globalmente — a fila converge
  sozinha para as imagens que passam. Comportamento que já existe, e que agora é o que faz a
  meta ser alcançável.

O par `sources: [primária, fallback]` do slot **desaparece**. Ele era o retry embutido em cada
slot, e a fila agora é o retry — manter os dois seria duas camadas de retry disputando o mesmo
orçamento. `plan_slots` vira o construtor da fila e devolve tentativas, não slots.
`RETRIABLE_ATTEMPTS` continua como está: é o retry de erro transitório (429/5xx) dentro de uma
tentativa, coisa diferente de recusa.

`SLOT_COUNT = 50` deixa de ser o alvo e vira `DEFAULT_CONVERT_TARGET = 100`.
`load_style_prompts` para de truncar em `SLOT_COUNT` e passa a exigir apenas que o arquivo
não esteja vazio — quem cicla é a fila.

### Prompts

`data/style_prompts.txt` recebe **50 linhas novas**, chegando a 100. As novas mudam só o
estilo — mídia, traço, sombreamento, era, tradição — nunca iluminação ou cor, porque cor e luz
não são conversão de estilo e o arquivo já tinha entradas assim. Cobrem linóleo, scratchboard,
têmpera com folha de ouro, afresco, encáustica, mosaico, bordado, batik, henna, giz em lousa,
caneta esferográfica em papel pautado, peças de plástico encaixáveis.

`tests/test_style_rush.py:63` afirma que os prompts são distintos entre si; segue valendo.

### UI

Campo **"Imagens para conversão"** no painel avançado (`web/index.html`), visível só no modo
Style Rush — mesmo padrão do `#adv-ltx-res-wrap`. Default 100, entra como
`overrides.convert_target`.

O dataset de restauração continua em 100 e sem input: é CPU local, custo zero, e o dono pediu
input só para a conversão.

### Custo

100 pares × $0,0142 = **~$1,42 por run**, contra ~$0,71 de hoje.

---

## 4. Refazer captions

Checkbox **"Refazer todas as captions"**, `off` por default, nos dois modos.

Ligado, antes da fase de caption: apaga todo `.txt` ao lado de um arquivo de mídia no
dataset, mais `.tagger_log.json` e `.caption_refused.json`. Os dois últimos são obrigatórios —
o tagger pula o que está no log, então sem apagá-lo o botão não faria nada nos arquivos que o
dono mandou refazer.

Não toca em `descartadas/`: aquilo já saiu do dataset e é a única cópia do que foi
quarentenado.

Como apaga caption escrita à mão, o clique pede confirmação na UI.

```python
def clear_captions(dataset_dir: Path, job: Job | None = None) -> int
```

Chega ao servidor como `redo` no `POST /api/captions` e como `overrides.redo_captions` no
`POST /api/train`.

---

## 5. Repeats

`STYLE_RUSH_SCHEDULE` em `trainero/presets.py:411`: `num_repeats` de 1 para **2**, valendo
para os três subsets (base, conversão, restauração), que é como o Style Rush já monta o
`dataset.toml`.

Com base de ~50 imagens: (50 + 100 + 100) × 2 = 500 steps por época × 5 épocas = **2500
steps**.

`tests/test_core.py:86` afirma o dicionário inteiro e precisa ser atualizado junto.

---

## Superfície de teste afetada

| arquivo | o que quebra |
|---|---|
| `tests/test_core.py:86` | `STYLE_RUSH_SCHEDULE` com `num_repeats: 1` |
| `tests/test_core.py:340` | `DEFAULT_CAPTION_MODEL` |
| `tests/test_captioner.py` | importa `FALLBACK_CAPTION_MODEL`, cascata de dois passes |
| `tests/test_style_rush.py` | `SLOT_COUNT`, `plan_slots`, nomes `slot_%02d` |

Testes novos que o design pede:

- a cascata para no primeiro modelo que zera as pendências, sem chamar os seguintes;
- `record_flagged` marca o que o primário recusou e um modelo posterior resgatou;
- a fila de conversão para exatamente na meta, com concorrência;
- o teto de `meta × 3` encerra e loga quando tudo é recusado;
- resume não repaga slot já `ok`;
- `clear_captions` apaga `.txt` e os dois logs, e não toca em `descartadas/`.

## Fora de escopo

- Mexer em `content_flagged` (resolvido: o Gemini fica de primário).
- Input para o dataset de restauração.
- `git pull` nos engines de treino.
- Ajustar o teto de comprimento do prompt por causa das 315 palavras do Muse Spark.
