# Pesquisa: receitas de LoRA para Krea 2 (musubi-tuner), 2026-11-03

Feita por um agente de pesquisa (web). **[K2]** = dado específico do Krea 2. **[T]** = transferido de Flux, Qwen ou Wan,
ou palpite, a validar com testes.

## Fatos firmes [K2]
- **Raw e Turbo:** treinar no Raw e inferir no Turbo ([musubi krea2.md](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/krea2.md)).
- **Base do doc:**
  - `lora_krea2` dim 32/32, adamw8bit, lr 1e-4, gradient checkpointing, sdpa;
  - `--timestep_sampling shift --discrete_flow_shift 2.5` em 1024 fixo;
  - em multi-resolução, `--timestep_sampling krea2_shift`, com shift de ~1.6 em 256² a ~3.2 em 1280².
- **Memória:** `--fp8_base --fp8_scaled` (os dois juntos) e `--blocks_to_swap` até 26.
- **Amostras:** `--turbo_dit` com 8 steps e cfg 1, sem block swap.
- **Inferência Turbo:** 8 steps, cfg 1, mu 1.15.
- **Steps ótimos por tamanho do dataset** (ai-toolkit, LoKr, [imagelora](https://imagelora.dev/experiments/krea2-dataset-size/)):

  | imagens | steps |
  |---|---|
  | 3 | ~1000 |
  | 8 | ~1440 |
  | 15 | ~2100 |

  Acima de ~2700 steps o modelo entra em colapso para headshot frontal.
- **Captions de personagem:** descrever cabelo, roupa, fundo e expressão, **sem descrever o rosto**.
  - Captions naturais, de 70 a 500 palavras ([guia imagelora](https://imagelora.dev/guides/how-to-train-krea-2-lora)).
  - 3 imagens bastam para fotorrealista; estilo e ângulos difíceis pedem mais imagens.
- **Estilo com 348 imagens** ([HF Zaytron40k](https://huggingface.co/Zaytron40k/krea2-style-v4)): rank 32, 4120 steps (batch 4), `krea2_shift`, multi-bucket 1024. A loss fica quase plana, então o checkpoint se escolhe visualmente.
- **Prodigy** com 41 imagens ([techtactician](https://techtactician.com/?p=59793)): o melhor foi o step 1600 de 2829, e o final saiu overtrained.
- **Likeness:** aparece entre os steps 600 e 900 com 10 a 20 imagens ([krea2-character-lora-trainer](https://github.com/bongobongo2020/krea2-character-lora-trainer)).
- **Velocidade (musubi):**

  | GPU | s/step em 1024, fp8 |
  |---|---|
  | RTX PRO 6000 Blackwell | 2.3 |
  | RTX 3090 | 7.1 |
  | RTX 5090 | ~2–3 (estimativa [T]) |

## Receitas propostas (validar)
| cenário | dim/alpha | steps | res | timestep | captions |
|---|---|---|---|---|---|
| personagem 1–5 (expandido) | 16/16 | ~1000–1500 | 1024 | shift 2.5 ou krea2_shift | naturais, sem identidade |
| personagem 5–15 | 32/32 | ~1500–2100 | 1024 | idem | idem |
| personagem 15–30 | 32/32 | ~2100–2700 | 1024 | idem | idem |
| estilo 50–200 | 32/32 | ~2000–4000 | 1024 multi-bucket | krea2_shift | conteúdo, sem descrever o estilo; prefixo "in the style of X" |
| misto ~450 | 32/32 ou 64/64 | ~5000–7000 (batch 2–4) | 1024 multi-bucket | krea2_shift | trigger único por personagem + classe; repeats balanceando personagens e estilo |

## Dataset sintético
- **Quantidade e variação:** 10 a 20 imagens, variando ângulo (frente, 3/4, perfil, costas, de cima, de baixo), expressão, roupa, fundo, luz e enquadramento (close-up e corpo inteiro).
- **Filtro:**
  - identidade (ArcFace para pessoa real, DINO/VLM para anime);
  - juiz VLM para artefatos;
  - remoção de duplicatas.
- **Armadilhas:**
  - rosto médio;
  - "cara do gerador" (pele lisa);
  - poses repetidas;
  - aprender os artefatos do gerador.
- **Modelos citados:** Qwen-Image-Edit 2511, Flux Kontext, Nano Banana (âncora headshot em fundo branco) e Seedream.
- **Disponibilidade no OpenRouter:** conferir pela API.
