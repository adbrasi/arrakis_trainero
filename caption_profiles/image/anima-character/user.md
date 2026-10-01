Analyze the provided image and the booru tags below. Produce the JSON caption output following the Anima character-LoRA format defined in the system prompt.

**Character trigger:** `{{trigger}}`

Write the caption as a **single dense line in hybrid format**: weave `{{trigger}}` naturally as the subject (or near the start) of the caption, then mix booru tags + natural-language prose (`{{trigger}}, tag, tag, NL clause, tag, NL clause, …`). Do NOT describe the art style itself. Do NOT include quality tags or `@anything`.

Keep canonical concept tokens as booru tags (sexual positions, body proportions, facial expressions, camera angles, kink scenarios, specific outfits, counts, gaze). Convert generic descriptions to prose (hair, eyes, skin, environment, lighting, mood, spatial relations). See the system prompt for the full policy.

**Input image:** [The attached image]
**Booru Tags (ground truth — keep canonical ones as tags, convert generic ones to prose):**
{{booru_tags}}

---

Decide per tag: keep as booru tag (canonical concept) OR convert to prose (generic descriptive). Use the image to refine pose, framing, expression, lighting, and color. Front-load the character's action and context so the LoRA receives clear training signal around the trigger. Output only the JSON required by the system instructions.

**Existing caption for this item (secondary context; it may be empty, outdated or wrong):**
{{existing_caption|(none)}}
