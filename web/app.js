/* Arrakis Trainero — single page, server state is authoritative (polling). */

"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
let pollTimer = null;

const state = {
  presets: null,
  order: [],
  status: null,
  model: null,
  mode: "lora",
  projectSet: false,
  advTouched: new Set(),
  uploading: false,
  samples: [],
  triggerHydrated: false,
  savedName: null,   // the project name the server last confirmed
  polling: false,
};

// ---------------------------------------------------------------- helpers
// Everything the server says can carry a file name from a third-party zip or a
// pasted URL. It reaches innerHTML only through here.
function esc(v) {
  return String(v ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function toast(msg, type = "info") {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = msg;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), 5200);
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let data = {};
  try { data = await res.json(); } catch { /* empty */ }
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

const post = (path, body) => api(path, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body || {}),
});

// ---------------------------------------------------------------- project
// Saved when the field is left (change), not per keystroke: a debounce on
// input created a project folder for every half-typed name ("m", "ma", "mak").
// Uploads, imports and TREINAR call saveProject() first anyway.
for (const id of ["#project-name", "#trigger-word"]) {
  $(id).addEventListener("change", () => saveProject());
  $(id).addEventListener("input", () => { renderStepState(); refreshTrainButton(); });
}
async function saveProject(extra = {}) {
  const name = $("#project-name").value.trim();
  if (!name) return;
  const switching = state.savedName !== null && name !== state.savedName;
  const body = { name, ...extra };
  // The trigger goes only with a save of the same project. Before the first
  // poll the field is empty because the stored value was not read yet, and on a
  // switch it still shows the previous project's word — both would overwrite
  // the right one. The server answers with the stored trigger either way.
  if (state.triggerHydrated && !switching) body.trigger = $("#trigger-word").value.trim();
  try {
    const r = await post("/api/project", body);
    state.projectSet = true;
    state.savedName = name;
    if (r.switched || switching) {
      $("#trigger-word").value = r.trigger || "";
      state.triggerHydrated = true;
      renderKind(r.kind);
    }
    refreshTrainButton();
  } catch (e) { toast(e.message, "error"); }
}

// ---------------------------------------------------------------- kind
// The one caption question a beginner can answer. The server turns it (plus
// the model) into a caption profile; nobody has to know what a profile is.
function renderKindButtons() {
  const kinds = state.presets?.lora_kinds || {};
  $("#kind-toggle").innerHTML = Object.entries(kinds).map(([k, label]) =>
    `<button type="button" data-kind="${esc(k)}">${esc(label)}</button>`).join("");
}
function renderKind(kind) {
  $$("#kind-toggle button").forEach((b) => b.classList.toggle("active", b.dataset.kind === kind));
}
$("#kind-toggle").addEventListener("click", async (ev) => {
  const btn = ev.target.closest("button[data-kind]");
  if (!btn || !requireProject()) return;
  renderKind(btn.dataset.kind);
  await saveProject({ kind: btn.dataset.kind });
  poll();
});

// ------------------------------------------------------- dataset de origem
// A dataset costs captions, conversion pairs and curation; none of that
// depends on the model. Training the same images on a second model is a copy
// into a new project, not a second dataset built from scratch.
function setSourceHelp(text, alert = false) {
  const el = $("#source-help");
  el.hidden = !text;
  el.textContent = text;
  el.className = `help${alert ? " alert" : ""}`;
}

$("#source-toggle").addEventListener("click", (ev) => {
  const btn = ev.target.closest("button[data-source]");
  if (!btn) return;
  $$("#source-toggle button").forEach((b) => b.classList.toggle("active", b === btn));
  const copying = btn.dataset.source === "copy";
  $("#source-copy").hidden = !copying;
  setSourceHelp("");
  if (copying) loadProjects();
});

async function loadProjects() {
  let data;
  try { data = await api("/api/projects"); }
  catch (e) { return setSourceHelp(e.message, true); }
  const sel = $("#source-project");
  const usable = (data.projects || [])
    .filter((p) => p.items > 0 && p.slug !== data.current);
  const previous = sel.value;
  sel.textContent = "";
  if (!usable.length) {
    sel.appendChild(new Option("nenhum outro projeto com imagens", ""));
    return;
  }
  for (const p of usable) {
    const bits = [`${p.items} imgs`];
    if (p.trigger) bits.push(`"${p.trigger}"`);
    const pairs = p.convert + p.restore;
    if (pairs) bits.push(`${pairs} pares`);
    sel.appendChild(new Option(`${p.name} · ${bits.join(" · ")}`, p.slug));
  }
  if (usable.some((p) => p.slug === previous)) sel.value = previous;
}

$("#btn-fork").addEventListener("click", async () => {
  const source = $("#source-project").value;
  if (!$("#project-name").value.trim()) return setSourceHelp("dê um nome ao projeto antes", true);
  if (!source) return setSourceHelp("escolha o projeto de origem", true);
  const btn = $("#btn-fork");
  btn.disabled = true;
  setSourceHelp("copiando...");
  try {
    await saveProject();  // the fork fills the project that is selected server-side
    const r = await post("/api/project/fork", { source });
    setSourceHelp(`${r.stats?.items || 0} imagens copiadas de ${source}`);
    // the trigger now comes from the source project, so let the poll read it
    state.triggerHydrated = false;
    await poll();
  } catch (e) { setSourceHelp(e.message, true); }
  finally { btn.disabled = false; }
});

// ---------------------------------------------------------------- mode
$("#mode-toggle").addEventListener("click", (ev) => {
  const btn = ev.target.closest("button[data-mode]");
  if (!btn) return;
  state.mode = btn.dataset.mode;
  $$("#mode-toggle button").forEach((b) => b.classList.toggle("active", b === btn));
  applyMode();
});

function sliderIsNative() {
  return state.mode === "slider" && state.model &&
    state.presets?.models[state.model]?.supports_slider_native;
}

function styleRush() { return state.mode === "style-rush"; }

// What each mode is called and what it needs. Visibility itself belongs to the
// CSS via <html data-mode>: a boolean per element per mode is exactly how the
// modes started leaking into each other.
const MODES = {
  "lora": {
    title: "Dataset",
    triggerHelp: "opcional — a palavra que invoca o estilo nos seus prompts",
  },
  "slider": {
    title: "Datasets do slider",
    nativeTitle: "Pares de prompt",
    triggerHelp: "opcional — a palavra que invoca o efeito nos seus prompts",
  },
  "style-rush": {
    title: "Dataset de estilo",
    triggerHelp: "obrigatória — entra em toda caption e no prompt de conversão",
  },
};

// An override means "this number instead of the one this preset suggests", so
// it only has meaning while that preset is on screen. Changing model or mode
// changes the preset, and a leftover override would silently describe a run
// nobody asked for (a rank typed on Krea reaching Anima, repeats reaching a
// Style Rush that has none).
function resetOverrides() {
  state.advTouched.clear();
}

function applyMode() {
  const root = document.documentElement;
  const native = sliderIsNative();
  if (root.dataset.mode !== state.mode) resetOverrides();
  root.dataset.mode = state.mode;
  root.dataset.native = native ? "1" : "0";
  const m = MODES[state.mode];
  $("#dataset-title").textContent = (native && m.nativeTitle) || m.title;
  $("#trigger-help").textContent = triggerHelpText();
  if (native && !$("#pair-list").children.length) addPair();
  if (state.status) renderDatasetStats();
  fillAdvanced();
  renderModelAvailability();
  renderControlZone();
  renderStepState();
  refreshTrainButton();
}

function triggerHelpText() {
  if (!styleRush() && captionNeedsTrigger()) {
    return "obrigatória — as captions que faltam vão começar com ela";
  }
  return MODES[state.mode].triggerHelp;
}

function renderControlZone() {
  const m = state.model && state.presets?.models[state.model];
  $("#control-zone").hidden = !(m && m.supports_control && state.mode === "lora");
}

function renderModelAvailability() {
  const allowed = styleRush() ? (state.status?.style_rush_models || []) : null;
  $$(".model-btn").forEach((b) => {
    const ok = !allowed || allowed.includes(b.dataset.key);
    b.disabled = !ok;
    b.classList.toggle("unavailable", !ok);
  });
  if (allowed && state.model && !allowed.includes(state.model)) {
    state.model = null;
    $$(".model-btn").forEach((b) => b.classList.remove("selected"));
    $("#preset-line").hidden = true;
  }
}

// slider prompt pairs (LTX)
$("#add-pair").addEventListener("click", () => addPair());
function addPair(pos = "", neg = "") {
  const div = document.createElement("div");
  div.className = "pair";
  div.innerHTML = `
    <input type="text" class="p-pos" placeholder="positivo (ex.: extremely detailed, sharp)" value="${pos}">
    <input type="text" class="p-neg" placeholder="negativo (ex.: blurry, simple)" value="${neg}">
    <button class="rm" title="remover">×</button>`;
  div.querySelector(".rm").addEventListener("click", () => div.remove());
  $("#pair-list").appendChild(div);
}
function sliderTargets() {
  return $$("#pair-list .pair").map((p) => ({
    positive: p.querySelector(".p-pos").value,
    negative: p.querySelector(".p-neg").value,
  })).filter((t) => t.positive.trim() && t.negative.trim());
}

// ---------------------------------------------------------------- dataset
function setupPanel(panel) {
  const side = panel.dataset.side;
  const dz = panel.querySelector(".dropzone");
  const input = panel.querySelector("input[type=file]");
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("drag");
    uploadFiles([...e.dataTransfer.files], side);
  });
  input.addEventListener("change", () => uploadFiles([...input.files], side));
  panel.querySelector(".import-btn").addEventListener("click", async () => {
    const src = panel.querySelector(".link-input").value.trim();
    if (!src) return toast("cole um link ou caminho primeiro", "error");
    if (!requireProject()) return;
    // the project name is saved on a 600ms debounce: importing right after
    // typing it would land the dataset in whatever project was current before
    await saveProject();
    try {
      await post(`/api/dataset/import?side=${side}`, { source: src });
      toast("Import iniciado — acompanhe no progresso.");
    } catch (e) { toast(e.message, "error"); }
  });
}
$$(".ds-panel").forEach(setupPanel);

function requireProject() {
  if (!$("#project-name").value.trim()) {
    toast("dê um nome ao projeto primeiro", "error");
    $("#project-name").focus();
    return false;
  }
  return true;
}

const stemOf = (name) => name.replace(/\.[^.]+$/, "");

// Edit training: control images go to dataset/control/ under their own names,
// which is how they pair with the result of the same name.
(() => {
  const dz = $("#control-drop");
  const input = dz.querySelector("input");
  dz.addEventListener("click", () => input.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("drag");
    uploadFiles([...e.dataTransfer.files], "pos", "control");
  });
  input.addEventListener("change", () => uploadFiles([...input.files], "pos", "control"));
})();

async function uploadFiles(files, side, forceDir = "") {
  if (!files.length || !requireProject()) return;
  if (state.status?.job?.status === "running") {
    return toast("há um job rodando — espere terminar ou cancele antes de enviar arquivos", "error");
  }
  await saveProject();
  state.uploading = true;
  refreshTrainButton();
  let done = 0, ok = 0, dup = 0;
  // Media first, captions after: the server never overwrites an image (a
  // second "image.jpg" becomes image_2.jpg), so a caption has to follow the
  // name its image actually got.
  const isTxt = (f) => /\.txt$/i.test(f.name);
  const renamed = {};
  const send = async (list) => {
    const queue = [...list];
    await Promise.all(Array.from({ length: 4 }, async () => {
      while (queue.length) {
        const f = queue.shift();
        const rel = f.webkitRelativePath || f.name;
        const dir = forceDir || (/(^|\/)control\//i.test(rel) ? "control" : "");
        let name = f.name;
        if (isTxt(f) && renamed[`${dir}/${stemOf(name)}`]) name = `${renamed[`${dir}/${stemOf(name)}`]}.txt`;
        try {
          const r = await api(`/api/dataset/file?side=${side}&name=${encodeURIComponent(name)}&dir=${dir}`,
            { method: "POST", body: f });
          if (r.duplicate) dup += 1;
          if (r.name && r.name !== name) renamed[`${dir}/${stemOf(name)}`] = stemOf(r.name);
          ok += 1;
        } catch (e) { toast(`${f.name}: ${e.message}`, "error"); }
        done += 1;
        $("#status-text").textContent = `enviando ${done}/${files.length}…`;
      }
    }));
  };
  await send(files.filter((f) => !isTxt(f)));
  await send(files.filter(isTxt));
  state.uploading = false;
  // "N enviados" after half of them failed is the report that lets someone
  // train on an incomplete dataset without ever knowing
  const failed = done - ok;
  const extra = [];
  const nRenamed = Object.keys(renamed).length;
  if (nRenamed) extra.push(`${nRenamed} renomeado${nRenamed === 1 ? "" : "s"} (nome já existia)`);
  if (dup) extra.push(`${dup} já estava${dup === 1 ? "" : "m"} no dataset`);
  const tail = extra.length ? ` · ${extra.join(" · ")}` : "";
  if (failed) toast(`${ok} de ${done} enviados — ${failed} falharam${tail}`, "error");
  else toast(`${ok} arquivo${ok === 1 ? "" : "s"} enviado${ok === 1 ? "" : "s"}${tail}`, "success");
  poll();
}

// ---------------------------------------------------------------- state lines
// The margin of every block says, in one short line, what is still missing.
// Reading the margin top to bottom has to be enough to know what to do next.
function renderStepState() {
  const s = state.status || {};
  const items = s.dataset?.items || 0;
  const named = !!$("#project-name").value.trim();
  const trigger = !!$("#trigger-word").value.trim();

  const project = $("#state-project");
  const triggerNeeded = captionNeedsTrigger();
  document.querySelector(".trigger-field").classList.toggle("needed", triggerNeeded);
  if (!named) setState(project, "falta o nome", false);
  else if (triggerNeeded && !trigger) setState(project, "falta a trigger word", false);
  else setState(project, named && trigger ? "nome e trigger" : "nomeado", true);

  const ds = $("#state-dataset");
  const negNeeded = state.mode === "slider" && !sliderIsNative();
  const negItems = s.dataset_neg?.items || 0;
  if (sliderIsNative()) setState(ds, `${sliderTargets().length} pares`, sliderTargets().length > 0);
  else if (!items) setState(ds, "vazio", false);
  else if (negNeeded && !negItems) setState(ds, "falta o lado (−)", false);
  else setState(ds, "pronto", true);

  $("#dataset-count").hidden = !items || sliderIsNative();
  $("#dataset-n").textContent = items;

  const model = $("#state-model");
  if (!state.model) setState(model, "nenhum escolhido", false);
  else setState(model, state.presets.models[state.model].label, true);
}

function setState(el, text, done) {
  el.textContent = text;
  el.classList.toggle("done", done);
}

// ---------------------------------------------------------------- contact sheet
// Seeing the images is the whole point: a count alone never told anyone whether
// they uploaded the right folder.
const sheetKey = {};
async function renderContactSheet(side, items) {
  const box = $(`#thumbs-${side}`);
  const panel = $(`.ds-panel[data-side=${side}]`);
  panel.classList.toggle("filled", items > 0);
  if (!items) { box.hidden = true; box.innerHTML = ""; sheetKey[side] = ""; return; }
  try {
    const { names, total } = await api(`/api/dataset/thumbs?side=${side}`);
    const key = `${total}:${names.join("|")}`;
    if (key === sheetKey[side]) return;
    sheetKey[side] = key;
    const cells = names.map((n) => {
      const src = `/api/dataset/thumb?side=${side}&name=${encodeURIComponent(n)}`;
      return `<img src="${src}" alt="" loading="lazy">`;
    });
    if (total > names.length) cells.push(`<div class="more">+${total - names.length}</div>`);
    box.innerHTML = cells.join("");
    box.hidden = false;
  } catch { /* transient */ }
}

function renderDatasetStats() {
  const s = state.status || {};
  for (const [side, stats] of [["pos", s.dataset], ["neg", s.dataset_neg]]) {
    const panel = $(`.ds-panel[data-side=${side}]`);
    const box = panel.querySelector(".ds-stats");
    renderContactSheet(side, stats?.images || 0);
    if (!stats || !stats.items) { box.hidden = true; continue; }
    box.hidden = false;
    const parts = [];
    if (stats.images) parts.push(`<span><b>${stats.images}</b> imagens</span>`);
    if (stats.videos) parts.push(`<span><b>${stats.videos}</b> vídeos</span>`);
    if (stats.control_images) parts.push(`<span><b>${stats.control_images}</b> controles</span>`);
    const ep = side === "pos" ? s.edit_pairs : null;
    if (ep && ep.paired !== undefined) {
      const refs = (ep.refs || []).join(" ou ");
      parts.push(`<span>edição: <b>${ep.paired}</b> itens com ${esc(refs)} controle(s)</span>`);
      if (ep.unpaired) parts.push(`<span class="warn">${ep.unpaired} sem controle (ex.: ${esc((ep.unpaired_sample || []).join(", "))})</span>`);
    }
    // Style Rush writes the captions during training, so counting them here is
    // noise at best and a false alarm at worst
    if (styleRush()) parts.push(`<span>captions escritas no treino</span>`);
    else {
      parts.push(`<span><b>${stats.captions}</b> captions</span>`);
      if (stats.missing_captions) parts.push(`<span class="warn">${stats.missing_captions} sem caption</span>`);
    }
    parts.push(`<button class="btn tiny danger clear">Limpar dataset</button>`);
    box.innerHTML = parts.join("");
    box.querySelector(".clear").addEventListener("click", async () => {
      const n = stats.items;
      if (!confirm(`Apagar ${n} ${n === 1 ? "item" : "itens"} deste dataset? `
        + "Os arquivos saem do pod. Para começar outro treino sem perder este, "
        + "troque o nome do projeto no passo 01.")) return;
      try {
        await post(`/api/dataset/clear?side=${side}`);
      } catch (e) { return toast(e.message, "error"); }
      sheetKey[side] = "";
      poll();
    });
  }
  renderCaptionCard();
}

// ---------------------------------------------------------------- captions
function profileInfo(key) {
  for (const list of Object.values(state.presets?.caption_profiles || {})) {
    const p = list.find((x) => x.key === key);
    if (p) return p;
  }
  return null;
}

// The profile the job will use for missing captions: the one picked here, or
// the server's plan for (model, kind). "auto" until a model is chosen.
function plannedProfile() {
  const s = state.status || {};
  if (s.caption_profile) return s.caption_profile;
  const plan = s.caption_plan || {};
  return state.model ? plan[state.model] : (Object.values(plan)[0] || null);
}

function captionNeedsTrigger() {
  const s = state.status || {};
  if (editDataset()) return false;
  const missing = (s.dataset?.missing_captions || 0)
    + (state.mode === "slider" ? (s.dataset_neg?.missing_captions || 0) : 0);
  if (styleRush()) return true;
  if (!missing || sliderIsNative()) return false;
  return !!profileInfo(plannedProfile())?.trigger;
}

function editDataset() {
  return (state.status?.edit_pairs?.paired || 0) > 0;
}

function renderCaptionCard() {
  const s = state.status || {};
  const negToo = state.mode === "slider";
  const missing = (s.dataset?.missing_captions || 0) + (negToo ? (s.dataset_neg?.missing_captions || 0) : 0);
  const items = (s.dataset?.items || 0) + (negToo ? (s.dataset_neg?.items || 0) : 0);
  const redo = $("#caption-redo").checked;
  // Style Rush writes its own captions during training, with the trigger word
  // from step 01 — offering this card there would offer a worse path.
  const show = !sliderIsNative() && !styleRush() && items > 0;
  $("#caption-card").hidden = !show;
  if (!show) return;
  $("#caption-redo-wrap").hidden = state.mode !== "lora";
  $("#caption-msg").textContent = missing > 0
    ? `${missing} ${missing === 1 ? "item sem caption" : "itens sem caption"}`
    : `${items} itens, todos com caption`;
  const isVideo = (s.dataset?.videos || 0) > 0 && !(s.dataset?.images || 0);
  const kind = isVideo ? "video" : "image";
  const sel = $("#caption-profile");
  const chosen = s.caption_profile || "";
  if (sel.dataset.kind !== kind) {
    sel.dataset.kind = kind;
    const profiles = state.presets?.caption_profiles[kind] || [];
    sel.innerHTML = `<option value="">Automático (recomendado)</option>`
      + profiles.map((p) => `<option value="${esc(p.key)}">${esc(p.label)}</option>`).join("");
  }
  if (document.activeElement !== sel) sel.value = chosen;
  const planned = profileInfo(plannedProfile());
  const label = planned ? planned.label : "o perfil automático";
  const plan = $("#caption-plan");
  if (editDataset()) {
    // an edit LoRA learns the instruction; describing the image would teach
    // the wrong thing, so the LLM path is closed here
    plan.textContent = missing > 0
      ? "dataset de edição: cada resultado precisa de um .txt com a INSTRUÇÃO (ex.: \"junte as duas pessoas na mesma foto\") — envie os .txt"
      : "dataset de edição: o .txt de cada item é a instrução";
    plan.classList.toggle("alert", missing > 0);
    $("#btn-captions").disabled = true;
    $("#caption-key-hint").hidden = true;
    renderCaptionTriggerNote();
    return;
  }
  if (!s.openrouter) {
    plan.textContent = missing > 0
      ? "sem OPENROUTER_API_KEY no pod: envie um .txt com o mesmo nome de cada imagem"
      : "";
    plan.classList.toggle("alert", missing > 0);
  } else {
    plan.classList.remove("alert");
    plan.textContent = missing > 0
      ? `o treino escreve as que faltam sozinho, com o perfil "${label}" — clique abaixo só se quiser revisar antes`
      : `perfil: "${label}"`;
  }
  $("#caption-key-hint").hidden = !!s.openrouter;
  $("#btn-captions").disabled = !s.openrouter || (missing === 0 && !redo);
  $("#btn-captions").textContent = redo ? "Refazer captions" : "Escrever agora";
  renderCaptionTriggerNote();
}

// The trigger word is one thing and it lives in step 01. A second field here
// meant the captions could be written with a word the training never uses.
function renderCaptionTriggerNote() {
  const needsVar = !!profileInfo($("#caption-profile").value || plannedProfile())?.trigger;
  const trig = $("#trigger-word").value.trim();
  const note = $("#caption-trigger-note");
  if (!needsVar) note.textContent = "este perfil não usa trigger word";
  else if (trig) note.textContent = `usa a trigger word do passo 01: ${trig}`;
  else note.textContent = "este perfil usa a trigger word — preencha-a no passo 01";
  note.classList.toggle("alert", needsVar && !trig);
}
$("#caption-profile").addEventListener("change", async () => {
  await saveProject({ caption_profile: $("#caption-profile").value });
  poll();
});
$("#caption-redo").addEventListener("change", renderCaptionCard);
$("#trigger-word").addEventListener("input", renderCaptionTriggerNote);

$("#btn-captions").addEventListener("click", async () => {
  const profile = $("#caption-profile").value || plannedProfile();
  const trigger = $("#trigger-word").value.trim();
  if (profileInfo(profile)?.trigger && !trigger) {
    toast("este perfil precisa da trigger word — preencha no passo 01", "error");
    $("#trigger-word").focus();
    return;
  }
  const redo = $("#caption-redo").checked;
  if (redo && !confirm("Apagar TODAS as captions deste dataset e escrever de novo? As que você mandou também somem.")) return;
  try {
    await saveProject();
    await post("/api/captions", {
      profile: $("#caption-profile").value || null,
      model: state.model,
      trigger,
      sides: state.mode === "slider" ? ["pos", "neg"] : ["pos"],
      redo,
    });
    toast("Geração de captions iniciada.");
  } catch (e) { toast(e.message, "error"); }
});

// ---------------------------------------------------------------- models
function renderModels() {
  const { models, order } = state.presets;
  for (const key of order) {
    const m = models[key];
    const btn = document.createElement("button");
    btn.className = "model-btn";
    btn.dataset.key = key;
    btn.innerHTML = `${esc(m.label)}<span class="eng">${esc(m.engine)}</span>`;
    btn.addEventListener("click", () => selectModel(key));
    $(m.group === "imagem" ? "#models-imagem" : "#models-video").appendChild(btn);
  }
}

function selectModel(key) {
  if (state.model !== key) resetOverrides();
  state.model = key;
  $$(".model-btn").forEach((b) => b.classList.toggle("selected", b.dataset.key === key));
  const m = state.presets.models[key];
  const t = m.train;
  $("#preset-line").hidden = false;
  $("#preset-line").textContent =
    `${m.label} · ${m.engine} · dim ${t.network_dim}/${t.network_alpha ?? "auto"} · lr ${t.learning_rate}`
    + (styleRush() ? " · 5 epochs" : " · epochs automáticos");
  fillAdvanced();
  applyMode();
}

// ---------------------------------------------------------------- advanced
const ADV_FIELDS = ["adv-dim", "adv-alpha", "adv-lr", "adv-epochs", "adv-repeats", "adv-save", "adv-ltx-res", "adv-convert-target", "adv-blocks-swap"];
// "input" too: "change" only fires on blur, and the poll refilled the field
// from the preset every 2.5 s while the owner was still typing in it
ADV_FIELDS.forEach((id) => {
  const el = document.getElementById(id);
  for (const ev of ["input", "change"]) el?.addEventListener(ev, () => state.advTouched.add(id));
});

function fillAdvanced() {
  // an open panel of empty boxes reads as broken; say what it is waiting for
  const waiting = $("#adv-waiting");
  waiting.hidden = !!state.model;
  $("#advanced .adv-grid").hidden = !state.model;
  $("#adv-foot").hidden = !state.model;
  if (!state.model) return;
  const m = state.presets.models[state.model];
  // Style Rush trains on a fixed schedule; the suggested one belongs to the
  // other modes and showing it here would describe a run that never happens
  const sched = styleRush() ? state.presets?.style_rush_schedule
                            : state.status?.schedules?.[state.model];
  if (!state.advTouched.has("adv-dim")) $("#adv-dim").value = m.train.network_dim ?? "";
  if (!state.advTouched.has("adv-alpha")) $("#adv-alpha").value = m.train.network_alpha ?? "";
  if (!state.advTouched.has("adv-lr")) $("#adv-lr").value = m.train.learning_rate ?? "";
  if (sched) {
    if (!state.advTouched.has("adv-epochs")) $("#adv-epochs").value = sched.epochs;
    if (!state.advTouched.has("adv-repeats")) $("#adv-repeats").value = sched.num_repeats;
    if (!state.advTouched.has("adv-save")) $("#adv-save").value = sched.save_every_n_epochs;
  }
  const isLtx = !!m.ltx;
  $("#adv-ltx-res-wrap").hidden = !isLtx;
  if (isLtx && !state.advTouched.has("adv-ltx-res")) $("#adv-ltx-res").value = m.ltx.resolution;
  // Only Style Rush pays gpt-image-2, and only Style Rush writes captions from
  // inside the training job — in every other mode both controls would describe
  // a run that never happens.
  const rush = styleRush();
  $("#adv-convert-target-wrap").hidden = !rush;
  $("#adv-redo-captions-wrap").hidden = !rush;
  if (rush && !state.advTouched.has("adv-convert-target")) {
    $("#adv-convert-target").value = state.presets?.default_convert_target ?? 100;
  }
}

function collectOverrides() {
  const o = {
    loraplus: $("#adv-loraplus").checked,
    hf_upload: $("#adv-hf").checked,
    hf_private: $("#adv-hf-private").checked,
  };
  const touched = state.advTouched;
  if (touched.has("adv-dim")) o.network_dim = parseInt($("#adv-dim").value, 10);
  if (touched.has("adv-alpha")) o.network_alpha = parseInt($("#adv-alpha").value, 10);
  if (touched.has("adv-lr")) o.learning_rate = parseFloat($("#adv-lr").value);
  if (touched.has("adv-epochs")) o.epochs = parseInt($("#adv-epochs").value, 10);
  if (touched.has("adv-repeats")) o.num_repeats = parseInt($("#adv-repeats").value, 10);
  if (touched.has("adv-save")) o.save_every_n_epochs = parseInt($("#adv-save").value, 10);
  if (touched.has("adv-ltx-res")) o.ltx_resolution = $("#adv-ltx-res").value.trim();
  if (touched.has("adv-convert-target")) o.convert_target = parseInt($("#adv-convert-target").value, 10);
  // 0 is a real value here (swap nothing), so an untouched field must stay
  // absent rather than send a zero the preset never chose
  if (touched.has("adv-blocks-swap") && $("#adv-blocks-swap").value !== "") {
    o.blocks_to_swap = parseInt($("#adv-blocks-swap").value, 10);
  }
  o.redo_captions = $("#adv-redo-captions").checked;
  o.sampling = $("#adv-sampling").checked;
  const sp = $("#adv-sample-prompt").value.trim();
  if (sp) o.sample_prompt = sp;
  return o;
}

// ---------------------------------------------------------------- train
$("#btn-train").addEventListener("click", async () => {
  if (!requireProject() || !state.model) return;
  if (captionNeedsTrigger() && !$("#trigger-word").value.trim()) {
    toast("preencha a trigger word no passo 01", "error");
    $("#trigger-word").focus();
    return;
  }
  await saveProject();
  try {
    await post("/api/train", {
      model: state.model,
      mode: state.mode,
      trigger: $("#trigger-word").value.trim(),
      overrides: collectOverrides(),
      slider_targets: sliderTargets(),
    });
    toast("Treino iniciado ⚔", "success");
    $("#card-progress").hidden = false;
  } catch (e) { toast(e.message, "error"); }
});

$("#btn-shutdown").addEventListener("click", async () => {
  const busy = state.status?.job?.status === "running";
  const msg = busy
    ? "Há um treino rodando. Desligar cancela o treino e encerra o servidor. Continuar?"
    : "Encerrar o servidor e liberar a porta 8090?";
  if (!confirm(msg)) return;
  try {
    await post("/api/shutdown", { force: busy });
  } catch (e) {
    // the socket usually dies before the response lands — that IS the success case
    if (!/Failed to fetch|NetworkError|load failed/i.test(e.message)) {
      return toast(e.message, "error");
    }
  }
  clearInterval(pollTimer);
  $("#curtain").hidden = false;
});

$("#btn-cancel").addEventListener("click", async () => {
  if (!confirm("Cancelar o job atual?")) return;
  try { await post("/api/cancel"); } catch (e) { toast(e.message, "error"); }
});

function refreshTrainButton() {
  const s = state.status || {};
  const busy = s.job && s.job.status === "running";
  // one ordered list of what is missing: the first entry becomes the hint next
  // to the button, so there is never a disabled control with no explanation
  const blockers = [];
  if (!$("#project-name").value.trim()) blockers.push("dê um nome ao projeto");
  if (captionNeedsTrigger() && !$("#trigger-word").value.trim()) {
    blockers.push(styleRush() ? "preencha a trigger word"
      : "preencha a trigger word (as captions que faltam vão usá-la)");
  }
  if (sliderIsNative()) {
    if (!sliderTargets().length) blockers.push("adicione um par de prompts");
  } else if (!(s.dataset?.items || 0)) {
    blockers.push("importe as imagens");
  }
  if (state.mode === "slider" && !sliderIsNative() && !(s.dataset_neg?.items || 0)) {
    blockers.push("importe o dataset do lado (−)");
  }
  if (!state.model) blockers.push("escolha um modelo");
  if (styleRush() && !s.openrouter) blockers.push("defina OPENROUTER_API_KEY no pod");
  const missing = (s.dataset?.missing_captions || 0)
    + (state.mode === "slider" ? (s.dataset_neg?.missing_captions || 0) : 0);
  if (editDataset() && missing) {
    blockers.push(`${missing} itens sem .txt de instrução (dataset de edição)`);
  } else if (editDataset() && s.edit_pairs.unpaired) {
    blockers.push(`${s.edit_pairs.unpaired} imagens sem controle — num treino de edição todas precisam`);
  } else if (!styleRush() && !sliderIsNative() && missing && !s.openrouter) {
    blockers.push(`${missing} itens sem caption — envie os .txt ou defina OPENROUTER_API_KEY`);
  }
  // the server refuses a train mid-upload; say so before the click, not after
  if (state.uploading) blockers.unshift("espere o upload terminar");

  $("#btn-train").disabled = !!busy || blockers.length > 0;
  $("#btn-cancel").hidden = !busy;
  const lowDisk = s.disk_free_gb >= 0 && s.disk_free_gb < 60;
  $("#next-action").textContent = busy ? `${s.job.title} — acompanhe abaixo`
    : blockers.length ? `falta: ${blockers[0]}`
    : lowDisk ? `pronto · atenção: só ${s.disk_free_gb} GB livres (modelos base ocupam 20–60 GB)`
    : "tudo pronto";
}

// ---------------------------------------------------------------- polling
async function poll() {
  // a slow answer must not stack polls behind it
  if (state.polling) return;
  state.polling = true;
  try {
    state.status = await api("/api/status");
  } catch {
    $("#status-dot").className = "dot failed";
    $("#status-text").textContent = "servidor fora do ar?";
    return;
  } finally {
    state.polling = false;
  }
  const s = state.status;
  if (!state.projectSet && s.project && !$("#project-name").value) {
    $("#project-name").value = s.project;
    state.projectSet = true;
  }
  if (state.savedName === null && s.project) state.savedName = s.project;
  renderKind(s.kind || "");
  // read the stored trigger exactly once: re-filling it every poll would undo
  // the owner clearing the field (the save is debounced 600ms behind it)
  if (!state.triggerHydrated) {
    $("#trigger-word").value = s.trigger || "";
    state.triggerHydrated = true;
  }
  // a copied dataset arrives with the trigger already inside every .txt on
  // disk: accepting another word here trains one thing and samples another
  const locked = !!s.trigger_locked;
  $("#trigger-word").readOnly = locked;
  $("#trigger-help").textContent = locked
    ? `herdada de ${s.origin} — as captions já têm essa palavra escrita dentro`
    : triggerHelpText();
  // chips
  const gpu = s.gpu || {};
  const gc = $("#gpu-chip");
  gc.hidden = !gpu.name;
  if (gpu.name) gc.textContent = `${gpu.name} · ${(gpu.vram_mb / 1024).toFixed(0)} GB`;
  const hc = $("#hf-chip");
  hc.hidden = false;
  hc.textContent = s.hf_token ? "HF ✓" : "HF token ausente";
  hc.className = `readout ${s.hf_token ? "ok" : "bad"}`;
  const oc = $("#or-chip");
  oc.hidden = false;
  oc.textContent = s.openrouter ? "OpenRouter ✓" : "OpenRouter —";
  oc.className = `readout ${s.openrouter ? "ok" : "bad"}`;
  const dc = $("#disk-chip");
  dc.hidden = !(s.disk_free_gb >= 0);
  dc.textContent = `${s.disk_free_gb} GB livres`;
  dc.className = `readout ${s.disk_free_gb >= 60 ? "ok" : "bad"}`;

  renderJob(s.job);
  if (!state.uploading) {
    const busy = s.job && s.job.status === "running";
    $("#status-dot").className = `dot ${busy ? "busy" : s.job?.status === "failed" ? "failed" : "idle"}`;
    $("#status-text").textContent = busy ? s.job.title : "pronto";
  }
  renderDatasetStats();
  // both depend on the status payload, so they also recover from the first poll
  // landing after the owner already picked a mode
  renderModelAvailability();
  renderStepState();
  fillAdvanced();
  refreshTrainButton();
}

function renderJob(job) {
  const card = $("#card-progress");
  if (!job) { card.hidden = true; return; }
  card.hidden = false;
  const other = job.project && state.savedName && job.project !== state.savedName
    ? ` (projeto ${job.project})` : "";
  $("#progress-title").textContent = job.title + other + (
    { done: " — concluído ✔", failed: " — FALHOU ✘", cancelled: " — cancelado" }[job.status] || ""
  );
  const chips = $("#phase-chips");
  chips.innerHTML = (job.phases || []).map((p) =>
    `<span class="phase-chip ${esc(p.status)}">${esc(p.name)}</span>`).join("");
  const pr = job.progress || {};
  const wrap = $("#bar-wrap");
  if (pr.total_steps) {
    wrap.hidden = false;
    const pct = Math.min(100, (pr.step / pr.total_steps) * 100);
    $("#bar-fill").style.width = `${pct}%`;
    $("#bar-label").textContent = `${pr.step}/${pr.total_steps} ${pr.unit || "steps"}`;
  } else wrap.hidden = true;
  const metrics = [];
  if (pr.epoch) metrics.push(`epoch <b>${pr.epoch}${pr.total_epochs ? "/" + pr.total_epochs : ""}</b>`);
  if (pr.loss != null) metrics.push(`loss <b>${pr.loss.toFixed(4)}</b>`);
  const cs = job.extra?.config_summary;
  if (cs) metrics.push(`<span class="hint">${esc(JSON.stringify(cs).slice(1, -1).replaceAll('"', ""))}</span>`);
  if (job.extra?.caption_cost) metrics.push(`captions <b>$${Number(job.extra.caption_cost).toFixed(4)}</b>`);
  if (job.error) metrics.push(`<b style="color:var(--signal)">${esc(job.error)}</b>`);
  $("#metrics").innerHTML = metrics.join(" · ");
  const hfRow = $("#hf-link-row");
  if (job.extra?.hf_repo) {
    hfRow.hidden = false;
    const files = job.extra.hf_files?.length ? ` · ${job.extra.hf_files.length} arquivos enviados` : "";
    const repo = esc(job.extra.hf_repo);
    hfRow.innerHTML = `🤗 <a href="https://huggingface.co/${repo}" target="_blank" rel="noopener">${repo}</a>${files}`;
  } else hfRow.hidden = true;
  // a finished job's log and gallery do not change: one last read, then stop
  const key = `${job.started_at}:${job.status}`;
  if (job.status === "running" || key !== lastJobKey) {
    lastJobKey = key;
    fetchSamples();
    fetchLog();
  }
}
let lastJobKey = "";

let lastLog = "";
async function fetchLog() {
  try {
    const { log } = await api("/api/logs");
    if (log !== lastLog) {
      lastLog = log;
      const el = $("#console");
      const stick = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
      el.textContent = log;
      if (stick) el.scrollTop = el.scrollHeight;
    }
  } catch { /* transient */ }
}

let lastSampleKey = "";
const SAMPLE_GRID = 12;
async function fetchSamples() {
  try {
    const { samples } = await api("/api/samples");
    state.samples = samples;
    const key = samples.map((s) => s.name).join("|");
    if (key === lastSampleKey) return;
    lastSampleKey = key;
    const box = $("#samples");
    const head = $("#samples-head");
    box.hidden = samples.length === 0;
    head.hidden = samples.length === 0;
    // a long run makes one sample per epoch: render the newest SAMPLE_GRID and
    // say so, instead of silently dropping the rest
    const shown = samples.slice(0, SAMPLE_GRID);
    head.textContent = samples.length > shown.length
      ? `${samples.length} amostras · mostrando as ${shown.length} mais recentes`
      : `${samples.length} amostra${samples.length === 1 ? "" : "s"}`;
    box.innerHTML = shown.map((s) => {
      const label = esc(s.epoch >= 0 ? `epoch ${s.epoch}` : s.name);
      const q = encodeURIComponent(s.name);
      // the grid pulls a thumbnail; the link opens the full-resolution PNG
      return `<figure>
        <a href="/api/sample?name=${q}" target="_blank" rel="noopener">
          <img src="/api/sample?name=${q}&thumb=1" alt="${label}" loading="lazy">
        </a>
        <figcaption>${label}</figcaption>
      </figure>`;
    }).join("");
  } catch { /* transient */ }
}

// ---------------------------------------------------------------- init
(async function init() {
  try {
    const p = await api("/api/presets");
    state.presets = p;
    state.order = p.order;
    renderModels();
    renderKindButtons();
  } catch (e) { toast(`presets: ${e.message}`, "error"); }
  await poll();
  pollTimer = setInterval(poll, 2500);
})();
