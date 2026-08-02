/* Arrakis Trainero — single page, server state is authoritative (polling). */

"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

const state = {
  presets: null,
  order: [],
  status: null,
  model: null,
  mode: "lora",
  projectSet: false,
  advTouched: new Set(),
  uploading: false,
};

// ---------------------------------------------------------------- helpers
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
let projectTimer = null;
$("#project-name").addEventListener("input", () => {
  clearTimeout(projectTimer);
  projectTimer = setTimeout(saveProject, 600);
});
async function saveProject() {
  const name = $("#project-name").value.trim();
  if (!name) return;
  try {
    await post("/api/project", { name });
    state.projectSet = true;
    refreshTrainButton();
  } catch (e) { toast(e.message, "error"); }
}

// ---------------------------------------------------------------- mode
$("#mode-toggle").addEventListener("click", (ev) => {
  const btn = ev.target.closest("button[data-mode]");
  if (!btn) return;
  state.mode = btn.dataset.mode;
  $$("#mode-toggle button").forEach((b) => b.classList.toggle("active", b === btn));
  renderModeUI();
});

function sliderIsNative() {
  return state.mode === "slider" && state.model &&
    state.presets?.models[state.model]?.supports_slider_native;
}

function renderModeUI() {
  const slider = state.mode === "slider";
  const native = sliderIsNative();
  $("#dataset-title").textContent = slider ? (native ? "Pares de prompt do slider" : "Datasets do slider") : "Dataset";
  $("#dataset-panels").hidden = slider && native;
  $("#slider-prompts").hidden = !(slider && native);
  $(".ds-panel[data-side=neg]").hidden = !slider || native;
  $(".ds-panel[data-side=pos] .ds-label").hidden = !slider || native;
  if (slider && native && !$("#pair-list").children.length) addPair();
  refreshTrainButton();
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

async function uploadFiles(files, side) {
  if (!files.length || !requireProject()) return;
  await saveProject();
  state.uploading = true;
  let done = 0;
  const queue = [...files];
  const workers = Array.from({ length: 4 }, async () => {
    while (queue.length) {
      const f = queue.shift();
      const rel = f.webkitRelativePath || f.name;
      const dir = /(^|\/)control\//i.test(rel) ? "control" : "";
      try {
        await api(`/api/dataset/file?side=${side}&name=${encodeURIComponent(f.name)}&dir=${dir}`,
          { method: "POST", body: f });
      } catch (e) { toast(`${f.name}: ${e.message}`, "error"); }
      done += 1;
      $("#status-text").textContent = `enviando ${done}/${files.length}…`;
    }
  });
  await Promise.all(workers);
  state.uploading = false;
  toast(`${done} arquivos enviados`, "success");
  poll();
}

function renderDatasetStats() {
  const s = state.status || {};
  for (const [side, stats] of [["pos", s.dataset], ["neg", s.dataset_neg]]) {
    const panel = $(`.ds-panel[data-side=${side}]`);
    const box = panel.querySelector(".ds-stats");
    if (!stats || !stats.items) { box.hidden = true; continue; }
    box.hidden = false;
    const parts = [];
    if (stats.images) parts.push(`<span><b>${stats.images}</b> imagens</span>`);
    if (stats.videos) parts.push(`<span><b>${stats.videos}</b> vídeos</span>`);
    parts.push(`<span><b>${stats.captions}</b> captions</span>`);
    if (stats.control_images) parts.push(`<span><b>${stats.control_images}</b> control</span>`);
    if (stats.missing_captions) parts.push(`<span class="warn">⚠ ${stats.missing_captions} sem caption</span>`);
    parts.push(`<span class="clear" data-side="${side}">limpar</span>`);
    box.innerHTML = parts.join("");
    box.querySelector(".clear").addEventListener("click", async () => {
      if (!confirm("Apagar o dataset importado deste projeto?")) return;
      await post(`/api/dataset/clear?side=${side}`);
      poll();
    });
  }
  renderCaptionCard();
}

// ---------------------------------------------------------------- captions
function renderCaptionCard() {
  const s = state.status || {};
  const missing = (s.dataset?.missing_captions || 0) + (s.dataset_neg?.missing_captions || 0);
  const show = missing > 0 && !(sliderIsNative());
  $("#caption-card").hidden = !show;
  if (!show) return;
  $("#caption-msg").textContent = `${missing} itens sem caption — gere com LLM ou suba os .txt`;
  $("#caption-key-hint").hidden = !!s.openrouter;
  $("#btn-captions").disabled = !s.openrouter;
  const isVideo = (s.dataset?.videos || 0) > 0;
  const profiles = state.presets?.caption_profiles[isVideo ? "video" : "image"] || [];
  const sel = $("#caption-profile");
  if (sel.dataset.kind !== (isVideo ? "video" : "image")) {
    sel.dataset.kind = isVideo ? "video" : "image";
    sel.innerHTML = profiles.map((p) => `<option value="${p.key}" data-var="${p.var || ""}">${p.label}</option>`).join("");
  }
}

$("#btn-captions").addEventListener("click", async () => {
  const sel = $("#caption-profile");
  const opt = sel.selectedOptions[0];
  try {
    const sides = [];
    if (state.status?.dataset?.missing_captions) sides.push("pos");
    if (state.status?.dataset_neg?.missing_captions) sides.push("neg");
    await post("/api/captions", {
      profile: sel.value,
      var_name: opt?.dataset.var || null,
      trigger: $("#caption-trigger").value.trim(),
      side: sides[0] || "pos",
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
    btn.innerHTML = `${m.label}<span class="eng">${m.engine}</span>`;
    btn.addEventListener("click", () => selectModel(key));
    $(m.group === "imagem" ? "#models-imagem" : "#models-video").appendChild(btn);
  }
}

function selectModel(key) {
  state.model = key;
  $$(".model-btn").forEach((b) => b.classList.toggle("selected", b.dataset.key === key));
  const m = state.presets.models[key];
  const t = m.train;
  $("#preset-line").hidden = false;
  $("#preset-line").textContent =
    `${m.label} · ${m.engine} · lora dim ${t.network_dim}/${t.network_alpha ?? "auto"} · lr ${t.learning_rate} · ${t.optimizer_type} · epochs automáticos`;
  fillAdvanced();
  renderModeUI();
}

// ---------------------------------------------------------------- advanced
const ADV_FIELDS = ["adv-net", "adv-dim", "adv-alpha", "adv-lr", "adv-epochs", "adv-repeats", "adv-save", "adv-ltx-res"];
ADV_FIELDS.forEach((id) => {
  const el = document.getElementById(id);
  el?.addEventListener("change", () => state.advTouched.add(id));
});

function fillAdvanced() {
  if (!state.model) return;
  const m = state.presets.models[state.model];
  const sched = state.status?.schedules?.[state.model];
  const net = $("#adv-net");
  if (net.dataset.model !== state.model) {
    net.dataset.model = state.model;
    net.innerHTML = m.net_types.map((n) =>
      `<option value="${n}">${{ lora: "LoRA (padrão)", loha: "LoHa", lokr: "LoKr" }[n] || n}</option>`).join("");
  }
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
}

function collectOverrides() {
  const o = {
    net_type: $("#adv-net").value || "lora",
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
  return o;
}

// ---------------------------------------------------------------- train
$("#btn-train").addEventListener("click", async () => {
  if (!requireProject() || !state.model) return;
  await saveProject();
  try {
    await post("/api/train", {
      model: state.model,
      mode: state.mode,
      overrides: collectOverrides(),
      slider_targets: sliderTargets(),
    });
    toast("Treino iniciado ⚔", "success");
    $("#card-progress").hidden = false;
  } catch (e) { toast(e.message, "error"); }
});

$("#btn-cancel").addEventListener("click", async () => {
  if (!confirm("Cancelar o job atual?")) return;
  await post("/api/cancel");
});

function refreshTrainButton() {
  const s = state.status || {};
  const busy = s.job && s.job.status === "running";
  const hasData = sliderIsNative()
    ? sliderTargets().length > 0 || state.mode !== "slider"
    : (s.dataset?.items || 0) > 0;
  const needNeg = state.mode === "slider" && !sliderIsNative();
  const negOk = !needNeg || (s.dataset_neg?.items || 0) > 0;
  $("#btn-train").disabled = !!busy || !state.model || !hasData || !negOk;
  $("#btn-cancel").hidden = !busy;
}

// ---------------------------------------------------------------- polling
async function poll() {
  try {
    state.status = await api("/api/status");
  } catch {
    $("#status-dot").className = "dot failed";
    $("#status-text").textContent = "servidor fora do ar?";
    return;
  }
  const s = state.status;
  if (!state.projectSet && s.project && !$("#project-name").value) {
    $("#project-name").value = s.project;
    state.projectSet = true;
  }
  // chips
  const gpu = s.gpu || {};
  const gc = $("#gpu-chip");
  gc.hidden = !gpu.name;
  if (gpu.name) gc.textContent = `${gpu.name} · ${(gpu.vram_mb / 1024).toFixed(0)} GB`;
  const hc = $("#hf-chip");
  hc.hidden = false;
  hc.textContent = s.hf_token ? "HF ✓" : "HF token ausente";
  hc.className = `chip ${s.hf_token ? "ok" : "bad"}`;
  const oc = $("#or-chip");
  oc.hidden = false;
  oc.textContent = s.openrouter ? "OpenRouter ✓" : "OpenRouter —";
  oc.className = `chip ${s.openrouter ? "ok" : ""}`;

  renderJob(s.job);
  if (!state.uploading) {
    const busy = s.job && s.job.status === "running";
    $("#status-dot").className = `dot ${busy ? "busy" : s.job?.status === "failed" ? "failed" : "idle"}`;
    $("#status-text").textContent = busy ? s.job.title : "pronto";
  }
  renderDatasetStats();
  fillAdvanced();
  refreshTrainButton();
}

function renderJob(job) {
  const card = $("#card-progress");
  if (!job) { card.hidden = true; return; }
  card.hidden = false;
  $("#progress-title").textContent = job.title + (
    { done: " — concluído ✔", failed: " — FALHOU ✘", cancelled: " — cancelado" }[job.status] || ""
  );
  const chips = $("#phase-chips");
  chips.innerHTML = (job.phases || []).map((p) =>
    `<span class="phase-chip ${p.status}">${p.name}</span>`).join("");
  const pr = job.progress || {};
  const wrap = $("#bar-wrap");
  if (pr.total_steps) {
    wrap.hidden = false;
    const pct = Math.min(100, (pr.step / pr.total_steps) * 100);
    $("#bar-fill").style.width = `${pct}%`;
    $("#bar-label").textContent = `${pr.step}/${pr.total_steps} steps`;
  } else wrap.hidden = true;
  const metrics = [];
  if (pr.epoch) metrics.push(`epoch <b>${pr.epoch}${pr.total_epochs ? "/" + pr.total_epochs : ""}</b>`);
  if (pr.loss != null) metrics.push(`loss <b>${pr.loss.toFixed(4)}</b>`);
  const cs = job.extra?.config_summary;
  if (cs) metrics.push(`<span class="hint">${JSON.stringify(cs).slice(1, -1).replaceAll('"', "")}</span>`);
  if (job.error) metrics.push(`<b style="color:var(--error)">${job.error}</b>`);
  $("#metrics").innerHTML = metrics.join(" · ");
  const hfRow = $("#hf-link-row");
  if (job.extra?.hf_repo) {
    hfRow.hidden = false;
    const files = job.extra.hf_files?.length ? ` · ${job.extra.hf_files.length} arquivos enviados` : "";
    hfRow.innerHTML = `🤗 <a href="https://huggingface.co/${job.extra.hf_repo}" target="_blank">${job.extra.hf_repo}</a>${files}`;
  } else hfRow.hidden = true;
  fetchLog();
}

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

// ---------------------------------------------------------------- init
(async function init() {
  try {
    const p = await api("/api/presets");
    state.presets = p;
    state.order = p.order;
    renderModels();
  } catch (e) { toast(`presets: ${e.message}`, "error"); }
  await poll();
  setInterval(poll, 2500);
})();
