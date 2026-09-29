const state = {
  tab: "installed",
  models: [],
  system: null,
  signature: "",
  histories: {},
  sending: false,
  controller: null,
};

const DEFAULT_SYSTEM = "Eres un asistente útil. Responde siempre en español, de forma clara, correcta y breve, salvo que el usuario pida otro idioma.";
// Últimos mensajes que se mandan al modelo: el contexto es corto (CTX_SIZE).
const HISTORY_LIMIT = 12;

const $ = (id) => document.getElementById(id);

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[ch]));
}

// Sesión firmada que devuelve /api/auth/login. Nunca se guarda el secreto maestro.
function session() {
  const token = localStorage.getItem("obrador-session") || "";
  const exp = Number(localStorage.getItem("obrador-session-exp") || 0);
  if (!token || exp * 1000 <= Date.now()) return null;
  return { token, exp };
}

function saveSession(created) {
  localStorage.setItem("obrador-session", created.token);
  localStorage.setItem("obrador-session-exp", String(created.expires_at));
}

function clearSession() {
  localStorage.removeItem("obrador-session");
  localStorage.removeItem("obrador-session-exp");
}

function headers(json) {
  const result = {};
  if (json) result["Content-Type"] = "application/json";
  const current = session();
  const admin = current?.token || localStorage.getItem("obrador-admin") || "";
  const key = current?.token || localStorage.getItem("obrador-api-key") || "";
  if (admin) result["X-Admin-Token"] = admin;
  if (key) result.Authorization = `Bearer ${key}`;
  return result;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { ...headers(Boolean(options.body)), ...(options.headers || {}) },
  });
  const text = await response.text();
  let payload = null;
  if (text) {
    try { payload = JSON.parse(text); } catch { payload = { detail: text }; }
  }
  if (!response.ok) {
    const message = payload?.detail || payload?.error?.message || response.statusText;
    if (response.status === 401 && !path.startsWith("/api/auth/")) needLogin();
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return payload;
}

function banner(message) {
  const node = $("banner");
  if (!message) {
    node.hidden = true;
    node.textContent = "";
    return;
  }
  node.hidden = false;
  node.textContent = message;
}

function meter(size) {
  const max = state.system?.max_model_bytes || 9000000000;
  if (!size) return `<div class="meter"><span style="width:0"></span></div>`;
  const over = size > max;
  const pct = Math.max(2, Math.min(100, (size / max) * 100));
  return `<div class="meter${over ? " over" : ""}" title="${over ? "Supera el límite" : "Dentro del límite"}"><span style="width:${pct}%"></span></div>`;
}

function statusLabel(model) {
  const runtime = model.runtime?.status;
  if (runtime === "running") return ["En marcha", "good"];
  if (runtime === "starting") return ["Cargando en memoria", "warn"];
  if (model.status === "downloading") return [`Descargando ${Math.round((model.progress || 0) * 100)}%`, "warn"];
  if (model.status === "queued") return ["En cola", "warn"];
  if (model.status === "error" || runtime === "error") return ["Error", "bad"];
  if (model.status === "ready") return ["Descargado", "good"];
  return [model.status, ""];
}

function renderPreset() {
  const preset = state.system?.preset;
  const host = $("preset");
  if (!preset) {
    host.innerHTML = "";
    return;
  }
  const installed = state.models.find((model) => model.repo_id === preset.repo_id && model.filename === preset.filename);
  const note = installed
    ? `Ya está en el panel (${esc(statusLabel(installed)[0])}).`
    : "Se descarga solo al arrancar el contenedor. También puedes lanzarla desde aquí.";
  host.innerHTML = `
    <div>
      <p class="kicker">Modelo incluido</p>
      <h3>${esc(preset.label || preset.filename)} · ${esc(preset.size_label)}</h3>
      <p class="meta">${esc(preset.repo_id)} / ${esc(preset.filename)}</p>
      <p class="meta">${esc(preset.note)} ${note}</p>
    </div>
    ${meter(preset.size_bytes)}
    <div class="preset-row">
      <span class="meta">Límite del panel: ${esc(state.system.max_model_label)}</span>
      <button type="button" id="preset-download" ${installed && installed.status !== "error" ? "disabled" : ""}>Descargar ${esc(preset.label || preset.filename)}</button>
    </div>`;
  $("preset-download")?.addEventListener("click", async () => {
    const limit = state.system.max_model_label;
    const ok = confirm(
      `${preset.label || preset.filename} pesa ${preset.size_label}, dentro del límite de ${limit}.\n¿Empezar la descarga?`,
    );
    if (!ok) return;
    try {
      banner("");
      await api("/api/models/download", {
        method: "POST",
        body: JSON.stringify({ repo_id: preset.repo_id, filename: preset.filename }),
      });
      showTab("downloads");
      await refresh();
    } catch (error) {
      banner(error.message);
    }
  });
}

function inFlight(model) {
  return model.status === "queued" || model.status === "downloading" || model.status === "error";
}

function renderDownloads() {
  const items = state.models.filter(inFlight);
  const tab = $("tab-downloads");
  tab.textContent = items.length ? `Descargas en curso (${items.length})` : "Descargas en curso";
  const host = $("download-list");
  if (!host) return;
  if (!items.length) {
    host.innerHTML = `<article class="card"><p class="meta">No hay descargas en curso.</p></article>`;
    return;
  }
  host.innerHTML = items.map((model) => {
    const [label, tone] = statusLabel(model);
    const pct = Math.round((model.progress || 0) * 100);
    const active = model.status === "queued" || model.status === "downloading";
    return `
      <article class="card">
        <div class="card-top">
          <div>
            <h3>${esc(model.filename)}</h3>
            <p class="meta">${esc(model.repo_id)} · ${esc(model.size_label)}</p>
            <p class="meta"><span class="${tone}">${esc(label)}</span>${model.bytes_downloaded ? ` · ${esc(formatBytes(model.bytes_downloaded))} recibidos` : ""}</p>
          </div>
          <button type="button" class="danger" data-action="cancel" data-slug="${esc(model.slug)}">${active ? "Detener descarga" : "Eliminar"}</button>
        </div>
        ${active ? `<div class="meter"><span style="width:${pct}%"></span></div>` : ""}
        ${model.error ? `<p class="bad">${esc(model.error)}</p>` : ""}
      </article>`;
  }).join("");
}

function formatBytes(n) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  return `${n} B`;
}

function renderInstalled() {
  const host = $("installed-list");
  const ready = state.models.filter((model) => model.status === "ready");
  if (!ready.length) {
    host.innerHTML = `<article class="card"><p class="meta">Todavía no hay modelos instalados. El modelo incluido aparece arriba y se descarga solo al arrancar; también puedes buscar otro en el catálogo.</p></article>`;
    return;
  }
  host.innerHTML = ready.map((model) => {
    const [label, tone] = statusLabel(model);
    const log = (model.runtime?.log_tail || []).slice(-6).join("\n");
    const error = model.error || model.runtime?.error || "";
    return `
      <article class="card">
        <div class="card-top">
          <div>
            <h3>${esc(model.slug)}</h3>
            <p class="meta">${esc(model.repo_id)} · ${esc(model.filename)}</p>
            <p class="meta">${esc(model.size_label)} ${model.quant ? "· " + esc(model.quant) : ""} · <span class="${tone}">${esc(label)}</span></p>
          </div>
          <div class="row-actions">
            <button type="button" class="primary" data-action="start" data-slug="${esc(model.slug)}" ${model.status === "ready" && !model.blocked ? "" : "disabled"}>Arrancar</button>
            <button type="button" data-action="stop" data-slug="${esc(model.slug)}">Detener</button>
            <button type="button" data-action="chat" data-slug="${esc(model.slug)}" ${model.blocked ? "disabled" : ""}>Chat</button>
            <button type="button" class="danger" data-action="delete" data-slug="${esc(model.slug)}">Eliminar</button>
          </div>
        </div>
        ${model.blocked ? `<p class="bad">${esc(model.blocked)}</p>` : ""}
        ${model.ram_warning ? `<p class="bad">${esc(model.ram_warning)}</p>` : ""}
        ${error && !model.blocked && error !== model.ram_warning ? `<p class="bad">${esc(error)}</p>` : ""}
        <div class="api">
          <p>OpenAI base <code>${esc(model.api.openai_base)}</code></p>
          <p>Este modelo <code>POST ${esc(model.api.chat)}</code></p>
          <p><code>{"messages":[{"role":"user","content":"Hola"}]}</code></p>
          <button type="button" data-action="copy" data-copy="${esc(model.api.chat)}">Copiar URL</button>
        </div>
        ${log ? `<pre class="api">${esc(log)}</pre>` : ""}
      </article>`;
  }).join("");
}

let chatOptions = "";

function fillChatModels() {
  const select = $("chat-model");
  const current = select.value;
  const ready = state.models
    .filter((model) => model.status === "ready" && !model.blocked)
    .sort((a, b) => rank(a) - rank(b));
  const html = ready.length
    ? ready.map((model) => `<option value="${esc(model.slug)}">${esc(model.slug)}</option>`).join("")
    : `<option value="">Sin modelos descargados</option>`;
  if (html !== chatOptions) {
    chatOptions = html;
    select.innerHTML = html;
    if (ready.some((model) => model.slug === current)) select.value = current;
    else if (ready.length) select.value = ready[0].slug;
  }
  const chosen = state.models.find((model) => model.slug === select.value);
  const status = $("chat-status");
  if (!chosen) {
    status.textContent = "Descarga un modelo para probarlo.";
    return;
  }
  const [label] = statusLabel(chosen);
  const warn = chosen.ram_warning ? ` ⚠ ${chosen.ram_warning}` : "";
  const err = chosen.runtime?.error && !chosen.ram_warning ? ` ⚠ ${chosen.runtime.error}` : "";
  status.textContent = `${chosen.size_label} · ${label}. URL ${chosen.api.chat}${warn}${err}`;
}

// Primero el que ya está en marcha, luego el modelo por defecto, luego el resto.
function rank(model) {
  if (model.ram_warning) return 3;
  if (model.runtime?.status === "running") return 0;
  if (model.is_default) return 1;
  return 2;
}

async function refresh() {
  const payload = await api("/api/models");
  state.models = payload.models || [];
  const signature = JSON.stringify(state.models.map((model) => [
    model.slug, model.status, model.progress, model.bytes_downloaded, model.error, model.runtime?.status, model.runtime?.error,
  ]));
  if (signature !== state.signature) {
    state.signature = signature;
    renderPreset();
    renderDownloads();
    if (state.tab === "installed") renderInstalled();
  }
  if (state.tab === "chat") fillChatModels();
}

async function loadSystem() {
  state.system = await api("/api/system");
  $("origin").textContent = window.location.origin;
  const llama = state.system.llama_server ? "llama-server listo" : "falta llama-server";
  $("system-line").textContent = `${llama} · RAM ${state.system.ram_limit_label} · ${state.system.cpu_limit} CPU · contexto ${state.system.ctx_size} · disco libre ${state.system.disk_free_label} · tope ${state.system.max_model_label}`;
  renderPreset();
}

function showTab(name) {
  state.tab = name;
  for (const button of document.querySelectorAll(".tab")) {
    button.classList.toggle("is-on", button.dataset.tab === name);
  }
  for (const view of ["installed", "downloads", "catalog", "chat"]) {
    $(`view-${view}`).hidden = view !== name;
  }
  if (name === "installed") renderInstalled();
  if (name === "downloads") renderDownloads();
  if (name === "chat") {
    fillChatModels();
    paintHistory();
  }
}

function paintHistory() {
  const slug = $("chat-model").value;
  const log = $("chat-log");
  log.innerHTML = "";
  for (const message of state.histories[slug] || []) {
    const bubble = document.createElement("div");
    bubble.className = `bubble ${message.role}`;
    bubble.textContent = message.content;
    log.appendChild(bubble);
  }
}

async function search(query) {
  banner("");
  const payload = await api(`/api/catalog/search?q=${encodeURIComponent(query)}`);
  const host = $("search-results");
  host.hidden = false;
  $("model-detail").innerHTML = "";
  if (!payload.results.length) {
    host.innerHTML = `<article class="card"><p class="meta">Sin resultados GGUF.</p></article>`;
    return;
  }
  host.innerHTML = payload.results.map((repo) => `
    <article class="repo" data-repo="${esc(repo.repo_id)}">
      <h3>${esc(repo.repo_id)}</h3>
      <p class="meta">${Number(repo.downloads).toLocaleString("es")} descargas · ${repo.likes} me gusta${repo.pipeline_tag ? " · " + esc(repo.pipeline_tag) : ""}${repo.gated ? " · acceso restringido" : ""}</p>
    </article>`).join("");
}

function formatWhen(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("es");
}

async function openModel(repoId) {
  banner("");
  const host = $("model-detail");
  $("search-results").hidden = true;
  host.innerHTML = `<article class="card"><p class="meta">Cargando la ficha y el peso de los archivos…</p></article>`;
  const payload = await api(`/api/catalog/model?repo_id=${encodeURIComponent(repoId)}`);
  const facts = [
    payload.author && `Autor ${payload.author}`,
    payload.pipeline_tag,
    payload.license && `Licencia ${payload.license}`,
    payload.library,
    payload.base_model && `Base ${payload.base_model}`,
    payload.last_modified && `Actualizado ${formatWhen(payload.last_modified)}`,
    payload.gated && "Acceso restringido",
  ].filter(Boolean);
  host.innerHTML = `
    <article class="card">
      <div class="detail-head">
        <div>
          <p class="kicker">${Number(payload.downloads || 0).toLocaleString("es")} descargas · ${payload.likes || 0} me gusta</p>
          <h3>${esc(payload.repo_id)}</h3>
          ${payload.summary ? `<p class="meta">${esc(payload.summary)}</p>` : ""}
        </div>
        <button type="button" id="detail-back" class="ghost">Volver a la búsqueda</button>
      </div>
      <div class="facts">${facts.map((fact) => `<span>${esc(fact)}</span>`).join("")}</div>
      <p class="meta">Solo aparecen modelos completos. El límite es ${esc(payload.limit_label)}.</p>
      ${payload.files.length ? payload.files.map((file) => `
        <div class="file-choice">
          <div class="card-top">
            <div>
              <p class="size-hero">${esc(file.size_label)}</p>
              <strong>${esc(file.label || file.filename)}</strong>
              <p class="meta">${file.quant ? esc(file.quant) + " · " : ""}${file.split ? "se descarga como un solo modelo" : "archivo único"}${file.allowed ? "" : " · supera el límite"}</p>
              ${file.reason ? `<p class="bad">${esc(file.reason)}</p>` : ""}
            </div>
            <button type="button" class="${file.allowed ? "primary" : "danger"}" data-action="download" data-repo="${esc(payload.repo_id)}" data-file="${esc(file.filename)}" data-label="${esc(file.label || file.filename)}" data-size="${esc(file.size_label)}" data-limit="${esc(payload.limit_label)}" data-parts="${file.part_total || 1}" data-allowed="${file.allowed ? "1" : "0"}" data-reason="${esc(file.reason || "")}">${file.allowed ? "Descargar" : "Ver límite"}</button>
          </div>
          ${meter(file.size_bytes)}
        </div>`).join("") : `<p class="meta">Este repositorio no publica un modelo GGUF completo.</p>`}
    </article>`;
  $("detail-back").addEventListener("click", () => {
    host.innerHTML = "";
    $("search-results").hidden = false;
  });
}

async function readSse(response, bubble, onFirst = () => {}) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let text = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (!data || data === "[DONE]") continue;
      let json;
      try { json = JSON.parse(data); } catch { continue; }
      if (json.error) {
        const failure = new Error(json.error.message || "Error del modelo");
        failure.partial = text;
        throw failure;
      }
      const delta = json.choices?.[0]?.delta?.content
        || json.choices?.[0]?.delta?.reasoning_content
        || json.choices?.[0]?.text
        || "";
      if (delta) {
        if (!text) onFirst();
        text += delta;
        bubble.textContent = text;
        bubble.classList.remove("pending");
        $("chat-log").scrollTop = $("chat-log").scrollHeight;
      }
    }
  }
  return text;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function ticker(bubble, text) {
  const started = Date.now();
  const paint = () => {
    const secs = Math.round((Date.now() - started) / 1000);
    bubble.textContent = `${text} ${secs} s`;
  };
  paint();
  const id = setInterval(paint, 1000);
  return () => clearInterval(id);
}

// Carga el modelo antes de mandar el mensaje y enseña el avance.
// Así el chat nunca se queda mudo mientras llama-server lee el GGUF.
async function ensureLoaded(slug, bubble, signal) {
  let model = state.models.find((item) => item.slug === slug);
  if (model?.runtime?.status === "running") return;
  if (model?.ram_warning) throw new Error(model.ram_warning);
  const stop = ticker(bubble, "Cargando el modelo en memoria…");
  try {
    await api(`/api/models/${encodeURIComponent(slug)}/start`, { method: "POST", signal });
    const limit = ((state.system?.load_timeout || 600) + 15) * 1000;
    const started = Date.now();
    while (Date.now() - started < limit) {
      if (signal.aborted) throw new DOMException("cancelado", "AbortError");
      model = await api(`/api/models/${encodeURIComponent(slug)}`, { signal });
      const runtime = model.runtime || {};
      if (runtime.status === "running") return;
      if (runtime.status === "error" || runtime.status === "stopped") {
        throw new Error(runtime.error || "El modelo no arrancó. Mira el registro en Instalados.");
      }
      await sleep(1500);
    }
    throw new Error(`El modelo no terminó de cargar en ${Math.round(limit / 1000)} s. Revisa la RAM libre o usa un modelo más pequeño.`);
  } finally {
    stop();
  }
}

async function sendChat(event) {
  event.preventDefault();
  if (state.sending) {
    // Solo el botón (que ahora dice "Detener") cancela; Enter no.
    if (event.submitter === $("chat-send")) state.controller?.abort();
    return;
  }
  const slug = $("chat-model").value;
  const input = $("chat-input");
  const content = input.value.trim();
  if (!slug || !content) return;
  const history = state.histories[slug] || (state.histories[slug] = []);
  history.push({ role: "user", content });
  input.value = "";
  paintHistory();
  const bubble = document.createElement("div");
  bubble.className = "bubble assistant pending";
  $("chat-log").appendChild(bubble);
  const controller = new AbortController();
  state.controller = controller;
  state.sending = true;
  $("chat-send").textContent = "Detener";
  let timedOut = false;
  let stopTicker = () => {};
  let timer = null;
  try {
    banner("");
    await ensureLoaded(slug, bubble, controller.signal);
    stopTicker = ticker(bubble, "Generando…");
    // El servidor corta a los GENERATION_TIMEOUT s; el navegador espera un poco más
    // y aborta por su cuenta si la conexión se queda colgada.
    const budget = ((state.system?.generation_timeout || 180) + 20) * 1000;
    timer = setTimeout(() => { timedOut = true; controller.abort(); }, budget);
    const system = ($("chat-system").value || "").trim();
    const messages = history.slice(-HISTORY_LIMIT).map((item) => ({ role: item.role, content: item.content }));
    if (system) messages.unshift({ role: "system", content: system });
    const response = await fetch(`/v1/models/${encodeURIComponent(slug)}/chat/completions`, {
      method: "POST",
      headers: headers(true),
      signal: controller.signal,
      body: JSON.stringify({
        messages,
        temperature: Number($("chat-temp").value),
        max_tokens: Number($("chat-max").value),
        stream: true,
        chat_template_kwargs: { enable_thinking: $("chat-think").checked },
      }),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error?.message || err.detail || `${response.status} ${response.statusText}`);
    }
    const type = response.headers.get("content-type") || "";
    let answer = "";
    if (type.includes("text/event-stream")) {
      answer = await readSse(response, bubble, () => stopTicker());
      stopTicker();
    } else {
      const data = await response.json();
      answer = data.choices?.[0]?.message?.content
        || data.choices?.[0]?.message?.reasoning_content
        || data.choices?.[0]?.text
        || "";
      stopTicker();
      bubble.textContent = answer;
    }
    bubble.classList.remove("pending");
    if (!answer) bubble.textContent = "(el modelo no devolvió texto)";
    history.push({ role: "assistant", content: bubble.textContent });
    await refresh();
  } catch (error) {
    stopTicker();
    bubble.classList.remove("pending");
    bubble.classList.add("error");
    let message = error.message || String(error);
    if (error.name === "AbortError") {
      message = timedOut
        ? `Sin respuesta en ${(state.system?.generation_timeout || 180) + 20} s. Se canceló la petición. Prueba con menos tokens o un modelo más pequeño.`
        : "Cancelado.";
    } else if (error instanceof TypeError) {
      message = `Se perdió la conexión con el servidor (${message}). Puede que el contenedor se haya quedado sin memoria.`;
    }
    bubble.textContent = error.partial ? `${error.partial}\n\n⚠ ${message}` : message;
    banner(message);
    // El mensaje sin respuesta no se reenvía: vuelve al cuadro para reintentarlo.
    history.pop();
    if (!input.value) input.value = content;
  } finally {
    if (timer) clearTimeout(timer);
    state.sending = false;
    state.controller = null;
    $("chat-send").textContent = "Enviar";
  }
}

document.querySelector(".tabs").addEventListener("click", (event) => {
  const button = event.target.closest(".tab");
  if (button) showTab(button.dataset.tab);
});

$("installed-list").addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const { action, slug, copy } = button.dataset;
  try {
    banner("");
    if (action === "copy") {
      try {
        await navigator.clipboard.writeText(copy);
        button.textContent = "Copiada";
      } catch {
        banner(copy);
      }
      return;
    }
    if (action === "chat") {
      showTab("chat");
      $("chat-model").value = slug;
      paintHistory();
      fillChatModels();
      return;
    }
    if (action === "delete") {
      if (!confirm(`¿Eliminar ${slug} y borrar sus archivos del volumen?`)) return;
      await api(`/api/models/${slug}`, { method: "DELETE" });
    } else if (action === "start") {
      await api(`/api/models/${slug}/start`, { method: "POST" });
    } else if (action === "stop") {
      await api(`/api/models/${slug}/stop`, { method: "POST" });
    }
    await refresh();
    renderInstalled();
  } catch (error) {
    banner(error.message);
  }
});

$("search-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  try { await search($("search-q").value); } catch (error) { banner(error.message); }
});

$("search-results").addEventListener("click", async (event) => {
  const card = event.target.closest("[data-repo]");
  if (!card) return;
  try { await openModel(card.dataset.repo); } catch (error) { banner(error.message); }
});

function confirmWithinLimit(button) {
  const label = button.dataset.label || button.dataset.file;
  const size = button.dataset.size || "tamaño desconocido";
  const limit = button.dataset.limit || "9.00 GB";
  const parts = Number(button.dataset.parts || "1");
  const reason = button.dataset.reason || "";
  if (button.dataset.allowed !== "1") {
    alert(`${label}\nPesa ${size}. El límite es ${limit}.\n${reason}\nNo se empieza la descarga.`);
    return false;
  }
  const together = parts > 1
    ? `Se descargarán ${parts} partes juntas, como un solo modelo.`
    : "Es un archivo completo.";
  return confirm(`${label}\nPesa ${size}, dentro del límite de ${limit}.\n${together}\n¿Empezar la descarga?`);
}

$("model-detail").addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-action='download']");
  if (!button) return;
  if (!confirmWithinLimit(button)) return;
  button.disabled = true;
  try {
    banner("");
    await api("/api/models/download", {
      method: "POST",
      body: JSON.stringify({ repo_id: button.dataset.repo, filename: button.dataset.file }),
    });
    showTab("downloads");
    await refresh();
  } catch (error) {
    button.disabled = false;
    banner(error.message);
  }
});

$("download-list").addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-action='cancel']");
  if (!button) return;
  try {
    banner("");
    await api(`/api/models/${button.dataset.slug}`, { method: "DELETE" });
    await refresh();
    renderDownloads();
  } catch (error) {
    banner(error.message);
  }
});

$("stop-all").addEventListener("click", async () => {
  const active = state.models.filter((model) => model.status === "queued" || model.status === "downloading");
  if (!active.length) {
    banner("No hay descargas activas.");
    return;
  }
  if (!confirm("¿Detener todas las descargas en curso?")) return;
  try {
    await api("/api/downloads/stop", { method: "POST" });
    await refresh();
    renderDownloads();
  } catch (error) {
    banner(error.message);
  }
});

$("chat-form").addEventListener("submit", sendChat);
$("chat-model").addEventListener("change", () => {
  paintHistory();
  fillChatModels();
});
$("chat-system").value = localStorage.getItem("obrador-system") ?? DEFAULT_SYSTEM;
$("chat-system").addEventListener("change", () => {
  localStorage.setItem("obrador-system", $("chat-system").value);
});
$("chat-clear").addEventListener("click", () => {
  const slug = $("chat-model").value;
  if (slug) state.histories[slug] = [];
  paintHistory();
});
$("chat-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("chat-form").requestSubmit();
  }
});

// ---------------------------------------------------------------- Acceso

let accessStatus = null;
let loginPrompted = false;

function fmtDate(epoch) {
  return new Date(epoch * 1000).toLocaleString("es", { dateStyle: "short", timeStyle: "short" });
}

function needLogin() {
  if (session() && accessStatus?.session) return;
  const text = accessStatus?.master_configured
    ? "Hace falta iniciar sesión: abre Acceso y escribe el secreto maestro."
    : "Hace falta una clave de administración: ábrela en Acceso.";
  banner(text);
  if (!loginPrompted && !$("access").open) {
    loginPrompted = true;
    openAccess().catch(() => {});
  }
}

function renderPill(status) {
  const pill = $("session-pill");
  pill.className = "pill";
  if (status.mode === "open") {
    pill.textContent = "⚠ Modo abierto";
    pill.classList.add("warn");
  } else if (status.session?.kind === "session") {
    pill.textContent = `Sesión activa · hasta ${fmtDate(status.session.expires_at)}`;
    pill.classList.add("good");
  } else if (status.session) {
    pill.textContent = "Admin con ADMIN_TOKEN";
    pill.classList.add("good");
  } else {
    pill.textContent = "Sin sesión";
    pill.classList.add("bad");
  }
}

function renderAccess(status) {
  const modes = {
    master: "Protegido con MASTER_SECRET: entra con el secreto maestro para administrar y crear API keys.",
    legacy: "Protegido con ADMIN_TOKEN / API_KEY fijos. Se recomienda definir MASTER_SECRET.",
    open: "Modo abierto: no hay MASTER_SECRET ni claves en el servidor.",
  };
  $("access-mode").textContent = modes[status.mode] || "";
  $("access-warnings").innerHTML = (status.warnings || []).map((item) => `<li>${esc(item)}</li>`).join("");
  const hasSession = status.session?.kind === "session";
  $("access-login").hidden = !status.master_configured || hasSession;
  $("access-session").hidden = !hasSession;
  if (hasSession) $("session-text").textContent = `Sesión de administración activa hasta ${fmtDate(status.session.expires_at)}.`;
  const canKeys = status.master_configured && Boolean(status.session);
  $("access-keys").hidden = !canKeys;
  $("access-legacy").open = status.mode === "legacy" && !status.session;
  if (canKeys) loadKeys().catch((error) => { $("access-note").textContent = error.message; });
}

async function loadAccess() {
  const status = await api("/api/auth/status");
  // Si la sesión guardada ya no vale (caducada, revocada o secreto rotado), se olvida.
  if (session() && status.session?.kind !== "session") clearSession();
  accessStatus = status;
  renderPill(status);
  if ($("access").open) renderAccess(status);
  if (status.mode === "open") banner(status.warnings?.[0] || "");
  else if (status.admin_required && !status.session) needLogin();
  return status;
}

async function loadKeys() {
  const payload = await api("/api/keys");
  const host = $("key-list");
  if (!payload.keys.length) {
    host.innerHTML = `<p class="meta">Todavía no hay API keys.</p>`;
    return;
  }
  host.innerHTML = payload.keys.map((key) => {
    const off = key.revoked || key.expired;
    const state = key.revoked ? "revocada" : key.expired ? "caducada" : key.expires_at ? `caduca ${fmtDate(key.expires_at)}` : "sin caducidad";
    const used = key.last_used_at ? ` · último uso ${new Date(key.last_used_at).toLocaleString("es")}` : "";
    return `
      <div class="key-item${off ? " off" : ""}">
        <div><strong>${esc(key.label)}</strong><br><span class="meta">${esc(key.id)} · ${esc(state)}${esc(used)}</span></div>
        ${off ? "" : `<button type="button" class="danger" data-revoke="${esc(key.id)}">Revocar</button>`}
      </div>`;
  }).join("");
}

async function openAccess() {
  $("admin-token").value = localStorage.getItem("obrador-admin") || "";
  $("api-key").value = localStorage.getItem("obrador-api-key") || "";
  $("access-note").textContent = "";
  $("key-new").hidden = true;
  $("key-value").value = "";
  if (!$("access").open) $("access").showModal();
  try {
    renderAccess(await loadAccess());
  } catch (error) {
    $("access-note").textContent = error.message;
  }
  if (!$("access-login").hidden) $("master-secret").focus();
}

async function afterAccessChange() {
  loginPrompted = false;
  await loadAccess();
  renderAccess(accessStatus);
  await loadSystem();
  await refresh();
  banner("");
}

$("open-access").addEventListener("click", () => {
  openAccess().catch((error) => banner(error.message));
});
$("access-close").addEventListener("click", () => $("access").close());

$("login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = $("master-secret");
  const secret = input.value;
  if (!secret) return;
  $("login-submit").disabled = true;
  try {
    const created = await api("/api/auth/login", { method: "POST", body: JSON.stringify({ secret }) });
    saveSession(created);
    input.value = "";
    $("access-note").textContent = "Sesión iniciada.";
    await afterAccessChange();
  } catch (error) {
    $("access-note").textContent = error.message;
  } finally {
    input.value = "";
    $("login-submit").disabled = false;
  }
});

$("logout").addEventListener("click", async () => {
  try {
    await api("/api/auth/logout", { method: "POST" });
  } catch {
    // Aunque falle la red, la sesión se borra de este navegador.
  }
  clearSession();
  $("access-note").textContent = "Sesión cerrada.";
  loginPrompted = true;
  await loadAccess().catch(() => {});
  if (accessStatus) renderAccess(accessStatus);
});

$("key-create").addEventListener("click", async () => {
  const days = $("key-exp").value;
  $("key-create").disabled = true;
  try {
    const created = await api("/api/keys", {
      method: "POST",
      body: JSON.stringify({ label: $("key-label").value.trim(), expires_in_days: days ? Number(days) : null }),
    });
    $("key-value").value = created.key;
    $("key-new").hidden = false;
    $("key-label").value = "";
    $("access-note").textContent = "";
    await loadKeys();
  } catch (error) {
    $("access-note").textContent = error.message;
  } finally {
    $("key-create").disabled = false;
  }
});

$("key-copy").addEventListener("click", async () => {
  const value = $("key-value").value;
  try {
    await navigator.clipboard.writeText(value);
    $("key-copy").textContent = "Copiada";
  } catch {
    $("key-value").select();
    document.execCommand("copy");
    $("key-copy").textContent = "Copiada";
  }
  setTimeout(() => { $("key-copy").textContent = "Copiar"; }, 2000);
});

$("key-list").addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-revoke]");
  if (!button) return;
  if (!confirm("¿Revocar esta API key? Los clientes que la usen dejarán de funcionar.")) return;
  try {
    await api(`/api/keys/${encodeURIComponent(button.dataset.revoke)}`, { method: "DELETE" });
    await loadKeys();
  } catch (error) {
    $("access-note").textContent = error.message;
  }
});

$("access-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  localStorage.setItem("obrador-admin", $("admin-token").value.trim());
  localStorage.setItem("obrador-api-key", $("api-key").value.trim());
  $("access-note").textContent = "Claves guardadas en este navegador.";
  try {
    await afterAccessChange();
  } catch (error) {
    $("access-note").textContent = error.message;
  }
});

loadAccess()
  .catch(() => null)
  .then(() => loadSystem())
  .then(refresh)
  .catch((error) => banner(error.message));

setInterval(() => {
  if (accessStatus?.admin_required && !accessStatus?.session) return;
  refresh().catch((error) => banner(error.message));
}, 2000);

// La sesión caduca sola: se comprueba cada minuto para avisar a tiempo.
setInterval(() => {
  loadAccess().catch(() => {});
}, 60000);
