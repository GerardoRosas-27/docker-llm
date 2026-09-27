const state = {
  tab: "installed",
  models: [],
  system: null,
  signature: "",
  histories: {},
  sending: false,
};

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

function headers(json) {
  const result = {};
  if (json) result["Content-Type"] = "application/json";
  const admin = localStorage.getItem("obrador-admin") || "";
  const key = localStorage.getItem("obrador-api-key") || "";
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
    throw new Error(message);
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
      <h3>Qwen 3.5 9B · Q4_K_M · ${esc(preset.size_label)}</h3>
      <p class="meta">${esc(preset.repo_id)} / ${esc(preset.filename)}</p>
      <p class="meta">${esc(preset.note)} ${note}</p>
    </div>
    ${meter(preset.size_bytes)}
    <div class="preset-row">
      <span class="meta">Límite del panel: ${esc(state.system.max_model_label)}</span>
      <button type="button" id="preset-download" ${installed && installed.status !== "error" ? "disabled" : ""}>Descargar Qwen 3.5 9B</button>
    </div>`;
  $("preset-download")?.addEventListener("click", async () => {
    const limit = state.system.max_model_label;
    const ok = confirm(
      `Qwen 3.5 9B Q4_K_M pesa ${preset.size_label}, dentro del límite de ${limit}.\n¿Empezar la descarga?`,
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
    host.innerHTML = `<article class="card"><p class="meta">Todavía no hay modelos instalados. El de Qwen 3.5 9B aparece arriba, o búscalo en el catálogo.</p></article>`;
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
        ${error && !model.blocked ? `<p class="bad">${esc(error)}</p>` : ""}
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
  const ready = state.models.filter((model) => model.status === "ready" && !model.blocked);
  const html = ready.length
    ? ready.map((model) => `<option value="${esc(model.slug)}">${esc(model.slug)}</option>`).join("")
    : `<option value="">Sin modelos descargados</option>`;
  if (html !== chatOptions) {
    chatOptions = html;
    select.innerHTML = html;
    if (ready.some((model) => model.slug === current)) select.value = current;
  }
  const chosen = state.models.find((model) => model.slug === select.value);
  const status = $("chat-status");
  if (!chosen) {
    status.textContent = "Descarga un modelo para probarlo.";
    return;
  }
  const [label] = statusLabel(chosen);
  status.textContent = `${chosen.size_label} · ${label}. URL ${chosen.api.chat}`;
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
  $("system-line").textContent = `${llama} · contexto ${state.system.ctx_size} · disco libre ${state.system.disk_free_label} · tope ${state.system.max_model_label}`;
  if (state.system.admin_token_required && !localStorage.getItem("obrador-admin")) {
    banner("El panel pide ADMIN_TOKEN. Ábrelo en Acceso.");
  }
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

async function readSse(response, bubble) {
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
      const delta = json.choices?.[0]?.delta?.content
        || json.choices?.[0]?.delta?.reasoning_content
        || json.choices?.[0]?.text
        || "";
      if (delta) {
        text += delta;
        bubble.textContent = text;
        bubble.classList.remove("pending");
        $("chat-log").scrollTop = $("chat-log").scrollHeight;
      }
    }
  }
  return text;
}

async function sendChat(event) {
  event.preventDefault();
  if (state.sending) return;
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
  bubble.textContent = "Cargando el modelo y generando…";
  $("chat-log").appendChild(bubble);
  state.sending = true;
  $("chat-send").disabled = true;
  try {
    const response = await fetch(`/v1/models/${encodeURIComponent(slug)}/chat/completions`, {
      method: "POST",
      headers: headers(true),
      body: JSON.stringify({
        messages: history.map((item) => ({ role: item.role, content: item.content })),
        temperature: Number($("chat-temp").value),
        max_tokens: Number($("chat-max").value),
        stream: true,
        chat_template_kwargs: { enable_thinking: $("chat-think").checked },
      }),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error?.message || err.detail || response.statusText);
    }
    const type = response.headers.get("content-type") || "";
    let answer = "";
    if (type.includes("text/event-stream")) {
      answer = await readSse(response, bubble);
    } else {
      const data = await response.json();
      answer = data.choices?.[0]?.message?.content
        || data.choices?.[0]?.message?.reasoning_content
        || data.choices?.[0]?.text
        || "";
      bubble.textContent = answer;
    }
    bubble.classList.remove("pending");
    if (!answer) bubble.textContent = "(sin texto)";
    history.push({ role: "assistant", content: bubble.textContent });
    await refresh();
  } catch (error) {
    bubble.classList.remove("pending");
    bubble.textContent = error.message;
    banner(error.message);
  } finally {
    state.sending = false;
    $("chat-send").disabled = false;
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
$("chat-model").addEventListener("change", paintHistory);
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

async function openAccess() {
  $("admin-token").value = localStorage.getItem("obrador-admin") || "";
  $("api-key").value = localStorage.getItem("obrador-api-key") || "";
  try {
    const status = await api("/api/access");
    $("access-note").textContent = status.admin_required || status.api_required
      ? "Hay claves activas. Si generas otras, sustituyen a las anteriores. Cópialas: no se vuelven a mostrar."
      : "El chat no pide clave hasta que pulses Generar. La clave nueva se guarda en este navegador.";
  } catch (error) {
    $("access-note").textContent = error.message;
  }
  $("access").showModal();
}

async function generateKeys(which) {
  const created = await api("/api/access/generate", {
    method: "POST",
    body: JSON.stringify({ which }),
  });
  if (created.admin_token) {
    localStorage.setItem("obrador-admin", created.admin_token);
    $("admin-token").value = created.admin_token;
  }
  if (created.api_key) {
    localStorage.setItem("obrador-api-key", created.api_key);
    $("api-key").value = created.api_key;
  }
  $("access-note").textContent = "Claves generadas. Cópialas ahora; el servidor no las vuelve a enseñar.";
  await loadSystem();
  await refresh();
}

$("open-access").addEventListener("click", () => {
  openAccess().catch((error) => banner(error.message));
});
$("gen-admin").addEventListener("click", () => {
  generateKeys("admin").catch((error) => banner(error.message));
});
$("gen-api").addEventListener("click", () => {
  generateKeys("api").catch((error) => banner(error.message));
});
$("gen-both").addEventListener("click", () => {
  generateKeys("both").catch((error) => banner(error.message));
});
$("access-close").addEventListener("click", () => $("access").close());
$("access-form").addEventListener("submit", async () => {
  localStorage.setItem("obrador-admin", $("admin-token").value.trim());
  localStorage.setItem("obrador-api-key", $("api-key").value.trim());
  try {
    await loadSystem();
    await refresh();
    banner("");
  } catch (error) {
    banner(error.message);
  }
});

loadSystem()
  .then(refresh)
  .catch((error) => banner(error.message));

setInterval(() => {
  refresh().catch((error) => banner(error.message));
}, 2000);
