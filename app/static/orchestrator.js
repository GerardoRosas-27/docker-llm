(() => {
  const { $, api, banner, copyText, esc, formatBytes, headers, session } = window.Obrador;

  const orch = {
    configData: null,
    runs: [],
    activeRunId: null,
    activeDetail: null,
    subtab: "timeline",
    selectedFile: null,
    selectedFileContent: "",
    streamAbort: null,
    loadedOnce: false,
  };

  const PRESETS = {
    code: {
      objective:
        "Implementar en Python una utilidad para calcular Fibonacci y conteo de frecuencias de palabras, ejecutar pruebas unitarias en el sandbox y guardar el reporte JSON en results/.",
      criteria: [
        "Script ejecutado en sandbox Python con código de salida 0",
        "Resultados estructurados depositados en results/",
        "Validación cruzada completada y registrada en NOTES.md",
      ],
    },
    math: {
      objective:
        "Calcular y verificar gcd(144, 60), lcm(18, 24) y los primeros 10 términos de Fibonacci mediante código Python verificable, documentando pasos y evidencia en results/.",
      criteria: [
        "Cálculos comprobados por ejecución en Python sin errores",
        "Evidencia numérica guardada en results/",
        "Verificador confirma exactitud matemática",
      ],
    },
    report: {
      objective:
        "Analizar los principios de auto-atención en Transformers y compararlo con arquitecturas de memoria eficiente (GQA y FlashAttention), dejando notas coordinadas y un informe consolidado en results/.",
      criteria: [
        "Resumen técnico estructurado depositado en results/",
        "Revisión cruzada registrada en NOTES.md",
        "Síntesis final clara en FINAL_RESULT.md",
      ],
    },
  };

  const STATUS_META = {
    pending: ["Pendiente", "warn"],
    running: ["En curso", "warn"],
    paused: ["Pausada", "warn"],
    completed: ["Objetivo cumplido", "good"],
    max_rounds: ["Límite de rondas", "warn"],
    stopped: ["Detenida", "bad"],
    error: ["Error", "bad"],
  };

  const PHASE_LABEL = {
    idle: "En espera",
    orchestrating: "Orquestador planificando la ronda y marcando el ritmo…",
    working: "Agente de resultados generando entregable…",
    executing: "Ejecutando código Python en el sandbox aislado…",
    verifying: "Verificador comprobando si ya se cumplió el objetivo…",
    synthesizing: "Orquestador consolidando FINAL_RESULT.md…",
    pacing: "Cadencia activa entre rondas…",
    paused: "Ritmo pausado — puedes dejar una nota en NOTES.md o avanzar la ronda",
    done: "Flujo finalizado",
    error: "Interrumpido por error",
  };

  function statusMeta(status) {
    return STATUS_META[status] || [status || "Desconocido", ""];
  }

  function formatTime(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleTimeString("es", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function modelOptionsHtml(selectedSlug) {
    const models = orch.configData?.available_models || [];
    if (!models.length) {
      return `<option value="">Sin modelos listos</option>`;
    }
    return models
      .map((m) => {
        const warn = m.ram_warning ? " ⚠ RAM" : "";
        const sel = m.slug === selectedSlug ? " selected" : "";
        return `<option value="${esc(m.slug)}"${sel}>${esc(m.slug)}${warn}</option>`;
      })
      .join("");
  }

  // ---------------------------------------------------------------- Configuración

  function renderWorkerRow(worker, index) {
    return `
      <div class="orch-worker-row" data-idx="${index}">
        <input type="text" class="orch-w-name" value="${esc(worker.name || `Agente ${index + 1}`)}" placeholder="Nombre / especialidad">
        <select class="orch-w-model">${modelOptionsHtml(worker.model)}</select>
        <input type="text" class="orch-w-role" value="${esc(worker.role || "")}" placeholder="Instrucción del rol (qué resultados produce)">
        <button type="button" class="danger orch-w-remove" title="Quitar agente">Quitar</button>
      </div>`;
  }

  function renderConfigForm() {
    const payload = orch.configData;
    if (!payload) return;
    const cfg = payload.config;
    $("orch-cfg-orchestrator").innerHTML = modelOptionsHtml(cfg.orchestrator_model);
    $("orch-cfg-verifier").innerHTML = modelOptionsHtml(cfg.verifier_model);
    $("orch-cfg-pacing").value = cfg.pacing_mode || "auto";
    $("orch-cfg-pace").value = cfg.pace_seconds ?? 1.5;
    $("orch-cfg-rounds").value = cfg.max_rounds ?? 4;
    $("orch-cfg-timeout").value = cfg.execution_timeout ?? 15;
    $("orch-cfg-temp").value = cfg.temperature ?? 0.4;
    $("orch-cfg-tokens").value = cfg.max_tokens ?? 768;
    $("orch-cfg-exec").checked = Boolean(cfg.execution_enabled);
    $("orch-swap-notice").hidden = !payload.sequential_swap_notice;
    $("orch-config-status").textContent = payload.saved
      ? "Configuración guardada"
      : "Usando valores automáticos";

    const workers = cfg.workers?.length ? cfg.workers : [{ id: "worker-1", name: "Especialista Técnico", role: "", model: cfg.orchestrator_model }];
    $("orch-workers-list").innerHTML = workers.map(renderWorkerRow).join("");
    renderTeamSummary();
  }

  function collectConfigFromForm() {
    const rows = [...document.querySelectorAll("#orch-workers-list .orch-worker-row")];
    const workers = rows.map((row, idx) => ({
      id: `worker-${idx + 1}`,
      name: row.querySelector(".orch-w-name").value.trim() || `Agente ${idx + 1}`,
      model: row.querySelector(".orch-w-model").value,
      role: row.querySelector(".orch-w-role").value.trim() || "Producir resultados verificables y dejarlos en results/.",
    }));
    return {
      orchestrator_model: $("orch-cfg-orchestrator").value,
      verifier_model: $("orch-cfg-verifier").value,
      workers,
      pacing_mode: $("orch-cfg-pacing").value,
      pace_seconds: Number($("orch-cfg-pace").value),
      max_rounds: Number($("orch-cfg-rounds").value),
      execution_enabled: $("orch-cfg-exec").checked,
      execution_timeout: Number($("orch-cfg-timeout").value),
      temperature: Number($("orch-cfg-temp").value),
      max_tokens: Number($("orch-cfg-tokens").value),
    };
  }

  function renderTeamSummary() {
    const host = $("orch-team-summary");
    const cfg = orch.configData?.config;
    if (!cfg || !cfg.orchestrator_model) {
      host.textContent = "Descarga o activa al menos un modelo en Instalados para iniciar el orquestador.";
      $("orch-launch").disabled = true;
      return;
    }
    $("orch-launch").disabled = false;
    const workerCount = (cfg.workers || []).length;
    const execTxt = cfg.execution_enabled ? "Sandbox Python activo" : "Sin sandbox";
    const paceTxt = cfg.pacing_mode === "step" ? "Paso a paso" : `Auto (${cfg.pace_seconds}s)`;
    host.textContent =
      `Orquestador: ${cfg.orchestrator_model} · ${workerCount} agente${workerCount === 1 ? "" : "s"} · ` +
      `Verificador: ${cfg.verifier_model} · ${paceTxt} · ${execTxt}`;
  }

  async function loadConfig() {
    orch.configData = await api("/api/orchestrator/config");
    renderConfigForm();
  }

  async function saveConfig() {
    const btn = $("orch-save-config");
    btn.disabled = true;
    try {
      banner("");
      const payload = collectConfigFromForm();
      orch.configData = await api("/api/orchestrator/config", {
        method: "PUT",
        body: JSON.stringify(payload),
      });
      renderConfigForm();
      $("orch-config-status").textContent = "Configuración guardada ✓";
    } catch (error) {
      banner(error.message);
    } finally {
      btn.disabled = false;
    }
  }

  // ---------------------------------------------------------------- Lista de misiones

  function renderRunsList() {
    const host = $("orch-runs-list");
    $("orch-runs-count").textContent = orch.runs.length ? `${orch.runs.length}` : "";
    if (!orch.runs.length) {
      host.innerHTML = `<article class="card"><p class="meta">Aún no hay misiones. Define un objetivo arriba para empezar.</p></article>`;
      return;
    }
    host.innerHTML = orch.runs
      .map((run) => {
        const [label, tone] = statusMeta(run.status);
        const selected = run.id === orch.activeRunId ? " is-selected" : "";
        return `
          <article class="card orch-run-item${selected}" data-run-id="${esc(run.id)}">
            <div class="card-top">
              <span class="pill ${tone}">${esc(label)}</span>
              <span class="meta">R ${run.current_round}/${run.max_rounds} · ${run.progress_pct}%</span>
            </div>
            <p class="orch-run-title">${esc(run.objective)}</p>
            <div class="meter"><span style="width:${Math.max(0, Math.min(100, run.progress_pct || 0))}%"></span></div>
            <div class="card-top orch-run-foot">
              <span class="meta">${esc(run.config?.orchestrator_model || "")}</span>
              <button type="button" class="danger orch-run-del" data-del-run="${esc(run.id)}">Eliminar</button>
            </div>
          </article>`;
      })
      .join("");
  }

  async function loadRuns() {
    const data = await api("/api/orchestrator/runs");
    orch.runs = data.runs || [];
    renderRunsList();
    if (!orch.activeRunId && orch.runs.length) {
      await selectRun(orch.runs[0].id);
    } else if (orch.activeRunId && !orch.runs.some((r) => r.id === orch.activeRunId)) {
      clearActiveRun();
    }
  }

  function clearActiveRun() {
    closeStream();
    orch.activeRunId = null;
    orch.activeDetail = null;
    orch.selectedFile = null;
    $("orch-empty").hidden = false;
    $("orch-mission").hidden = true;
  }

  // ---------------------------------------------------------------- Vista de misión activa

  function showSubtab(name) {
    orch.subtab = name;
    for (const btn of document.querySelectorAll(".orch-subtab")) {
      btn.classList.toggle("is-on", btn.dataset.subtab === name);
    }
    for (const pane of ["timeline", "notes", "results"]) {
      $(`orch-pane-${pane}`).hidden = pane !== name;
    }
  }

  function renderMissionHeader() {
    const run = orch.activeDetail?.run;
    if (!run) return;
    $("orch-empty").hidden = true;
    $("orch-mission").hidden = false;
    $("orch-m-id").textContent = `Misión ${run.id}`;
    $("orch-m-objective").textContent = run.objective;

    const [label, tone] = statusMeta(run.status);
    const pill = $("orch-m-status");
    pill.textContent = label;
    pill.className = `pill ${tone}`;

    $("orch-m-phase").textContent = run.error ? `Error: ${run.error}` : (PHASE_LABEL[run.phase] || run.phase);
    $("orch-m-round").textContent = `Ronda ${run.current_round} de ${run.max_rounds}`;
    const pct = Math.max(0, Math.min(100, Number(run.progress_pct || 0)));
    $("orch-m-pct").textContent = `${pct}%`;
    $("orch-m-meter").innerHTML = `<span style="width:${pct}%"></span>`;

    const cfg = run.config || {};
    const workerPills = (cfg.workers || [])
      .map((w) => `<span class="orch-tag">Agente · ${esc(w.name)} (<code>${esc(w.model)}</code>)</span>`)
      .join("");
    $("orch-m-team").innerHTML = `
      <span class="orch-tag is-orch">Orquestador · <code>${esc(cfg.orchestrator_model || "")}</code></span>
      ${workerPills}
      <span class="orch-tag is-ver">Verificador · <code>${esc(cfg.verifier_model || "")}</code></span>
      <span class="orch-tag">${cfg.execution_enabled ? "Sandbox Python ON" : "Sandbox OFF"}</span>
    `;

    const isRunning = run.status === "running" && !run.paused;
    const isPaused = run.status === "paused" || run.paused;
    const isCompleted = run.status === "completed";
    const isStoppedOrErr = run.status === "stopped" || run.status === "error";
    const isMaxRounds = run.status === "max_rounds";

    $("orch-btn-pause").disabled = !isRunning;
    $("orch-btn-resume").disabled = isRunning || isCompleted;
    $("orch-btn-step").disabled = isRunning;
    $("orch-btn-extend").disabled = isRunning || (!isMaxRounds && !isPaused && !isCompleted && !isStoppedOrErr);
    $("orch-btn-stop").disabled = !isRunning && !isPaused;
  }

  function renderEventCard(ev) {
    const meta = ev.metadata || {};
    const badgeMap = {
      orchestrator: ["Orquestador", "is-orch"],
      worker: ["Agente", "is-worker"],
      execution: [meta.ok ? "Sandbox · OK" : `Sandbox · exit ${meta.exit_code ?? "?"}`, meta.ok ? "is-exec-ok" : "is-exec-bad"],
      verifier: [meta.achieved ? "Verificador · Cumplido" : "Verificador", meta.achieved ? "is-ver-ok" : "is-ver"],
      human_note: ["Nota humana", "is-human"],
      final: ["Resultado final", "is-final"],
      system: ["Sistema", "is-sys"],
    };
    const [badgeText, badgeClass] = badgeMap[ev.kind] || [ev.kind, "is-sys"];
    const modelChip = ev.model_slug ? `<code class="orch-ev-model">${esc(ev.model_slug)}</code>` : "";

    let extraHtml = "";
    if (ev.kind === "execution") {
      const created = (meta.created_files || [])
        .map((f) => `<button type="button" class="orch-file-ref" data-open-file="${esc(f)}">results/${esc(f)}</button>`)
        .join("");
      const codeBlock = meta.code
        ? `<details class="orch-code-fold"><summary>Ver código ejecutado (${esc(meta.script_file || "script.py")})</summary><pre class="orch-code">${esc(meta.code)}</pre></details>`
        : "";
      const outBlock = meta.stdout
        ? `<div class="orch-term"><span class="orch-term-label">stdout</span><pre>${esc(meta.stdout)}</pre></div>`
        : "";
      const errBlock = meta.stderr
        ? `<div class="orch-term is-err"><span class="orch-term-label">stderr</span><pre>${esc(meta.stderr)}</pre></div>`
        : "";
      extraHtml = `${codeBlock}${outBlock}${errBlock}${created ? `<div class="orch-refs">Archivos creados: ${created}</div>` : ""}`;
    } else if (ev.kind === "verifier") {
      const checks = (meta.criteria || [])
        .map((c) => `<li class="${c.passed ? "good" : "warn"}">${c.passed ? "✓" : "○"} ${esc(c.text)}</li>`)
        .join("");
      const feedback = meta.feedback ? `<p class="orch-ev-feedback"><strong>Siguiente paso:</strong> ${esc(meta.feedback)}</p>` : "";
      const ref = meta.report_file
        ? `<div class="orch-refs"><button type="button" class="orch-file-ref" data-open-file="${esc(meta.report_file)}">Ver dictamen (${esc(meta.report_file)})</button></div>`
        : "";
      extraHtml = `${checks ? `<ul class="orch-checklist">${checks}</ul>` : ""}${feedback}${ref}`;
    } else if (ev.kind === "worker") {
      const refs = (meta.artifacts || [])
        .map((f) => `<button type="button" class="orch-file-ref" data-open-file="${esc(f)}">results/${esc(f)}</button>`)
        .join("");
      if (refs) extraHtml = `<div class="orch-refs">Depositado: ${refs}</div>`;
    } else if (ev.kind === "final" && meta.file) {
      extraHtml = `<div class="orch-refs"><button type="button" class="orch-file-ref" data-open-file="${esc(meta.file)}">Abrir ${esc(meta.file)}</button></div>`;
    }

    const bodyHtml =
      ev.kind === "execution"
        ? ""
        : `<div class="orch-ev-body">${esc(ev.content)}</div>`;

    return `
      <article class="orch-event ${badgeClass}" data-event-id="${ev.id}">
        <div class="orch-ev-head">
          <div class="orch-ev-who">
            <span class="orch-badge ${badgeClass}">${esc(badgeText)}</span>
            <strong>${esc(ev.title)}</strong>
            ${modelChip}
          </div>
          <span class="meta">R${ev.round} · ${esc(formatTime(ev.created_at))}</span>
        </div>
        ${bodyHtml}
        ${extraHtml}
      </article>`;
  }

  function renderTimeline(fullReset = false) {
    const host = $("orch-timeline");
    const events = orch.activeDetail?.events || [];
    if (!events.length) {
      host.innerHTML = `<p class="meta">Iniciando la orquestación…</p>`;
      return;
    }
    if (fullReset) {
      host.innerHTML = events.map(renderEventCard).join("");
    } else {
      const existing = new Set([...host.querySelectorAll("[data-event-id]")].map((n) => Number(n.dataset.eventId)));
      let appended = false;
      for (const ev of events) {
        if (!existing.has(ev.id)) {
          host.insertAdjacentHTML("beforeend", renderEventCard(ev));
          appended = true;
        }
      }
      if (appended) host.scrollTop = host.scrollHeight;
    }
  }

  function renderNotes() {
    const detail = orch.activeDetail;
    if (!detail) return;
    $("orch-notes-path").textContent = detail.notes_path || "NOTES.md";
    $("orch-notes-content").textContent = detail.notes || "(El cuaderno NOTES.md está vacío.)";
  }

  function renderResultsList() {
    const files = orch.activeDetail?.results || [];
    const realCount = files.filter((f) => f.filename !== "INDEX.md").length;
    $("orch-subtab-results").textContent = realCount
      ? `Resultados (${realCount})`
      : "Resultados (results/)";

    const host = $("orch-results-list");
    if (!files.length) {
      host.innerHTML = `<p class="meta">Todavía no hay archivos en results/.</p>`;
      return;
    }
    host.innerHTML = files
      .map((item) => {
        const active = item.filename === orch.selectedFile ? " is-on" : "";
        return `
          <button type="button" class="orch-file-btn${active}" data-file="${esc(item.filename)}">
            <strong>${esc(item.filename)}</strong>
            <span class="meta">R${item.round || 0} · ${esc(item.author || "agente")} · ${esc(formatBytes(item.size_bytes || 0))}</span>
          </button>`;
      })
      .join("");
  }

  async function openResultFile(filename) {
    if (!orch.activeRunId || !filename) return;
    orch.selectedFile = filename;
    renderResultsList();
    $("orch-viewer-name").textContent = `results/${filename}`;
    $("orch-viewer-meta").textContent = "Cargando…";
    $("orch-viewer-content").textContent = "";
    $("orch-copy-file").disabled = true;
    try {
      const data = await api(`/api/orchestrator/runs/${encodeURIComponent(orch.activeRunId)}/files/${encodeURIComponent(filename)}`);
      orch.selectedFileContent = data.content || "";
      $("orch-viewer-name").textContent = filename === "NOTES.md" ? "NOTES.md" : `results/${data.filename}`;
      $("orch-viewer-meta").textContent = `Autor: ${data.author || "agente"} · Ronda ${data.round || 0} · ${formatBytes(data.size_bytes || 0)}`;
      $("orch-viewer-content").textContent = orch.selectedFileContent || "(archivo vacío)";
      $("orch-copy-file").disabled = false;
    } catch (error) {
      $("orch-viewer-meta").textContent = error.message;
    }
  }

  // ---------------------------------------------------------------- SSE en vivo

  function closeStream() {
    if (orch.streamAbort) {
      orch.streamAbort.abort();
      orch.streamAbort = null;
    }
  }

  async function connectStream(runId) {
    closeStream();
    const controller = new AbortController();
    orch.streamAbort = controller;
    try {
      const response = await fetch(`/api/orchestrator/runs/${encodeURIComponent(runId)}/stream`, {
        method: "GET",
        headers: headers(false),
        signal: controller.signal,
      });
      if (!response.ok) return;
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const raw = line.slice(5).trim();
          if (!raw || raw === "[DONE]") continue;
          let msg;
          try { msg = JSON.parse(raw); } catch { continue; }
          if (orch.activeRunId !== runId) return;
          applyStreamMessage(msg);
        }
      }
    } catch (error) {
      if (error.name !== "AbortError") {
        // Reconciliar estado si el stream se cerró por red
        loadRuns().catch(() => {});
      }
    }
  }

  function applyStreamMessage(msg) {
    if (!orch.activeDetail) return;
    if (msg.type === "snapshot") {
      orch.activeDetail = {
        ...orch.activeDetail,
        run: msg.data.run,
        events: msg.data.events || [],
        notes: msg.data.notes ?? orch.activeDetail.notes,
        results: msg.data.results || [],
      };
      syncRunInList(msg.data.run);
      renderMissionHeader();
      renderTimeline(true);
      renderNotes();
      renderResultsList();
    } else if (msg.type === "run") {
      orch.activeDetail.run = msg.data;
      syncRunInList(msg.data);
      renderMissionHeader();
    } else if (msg.type === "event") {
      const ev = msg.data;
      if (!orch.activeDetail.events.some((e) => e.id === ev.id)) {
        orch.activeDetail.events.push(ev);
      }
      renderTimeline(false);
    } else if (msg.type === "workspace") {
      orch.activeDetail.notes = msg.data.notes ?? orch.activeDetail.notes;
      orch.activeDetail.results = msg.data.results || orch.activeDetail.results;
      renderNotes();
      renderResultsList();
      if (orch.selectedFile && orch.activeDetail.results.some((f) => f.filename === orch.selectedFile)) {
        openResultFile(orch.selectedFile).catch(() => {});
      }
    }
  }

  function syncRunInList(updatedRun) {
    if (!updatedRun) return;
    const idx = orch.runs.findIndex((r) => r.id === updatedRun.id);
    if (idx >= 0) orch.runs[idx] = updatedRun;
    else orch.runs.unshift(updatedRun);
    renderRunsList();
  }

  async function selectRun(runId) {
    closeStream();
    orch.activeRunId = runId;
    orch.selectedFile = null;
    $("orch-viewer-name").textContent = "Selecciona un resultado";
    $("orch-viewer-meta").textContent = "";
    $("orch-viewer-content").textContent = "Haz clic en cualquier archivo de la izquierda para inspeccionar su contenido.";
    $("orch-copy-file").disabled = true;

    const detail = await api(`/api/orchestrator/runs/${encodeURIComponent(runId)}`);
    orch.activeDetail = detail;
    syncRunInList(detail.run);
    renderMissionHeader();
    renderTimeline(true);
    renderNotes();
    renderResultsList();

    // Abrir automáticamente FINAL_RESULT.md si ya existe
    const finalFile = (detail.results || []).find((f) => f.filename === "FINAL_RESULT.md");
    if (finalFile) openResultFile(finalFile.filename).catch(() => {});

    if (detail.run.active || detail.run.status === "running" || detail.run.status === "paused") {
      connectStream(runId);
    }
  }

  // ---------------------------------------------------------------- Eventos DOM

  $("orch-toggle-config").addEventListener("click", () => {
    const panel = $("orch-config-panel");
    panel.hidden = !panel.hidden;
  });
  $("orch-close-config").addEventListener("click", () => {
    $("orch-config-panel").hidden = true;
  });
  $("orch-save-config").addEventListener("click", saveConfig);

  $("orch-add-worker").addEventListener("click", () => {
    const list = $("orch-workers-list");
    const count = list.querySelectorAll(".orch-worker-row").length;
    if (count >= 8) {
      banner("Máximo 8 agentes de resultados por equipo.");
      return;
    }
    const defaultModel = orch.configData?.config?.orchestrator_model || orch.configData?.available_models?.[0]?.slug || "";
    list.insertAdjacentHTML(
      "beforeend",
      renderWorkerRow(
        {
          id: `worker-${count + 1}`,
          name: `Agente ${count + 1}`,
          model: defaultModel,
          role: "Revisar NOTES.md y results/, ejecutar pruebas y aportar resultados.",
        },
        count,
      ),
    );
  });

  $("orch-workers-list").addEventListener("click", (event) => {
    const btn = event.target.closest(".orch-w-remove");
    if (!btn) return;
    const rows = $("orch-workers-list").querySelectorAll(".orch-worker-row");
    if (rows.length <= 1) {
      banner("Debe existir al menos un agente de resultados.");
      return;
    }
    btn.closest(".orch-worker-row").remove();
  });

  document.querySelector(".orch-presets").addEventListener("click", (event) => {
    const chip = event.target.closest("[data-preset]");
    if (!chip) return;
    const preset = PRESETS[chip.dataset.preset];
    if (!preset) return;
    $("orch-objective").value = preset.objective;
    $("orch-criteria").value = preset.criteria.join("\n");
  });

  $("orch-new-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const objective = $("orch-objective").value.trim();
    if (!objective) {
      banner("Escribe un objetivo para lanzar el orquestador.");
      return;
    }
    const criteria = $("orch-criteria").value
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    const launchPacing = $("orch-launch-pacing").value || null;
    const btn = $("orch-launch");
    btn.disabled = true;
    try {
      banner("");
      const created = await api("/api/orchestrator/runs", {
        method: "POST",
        body: JSON.stringify({
          objective,
          criteria,
          pacing_mode: launchPacing,
          start_immediately: true,
        }),
      });
      orch.activeRunId = created.run.id;
      orch.activeDetail = created;
      syncRunInList(created.run);
      showSubtab("timeline");
      renderMissionHeader();
      renderTimeline(true);
      renderNotes();
      renderResultsList();
      connectStream(created.run.id);
    } catch (error) {
      banner(error.message);
    } finally {
      btn.disabled = false;
    }
  });

  $("orch-runs-list").addEventListener("click", async (event) => {
    const delBtn = event.target.closest("[data-del-run]");
    if (delBtn) {
      event.stopPropagation();
      const id = delBtn.dataset.delRun;
      if (!confirm("¿Eliminar esta misión, su fichero NOTES.md y sus archivos de results/?")) return;
      try {
        await api(`/api/orchestrator/runs/${encodeURIComponent(id)}`, { method: "DELETE" });
        if (orch.activeRunId === id) clearActiveRun();
        await loadRuns();
      } catch (error) {
        banner(error.message);
      }
      return;
    }
    const card = event.target.closest("[data-run-id]");
    if (card) {
      try {
        await selectRun(card.dataset.runId);
      } catch (error) {
        banner(error.message);
      }
    }
  });

  document.querySelector(".orch-subtabs").addEventListener("click", (event) => {
    const btn = event.target.closest(".orch-subtab");
    if (btn) showSubtab(btn.dataset.subtab);
  });

  $("orch-timeline").addEventListener("click", (event) => {
    const ref = event.target.closest("[data-open-file]");
    if (!ref) return;
    showSubtab("results");
    openResultFile(ref.dataset.openFile).catch((error) => banner(error.message));
  });

  $("orch-results-list").addEventListener("click", (event) => {
    const btn = event.target.closest("[data-file]");
    if (!btn) return;
    openResultFile(btn.dataset.file).catch((error) => banner(error.message));
  });

  $("orch-copy-notes").addEventListener("click", () => {
    copyText($("orch-copy-notes"), orch.activeDetail?.notes || "");
  });
  $("orch-copy-file").addEventListener("click", () => {
    copyText($("orch-copy-file"), orch.selectedFileContent || "");
  });

  $("orch-note-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!orch.activeRunId) return;
    const input = $("orch-note-input");
    const text = input.value.trim();
    if (!text) return;
    try {
      banner("");
      const res = await api(`/api/orchestrator/runs/${encodeURIComponent(orch.activeRunId)}/notes`, {
        method: "POST",
        body: JSON.stringify({ text, author: "Operador" }),
      });
      input.value = "";
      orch.activeDetail.notes = res.notes;
      renderNotes();
      const notesBox = $("orch-notes-content");
      if (notesBox) notesBox.scrollTop = notesBox.scrollHeight;
      if (res.event && !orch.activeDetail.events.some((e) => e.id === res.event.id)) {
        orch.activeDetail.events.push(res.event);
        renderTimeline(false);
      }
    } catch (error) {
      banner(error.message);
    }
  });

  $("orch-file-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!orch.activeRunId) return;
    const filename = $("orch-file-name").value.trim();
    const content = $("orch-file-content").value;
    if (!filename || !content.trim()) {
      banner("Indica un nombre de archivo y contenido para depositar en results/.");
      return;
    }
    try {
      banner("");
      const res = await api(`/api/orchestrator/runs/${encodeURIComponent(orch.activeRunId)}/files`, {
        method: "POST",
        body: JSON.stringify({ filename, content, summary: "Depositado por el operador" }),
      });
      $("orch-file-name").value = "";
      $("orch-file-content").value = "";
      orch.activeDetail.results = res.results;
      orch.activeDetail.notes = res.notes;
      renderResultsList();
      renderNotes();
      if (res.file?.filename) await openResultFile(res.file.filename);
    } catch (error) {
      banner(error.message);
    }
  });

  async function controlAction(action, body = null) {
    if (!orch.activeRunId) return;
    try {
      banner("");
      const detail = await api(`/api/orchestrator/runs/${encodeURIComponent(orch.activeRunId)}/${action}`, {
        method: "POST",
        ...(body ? { body: JSON.stringify(body) } : {}),
      });
      orch.activeDetail = detail;
      syncRunInList(detail.run);
      renderMissionHeader();
      renderTimeline(true);
      renderNotes();
      renderResultsList();
      if (detail.run.active || detail.run.status === "running" || detail.run.status === "paused") {
        connectStream(detail.run.id);
      }
    } catch (error) {
      banner(error.message);
    }
  }

  $("orch-btn-pause").addEventListener("click", () => controlAction("pause"));
  $("orch-btn-resume").addEventListener("click", () => controlAction("resume", { extra_rounds: 0 }));
  $("orch-btn-step").addEventListener("click", () => controlAction("step"));
  $("orch-btn-extend").addEventListener("click", () => controlAction("resume", { extra_rounds: 2 }));
  $("orch-btn-stop").addEventListener("click", () => controlAction("stop"));

  async function initOrRefresh() {
    if (!session()) return;
    try {
      await loadConfig();
      await loadRuns();
      orch.loadedOnce = true;
    } catch (error) {
      banner(error.message);
    }
  }

  window.ObradorOrch = {
    onTabShow() {
      initOrRefresh();
    },
    onModelsUpdated() {
      if (window.Obrador.state.tab === "orchestrator" || orch.loadedOnce) {
        loadConfig().catch(() => {});
      }
    },
  };
})();
