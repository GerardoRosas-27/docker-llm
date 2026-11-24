"""Motor de orquestación multi-agente por objetivos para Obrador.

Implementa un flujo coordinado con:
- Un modelo **Orquestador** configurable que marca el ritmo, divide el objetivo
  en fases y asigna tareas en cada ronda.
- Uno o más modelos **Agentes de Resultados (Workers)** configurables que leen
  el cuaderno compartido `NOTES.md` y la carpeta común `results/`, ejecutan
  código Python en un sandbox aislado y depositan sus resultados para que otros
  agentes puedan leerlos y construir sobre ellos.
- Un modelo **Verificador** configurable que evalúa independientemente el
  cuaderno, las ejecuciones y los resultados para decidir si el objetivo se ha
  cumplido o qué falta por corregir en la siguiente ronda.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx

from app import config, db, resources
from app.runner import runner
from app.sandbox import SharedWorkspace, sanitize_filename

log = logging.getLogger("obrador.orchestrator")

CONFIG_META_KEY = "orchestrator_config"


class OrchestratorError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


# --------------------------------------------------------------------------
# Configuración persistida (modelos por defecto, ritmo y ejecución)

def _ready_models() -> list[dict[str, Any]]:
    models = []
    for item in db.list_models():
        if item["status"] != "ready":
            continue
        is_hybrid = item["slug"] == config.HYBRID_SLUG or item["repo_id"] == config.HYBRID_REPO
        ram_warn = None
        if not is_hybrid and config.ram_check():
            ram_warn = resources.ram_block_reason(item.get("size_bytes"), config.llama_repack())
        models.append(
            {
                "slug": item["slug"],
                "repo_id": item["repo_id"],
                "filename": item["filename"],
                "is_default": item["slug"] == config.default_slug(),
                "ram_warning": ram_warn,
                "runtime_status": runner.public_status(item["slug"])["status"],
            }
        )
    models.sort(
        key=lambda m: (
            1 if m["ram_warning"] else 0,
            0 if m["runtime_status"] == "running" else 1,
            0 if m["slug"] == config.default_slug() else 1,
            m["slug"],
        )
    )
    return models


def _default_workers(ready_slugs: list[str]) -> list[dict[str, str]]:
    primary = ready_slugs[0] if ready_slugs else ""
    secondary = ready_slugs[1] if len(ready_slugs) > 1 else primary
    workers = [
        {
            "id": "worker-1",
            "name": "Especialista Técnico",
            "role": "Diseñar, implementar y ejecutar soluciones verificables en el sandbox y guardar resultados en results/.",
            "model": primary,
        }
    ]
    if primary:
        workers.append(
            {
                "id": "worker-2",
                "name": "Analista y Revisor",
                "role": "Leer los resultados de otros agentes en results/, comprobar casos borde, complementar evidencias y refinar el entregable.",
                "model": secondary,
            }
        )
    return workers


def default_config() -> dict[str, Any]:
    ready = _ready_models()
    slugs = [m["slug"] for m in ready]
    primary = slugs[0] if slugs else ""
    verifier = slugs[-1] if len(slugs) > 2 else primary
    return {
        "orchestrator_model": primary,
        "verifier_model": verifier,
        "workers": _default_workers(slugs),
        "pacing_mode": "auto",
        "pace_seconds": 1.5,
        "max_rounds": config.orchestrator_max_rounds(),
        "execution_enabled": True,
        "execution_timeout": config.orchestrator_exec_timeout(),
        "temperature": 0.4,
        "max_tokens": 768,
    }


def _normalize_workers(raw_workers: Any, fallback_slug: str, valid_slugs: set[str], strict: bool) -> list[dict[str, str]]:
    if not isinstance(raw_workers, list) or not raw_workers:
        if strict:
            raise OrchestratorError(400, "Configura al menos un modelo de resultados (agente).")
        return _default_workers([fallback_slug] if fallback_slug else [])
    normalized: list[dict[str, str]] = []
    used_ids: set[str] = set()
    for idx, item in enumerate(raw_workers[:8], start=1):
        if isinstance(item, str):
            item = {"model": item}
        if not isinstance(item, dict):
            continue
        model = str(item.get("model") or "").strip()
        if strict:
            if not model or model not in valid_slugs:
                raise OrchestratorError(
                    400,
                    f"El modelo de resultados '{model or '(vacío)'}' no está instalado y listo.",
                )
        else:
            if model not in valid_slugs:
                model = fallback_slug
        raw_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(item.get("id") or f"worker-{idx}").strip()).strip("-").lower()
        worker_id = raw_id or f"worker-{idx}"
        while worker_id in used_ids:
            worker_id = f"{worker_id}-{idx}"
        used_ids.add(worker_id)
        name = str(item.get("name") or f"Agente {idx}").strip()[:60] or f"Agente {idx}"
        role = str(item.get("role") or "Producir resultados verificables y dejarlos en results/.").strip()[:280]
        normalized.append({"id": worker_id, "name": name, "role": role, "model": model})
    if not normalized:
        if strict:
            raise OrchestratorError(400, "Configura al menos un modelo de resultados válido.")
        return _default_workers([fallback_slug] if fallback_slug else [])
    return normalized


def validate_config(raw: dict[str, Any], *, strict: bool = True) -> dict[str, Any]:
    ready = _ready_models()
    slugs = [m["slug"] for m in ready]
    valid = set(slugs)
    fallback = slugs[0] if slugs else ""
    base = default_config()
    merged = {**base, **(raw or {})}

    orch = str(merged.get("orchestrator_model") or "").strip()
    if strict:
        if not slugs:
            raise OrchestratorError(409, "No hay modelos instalados y listos para usar el orquestador.")
        if not orch or orch not in valid:
            raise OrchestratorError(400, f"El modelo orquestador '{orch or '(vacío)'}' no está instalado y listo.")
    elif orch not in valid:
        orch = fallback

    verifier = str(merged.get("verifier_model") or "").strip()
    if strict:
        if not verifier or verifier not in valid:
            raise OrchestratorError(400, f"El modelo verificador '{verifier or '(vacío)'}' no está instalado y listo.")
    elif verifier not in valid:
        verifier = orch or fallback

    workers = _normalize_workers(merged.get("workers"), orch or fallback, valid, strict=strict)

    pacing_mode = str(merged.get("pacing_mode") or "auto").strip().lower()
    if pacing_mode not in ("auto", "step"):
        if strict:
            raise OrchestratorError(400, "pacing_mode debe ser 'auto' o 'step'.")
        pacing_mode = "auto"

    try:
        pace_seconds = round(max(0.0, min(30.0, float(merged.get("pace_seconds", 1.5)))), 2)
    except (TypeError, ValueError):
        pace_seconds = 1.5
    try:
        max_rounds = max(1, min(20, int(merged.get("max_rounds", config.orchestrator_max_rounds()))))
    except (TypeError, ValueError):
        max_rounds = config.orchestrator_max_rounds()
    try:
        execution_timeout = max(2, min(60, int(merged.get("execution_timeout", config.orchestrator_exec_timeout()))))
    except (TypeError, ValueError):
        execution_timeout = config.orchestrator_exec_timeout()
    try:
        temperature = round(max(0.0, min(1.5, float(merged.get("temperature", 0.4)))), 2)
    except (TypeError, ValueError):
        temperature = 0.4
    try:
        max_tokens = max(128, min(4096, int(merged.get("max_tokens", 768))))
    except (TypeError, ValueError):
        max_tokens = 768

    return {
        "orchestrator_model": orch,
        "verifier_model": verifier,
        "workers": workers,
        "pacing_mode": pacing_mode,
        "pace_seconds": pace_seconds,
        "max_rounds": max_rounds,
        "execution_enabled": bool(merged.get("execution_enabled", True)),
        "execution_timeout": execution_timeout,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def get_config() -> dict[str, Any]:
    raw = db.meta_get(CONFIG_META_KEY)
    saved: dict[str, Any] = {}
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                saved = parsed
        except ValueError:
            saved = {}
    resolved = validate_config(saved, strict=False)
    ready = _ready_models()
    distinct = {
        m
        for m in [
            resolved["orchestrator_model"],
            resolved["verifier_model"],
            *[w["model"] for w in resolved["workers"]],
        ]
        if m
    }
    return {
        "config": resolved,
        "saved": bool(raw),
        "available_models": ready,
        "max_loaded_models": config.max_loaded_models(),
        "sequential_swap_notice": (
            len(distinct) > config.max_loaded_models()
            and config.max_loaded_models() == 1
        ),
    }


def save_config(payload: dict[str, Any]) -> dict[str, Any]:
    validated = validate_config(payload, strict=True)
    db.meta_set(CONFIG_META_KEY, json.dumps(validated, ensure_ascii=False))
    return get_config()


# --------------------------------------------------------------------------
# Llamada interna a modelos desplegados en Obrador

async def call_model(
    slug: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.4,
    max_tokens: int = 768,
) -> str:
    """Invoca un modelo instalado vía Runner y su servidor local OpenAI-compatible."""
    model = db.get(slug)
    if model is None or model["status"] != "ready":
        raise RuntimeError(f"El modelo '{slug}' no está instalado y listo.")
    running = await runner.ensure(slug)
    total = config.generation_timeout()
    payload: dict[str, Any] = {
        "model": slug,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = f"http://127.0.0.1:{running.port}/v1/chat/completions"
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(total, connect=10.0, read=total, write=30.0),
        trust_env=False,
    ) as client:
        resp = await client.post(url, json=payload)
        if resp.status_code >= 400:
            lower = resp.text.lower()
            if "chat_template" in lower or "unknown" in lower or "extra" in lower:
                payload.pop("chat_template_kwargs", None)
                resp = await client.post(url, json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(f"Modelo '{slug}' devolvió HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    text = msg.get("content") or msg.get("reasoning_content") or choices[0].get("text") or ""
    return str(text).strip()


# --------------------------------------------------------------------------
# Prompts y parsers estructurados (tolerantes con modelos locales pequeños)

def _extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def _try_parse_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    fence = re.search(r"```json\s*(\{.*?\})\s*```", stripped, re.IGNORECASE | re.DOTALL)
    candidate = fence.group(1) if fence else stripped
    if not (candidate.startswith("{") and candidate.endswith("}")):
        return None
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except ValueError:
        return None


def _strip_protocol_tags(text: str) -> str:
    cleaned = re.sub(r"</?(?:plan|note|pace|dispatch|exec|result|verdict|progress|summary|criteria|feedback)[^>]*>", "", text, flags=re.IGNORECASE)
    return cleaned.strip()


def _orchestrator_messages(
    *,
    objective: str,
    criteria: list[str],
    round_num: int,
    max_rounds: int,
    workers: list[dict[str, str]],
    notes: str,
    results_digest: str,
    last_verdict: dict[str, Any] | None,
    execution_enabled: bool,
) -> list[dict[str, str]]:
    workers_desc = "\n".join(
        f"- id=`{w['id']}` | nombre={w['name']} | modelo=`{w['model']}` | especialidad={w['role']}"
        for w in workers
    )
    criteria_desc = "\n".join(f"- {c}" for c in criteria) if criteria else "- Cumplir completamente el objetivo con evidencia verificable."
    prev_block = ""
    if last_verdict:
        prev_block = (
            f"\nÚltimo dictamen del Verificador (ronda {round_num - 1}):\n"
            f"- Estado: {last_verdict.get('verdict')} ({last_verdict.get('progress', 0)}%)\n"
            f"- Resumen: {last_verdict.get('summary')}\n"
            f"- Feedback pendiente: {last_verdict.get('feedback')}\n"
        )
    system = (
        "[OBRADOR_ROLE:ORCHESTRATOR]\n"
        "Eres el ORQUESTADOR de un equipo multi-agente por objetivos.\n"
        "Tu responsabilidad es marcar el ritmo, coordinar el plan por rondas, leer las notas "
        "compartidas en NOTES.md y los archivos de results/, y asignar tareas concretas a los "
        "agentes de resultados para converger al objetivo.\n"
        "Responde SIEMPRE en español usando exactamente estas etiquetas:\n"
        "<plan>Estrategia clara y concreta para esta ronda</plan>\n"
        "<note>Nota breve que quedará registrada en NOTES.md para todos los agentes</note>\n"
        "<pace>continue | refine | converge</pace>\n"
        "<dispatch worker=\"ID_DEL_AGENTE\">Instrucción concreta, qué debe leer de results/ y qué debe entregar</dispatch>\n"
        "Incluye al menos un bloque <dispatch> con el id exacto de uno o más agentes disponibles."
    )
    user = (
        f"RONDA ACTUAL: {round_num} de {max_rounds}\n"
        f"ENTORNO DE EJECUCIÓN PYTHON: {'HABILITADO' if execution_enabled else 'DESHABILITADO'}\n\n"
        f"OBJETIVO:\n{objective}\n\n"
        f"CRITERIOS DE ÉXITO:\n{criteria_desc}\n\n"
        f"AGENTES DISPONIBLES:\n{workers_desc}\n"
        f"{prev_block}\n"
        f"CUADERNO COMPARTIDO (NOTES.md):\n{notes}\n\n"
        f"CARPETA COMPARTIDA (results/):\n{results_digest}\n\n"
        f"Define el plan, marca el ritmo y asigna las tareas de la ronda {round_num}."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_orchestrator_output(
    raw: str,
    workers: list[dict[str, str]],
    objective: str,
    round_num: int,
) -> dict[str, Any]:
    by_lookup: dict[str, dict[str, str]] = {}
    for w in workers:
        by_lookup[w["id"].lower()] = w
        by_lookup[w["name"].lower()] = w
        by_lookup[w["model"].lower()] = w

    js = _try_parse_json(raw)
    assignments: list[dict[str, Any]] = []
    if js:
        plan = str(js.get("plan") or "").strip()
        note = str(js.get("note") or plan).strip()
        pace = str(js.get("pace") or "continue").strip().lower()
        for entry in js.get("assignments") or js.get("dispatch") or []:
            if not isinstance(entry, dict):
                continue
            key = str(entry.get("worker") or entry.get("id") or "").strip().lower()
            task = str(entry.get("task") or entry.get("instruction") or "").strip()
            matched = by_lookup.get(key)
            if matched and task:
                assignments.append({"worker": matched, "task": task})
    else:
        plan = _extract_tag(raw, "plan")
        note = _extract_tag(raw, "note")
        pace = _extract_tag(raw, "pace").lower() or "continue"
        for m in re.finditer(
            r"<dispatch\s+[^>]*worker\s*=\s*[\"']?([^\"'>\s]+)[\"']?[^>]*>(.*?)</dispatch>",
            raw,
            re.IGNORECASE | re.DOTALL,
        ):
            key = m.group(1).strip().lower()
            task = m.group(2).strip()
            matched = by_lookup.get(key)
            if matched and task:
                assignments.append({"worker": matched, "task": task})

    if pace not in ("continue", "refine", "converge"):
        pace = "continue"
    if not plan:
        plan = _strip_protocol_tags(raw)[:700] or f"Coordinar la ronda {round_num} para avanzar hacia el objetivo."
    if not note:
        note = plan[:450]
    if not assignments:
        # Respaldo seguro: si un modelo pequeño olvidó las etiquetas <dispatch>, asignamos a los workers activos.
        for w in workers:
            assignments.append(
                {
                    "worker": w,
                    "task": (
                        f"Contribuye en la ronda {round_num} según tu especialidad ({w['role']}) "
                        f"siguiendo el plan del orquestador: {plan[:350]}. Objetivo: {objective}"
                    ),
                }
            )
    return {
        "plan": plan,
        "note": note,
        "pace": pace,
        "assignments": assignments,
    }


def _worker_messages(
    *,
    worker: dict[str, str],
    objective: str,
    criteria: list[str],
    round_num: int,
    max_rounds: int,
    assignment: str,
    notes: str,
    results_digest: str,
    execution_enabled: bool,
) -> list[dict[str, str]]:
    exec_hint = (
        "Tienes un entorno de ejecución Python aislado. Si escribes un bloque:\n"
        "<exec lang=\"python\">\n# código ejecutable aquí\n</exec>\n"
        "se ejecutará realmente en `workspace/`. Puedes leer/escribir archivos compartidos en "
        "`os.environ['ORCH_RESULTS_DIR']` y leer `os.environ['ORCH_NOTES_FILE']`. "
        "La salida (stdout, stderr y archivos creados) se guardará automáticamente en `results/`.\n"
        if execution_enabled
        else "El entorno de ejecución está deshabilitado para esta misión; entrega tu análisis directamente en `<result>`.\n"
    )
    criteria_desc = "\n".join(f"- {c}" for c in criteria) if criteria else "- Cumplir el objetivo de manera completa y verificable."
    system = (
        f"[OBRADOR_ROLE:WORKER]\n"
        f"[OBRADOR_WORKER_ID:{worker['id']}]\n"
        f"[OBRADOR_WORKER_NAME:{worker['name']}]\n"
        f"Eres el agente '{worker['name']}' (especialidad: {worker['role']}).\n"
        "Lee cuidadosamente las notas previas en NOTES.md y los resultados depositados por otros agentes en results/ "
        "para no repetir trabajo y construir sobre sus avances.\n"
        f"{exec_hint}"
        "Responde SIEMPRE en español usando estas etiquetas:\n"
        "<note>Nota breve para NOTES.md indicando qué lograste, hallazgos clave y avisos para otros agentes</note>\n"
        + ("<exec lang=\"python\">Código Python opcional para calcular, validar o generar archivos en ORCH_RESULTS_DIR</exec>\n" if execution_enabled else "")
        + "<result file=\"nombre_descriptivo.md\">Contenido completo de tu entregable para depositar en results/</result>"
    )
    user = (
        f"RONDA: {round_num} de {max_rounds}\n"
        f"OBJETIVO GLOBAL:\n{objective}\n\n"
        f"CRITERIOS DE ÉXITO:\n{criteria_desc}\n\n"
        f"TU ASIGNACIÓN DEL ORQUESTADOR:\n{assignment}\n\n"
        f"CUADERNO COMPARTIDO (NOTES.md):\n{notes}\n\n"
        f"RESULTADOS COMPARTIDOS ACTUALES (results/):\n{results_digest}\n\n"
        f"Ejecuta tu tarea, deja una nota útil en NOTES.md y deposita tu resultado en results/."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_worker_output(
    raw: str,
    worker: dict[str, str],
    round_num: int,
    execution_enabled: bool,
) -> dict[str, Any]:
    js = _try_parse_json(raw)
    exec_blocks: list[str] = []
    result_files: list[tuple[str, str]] = []
    if js:
        note = str(js.get("note") or "").strip()
        code = str(js.get("code") or js.get("exec") or "").strip()
        if code and execution_enabled:
            exec_blocks.append(code)
        if js.get("result") or js.get("content"):
            fname = str(js.get("file") or js.get("filename") or f"r{round_num}_{worker['id']}.md")
            result_files.append((fname, str(js.get("result") or js.get("content"))))
    else:
        note = _extract_tag(raw, "note")
        if execution_enabled:
            for m in re.finditer(r"<exec[^>]*>(.*?)</exec>", raw, re.IGNORECASE | re.DOTALL):
                code = m.group(1).strip()
                # Si el modelo metió cercas markdown dentro de <exec>, las limpiamos.
                code = re.sub(r"^```(?:python|py)?\s*", "", code, flags=re.IGNORECASE)
                code = re.sub(r"\s*```$", "", code)
                if code:
                    exec_blocks.append(code)
            if not exec_blocks:
                # Respaldo: extraer bloques ```python ... ``` si no usó <exec>
                for m in re.finditer(r"```(?:python|py)\s*\n(.*?)```", raw, re.IGNORECASE | re.DOTALL):
                    code = m.group(1).strip()
                    if code:
                        exec_blocks.append(code)
        for m in re.finditer(
            r"<result(?:\s+[^>]*file\s*=\s*[\"']?([^\"'>]+)[\"']?)?[^>]*>(.*?)</result>",
            raw,
            re.IGNORECASE | re.DOTALL,
        ):
            fname = (m.group(1) or "").strip() or f"r{round_num}_{worker['id']}.md"
            body = m.group(2).strip()
            if body:
                result_files.append((fname, body))

    cleaned = _strip_protocol_tags(raw)
    if not result_files and cleaned:
        result_files.append((f"r{round_num}_{worker['id']}.md", cleaned))
    if not note:
        first_para = cleaned.split("\n\n")[0].strip() if cleaned else ""
        note = (first_para[:420] or f"Entrega completada por {worker['name']} en la ronda {round_num}.")

    # Limitar a máximo 3 bloques ejecutables por turno para mantener ritmo controlado.
    return {
        "note": note,
        "exec_blocks": exec_blocks[:3],
        "result_files": result_files[:4],
        "raw": raw,
    }


def _verifier_messages(
    *,
    objective: str,
    criteria: list[str],
    round_num: int,
    max_rounds: int,
    notes: str,
    results_digest: str,
    round_executions: list[dict[str, Any]],
) -> list[dict[str, str]]:
    criteria_desc = "\n".join(f"- {c}" for c in criteria) if criteria else "- El objetivo principal está resuelto con claridad y evidencia comprobable en results/."
    if round_executions:
        exec_lines = []
        for ex in round_executions:
            code_label = "OK (exit 0)" if ex["ok"] else f"ERROR (exit {ex['exit_code']})"
            stdout_preview = (ex.get("stdout") or "(vacío)")[:260]
            stderr_part = f" · stderr: {ex['stderr'][:200]!r}" if ex.get("stderr") else ""
            exec_lines.append(
                f"- {ex['script_file']}: {code_label} en {ex['duration_ms']} ms · stdout: {stdout_preview!r}{stderr_part}"
            )
        exec_summary = "\n".join(exec_lines)
    else:
        exec_summary = "(No hubo ejecuciones de código en esta ronda.)"

    system = (
        "[OBRADOR_ROLE:VERIFIER]\n"
        "Eres el VERIFICADOR INDEPENDIENTE de objetivos.\n"
        "Inspecciona críticamente las notas en NOTES.md, los archivos depositados en results/ y las "
        "salidas de ejecución del sandbox para determinar si el objetivo y sus criterios de éxito "
        "ya están verdaderamente cumplidos.\n"
        "No apruebes si faltan evidencias, hay errores sin corregir o el resultado está incompleto.\n"
        "Responde SIEMPRE en español usando exactamente estas etiquetas:\n"
        "<verdict>achieved | in_progress | blocked</verdict>\n"
        "<progress>Número entero de 0 a 100</progress>\n"
        "<summary>Evaluación objetiva del estado actual</summary>\n"
        "<criteria>\n"
        "- [PASS] Criterio cumplido: evidencia concreta\n"
        "- [FAIL] Criterio pendiente: qué falta\n"
        "</criteria>\n"
        "<feedback>Instrucción concreta para el orquestador en la siguiente ronda (o confirmación final)</feedback>"
    )
    user = (
        f"RONDA EVALUADA: {round_num} de {max_rounds}\n\n"
        f"OBJETIVO:\n{objective}\n\n"
        f"CRITERIOS DE ÉXITO:\n{criteria_desc}\n\n"
        f"EJECUCIONES DE ESTA RONDA:\n{exec_summary}\n\n"
        f"CUADERNO COMPARTIDO (NOTES.md):\n{notes}\n\n"
        f"RESULTADOS COMPARTIDOS (results/):\n{results_digest}\n\n"
        f"Emite tu dictamen de verificación para la ronda {round_num}."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_verifier_output(raw: str) -> dict[str, Any]:
    js = _try_parse_json(raw)
    criteria_items: list[dict[str, Any]] = []
    if js:
        verdict_raw = str(js.get("verdict") or ("achieved" if js.get("achieved") else "in_progress")).strip().lower()
        progress_raw = js.get("progress", js.get("progress_pct", 0))
        summary = str(js.get("summary") or "").strip()
        feedback = str(js.get("feedback") or js.get("next_guidance") or "").strip()
        for c in js.get("criteria") or []:
            if isinstance(c, dict):
                criteria_items.append({"passed": bool(c.get("passed")), "text": str(c.get("text") or c.get("criterion") or "")})
            elif isinstance(c, str):
                criteria_items.append({"passed": "pass" in c.lower() or "✓" in c, "text": c})
    else:
        verdict_raw = _extract_tag(raw, "verdict").strip().lower()
        progress_raw = _extract_tag(raw, "progress").strip()
        summary = _extract_tag(raw, "summary")
        feedback = _extract_tag(raw, "feedback")
        crit_block = _extract_tag(raw, "criteria")
        for line in crit_block.splitlines():
            clean = line.strip().lstrip("-*•").strip()
            if not clean:
                continue
            low = clean.lower()
            passed = low.startswith("[pass]") or low.startswith("[ok]") or low.startswith("[x]") or low.startswith("✓") or low.startswith("cumplido")
            text = re.sub(r"^\[(?:pass|fail|ok|x| )\]\s*", "", clean, flags=re.IGNORECASE).strip()
            criteria_items.append({"passed": passed, "text": text or clean})

    if any(token in verdict_raw for token in ("achieved", "cumplido", "complet", "done", "aprob", "pass", "sí", "si", "yes")):
        verdict = "achieved"
    elif any(token in verdict_raw for token in ("blocked", "bloque", "imposib")):
        verdict = "blocked"
    else:
        verdict = "in_progress"

    m_pct = re.search(r"\d+", str(progress_raw))
    if m_pct:
        progress = max(0, min(100, int(m_pct.group(0))))
    else:
        progress = 100 if verdict == "achieved" else 50
    if verdict == "achieved":
        progress = 100

    cleaned = _strip_protocol_tags(raw)
    if not summary:
        summary = cleaned[:500] or ("Objetivo verificado como cumplido." if verdict == "achieved" else "Se requieren ajustes adicionales.")
    if not feedback:
        feedback = "Objetivo completado satisfactoriamente." if verdict == "achieved" else "Completar los criterios pendientes y validar en el sandbox."

    return {
        "verdict": verdict,
        "achieved": verdict == "achieved",
        "progress": progress,
        "summary": summary,
        "criteria": criteria_items,
        "feedback": feedback,
    }


def _synthesizer_messages(
    *,
    objective: str,
    criteria: list[str],
    achieved: bool,
    round_num: int,
    notes: str,
    results_digest: str,
) -> list[dict[str, str]]:
    status_txt = "CUMPLIDO Y VERIFICADO" if achieved else f"LÍMITE DE RONDAS ALCANZADO ({round_num})"
    system = (
        "[OBRADOR_ROLE:SYNTHESIZER]\n"
        "Eres el ORQUESTADOR redactando el documento final `FINAL_RESULT.md`.\n"
        "Sintetiza de forma clara, estructurada y en español la respuesta final al objetivo "
        "apoyándote en las notas de `NOTES.md`, los resultados en `results/` y las ejecuciones "
        "verificadas. Indica los archivos clave producidos en `results/`."
    )
    user = (
        f"ESTADO FINAL: {status_txt}\n\n"
        f"OBJETIVO:\n{objective}\n\n"
        f"CRITERIOS:\n" + ("\n".join(f"- {c}" for c in criteria) if criteria else "- Cumplir el objetivo.") + "\n\n"
        f"CUADERNO COMPARTIDO (NOTES.md):\n{notes}\n\n"
        f"RESULTADOS COMPARTIDOS (results/):\n{results_digest}\n\n"
        "Redacta el documento final en Markdown limpio."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------
# Gestor de misiones en ejecución, ritmo y streaming SSE

@dataclass
class _RunControl:
    run_id: str
    task: asyncio.Task | None = None
    resume_event: asyncio.Event = field(default_factory=asyncio.Event)
    paused: bool = False
    step_once: bool = False
    cancelled: bool = False


class OrchestratorManager:
    def __init__(self) -> None:
        self._controls: dict[str, _RunControl] = {}
        self._subscribers: dict[str, set[asyncio.Queue]] = {}

    def workspace_for(self, run: dict[str, Any]) -> SharedWorkspace:
        return SharedWorkspace(Path(run["workspace_dir"]))

    def _broadcast(self, run_id: str, event_type: str, data: dict[str, Any]) -> None:
        queues = list(self._subscribers.get(run_id, ()))
        if not queues:
            return
        payload = {"type": event_type, "data": data}
        for queue in queues:
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    def _record_event(
        self,
        run_id: str,
        *,
        round_num: int,
        kind: str,
        agent_name: str,
        model_slug: str | None,
        title: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ev = db.add_orchestrator_event(
            run_id=run_id,
            round_num=round_num,
            kind=kind,
            agent_name=agent_name,
            model_slug=model_slug,
            title=title,
            content=content,
            metadata=metadata,
        )
        self._broadcast(run_id, "event", ev)
        return ev

    def _update_run(self, run_id: str, **fields: Any) -> dict[str, Any] | None:
        updated = db.update_orchestrator_run(run_id, **fields)
        if updated is not None:
            self._broadcast(run_id, "run", self._public_run_summary(updated))
        return updated

    def _public_run_summary(self, run: dict[str, Any]) -> dict[str, Any]:
        ctrl = self._controls.get(run["id"])
        active = bool(ctrl and ctrl.task and not ctrl.task.done())
        return {
            **run,
            "active": active,
            "paused": bool(ctrl.paused) if ctrl else run["status"] == "paused",
        }

    def list_runs(self, limit: int = 30) -> list[dict[str, Any]]:
        return [self._public_run_summary(r) for r in db.list_orchestrator_runs(limit=limit)]

    def get_run_detail(self, run_id: str) -> dict[str, Any]:
        run = db.get_orchestrator_run(run_id)
        if run is None:
            raise OrchestratorError(404, "Misión de orquestación no encontrada.")
        ws = self.workspace_for(run)
        events = db.list_orchestrator_events(run_id)
        return {
            "run": self._public_run_summary(run),
            "events": events,
            "notes": ws.read_notes_full(),
            "notes_path": str(ws.notes_path),
            "results": ws.list_results(),
        }

    def create_run(
        self,
        *,
        objective: str,
        criteria: list[str] | None = None,
        overrides: dict[str, Any] | None = None,
        start_immediately: bool = True,
    ) -> dict[str, Any]:
        clean_objective = (objective or "").strip()
        if not clean_objective:
            raise OrchestratorError(400, "Escribe un objetivo para iniciar el orquestador.")
        clean_criteria = [str(c).strip() for c in (criteria or []) if str(c).strip()][:12]
        base = get_config()["config"]
        merged = {**base, **{k: v for k, v in (overrides or {}).items() if v is not None}}
        validated = validate_config(merged, strict=True)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_id = f"orch-{stamp}-{secrets.token_hex(3)}"
        ws_root = config.orchestrator_dir() / run_id
        ws = SharedWorkspace(ws_root)
        ws.initialize(clean_objective, clean_criteria, validated)

        initial_status = "running" if start_immediately else "paused"
        initial_phase = "orchestrating" if start_immediately else "paused"
        run = db.create_orchestrator_run(
            run_id=run_id,
            objective=clean_objective,
            criteria=clean_criteria,
            run_config=validated,
            workspace_dir=str(ws.root),
            status=initial_status,
            phase=initial_phase,
        )
        self._record_event(
            run_id,
            round_num=0,
            kind="system",
            agent_name="Sistema",
            model_slug=None,
            title="Espacio compartido inicializado",
            content=(
                f"Cuaderno `NOTES.md` y carpeta compartida `results/` preparados.\n"
                f"Orquestador: `{validated['orchestrator_model']}` · "
                f"Verificador: `{validated['verifier_model']}` · "
                f"Agentes: {len(validated['workers'])} · "
                f"Sandbox: {'activo' if validated['execution_enabled'] else 'inactivo'}."
            ),
            metadata={"notes_path": str(ws.notes_path)},
        )

        if start_immediately:
            self._spawn_loop(run_id, paused=False, step_once=False)
        else:
            ctrl = _RunControl(run_id=run_id, paused=True)
            self._controls[run_id] = ctrl

        return self.get_run_detail(run_id)

    def _spawn_loop(self, run_id: str, *, paused: bool = False, step_once: bool = False) -> None:
        ctrl = self._controls.get(run_id)
        if ctrl and ctrl.task and not ctrl.task.done():
            ctrl.paused = paused
            ctrl.step_once = step_once
            if not paused:
                ctrl.resume_event.set()
            return
        ctrl = _RunControl(run_id=run_id, paused=paused, step_once=step_once)
        if not paused:
            ctrl.resume_event.set()
        loop = asyncio.get_running_loop()
        ctrl.task = loop.create_task(self._run_loop(run_id, ctrl))
        self._controls[run_id] = ctrl

    async def _wait_if_paused(self, run_id: str, ctrl: _RunControl) -> None:
        if not ctrl.paused:
            return
        self._update_run(run_id, status="paused", phase="paused")
        ctrl.resume_event.clear()
        await ctrl.resume_event.wait()
        if not ctrl.cancelled:
            self._update_run(run_id, status="running", phase="orchestrating")

    async def _sleep_paced(self, run_id: str, ctrl: _RunControl, seconds: float) -> None:
        if seconds <= 0:
            return
        self._update_run(run_id, status="running", phase="pacing")
        remaining = float(seconds)
        while remaining > 0:
            if ctrl.cancelled:
                return
            if ctrl.paused:
                await self._wait_if_paused(run_id, ctrl)
                return
            slice_s = min(0.2, remaining)
            await asyncio.sleep(slice_s)
            remaining -= slice_s

    async def _run_loop(self, run_id: str, ctrl: _RunControl) -> None:
        last_verdict: dict[str, Any] | None = None
        try:
            while not ctrl.cancelled:
                run = db.get_orchestrator_run(run_id)
                if run is None:
                    return
                if ctrl.paused:
                    await self._wait_if_paused(run_id, ctrl)
                    if ctrl.cancelled:
                        return
                    run = db.get_orchestrator_run(run_id)
                    if run is None:
                        return

                cfg = run["config"]
                ws = self.workspace_for(run)
                round_num = int(run["current_round"]) + 1
                max_rounds = int(run["max_rounds"])
                if round_num > max_rounds:
                    self._update_run(run_id, status="max_rounds", phase="done")
                    return

                objective = run["objective"]
                criteria = run["criteria"]
                workers = cfg["workers"]
                orch_model = cfg["orchestrator_model"]
                ver_model = cfg["verifier_model"]
                exec_enabled = bool(cfg.get("execution_enabled", True))
                exec_timeout = int(cfg.get("execution_timeout", 15))
                temp = float(cfg.get("temperature", 0.4))
                max_tokens = int(cfg.get("max_tokens", 768))

                # ------------------------------------------------ FASE 1: ORQUESTADOR
                self._update_run(run_id, status="running", phase="orchestrating", current_round=round_num)
                orch_msgs = _orchestrator_messages(
                    objective=objective,
                    criteria=criteria,
                    round_num=round_num,
                    max_rounds=max_rounds,
                    workers=workers,
                    notes=ws.read_notes(),
                    results_digest=ws.results_context(),
                    last_verdict=last_verdict,
                    execution_enabled=exec_enabled,
                )
                orch_raw = await call_model(orch_model, orch_msgs, temperature=temp, max_tokens=max_tokens)
                orch_parsed = parse_orchestrator_output(orch_raw, workers, objective, round_num)

                dispatch_lines = "\n".join(
                    f"- **{item['worker']['name']}** (`{item['worker']['model']}`): {item['task']}"
                    for item in orch_parsed["assignments"]
                )
                ws.append_note(
                    round_num=round_num,
                    agent="Orquestador",
                    role=f"ritmo:{orch_parsed['pace']}",
                    model=orch_model,
                    summary=orch_parsed["note"],
                    details=f"**Plan de ronda {round_num}:** {orch_parsed['plan']}\n\n**Asignaciones:**\n{dispatch_lines}",
                )
                self._record_event(
                    run_id,
                    round_num=round_num,
                    kind="orchestrator",
                    agent_name="Orquestador",
                    model_slug=orch_model,
                    title=f"Ronda {round_num} · Plan y ritmo ({orch_parsed['pace']})",
                    content=f"{orch_parsed['plan']}\n\n{dispatch_lines}",
                    metadata={
                        "pace": orch_parsed["pace"],
                        "note": orch_parsed["note"],
                        "assignments": [
                            {
                                "worker_id": a["worker"]["id"],
                                "worker_name": a["worker"]["name"],
                                "model": a["worker"]["model"],
                                "task": a["task"],
                            }
                            for a in orch_parsed["assignments"]
                        ],
                    },
                )
                self._broadcast(run_id, "workspace", {"notes": ws.read_notes_full(), "results": ws.list_results()})

                # ------------------------------------------------ FASE 2: WORKERS + SANDBOX
                round_executions: list[dict[str, Any]] = []
                for idx, item in enumerate(orch_parsed["assignments"], start=1):
                    if ctrl.cancelled:
                        return
                    if ctrl.paused:
                        await self._wait_if_paused(run_id, ctrl)
                        if ctrl.cancelled:
                            return

                    worker = item["worker"]
                    assignment = item["task"]
                    self._update_run(run_id, status="running", phase="working")
                    w_msgs = _worker_messages(
                        worker=worker,
                        objective=objective,
                        criteria=criteria,
                        round_num=round_num,
                        max_rounds=max_rounds,
                        assignment=assignment,
                        notes=ws.read_notes(),
                        results_digest=ws.results_context(),
                        execution_enabled=exec_enabled,
                    )
                    w_raw = await call_model(worker["model"], w_msgs, temperature=temp, max_tokens=max_tokens)
                    w_parsed = parse_worker_output(w_raw, worker, round_num, exec_enabled)

                    saved_refs: list[str] = []
                    worker_execs: list[dict[str, Any]] = []

                    # Ejecutar bloques de código en sandbox aislado
                    if exec_enabled and w_parsed["exec_blocks"]:
                        self._update_run(run_id, status="running", phase="executing")
                        for ex_i, code in enumerate(w_parsed["exec_blocks"], start=1):
                            if ctrl.cancelled:
                                return
                            step_tag = f"r{round_num}_{worker['id']}_e{ex_i}"
                            ex_res = await ws.execute_python_async(
                                code,
                                step_tag=step_tag,
                                timeout_sec=exec_timeout,
                                author=worker["name"],
                                round_num=round_num,
                            )
                            worker_execs.append(ex_res)
                            round_executions.append(ex_res)
                            saved_refs.append(ex_res["transcript_file"])
                            saved_refs.extend(ex_res["created_files"])
                            self._record_event(
                                run_id,
                                round_num=round_num,
                                kind="execution",
                                agent_name=worker["name"],
                                model_slug=worker["model"],
                                title=(
                                    f"Sandbox · {ex_res['script_file']} "
                                    f"({'OK' if ex_res['ok'] else 'ERROR exit ' + str(ex_res['exit_code'])}, "
                                    f"{ex_res['duration_ms']} ms)"
                                ),
                                content=ex_res["stdout"] or ex_res["stderr"] or "(sin salida en consola)",
                                metadata={
                                    **ex_res,
                                    "code": code,
                                },
                            )

                    # Guardar resultados declarados en results/
                    for fname, body in w_parsed["result_files"]:
                        target_name = sanitize_filename(fname, default=f"r{round_num}_{worker['id']}.md")
                        final_body = body
                        if worker_execs and "## Salida de ejecución" not in body:
                            exec_md = "\n\n".join(
                                f"### Ejecución `{ex['script_file']}` (exit {ex['exit_code']}, {ex['duration_ms']} ms)\n"
                                f"```text\n{(ex['stdout'] or ex['stderr'] or '(sin salida)').strip()}\n```"
                                for ex in worker_execs
                            )
                            final_body = f"{body.rstrip()}\n\n## Salida de ejecución\n\n{exec_md}\n"
                        entry = ws.save_result(
                            target_name,
                            final_body,
                            author=f"{worker['name']} ({worker['model']})",
                            round_num=round_num,
                            summary=w_parsed["note"][:180],
                        )
                        saved_refs.append(entry["filename"])

                    # Dedup manteniendo orden
                    unique_refs = list(dict.fromkeys(saved_refs))
                    exec_note_extra = ""
                    if worker_execs:
                        note_lines = []
                        for ex in worker_execs:
                            status_txt = "exit 0" if ex["ok"] else f"ERROR exit {ex['exit_code']}"
                            first_out = ex["stdout"].strip().splitlines()[0][:140] if ex["stdout"].strip() else ""
                            out_part = f" — salida: `{first_out}`" if first_out else ""
                            note_lines.append(f"- Ejecución `{ex['script_file']}`: {status_txt}{out_part}")
                        exec_note_extra = "\n".join(note_lines)
                    ws.append_note(
                        round_num=round_num,
                        agent=worker["name"],
                        role="agente de resultados",
                        model=worker["model"],
                        summary=w_parsed["note"],
                        details=exec_note_extra,
                        refs=unique_refs,
                    )
                    self._record_event(
                        run_id,
                        round_num=round_num,
                        kind="worker",
                        agent_name=worker["name"],
                        model_slug=worker["model"],
                        title=f"Ronda {round_num} · {worker['name']} ({idx}/{len(orch_parsed['assignments'])})",
                        content=w_parsed["note"],
                        metadata={
                            "worker_id": worker["id"],
                            "artifacts": unique_refs,
                            "executions_count": len(worker_execs),
                        },
                    )
                    self._broadcast(run_id, "workspace", {"notes": ws.read_notes_full(), "results": ws.list_results()})

                if ctrl.cancelled:
                    return

                # ------------------------------------------------ FASE 3: VERIFICADOR
                self._update_run(run_id, status="running", phase="verifying")
                v_msgs = _verifier_messages(
                    objective=objective,
                    criteria=criteria,
                    round_num=round_num,
                    max_rounds=max_rounds,
                    notes=ws.read_notes(),
                    results_digest=ws.results_context(),
                    round_executions=round_executions,
                )
                v_raw = await call_model(ver_model, v_msgs, temperature=min(temp, 0.3), max_tokens=max_tokens)
                verdict = parse_verifier_output(v_raw)
                last_verdict = verdict

                crit_md = "\n".join(
                    f"- [{'PASS' if c['passed'] else 'FAIL'}] {c['text']}" for c in verdict["criteria"]
                )
                ver_report = (
                    f"# Dictamen de Verificación — Ronda {round_num}\n\n"
                    f"- **Veredicto**: `{verdict['verdict']}`\n"
                    f"- **Progreso estimado**: {verdict['progress']}%\n"
                    f"- **Modelo verificador**: `{ver_model}`\n\n"
                    f"## Resumen\n{verdict['summary']}\n\n"
                    + (f"## Criterios evaluados\n{crit_md}\n\n" if crit_md else "")
                    + f"## Feedback para el orquestador\n{verdict['feedback']}\n"
                )
                ver_entry = ws.save_result(
                    f"verificacion_ronda_{round_num}.md",
                    ver_report,
                    author=f"Verificador ({ver_model})",
                    round_num=round_num,
                    summary=f"{verdict['verdict']} ({verdict['progress']}%) — {verdict['summary'][:120]}",
                )
                ws.append_note(
                    round_num=round_num,
                    agent="Verificador",
                    role=f"dictamen:{verdict['verdict']} ({verdict['progress']}%)",
                    model=ver_model,
                    summary=verdict["summary"],
                    details=(f"{crit_md}\n\n**Siguiente paso:** {verdict['feedback']}" if crit_md else f"**Siguiente paso:** {verdict['feedback']}"),
                    refs=[ver_entry["filename"]],
                )
                self._update_run(run_id, progress_pct=verdict["progress"])
                self._record_event(
                    run_id,
                    round_num=round_num,
                    kind="verifier",
                    agent_name="Verificador",
                    model_slug=ver_model,
                    title=(
                        f"Ronda {round_num} · Verificación: "
                        + ("OBJETIVO CUMPLIDO" if verdict["achieved"] else f"{verdict['progress']}% ({verdict['verdict']})")
                    ),
                    content=verdict["summary"],
                    metadata={
                        **verdict,
                        "report_file": ver_entry["filename"],
                    },
                )
                self._broadcast(run_id, "workspace", {"notes": ws.read_notes_full(), "results": ws.list_results()})

                # ------------------------------------------------ FASE 4: RITMO / CIERRE
                if verdict["achieved"] or round_num >= max_rounds:
                    self._update_run(run_id, status="running", phase="synthesizing")
                    synth_msgs = _synthesizer_messages(
                        objective=objective,
                        criteria=criteria,
                        achieved=verdict["achieved"],
                        round_num=round_num,
                        notes=ws.read_notes(),
                        results_digest=ws.results_context(),
                    )
                    try:
                        final_md = await call_model(orch_model, synth_msgs, temperature=min(temp, 0.3), max_tokens=max_tokens)
                    except Exception as exc:  # noqa: BLE001 - respaldo si falla síntesis
                        log.warning("síntesis final falló en %s: %s", run_id, exc)
                        final_md = f"# Resultado Final\n\n{verdict['summary']}\n\nConsulta `results/` y `NOTES.md` para los artefactos generados."
                    if not final_md.strip():
                        final_md = f"# Resultado Final\n\n{verdict['summary']}"
                    ws.save_result(
                        "FINAL_RESULT.md",
                        final_md,
                        author=f"Orquestador ({orch_model})",
                        round_num=round_num,
                        summary="Síntesis final del objetivo",
                        allow_final=True,
                    )
                    ws.append_note(
                        round_num=round_num,
                        agent="Orquestador",
                        role="cierre",
                        model=orch_model,
                        summary=(
                            "Objetivo cumplido y verificado. Resultado consolidado en FINAL_RESULT.md."
                            if verdict["achieved"]
                            else f"Límite de {max_rounds} rondas alcanzado. Estado consolidado en FINAL_RESULT.md."
                        ),
                        refs=["FINAL_RESULT.md"],
                    )
                    final_status = "completed" if verdict["achieved"] else "max_rounds"
                    final_progress = 100 if verdict["achieved"] else verdict["progress"]
                    self._update_run(
                        run_id,
                        status=final_status,
                        phase="done",
                        progress_pct=final_progress,
                        final_result=final_md,
                    )
                    self._record_event(
                        run_id,
                        round_num=round_num,
                        kind="final",
                        agent_name="Orquestador",
                        model_slug=orch_model,
                        title=(
                            "Objetivo cumplido y consolidado"
                            if verdict["achieved"]
                            else f"Límite de {max_rounds} rondas alcanzado"
                        ),
                        content=final_md,
                        metadata={"achieved": verdict["achieved"], "file": "FINAL_RESULT.md"},
                    )
                    self._broadcast(run_id, "workspace", {"notes": ws.read_notes_full(), "results": ws.list_results()})
                    return

                # Pausa tras 1 ronda en modo paso a paso o si el usuario pidió avanzar 1 ronda
                current_cfg = (db.get_orchestrator_run(run_id) or {}).get("config", cfg)
                if current_cfg.get("pacing_mode") == "step" or ctrl.step_once:
                    ctrl.step_once = False
                    ctrl.paused = True
                    self._record_event(
                        run_id,
                        round_num=round_num,
                        kind="system",
                        agent_name="Orquestador",
                        model_slug=orch_model,
                        title=f"Ronda {round_num} pausada (control de ritmo)",
                        content="Esperando orden para iniciar la siguiente ronda. Puedes dejar una nota en NOTES.md antes de continuar.",
                        metadata={"paused": True},
                    )
                    await self._wait_if_paused(run_id, ctrl)
                else:
                    await self._sleep_paced(run_id, ctrl, float(current_cfg.get("pace_seconds", 1.5)))

        except asyncio.CancelledError:
            if ctrl.cancelled:
                self._update_run(run_id, status="stopped", phase="done")
            else:
                self._update_run(run_id, status="paused", phase="paused")
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("error en orquestación %s", run_id)
            message = str(exc) or "Error inesperado durante la orquestación."
            self._update_run(run_id, status="error", phase="error", error=message[:1000])
            self._record_event(
                run_id,
                round_num=(db.get_orchestrator_run(run_id) or {}).get("current_round", 0),
                kind="system",
                agent_name="Sistema",
                model_slug=None,
                title="Error en la orquestación",
                content=message,
                metadata={"error": True},
            )
        finally:
            self._broadcast(run_id, "done", {"run_id": run_id})

    # ------------------------------------------------------------------ Controles de ritmo

    def pause(self, run_id: str) -> dict[str, Any]:
        run = db.get_orchestrator_run(run_id)
        if run is None:
            raise OrchestratorError(404, "Misión no encontrada.")
        ctrl = self._controls.get(run_id)
        if ctrl and ctrl.task and not ctrl.task.done():
            ctrl.paused = True
            ctrl.resume_event.clear()
        self._update_run(run_id, status="paused", phase="paused")
        return self.get_run_detail(run_id)

    def resume(
        self,
        run_id: str,
        *,
        extra_rounds: int = 0,
        pacing_mode: str | None = None,
    ) -> dict[str, Any]:
        run = db.get_orchestrator_run(run_id)
        if run is None:
            raise OrchestratorError(404, "Misión no encontrada.")
        cfg = dict(run["config"])
        max_rounds = int(run["max_rounds"])
        if extra_rounds > 0:
            max_rounds = min(20, max_rounds + int(extra_rounds))
            cfg["max_rounds"] = max_rounds
        elif run["current_round"] >= max_rounds:
            max_rounds = min(20, max_rounds + 2)
            cfg["max_rounds"] = max_rounds
        if pacing_mode in ("auto", "step"):
            cfg["pacing_mode"] = pacing_mode
        self._update_run(
            run_id,
            status="running",
            phase="orchestrating",
            max_rounds=max_rounds,
            config=cfg,
            error=None,
        )
        ctrl = self._controls.get(run_id)
        if ctrl and ctrl.task and not ctrl.task.done():
            ctrl.cancelled = False
            ctrl.paused = False
            ctrl.step_once = False
            ctrl.resume_event.set()
        else:
            self._spawn_loop(run_id, paused=False, step_once=False)
        return self.get_run_detail(run_id)

    def step(self, run_id: str) -> dict[str, Any]:
        run = db.get_orchestrator_run(run_id)
        if run is None:
            raise OrchestratorError(404, "Misión no encontrada.")
        max_rounds = int(run["max_rounds"])
        cfg = dict(run["config"])
        if run["current_round"] >= max_rounds:
            max_rounds = min(20, max_rounds + 1)
            cfg["max_rounds"] = max_rounds
        self._update_run(
            run_id,
            status="running",
            phase="orchestrating",
            max_rounds=max_rounds,
            config=cfg,
            error=None,
        )
        ctrl = self._controls.get(run_id)
        if ctrl and ctrl.task and not ctrl.task.done():
            ctrl.cancelled = False
            ctrl.step_once = True
            ctrl.paused = False
            ctrl.resume_event.set()
        else:
            self._spawn_loop(run_id, paused=False, step_once=True)
        return self.get_run_detail(run_id)

    async def stop(self, run_id: str) -> dict[str, Any]:
        run = db.get_orchestrator_run(run_id)
        if run is None:
            raise OrchestratorError(404, "Misión no encontrada.")
        ctrl = self._controls.get(run_id)
        if ctrl:
            ctrl.cancelled = True
            ctrl.paused = False
            ctrl.resume_event.set()
            if ctrl.task and not ctrl.task.done():
                ctrl.task.cancel()
                try:
                    await ctrl.task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
        self._update_run(run_id, status="stopped", phase="done")
        self._record_event(
            run_id,
            round_num=int(run["current_round"]),
            kind="system",
            agent_name="Usuario",
            model_slug=None,
            title="Orquestación detenida",
            content="La misión fue detenida manualmente. Las notas en NOTES.md y archivos en results/ se conservan.",
        )
        return self.get_run_detail(run_id)

    async def delete_run(self, run_id: str) -> bool:
        run = db.get_orchestrator_run(run_id)
        if run is None:
            return False
        ctrl = self._controls.pop(run_id, None)
        if ctrl:
            ctrl.cancelled = True
            ctrl.resume_event.set()
            if ctrl.task and not ctrl.task.done():
                ctrl.task.cancel()
                try:
                    await ctrl.task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
        db.delete_orchestrator_run(run_id)
        self._broadcast(run_id, "done", {"run_id": run_id, "deleted": True})
        ws_dir = Path(run["workspace_dir"])
        if ws_dir.exists():
            shutil.rmtree(ws_dir, ignore_errors=True)
        return True

    def stop_all(self) -> None:
        for run_id, ctrl in list(self._controls.items()):
            ctrl.cancelled = False
            if ctrl.task and not ctrl.task.done():
                ctrl.task.cancel()
            run = db.get_orchestrator_run(run_id)
            if run and run["status"] == "running":
                db.update_orchestrator_run(run_id, status="paused", phase="paused")
        self._controls.clear()

    # ------------------------------------------------------------------ Notas y archivos compartidos

    def add_human_note(self, run_id: str, text: str, author: str = "Operador") -> dict[str, Any]:
        run = db.get_orchestrator_run(run_id)
        if run is None:
            raise OrchestratorError(404, "Misión no encontrada.")
        clean = (text or "").strip()
        if not clean:
            raise OrchestratorError(400, "La nota no puede estar vacía.")
        ws = self.workspace_for(run)
        ws.append_note(
            round_num=int(run["current_round"]),
            agent=(author or "Operador").strip()[:40],
            role="nota humana",
            model=None,
            summary=clean,
        )
        ev = self._record_event(
            run_id,
            round_num=int(run["current_round"]),
            kind="human_note",
            agent_name=(author or "Operador").strip()[:40],
            model_slug=None,
            title="Nota añadida a NOTES.md",
            content=clean,
        )
        self._broadcast(run_id, "workspace", {"notes": ws.read_notes_full(), "results": ws.list_results()})
        return {"event": ev, "notes": ws.read_notes_full()}

    def deposit_shared_file(self, run_id: str, filename: str, content: str, summary: str = "") -> dict[str, Any]:
        run = db.get_orchestrator_run(run_id)
        if run is None:
            raise OrchestratorError(404, "Misión no encontrada.")
        ws = self.workspace_for(run)
        safe = sanitize_filename(filename, default="archivo_usuario.txt")
        entry = ws.save_result(
            safe,
            content,
            author="Operador",
            round_num=int(run["current_round"]),
            summary=summary or "Archivo depositado por el usuario",
        )
        ws.append_note(
            round_num=int(run["current_round"]),
            agent="Operador",
            role="archivo compartido",
            model=None,
            summary=f"Se depositó el archivo `results/{entry['filename']}` para uso del equipo.",
            refs=[entry["filename"]],
        )
        self._record_event(
            run_id,
            round_num=int(run["current_round"]),
            kind="human_note",
            agent_name="Operador",
            model_slug=None,
            title=f"Archivo depositado: results/{entry['filename']}",
            content=summary or f"Archivo `{entry['filename']}` ({entry['size_bytes']} B) disponible en results/.",
            metadata={"file": entry["filename"]},
        )
        self._broadcast(run_id, "workspace", {"notes": ws.read_notes_full(), "results": ws.list_results()})
        return {"file": entry, "results": ws.list_results(), "notes": ws.read_notes_full()}

    def read_file(self, run_id: str, filename: str) -> dict[str, Any]:
        run = db.get_orchestrator_run(run_id)
        if run is None:
            raise OrchestratorError(404, "Misión no encontrada.")
        ws = self.workspace_for(run)
        try:
            content, meta = ws.read_result(filename)
        except ValueError as exc:
            raise OrchestratorError(400, str(exc)) from exc
        except FileNotFoundError as exc:
            raise OrchestratorError(404, str(exc)) from exc
        return {"content": content, **meta}

    # ------------------------------------------------------------------ Streaming SSE

    async def stream_events(self, run_id: str, after_id: int = 0) -> AsyncGenerator[bytes, None]:
        run = db.get_orchestrator_run(run_id)
        if run is None:
            raise OrchestratorError(404, "Misión no encontrada.")
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        subs = self._subscribers.setdefault(run_id, set())
        subs.add(queue)
        try:
            ws = self.workspace_for(run)
            summary = self._public_run_summary(run)
            snapshot = {
                "type": "snapshot",
                "data": {
                    "run": summary,
                    "events": db.list_orchestrator_events(run_id, after_id=after_id),
                    "notes": ws.read_notes_full(),
                    "results": ws.list_results(),
                },
            }
            yield f"data: {json.dumps(snapshot, ensure_ascii=False)}\n\n".encode("utf-8")

            if summary["status"] in ("completed", "max_rounds", "stopped", "error") and not summary["active"]:
                yield b"data: [DONE]\n\n"
                return

            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n".encode("utf-8")
                    if msg.get("type") == "done":
                        current = db.get_orchestrator_run(run_id)
                        if current is None or current["status"] in ("completed", "max_rounds", "stopped", "error"):
                            yield b"data: [DONE]\n\n"
                            return
                except asyncio.TimeoutError:
                    yield b": keepalive\n\n"
                    current = db.get_orchestrator_run(run_id)
                    ctrl = self._controls.get(run_id)
                    if current is None or (
                        current["status"] in ("completed", "max_rounds", "stopped", "error")
                        and not (ctrl and ctrl.task and not ctrl.task.done())
                    ):
                        yield b"data: [DONE]\n\n"
                        return
        finally:
            subs.discard(queue)
            if not subs:
                self._subscribers.pop(run_id, None)


manager = OrchestratorManager()
