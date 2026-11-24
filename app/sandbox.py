"""Espacio de trabajo compartido y entorno de ejecución aislado para el orquestador.

Cada misión recibe su propia carpeta bajo `<DATA_DIR>/orchestrator/<run_id>/`:
- `NOTES.md`: fichero compartido de notas donde cada agente (orquestador, workers,
  verificador y el propio usuario) deja observaciones y lee las de los demás.
- `results/`: carpeta común donde cada agente deposita resultados, artefactos y
  salidas de ejecución para que otros agentes puedan leerlos y construir sobre ellos.
- `workspace/`: directorio de trabajo aislado donde se ejecuta código Python con
  límites de tiempo/recursos y sin acceso a variables sensibles del servidor.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import resource as _resource
except ImportError:  # pragma: no cover - sistemas no POSIX
    _resource = None

_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_OUTPUT_CHARS = 24_000
_MAX_EXEC_FILE_BYTES = 10 * 1024 * 1024  # 10 MiB por archivo creado en el sandbox
_MAX_EXEC_MEM_BYTES = 768 * 1024 * 1024  # 768 MiB espacio virtual para Python


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def sanitize_filename(name: str, default: str = "resultado.md") -> str:
    """Convierte un nombre propuesto por un agente en un nombre de archivo seguro."""
    raw = (name or "").strip().replace("\\", "/")
    raw = raw.lstrip("/")
    if not raw or "\x00" in raw:
        return default
    # Permitir como mucho un subdirectorio simple o nombre plano sin '..'.
    parts = [part for part in raw.split("/") if part not in ("", ".")]
    cleaned: list[str] = []
    for part in parts:
        if part == ".." or part.startswith("."):
            continue
        safe = _SAFE_SEGMENT.sub("_", part).strip("._")
        if safe:
            cleaned.append(safe[:80])
    if not cleaned:
        return default
    # Aplanar a nombre único en results/ si hay demasiada profundidad.
    if len(cleaned) > 2:
        cleaned = ["_".join(cleaned[:-1])[:40], cleaned[-1]]
    result = "/".join(cleaned)
    if result in (".manifest.json", "INDEX.md"):
        return default
    return result


def resolve_inside(base: Path, rel: str) -> Path:
    """Resuelve `rel` dentro de `base` rechazando cualquier escape de ruta."""
    if not rel or "\x00" in rel:
        raise ValueError("Nombre de archivo inválido.")
    cleaned = rel.replace("\\", "/").lstrip("/")
    if cleaned.startswith("..") or "/../" in f"/{cleaned}/" or cleaned.endswith("/.."):
        raise ValueError("Ruta fuera del espacio compartido.")
    base_resolved = base.resolve()
    target = (base_resolved / cleaned).resolve()
    if not target.is_relative_to(base_resolved) or target == base_resolved:
        raise ValueError("Ruta fuera del espacio compartido.")
    return target


def _sandbox_env(workspace_dir: Path, results_dir: Path, notes_path: Path, tmp_dir: Path) -> dict[str, str]:
    """Entorno mínimo limpio para el subproceso: nunca hereda MASTER_SECRET ni HF_TOKEN."""
    return {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": str(workspace_dir),
        "TMPDIR": str(tmp_dir),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "ORCH_WORKSPACE": str(workspace_dir),
        "ORCH_RESULTS_DIR": str(results_dir),
        "ORCH_NOTES_FILE": str(notes_path),
    }


def _make_preexec(timeout_sec: int):
    def _apply() -> None:
        if os.name != "nt":
            os.setsid()
        if _resource is None:
            return
        cpu_limit = max(2, int(timeout_sec) + 2)
        for kind, limits in (
            (_resource.RLIMIT_CPU, (cpu_limit, cpu_limit)),
            (_resource.RLIMIT_FSIZE, (_MAX_EXEC_FILE_BYTES, _MAX_EXEC_FILE_BYTES)),
            (_resource.RLIMIT_AS, (_MAX_EXEC_MEM_BYTES, _MAX_EXEC_MEM_BYTES)),
        ):
            try:
                soft, hard = _resource.getrlimit(kind)
                new_hard = limits[1] if hard == _resource.RLIM_INFINITY else min(hard, limits[1])
                new_soft = min(limits[0], new_hard)
                _resource.setrlimit(kind, (new_soft, new_hard))
            except (OSError, ValueError):
                pass

    return _apply


def _kill_process_group(pid: int | None) -> None:
    if not pid:
        return
    try:
        if os.name != "nt":
            os.killpg(pid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGTERM)
    except OSError:
        pass


def _truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    extra = len(text) - limit
    return f"{text[:limit]}\n\n[... salida truncada: {extra} caracteres omitidos ...]"


class SharedWorkspace:
    """Gestiona NOTES.md, results/ y el sandbox de ejecución de una misión."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.notes_path = self.root / "NOTES.md"
        self.results_dir = self.root / "results"
        self.workspace_dir = self.root / "workspace"
        self.tmp_dir = self.workspace_dir / "tmp"
        self.manifest_path = self.results_dir / ".manifest.json"
        self._lock = threading.Lock()

    def ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def initialize(self, objective: str, criteria: list[str], run_config: dict[str, Any]) -> None:
        with self._lock:
            self.ensure_layout()
            workers = run_config.get("workers") or []
            worker_lines = "\n".join(
                f"- **{w.get('name') or w.get('id')}** (`{w.get('model')}`): {w.get('role') or 'Agente de resultados'}"
                for w in workers
            ) or "- *(sin agentes configurados)*"
            criteria_lines = "\n".join(f"- [ ] {item}" for item in criteria) if criteria else "- [ ] Cumplir el objetivo de forma completa y verificable."
            exec_label = "Habilitado (Python sandbox)" if run_config.get("execution_enabled", True) else "Deshabilitado"
            pacing_label = (
                "Paso a paso (manual)"
                if run_config.get("pacing_mode") == "step"
                else f"Automático (cadencia {run_config.get('pace_seconds', 1.5)} s)"
            )
            header = (
                f"# Cuaderno Compartido de Agentes (`NOTES.md`)\n\n"
                f"> Creado: {_now_iso()}\n\n"
                f"## Objetivo\n{objective.strip()}\n\n"
                f"## Criterios de verificación\n{criteria_lines}\n\n"
                f"## Equipo y configuración\n"
                f"- **Orquestador (ritmo y dirección)**: `{run_config.get('orchestrator_model')}`\n"
                f"- **Verificador (criterio de cumplimiento)**: `{run_config.get('verifier_model')}`\n"
                f"- **Ritmo**: {pacing_label} · Máx. rondas: {run_config.get('max_rounds')}\n"
                f"- **Entorno de ejecución**: {exec_label}\n"
                f"- **Carpeta de resultados compartidos**: `results/`\n\n"
                f"### Agentes de resultados\n{worker_lines}\n\n"
                f"---\n\n"
                f"## Registro de notas compartidas\n"
            )
            self.notes_path.write_text(header, encoding="utf-8")
            self._save_manifest({})
            self._rebuild_index({})

    # ------------------------------------------------------------------ NOTES.md

    def append_note(
        self,
        *,
        round_num: int,
        agent: str,
        role: str,
        model: str | None,
        summary: str,
        details: str = "",
        refs: list[str] | None = None,
    ) -> str:
        with self._lock:
            self.ensure_layout()
            model_tag = f" · modelo `{model}`" if model else ""
            round_tag = f"Ronda {round_num}" if round_num > 0 else "Inicial"
            parts = [
                f"\n### [{round_tag}] {agent} ({role}{model_tag}) — {_now_iso()}\n",
                f"{summary.strip()}\n",
            ]
            if details.strip():
                parts.append(f"\n{details.strip()}\n")
            if refs:
                ref_list = ", ".join(f"`results/{r}`" for r in refs)
                parts.append(f"\n**Archivos depositados / referenciados:** {ref_list}\n")
            block = "".join(parts)
            with self.notes_path.open("a", encoding="utf-8") as handle:
                handle.write(block)
            return block.strip()

    def read_notes_full(self) -> str:
        with self._lock:
            if not self.notes_path.is_file():
                return ""
            return self.notes_path.read_text(encoding="utf-8", errors="replace")

    def read_notes(self, max_chars: int = 10_000) -> str:
        text = self.read_notes_full()
        if len(text) <= max_chars:
            return text
        head_budget = min(2200, max_chars // 3)
        tail_budget = max_chars - head_budget - 80
        return (
            f"{text[:head_budget].rstrip()}\n\n"
            f"[... notas intermedias omitidas para caber en el contexto ...]\n\n"
            f"{text[-tail_budget:].lstrip()}"
        )

    # ------------------------------------------------------------------ results/

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        if not self.manifest_path.is_file():
            return {}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_manifest(self, manifest: dict[str, dict[str, Any]]) -> None:
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _rebuild_index(self, manifest: dict[str, dict[str, Any]]) -> None:
        lines = [
            "# Índice de Resultados Compartidos (`results/`)",
            "",
            f"Actualizado: {_now_iso()}",
            "",
        ]
        entries = sorted(
            manifest.items(),
            key=lambda item: (
                0 if item[0] == "FINAL_RESULT.md" else 1,
                item[1].get("round", 0),
                item[0],
            ),
        )
        if not entries:
            lines.append("_Todavía no hay archivos depositados en `results/`._")
        else:
            lines.append("| Archivo | Ronda | Autor | Tamaño | Resumen |")
            lines.append("| --- | --- | --- | --- | --- |")
            for fname, meta in entries:
                summary = str(meta.get("summary") or "").replace("|", "/").replace("\n", " ")[:120]
                size = int(meta.get("size_bytes") or 0)
                lines.append(
                    f"| `{fname}` | {meta.get('round', 0)} | {meta.get('author') or 'agente'} | {size} B | {summary} |"
                )
        (self.results_dir / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def save_result(
        self,
        filename: str,
        content: str,
        *,
        author: str,
        round_num: int,
        summary: str = "",
        allow_final: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            self.ensure_layout()
            if allow_final and filename == "FINAL_RESULT.md":
                safe = "FINAL_RESULT.md"
            else:
                safe = sanitize_filename(filename, default=f"resultado_r{round_num}.md")
            target = resolve_inside(self.results_dir, safe)
            target.parent.mkdir(parents=True, exist_ok=True)
            encoded = content.encode("utf-8")
            target.write_bytes(encoded)
            manifest = self._load_manifest()
            entry = {
                "filename": safe,
                "author": author,
                "round": int(round_num),
                "summary": (summary or "").strip()[:240],
                "size_bytes": len(encoded),
                "updated_at": _now_iso(),
            }
            manifest[safe] = entry
            self._save_manifest(manifest)
            self._rebuild_index(manifest)
            return entry

    def _snapshot_results(self) -> dict[str, tuple[int, int]]:
        snap: dict[str, tuple[int, int]] = {}
        if not self.results_dir.is_dir():
            return snap
        for path in self.results_dir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(self.results_dir).as_posix()
            if rel in (".manifest.json", "INDEX.md") or rel.startswith("."):
                continue
            stat = path.stat()
            snap[rel] = (stat.st_mtime_ns, stat.st_size)
        return snap

    def list_results(self) -> list[dict[str, Any]]:
        with self._lock:
            self.ensure_layout()
            manifest = self._load_manifest()
            changed = False
            found_files: set[str] = set()
            for path in sorted(self.results_dir.rglob("*")):
                if not path.is_file():
                    continue
                rel = path.relative_to(self.results_dir).as_posix()
                if rel == ".manifest.json" or any(part.startswith(".") for part in path.relative_to(self.results_dir).parts):
                    continue
                found_files.add(rel)
                if rel == "INDEX.md":
                    continue
                stat = path.stat()
                if rel not in manifest or manifest[rel].get("size_bytes") != stat.st_size:
                    existing = manifest.get(rel, {})
                    manifest[rel] = {
                        "filename": rel,
                        "author": existing.get("author") or "sandbox",
                        "round": existing.get("round", 0),
                        "summary": existing.get("summary") or "Archivo depositado en results/",
                        "size_bytes": stat.st_size,
                        "updated_at": existing.get("updated_at") or _now_iso(),
                    }
                    changed = True
            for stale in list(manifest):
                if stale not in found_files:
                    manifest.pop(stale, None)
                    changed = True
            if changed:
                self._save_manifest(manifest)
                self._rebuild_index(manifest)

            items = list(manifest.values())
            index_path = self.results_dir / "INDEX.md"
            if index_path.is_file():
                items.append(
                    {
                        "filename": "INDEX.md",
                        "author": "sistema",
                        "round": 0,
                        "summary": "Índice automático de resultados compartidos",
                        "size_bytes": index_path.stat().st_size,
                        "updated_at": _now_iso(),
                    }
                )
            items.sort(
                key=lambda item: (
                    0 if item["filename"] == "FINAL_RESULT.md" else (1 if item["filename"] == "INDEX.md" else 2),
                    -int(item.get("round") or 0),
                    item["filename"],
                )
            )
            return items

    def read_result(self, filename: str, max_chars: int | None = None) -> tuple[str, dict[str, Any]]:
        if filename == "NOTES.md":
            content = self.read_notes_full()
            if max_chars is not None and len(content) > max_chars:
                content = _truncate(content, max_chars)
            return content, {
                "filename": "NOTES.md",
                "size_bytes": len(content.encode("utf-8")),
                "author": "compartido",
                "round": 0,
            }
        target = resolve_inside(self.results_dir, filename)
        if not target.is_file():
            raise FileNotFoundError(f"No existe el archivo '{filename}' en results/.")
        raw = target.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        if max_chars is not None and len(text) > max_chars:
            text = _truncate(text, max_chars)
        manifest = self._load_manifest()
        rel = target.relative_to(self.results_dir).as_posix()
        meta = manifest.get(
            rel,
            {
                "filename": rel,
                "author": "sistema" if rel == "INDEX.md" else "sandbox",
                "round": 0,
                "size_bytes": len(raw),
            },
        )
        meta["size_bytes"] = len(raw)
        return text, meta

    def results_context(self, max_total_chars: int = 6500) -> str:
        """Resumen compacto de los resultados existentes para inyectar a los agentes."""
        items = [item for item in self.list_results() if item["filename"] != "INDEX.md"]
        if not items:
            return "(Todavía no hay archivos en results/. Puedes crear el primero.)"
        lines = ["Archivos actuales en `results/`:"]
        for item in items:
            lines.append(
                f"- `{item['filename']}` (ronda {item.get('round', 0)}, autor: {item.get('author')}, "
                f"{item.get('size_bytes', 0)} B): {item.get('summary') or ''}"
            )
        header = "\n".join(lines)
        remaining = max_total_chars - len(header) - 200
        if remaining <= 200:
            return header
        excerpts: list[str] = []
        per_file = max(400, min(1400, remaining // max(1, min(len(items), 6))))
        used = 0
        for item in items[:8]:
            if used >= remaining:
                break
            try:
                content, _meta = self.read_result(item["filename"], max_chars=per_file)
            except OSError:
                continue
            block = f"\n--- Contenido de `results/{item['filename']}` ---\n{content.strip()}\n"
            if used + len(block) > remaining and excerpts:
                break
            excerpts.append(block)
            used += len(block)
        return header + "\n" + "".join(excerpts)

    # ------------------------------------------------------------------ Sandbox

    def _prepare_script(self, code: str, step_tag: str) -> tuple[str, Path, dict[str, tuple[int, int]]]:
        self.ensure_layout()
        safe_tag = sanitize_filename(step_tag, default="step").replace("/", "_")
        if not safe_tag.endswith(".py"):
            script_name = f"{safe_tag}.py"
        else:
            script_name = safe_tag
            safe_tag = safe_tag[:-3]
        script_path = resolve_inside(self.workspace_dir, script_name)
        script_path.write_text(code, encoding="utf-8")
        before = self._snapshot_results()
        return safe_tag, script_path, before

    def _finalize_execution(
        self,
        *,
        safe_tag: str,
        script_name: str,
        before: dict[str, tuple[int, int]],
        exit_code: int,
        timed_out: bool,
        duration_ms: int,
        stdout: str,
        stderr: str,
        author: str,
        round_num: int,
    ) -> dict[str, Any]:
        stdout_clean = _truncate(stdout)
        stderr_clean = _truncate(stderr)
        after = self._snapshot_results()
        created_or_updated = sorted(
            rel for rel, info in after.items() if rel not in before or before[rel] != info
        )
        with self._lock:
            manifest = self._load_manifest()
            for rel in created_or_updated:
                path = self.results_dir / rel
                if path.is_file():
                    manifest[rel] = {
                        "filename": rel,
                        "author": author or "sandbox",
                        "round": int(round_num),
                        "summary": f"Generado por ejecución {script_name} (exit {exit_code})",
                        "size_bytes": path.stat().st_size,
                        "updated_at": _now_iso(),
                    }
            self._save_manifest(manifest)
            self._rebuild_index(manifest)

        transcript_name = f"exec_{safe_tag}.log"
        transcript_body = (
            f"# Ejecución Sandbox: {script_name}\n"
            f"Autor: {author or 'agente'} | Ronda: {round_num} | Código de salida: {exit_code} | "
            f"Duración: {duration_ms} ms | Timeout: {'sí' if timed_out else 'no'}\n"
            f"Archivos creados/actualizados en results/: {', '.join(created_or_updated) or '(ninguno)'}\n\n"
            f"=== STDOUT ===\n{stdout_clean or '(vacío)'}\n\n"
            f"=== STDERR ===\n{stderr_clean or '(vacío)'}\n"
        )
        self.save_result(
            transcript_name,
            transcript_body,
            author=f"{author or 'agente'} (sandbox)",
            round_num=round_num,
            summary=f"Salida de ejecución Python (exit {exit_code}, {duration_ms} ms)",
        )
        return {
            "ok": exit_code == 0 and not timed_out,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "duration_ms": duration_ms,
            "stdout": stdout_clean,
            "stderr": stderr_clean,
            "script_file": script_name,
            "created_files": created_or_updated,
            "transcript_file": transcript_name,
        }

    def execute_python(
        self,
        code: str,
        *,
        step_tag: str,
        timeout_sec: int = 15,
        author: str = "",
        round_num: int = 0,
    ) -> dict[str, Any]:
        """Ejecuta código Python de forma síncrona en el sandbox aislado."""
        timeout_sec = max(2, min(60, int(timeout_sec)))
        safe_tag, script_path, before = self._prepare_script(code, step_tag)
        env = _sandbox_env(self.workspace_dir, self.results_dir, self.notes_path, self.tmp_dir)
        started = time.monotonic()
        timed_out = False
        exit_code = -1
        stdout = ""
        stderr = ""
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(self.workspace_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=_make_preexec(timeout_sec) if os.name != "nt" else None,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_sec)
            exit_code = int(proc.returncode or 0)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_group(proc.pid)
            try:
                stdout, stderr = proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            stderr = (stderr + f"\nTiempo máximo de ejecución superado ({timeout_sec} s).").strip()
            exit_code = 124
        duration_ms = int((time.monotonic() - started) * 1000)
        return self._finalize_execution(
            safe_tag=safe_tag,
            script_name=script_path.name,
            before=before,
            exit_code=exit_code,
            timed_out=timed_out,
            duration_ms=duration_ms,
            stdout=stdout,
            stderr=stderr,
            author=author,
            round_num=round_num,
        )

    async def execute_python_async(
        self,
        code: str,
        *,
        step_tag: str,
        timeout_sec: int = 15,
        author: str = "",
        round_num: int = 0,
    ) -> dict[str, Any]:
        """Ejecuta código Python asíncronamente, matando el grupo de procesos si se cancela."""
        timeout_sec = max(2, min(60, int(timeout_sec)))
        safe_tag, script_path, before = self._prepare_script(code, step_tag)
        env = _sandbox_env(self.workspace_dir, self.results_dir, self.notes_path, self.tmp_dir)
        started = time.monotonic()
        timed_out = False
        exit_code = -1
        stdout_b = b""
        stderr_b = b""
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script_path),
            cwd=str(self.workspace_dir),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=_make_preexec(timeout_sec) if os.name != "nt" else None,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
            exit_code = int(proc.returncode or 0)
        except asyncio.TimeoutError:
            timed_out = True
            _kill_process_group(proc.pid)
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=2)
            except asyncio.TimeoutError:
                stdout_b, stderr_b = b"", b""
            exit_code = 124
        except asyncio.CancelledError:
            _kill_process_group(proc.pid)
            raise
        duration_ms = int((time.monotonic() - started) * 1000)
        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        if timed_out:
            stderr = (stderr + f"\nTiempo máximo de ejecución superado ({timeout_sec} s).").strip()
        return self._finalize_execution(
            safe_tag=safe_tag,
            script_name=script_path.name,
            before=before,
            exit_code=exit_code,
            timed_out=timed_out,
            duration_ms=duration_ms,
            stdout=stdout,
            stderr=stderr,
            author=author,
            round_num=round_num,
        )
