"""Arranca un llama-server por modelo y lo deja en un puerto interno."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from app import config

log = logging.getLogger("obrador.runner")


@dataclass
class Running:
    slug: str
    port: int
    proc: subprocess.Popen
    started: float
    healthy: bool = False
    logs: deque = field(default_factory=lambda: deque(maxlen=400))


class Runner:
    def __init__(self) -> None:
        self._running: dict[str, Running] = {}
        self._errors: dict[str, str] = {}
        self._pending: set[str] = set()
        self._tasks: dict[str, asyncio.Task] = {}
        self._help_cache: dict[str, str] = {}
        self._binary: str | None = None

    def resolve(self) -> str | None:
        if self._binary and Path(self._binary).exists():
            return self._binary
        configured = config.llama_server_bin()
        if configured and Path(configured).is_file():
            self._binary = configured
            return configured
        found = shutil.which(configured) or shutil.which("llama-server")
        self._binary = found
        return found

    def public_status(self, slug: str) -> dict:
        running = self._running.get(slug)
        if running and running.proc.poll() is None:
            state = "running" if running.healthy else "starting"
            return {
                "status": state,
                "port": running.port,
                "pid": running.proc.pid,
                "log_tail": list(running.logs)[-30:],
                "error": None,
            }
        if slug in self._pending:
            return {"status": "starting", "port": None, "pid": None, "log_tail": [], "error": None}
        error = self._errors.get(slug)
        tail: list[str] = []
        if running is not None:
            tail = list(running.logs)[-30:]
        return {
            "status": "error" if error else "stopped",
            "port": None,
            "pid": None,
            "log_tail": tail,
            "error": error,
        }

    def schedule(self, slug: str) -> asyncio.Task:
        """Encola la carga. Se llama desde el event loop, sin await interno."""
        current = self._running.get(slug)
        if current and current.proc.poll() is None and current.healthy:
            task = self._tasks.get(slug)
            if task is not None and not task.done():
                return task
            return asyncio.get_running_loop().create_task(asyncio.sleep(0, result=current))
        self._pending.add(slug)
        task = self._tasks.get(slug)
        if task is not None and not task.done():
            return task
        task = asyncio.get_running_loop().create_task(self._wrap(slug))
        task.add_done_callback(_consume_task_error)
        self._tasks[slug] = task
        return task

    async def ensure(self, slug: str) -> Running:
        current = self._running.get(slug)
        if current and current.proc.poll() is None and current.healthy:
            return current
        return await self.schedule(slug)

    async def _wrap(self, slug: str) -> Running:
        try:
            return await self._start(slug)
        finally:
            self._pending.discard(slug)

    async def stop(self, slug: str) -> None:
        running = self._running.pop(slug, None)
        self._pending.discard(slug)
        if running is None:
            return
        _terminate(running.proc)
        self._errors.pop(slug, None)

    def stop_all(self) -> None:
        for slug in list(self._running):
            running = self._running.pop(slug, None)
            if running is not None:
                _terminate(running.proc)

    async def _start(self, slug: str) -> Running:
        from app import db

        current = self._running.get(slug)
        if current and current.proc.poll() is None and current.healthy:
            return current
        model = db.get(slug)
        if model is None or model["status"] != "ready" or not model.get("local_path"):
            self._errors[slug] = "El modelo todavía no está descargado."
            raise RuntimeError(self._errors[slug])
        path = Path(model["local_path"])
        if not path.is_file():
            self._errors[slug] = "Falta el archivo GGUF en el volumen."
            raise RuntimeError(self._errors[slug])
        binary = self.resolve()
        if not binary:
            self._errors[slug] = "llama-server no está disponible en este contenedor."
            raise RuntimeError(self._errors[slug])
        while len([item for item in self._running.values() if item.proc.poll() is None]) >= config.max_loaded_models():
            victim = min(self._running.values(), key=lambda item: item.started)
            if victim.slug == slug:
                break
            log.info("descargando %s de memoria para cargar %s", victim.slug, slug)
            await self.stop(victim.slug)
        port = self._allocate_port()
        cmd = self._command(binary, path, port, slug)
        log.info("arrancando %s", " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=os.name != "nt",
        )
        running = Running(slug=slug, port=port, proc=proc, started=time.time())
        running.logs.append("$ " + " ".join(cmd))
        self._running[slug] = running
        self._errors.pop(slug, None)
        threading.Thread(target=self._pump, args=(slug, proc), daemon=True).start()
        try:
            await self._wait_healthy(running)
        except Exception as exc:
            message = str(exc) or "No se pudo cargar el modelo."
            useful = [line for line in running.logs if "unused tensor" not in line]
            tail = "\n".join(useful[-8:])
            if tail:
                message = f"{message}\n{tail}"
            self._errors[slug] = message[:1200]
            self._running.pop(slug, None)
            _terminate(proc)
            raise RuntimeError(self._errors[slug]) from exc
        running.healthy = True
        return running

    def _allocate_port(self) -> int:
        used = {item.port for item in self._running.values()}
        for port in range(18080, 18280):
            if port not in used:
                return port
        raise RuntimeError("No hay puertos internos libres para otro modelo.")

    def _command(self, binary: str, model_path: Path, port: int, alias: str) -> list[str]:
        help_text = self._help(binary)
        cmd = [binary, "-m", str(model_path), "--host", "127.0.0.1", "--port", str(port)]
        if _has(help_text, "--ctx-size"):
            cmd.extend(["-c", str(config.ctx_size())])
        if _has(help_text, "--alias"):
            cmd.extend(["-a", alias])
        if _has(help_text, "--jinja"):
            cmd.append("--jinja")
        if _has(help_text, "--no-webui"):
            cmd.append("--no-webui")
        if _has(help_text, "--no-mmproj"):
            cmd.append("--no-mmproj")
        if _has(help_text, "--parallel"):
            cmd.extend(["-np", "1"])
        # El valor por defecto de llama-server es "auto" y puede reservar la GPU
        # o ampliar el contexto. 0 deja el modelo en CPU.
        if _has(help_text, "--n-gpu-layers"):
            cmd.extend(["--n-gpu-layers", str(config.n_gpu_layers())])
        if _has(help_text, "--fit"):
            cmd.extend(["--fit", "off"])
        if _has(help_text, "--batch-size"):
            cmd.extend(["-b", "512", "-ub", "128"])
        if _has(help_text, "--cache-type-k"):
            cmd.extend(["-ctk", "q8_0", "-ctv", "q8_0"])
        threads = config.llama_threads()
        if threads and _has(help_text, "--threads"):
            cmd.extend(["-t", str(threads)])
        return cmd

    def _help(self, binary: str) -> str:
        cached = self._help_cache.get(binary)
        if cached is not None:
            return cached
        try:
            result = subprocess.run(
                [binary, "--help"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            text = (result.stdout or "") + "\n" + (result.stderr or "")
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.warning("llama-server --help falló: %s", exc)
            text = "--ctx-size --alias --jinja --no-webui --no-mmproj --parallel --threads --n-gpu-layers --host --port"
        self._help_cache[binary] = text
        return text

    async def _wait_healthy(self, running: Running) -> None:
        deadline = time.time() + config.load_timeout()
        url_health = f"http://127.0.0.1:{running.port}/health"
        url_models = f"http://127.0.0.1:{running.port}/v1/models"
        async with httpx.AsyncClient(timeout=3) as client:
            while time.time() < deadline:
                code = running.proc.poll()
                if code is not None:
                    raise RuntimeError(f"llama-server terminó al cargar el modelo (código {code}).")
                for url in (url_health, url_models):
                    try:
                        response = await client.get(url)
                    except httpx.HTTPError:
                        continue
                    if response.status_code == 200:
                        return
                await asyncio.sleep(1)
        raise RuntimeError("El modelo no respondió a tiempo. Revisa la RAM libre del equipo.")

    def _pump(self, slug: str, proc: subprocess.Popen) -> None:
        stream = proc.stdout
        if stream is None:
            return
        for line in stream:
            running = self._running.get(slug)
            if running and running.proc is proc:
                running.logs.append(line.rstrip())
        try:
            code = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            code = proc.poll()
        running = self._running.get(slug)
        if running and running.proc is proc:
            self._errors[slug] = f"llama-server terminó (código {code})."
            self._running.pop(slug, None)


def _consume_task_error(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    task.exception()


def _has(help_text: str, flag: str) -> bool:
    return flag in help_text


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=8)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name != "nt":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except OSError:
            pass


runner = Runner()
