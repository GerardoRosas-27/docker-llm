"""Descarga GGUF al volumen, con tope de tamaño y reanudación."""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
from pathlib import Path
from urllib.parse import quote

import httpx

from app import config, db
from app.hfclient import HFError
from app.policy import PolicyError, format_bytes, max_model_bytes, require_allowed

log = logging.getLogger("obrador.downloader")

_USER_AGENT = "obrador-llm/1.0"


class DownloadCancelled(Exception):
    pass


class Downloader:
    def __init__(self) -> None:
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._pending: set[str] = set()
        self._cancels: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, name="downloader", daemon=True)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(None)

    def enqueue(self, slug: str) -> None:
        with self._lock:
            if slug in self._pending:
                return
            self._pending.add(slug)
            self._cancels[slug] = threading.Event()
            self._queue.put(slug)

    def cancel(self, slug: str) -> None:
        event = self._cancels.get(slug)
        if event is not None:
            event.set()

    def resume_queued(self) -> None:
        for model in db.list_models():
            if model["status"] in ("queued", "downloading"):
                db.mark_queued(model["slug"])
                self.enqueue(model["slug"])

    def _loop(self) -> None:
        while True:
            slug = self._queue.get()
            if slug is None:
                return
            try:
                self._download(slug)
            except Exception as exc:  # noqa: BLE001 - el hilo no puede morir
                log.exception("descarga %s", slug)
                if db.get(slug):
                    db.mark_error(slug, str(exc) or "Error de descarga")
            finally:
                with self._lock:
                    self._pending.discard(slug)
                    self._cancels.pop(slug, None)

    def _download(self, slug: str) -> None:
        model = db.get(slug)
        if model is None or model["status"] == "ready":
            return
        try:
            # El tope se aplica antes de abrir la conexión, no a mitad del archivo.
            require_allowed(model.get("size_bytes"))
        except PolicyError as exc:
            if db.get(slug):
                db.mark_error(slug, str(exc))
            return
        cancel = self._cancels.get(slug) or threading.Event()
        db.mark_downloading(slug)
        dest_dir = config.models_dir() / slug
        dest_dir.mkdir(parents=True, exist_ok=True)
        parts = _parts(model)
        grand = int(model["size_bytes"] or 0)
        done = 0
        first_final: Path | None = None
        partial: Path | None = None
        try:
            for name in parts:
                final = dest_dir / name
                final.parent.mkdir(parents=True, exist_ok=True)
                partial = Path(str(final) + ".partial")
                self._stream(model, name, partial, cancel, done, grand)
                if cancel.is_set() or db.get(slug) is None:
                    partial.unlink(missing_ok=True)
                    return
                self._assert_gguf(partial)
                self._verify_sha256(model, name, partial, cancel)
                partial.replace(final)
                done += final.stat().st_size
                if done > max_model_bytes():
                    raise PolicyError(
                        f"La descarga superó el límite de {format_bytes(max_model_bytes())} y se detuvo."
                    )
                if first_final is None:
                    first_final = final
                if db.get(slug):
                    db.mark_progress(slug, done, grand or done)
            if first_final is None or db.get(slug) is None:
                return
            require_allowed(done)
            db.mark_ready(slug, str(first_final), done)
            log.info("listo %s (%s bytes, %s archivos)", slug, done, len(parts))
            _preload_if_default(model, slug)
        except DownloadCancelled:
            if partial is not None:
                partial.unlink(missing_ok=True)
            if db.get(slug):
                db.mark_error(slug, "Descarga cancelada.")
        except PolicyError as exc:
            if partial is not None:
                partial.unlink(missing_ok=True)
            if db.get(slug):
                db.mark_error(slug, str(exc))
        except Exception as exc:
            if db.get(slug):
                db.mark_error(slug, str(exc) or "Error de descarga")
            raise

    def _stream(
        self,
        model: dict,
        filename: str,
        partial: Path,
        cancel: threading.Event,
        base_done: int,
        grand_total: int,
    ) -> None:
        url = (
            f"https://huggingface.co/{model['repo_id']}/resolve/main/"
            f"{quote(filename, safe='/')}"
        )
        headers = {"User-Agent": _USER_AGENT}
        token = config.hf_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        already = partial.stat().st_size if partial.exists() else 0
        if already:
            headers["Range"] = f"bytes={already}-"
        timeout = httpx.Timeout(30.0, read=120.0)
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            with client.stream("GET", url, headers=headers) as response:
                if response.status_code == 401:
                    raise HFError("Ese modelo pide una cuenta o un token de Hugging Face (HF_TOKEN).")
                if response.status_code == 404:
                    raise HFError("El archivo no existe en Hugging Face.")
                if response.status_code == 416 and partial.exists():
                    return
                if response.status_code not in (200, 206):
                    raise HFError(f"Hugging Face respondió {response.status_code}.")
                if response.status_code == 200:
                    already = 0
                    mode = "wb"
                else:
                    mode = "ab"
                part_total = _total_size(response, already)
                if base_done == 0 and part_total is not None:
                    require_allowed(part_total)
                last_write = 0
                downloaded = already
                with partial.open(mode) as handle:
                    for chunk in response.iter_bytes(256 * 1024):
                        if cancel.is_set():
                            raise DownloadCancelled()
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if base_done + downloaded > max_model_bytes():
                            raise PolicyError(
                                f"La descarga superó el límite de {format_bytes(max_model_bytes())} y se detuvo."
                            )
                        if downloaded - last_write >= 8 * 1024 * 1024:
                            last_write = downloaded
                            if db.get(model["slug"]):
                                db.mark_progress(
                                    model["slug"],
                                    base_done + downloaded,
                                    grand_total or (base_done + downloaded),
                                )
                if db.get(model["slug"]):
                    db.mark_progress(
                        model["slug"],
                        base_done + downloaded,
                        grand_total or (base_done + downloaded),
                    )

    def _verify_sha256(self, model: dict, name: str, path: Path, cancel: threading.Event) -> None:
        expected = _expected_sha256(model["repo_id"], name)
        if not expected:
            log.info("sin sha256 publicado para %s; solo se comprueba la cabecera GGUF", name)
            return
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                if cancel.is_set():
                    raise DownloadCancelled()
                block = handle.read(4 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected:
            path.unlink(missing_ok=True)
            raise RuntimeError(
                f"{name} no coincide con el sha256 publicado ({actual[:12]}… en vez de "
                f"{expected[:12]}…). Se borró el archivo; vuelve a descargarlo."
            )
        log.info("sha256 correcto para %s", name)

    def _assert_gguf(self, path: Path) -> None:
        with path.open("rb") as handle:
            magic = handle.read(4)
        if magic != b"GGUF":
            path.unlink(missing_ok=True)
            raise RuntimeError("El archivo descargado no es un GGUF válido.")


def _expected_sha256(repo_id: str, name: str) -> str | None:
    if repo_id == config.default_repo() and name == config.default_file():
        known = config.default_sha256()
        if known:
            return known
    try:
        from app import hfclient

        return hfclient.file_sha256(repo_id, name)
    except Exception as exc:  # noqa: BLE001 - sin red no se bloquea la descarga
        log.warning("no se pudo leer el sha256 de %s: %s", name, exc)
        return None


def _preload_if_default(model: dict, slug: str) -> None:
    """Carga en memoria el modelo por defecto en cuanto termina de bajar."""
    if not config.preload_default():
        return
    if model["repo_id"] != config.default_repo() or model["filename"] != config.default_file():
        return
    from app.runner import runner

    runner.schedule_threadsafe(slug)


def _parts(model: dict) -> list[str]:
    raw = model.get("parts")
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list) and parsed:
            return [str(item) for item in parsed]
    return [model["filename"]]


def _total_size(response: httpx.Response, already: int) -> int | None:
    content_range = response.headers.get("content-range")
    if content_range and "/" in content_range:
        total_text = content_range.rsplit("/", 1)[-1]
        if total_text.isdigit():
            return int(total_text)
    length = response.headers.get("content-length")
    if length and length.isdigit():
        return already + int(length) if response.status_code == 206 else int(length)
    return None


service = Downloader()
