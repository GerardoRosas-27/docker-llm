"""Búsqueda de modelos GGUF públicos en Hugging Face."""

from __future__ import annotations

import threading
import time
from urllib.parse import quote

import httpx

from app.config import hf_token

_CACHE_SECONDS = 120
_cache: dict[str, tuple[float, list[dict]]] = {}
_cache_lock = threading.Lock()
_USER_AGENT = "obrador-llm/1.0"


class HFError(RuntimeError):
    pass


def _headers() -> dict[str, str]:
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
    token = hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _client() -> httpx.Client:
    return httpx.Client(timeout=30, follow_redirects=True, headers=_headers())


def search_models(query: str, limit: int = 20) -> list[dict]:
    params: dict[str, str] = {
        "filter": "gguf",
        "sort": "downloads",
        "direction": "-1",
        "limit": str(max(1, min(limit, 30))),
    }
    if query.strip():
        params["search"] = query.strip()
    try:
        with _client() as client:
            response = client.get("https://huggingface.co/api/models", params=params)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        raise HFError(f"No se pudo buscar en Hugging Face: {exc}") from exc
    results = []
    for item in payload:
        repo_id = item.get("id") or item.get("modelId")
        if not repo_id or item.get("private"):
            continue
        results.append(
            {
                "repo_id": repo_id,
                "downloads": item.get("downloads") or 0,
                "likes": item.get("likes") or 0,
                "pipeline_tag": item.get("pipeline_tag"),
                "gated": bool(item.get("gated")),
            }
        )
    return results


def _real_size(item: dict) -> int | None:
    lfs = item.get("lfs") or {}
    candidates = [item.get("size") or 0, lfs.get("size") or 0]
    size = max(int(value) for value in candidates)
    return size or None


def repo_gguf_files(repo_id: str) -> list[dict]:
    now = time.time()
    with _cache_lock:
        cached = _cache.get(repo_id)
        if cached and now - cached[0] < _CACHE_SECONDS:
            return cached[1]
    url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
    try:
        with _client() as client:
            response = client.get(url)
    except httpx.HTTPError as exc:
        raise HFError(f"No se pudo leer el repositorio: {exc}") from exc
    if response.status_code == 404:
        raise HFError("Repositorio no encontrado en Hugging Face.")
    if response.status_code == 401:
        raise HFError("Ese modelo pide una cuenta o un token de Hugging Face.")
    try:
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        raise HFError(f"No se pudo leer el repositorio: {exc}") from exc
    files = []
    for item in payload:
        if item.get("type") != "file":
            continue
        path = item.get("path") or ""
        if not path.lower().endswith(".gguf"):
            continue
        lfs = item.get("lfs") or {}
        files.append(
            {
                "filename": path,
                "size_bytes": _real_size(item),
                "sha256": (lfs.get("oid") or "").lower() or None,
            }
        )
    files.sort(key=lambda entry: (entry["size_bytes"] or 0, entry["filename"]))
    with _cache_lock:
        _cache[repo_id] = (time.time(), files)
    return files


def file_size(repo_id: str, filename: str) -> int | None:
    for entry in repo_gguf_files(repo_id):
        if entry["filename"] == filename:
            return entry["size_bytes"]
    return None


def file_sha256(repo_id: str, filename: str) -> str | None:
    """sha256 que Hugging Face publica para el archivo LFS (lfs.oid)."""
    for entry in repo_gguf_files(repo_id):
        if entry["filename"] == filename:
            return entry.get("sha256")
    return None


def head_size(repo_id: str, filename: str) -> int | None:
    url = f"https://huggingface.co/{repo_id}/resolve/main/{quote(filename, safe='/')}"
    try:
        with _client() as client:
            response = client.head(url)
    except httpx.HTTPError:
        return None
    if response.status_code >= 400:
        return None
    length = response.headers.get("content-length")
    if not length:
        return None
    try:
        return int(length)
    except ValueError:
        return None


def model_overview(repo_id: str) -> dict:
    """Ficha corta del repositorio, sin la plantilla de chat ni los pesos."""
    url = f"https://huggingface.co/api/models/{repo_id}"
    try:
        with _client() as client:
            response = client.get(url)
    except httpx.HTTPError as exc:
        raise HFError(f"No se pudo leer el modelo: {exc}") from exc
    if response.status_code == 404:
        raise HFError("Repositorio no encontrado en Hugging Face.")
    if response.status_code == 401:
        raise HFError("Ese modelo pide una cuenta o un token de Hugging Face.")
    try:
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as exc:
        raise HFError(f"No se pudo leer el modelo: {exc}") from exc
    card = payload.get("cardData") or {}
    tags = payload.get("tags") or []
    license_name = card.get("license")
    if not isinstance(license_name, str):
        license_name = None
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("license:"):
                license_name = tag.split(":", 1)[1]
                break
    base_model = card.get("base_model")
    if isinstance(base_model, list):
        base_model = ", ".join(str(item) for item in base_model[:3])
    elif not isinstance(base_model, str):
        base_model = None
    summary = ""
    for key in ("description", "summary", "model_name"):
        value = card.get(key)
        if isinstance(value, str) and value.strip():
            summary = value.strip()
            break
    return {
        "repo_id": payload.get("id") or repo_id,
        "author": payload.get("author") or repo_id.split("/", 1)[0],
        "downloads": payload.get("downloads") or 0,
        "likes": payload.get("likes") or 0,
        "pipeline_tag": payload.get("pipeline_tag"),
        "gated": bool(payload.get("gated")),
        "last_modified": payload.get("lastModified"),
        "license": license_name,
        "base_model": base_model,
        "library": card.get("library_name") or payload.get("library_name"),
        "summary": summary[:700],
    }


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()
