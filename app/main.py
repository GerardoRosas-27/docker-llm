"""API de administración y API compatible con OpenAI, una ruta por modelo."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import access, config, db, hfclient, resources
from app.downloader import service
from app.policy import (
    PolicyError,
    evaluate,
    format_bytes,
    group_gguf_entries,
    max_model_bytes,
    quant_label,
    non_model_reason,
    split_info,
    split_load_reason,
    validate_filename,
    validate_repo,
)
from app.runner import runner

log = logging.getLogger("obrador")
STATIC = Path(__file__).resolve().parent / "static"
_OPENAI_FIELDS = {
    "model",
    "messages",
    "prompt",
    "temperature",
    "top_p",
    "max_tokens",
    "stream",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "n",
    "user",
    "chat_template_kwargs",
    "response_format",
}


class DownloadRequest(BaseModel):
    repo_id: str
    filename: str


class GenerateRequest(BaseModel):
    which: str = "api"
    label: str | None = None
    expires_in_days: float | None = None


class LoginRequest(BaseModel):
    secret: str = ""


class ApiKeyRequest(BaseModel):
    label: str = ""
    expires_in_days: float | None = None


def _bootstrap_key(repo: str, filename: str) -> str:
    # Una marca por modelo por defecto: al cambiar de modelo (9B -> 2B) los
    # volúmenes que ya existían también reciben el nuevo sin tocar nada.
    return f"bootstrap:{repo}/{filename}"


def _preload_default() -> None:
    if not config.preload_default():
        return
    model = db.find(config.default_repo(), config.default_file())
    if model and model["status"] == "ready" and not _load_block_reason(model):
        runner.schedule(model["slug"])


def _bootstrap() -> None:
    service.resume_queued()
    if not config.auto_download():
        return
    repo = config.default_repo()
    filename = config.default_file()
    key = _bootstrap_key(repo, filename)
    if db.meta_get(key) == "1":
        return
    if db.find(repo, filename):
        db.meta_set(key, "1")
        return
    try:
        repo = validate_repo(repo)
        filename = validate_filename(filename)
        size = hfclient.file_size(repo, filename)
        if size is None:
            size = hfclient.head_size(repo, filename)
        allowed, reason = evaluate(size)
        if not allowed:
            log.error("modelo inicial rechazado: %s", reason)
            db.meta_set(key, "1")
            return
    except Exception as exc:  # noqa: BLE001 - se reintenta en el próximo arranque
        log.warning("el modelo inicial no se encoló todavía: %s", exc)
        return
    record = db.upsert_download(repo, filename, int(size))
    if record["status"] == "queued":
        service.enqueue(record["slug"])
    db.meta_set(key, "1")
    log.info("descarga inicial encolada: %s", record["slug"])


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config.ensure_dirs()
    db.init()
    runner.bind_loop(asyncio.get_running_loop())
    service.start()
    _bootstrap()
    _preload_default()
    yield
    service.stop()
    runner.stop_all()


app = FastAPI(
    title="Obrador",
    summary="Panel local para buscar, descargar y servir modelos GGUF.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(PolicyError)
async def _policy_error(_request: Request, exc: PolicyError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(access.AuthError)
async def _auth_error(_request: Request, exc: access.AuthError) -> JSONResponse:
    headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
    return JSONResponse(
        status_code=exc.status,
        content={"detail": exc.message, "code": exc.code},
        headers=headers,
    )


@app.exception_handler(hfclient.HFError)
async def _hf_error(_request: Request, exc: hfclient.HFError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


def _bearer(request: Request) -> str:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return ""


def _credentials(request: Request) -> list[str]:
    values = [request.headers.get("x-admin-token", "").strip(), _bearer(request)]
    return [value for value in values if value]


def _client_ip(request: Request) -> str:
    # Railway añade la IP real al final de X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[-1].strip() or "?"
    return request.client.host if request.client else "?"


def _require_admin(request: Request) -> None:
    if access.admin_ok(_credentials(request)):
        return
    raise HTTPException(
        status_code=401,
        detail="Sesión de administración inválida o caducada. Entra con el secreto maestro en Acceso.",
    )


def _require_api_key(request: Request) -> JSONResponse | None:
    if access.api_ok(_credentials(request)):
        return None
    return JSONResponse(
        status_code=401,
        content={"error": {"message": "API key inválida, revocada o caducada.", "type": "invalid_request_error"}},
    )


def _base(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-proto")
    host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    if forwarded and host:
        return f"{forwarded.split(',')[0].strip()}://{host.split(',')[0].strip()}"
    return str(request.base_url).rstrip("/")


def _present(model: dict, request: Request) -> dict:
    slug = model["slug"]
    base = _base(request)
    runtime = runner.public_status(slug)
    size = model.get("size_bytes")
    blocked = _load_block_reason(model)
    return {
        **model,
        "quant": quant_label(model["filename"]),
        "size_label": format_bytes(size),
        "blocked": blocked,
        "ram_warning": _ram_reason(model),
        "is_default": model["repo_id"] == config.default_repo()
        and model["filename"] == config.default_file(),
        "runtime": runtime,
        "api": {
            "model_id": slug,
            "openai_base": f"{base}/v1",
            "chat": f"{base}/v1/models/{slug}/chat/completions",
            "completions": f"{base}/v1/models/{slug}/completions",
        },
    }


def _oai_error(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": "invalid_request_error"}},
    )


def _bundle(repo_id: str, filename: str) -> tuple[str, list[str], int]:
    """Resuelve un archivo suelto o todas las partes de un GGUF dividido.

    El tamaño total se comprueba antes de devolver la lista, así no empieza
    la descarga de un conjunto que pasa de 9 GB.
    """
    filename = validate_filename(filename)
    accessory = non_model_reason(filename)
    if accessory:
        raise PolicyError(accessory)
    info = split_info(filename)
    if not info:
        size = hfclient.file_size(repo_id, filename)
        if size is None:
            size = hfclient.head_size(repo_id, filename)
        _ensure_fits(size)
        return filename, [filename], int(size)
    published = {item["filename"]: item.get("size_bytes") for item in hfclient.repo_gguf_files(repo_id)}
    by_base = {name.split("/")[-1].lower(): (name, size) for name, size in published.items()}
    parts: list[str] = []
    sizes: list[int] = []
    for index in range(1, info["total"] + 1):
        expected = f"{info['stem']}-{index:05d}-of-{info['total']:05d}.gguf".lower()
        found = by_base.get(expected)
        if not found or found[1] is None:
            raise PolicyError(
                f"El modelo está partido en {info['total']} archivos y falta la parte {index}."
            )
        parts.append(found[0])
        sizes.append(int(found[1]))
    total = sum(sizes)
    _ensure_fits(total)
    return parts[0], parts, total


def _ensure_fits(size: int | None) -> None:
    allowed, reason = evaluate(size)
    if not allowed:
        raise PolicyError(reason)
    assert size is not None
    free = shutil.disk_usage(config.data_dir()).free
    if free < size + 512 * 1024 * 1024:
        raise PolicyError(
            f"Quedan {format_bytes(free)} libres y el archivo pesa {format_bytes(size)}."
        )


def _ram_reason(model: dict) -> str | None:
    if not config.ram_check():
        return None
    return resources.ram_block_reason(model.get("size_bytes"), config.llama_repack())


def _load_block_reason(model: dict) -> str | None:
    names = None
    info = split_info(model["filename"])
    local_path = model.get("local_path")
    if info and info["index"] == 1 and local_path:
        directory = Path(local_path).parent
        if directory.is_dir():
            names = [item.name for item in directory.iterdir()]
    return split_load_reason(model["filename"], names)


@app.get("/api/access")
@app.get("/api/auth/status")
def access_status(request: Request) -> dict:
    return access.status(_credentials(request))


@app.post("/api/auth/login")
def auth_login(body: LoginRequest, request: Request) -> dict:
    # def normal (no async): el pequeño retardo tras un fallo no bloquea el event loop.
    return access.login(body.secret, _client_ip(request))


@app.post("/api/auth/logout")
def auth_logout(request: Request) -> dict:
    closed = any(access.logout(value) for value in _credentials(request))
    return {"logged_out": closed}


def _require_master_admin(request: Request) -> None:
    if not access.master_configured():
        raise access.AuthError(
            409,
            "Para crear API keys hace falta MASTER_SECRET en las variables de entorno del servidor.",
            "master_secret_not_configured",
        )
    _require_admin(request)
    if not access.admin_required():  # nunca debería pasar con secreto maestro
        raise HTTPException(status_code=401, detail="Hace falta una sesión de administración.")


@app.get("/api/keys")
def keys_list(request: Request) -> dict:
    _require_master_admin(request)
    return {"keys": access.list_api_keys()}


@app.post("/api/keys")
def keys_create(body: ApiKeyRequest, request: Request) -> dict:
    _require_master_admin(request)
    if body.expires_in_days is not None and not (0 < body.expires_in_days <= 3650):
        raise HTTPException(status_code=400, detail="La caducidad va de 1 a 3650 días.")
    return access.create_api_key(body.label, body.expires_in_days)


@app.delete("/api/keys/{key_id}")
def keys_revoke(key_id: str, request: Request) -> dict:
    _require_master_admin(request)
    if not access.revoke_api_key(key_id):
        raise HTTPException(status_code=404, detail="API key no encontrada o ya revocada.")
    return {"revoked": key_id}


@app.post("/api/access/generate")
def access_generate(body: GenerateRequest, request: Request) -> dict:
    """Ruta antigua del panel. Ahora solo crea API keys derivadas y exige sesión."""
    _require_master_admin(request)
    if body.which not in ("api", "both"):
        raise HTTPException(
            status_code=400,
            detail="El acceso de administración ya no se genera: entra con el secreto maestro.",
        )
    created = access.create_api_key(body.label or "panel", body.expires_in_days)
    return {"api_key": created["key"], "id": created["id"], "expires_at": created["expires_at"]}


@app.get("/health")
def health() -> dict:
    models = db.list_models()
    running = 0
    for model in models:
        if runner.public_status(model["slug"])["status"] == "running":
            running += 1
    return {
        "status": "ok",
        "auth": {
            "mode": access.mode(),
            "master_secret": access.master_configured(),
            "warnings": access.warnings(),
        },
        "llama_server": runner.resolve() is not None,
        "models_ready": sum(1 for model in models if model["status"] == "ready"),
        "models_running": running,
        "max_model_bytes": max_model_bytes(),
    }


@app.get("/api/system")
def system(request: Request) -> dict:
    _require_admin(request)
    usage = shutil.disk_usage(config.data_dir())
    ram = resources.memory_limit()
    return {
        "ram_limit": ram,
        "ram_limit_label": format_bytes(ram) if ram else "desconocida",
        "cpu_limit": resources.cpu_limit(),
        "generation_timeout": config.generation_timeout(),
        "stream_idle_timeout": config.stream_idle_timeout(),
        "load_timeout": config.load_timeout(),
        "max_model_bytes": max_model_bytes(),
        "max_model_label": format_bytes(max_model_bytes()),
        "disk_free": usage.free,
        "disk_free_label": format_bytes(usage.free),
        "llama_server": runner.resolve() is not None,
        "ctx_size": config.ctx_size(),
        "max_loaded_models": config.max_loaded_models(),
        "api_key_required": access.api_required(),
        "admin_token_required": access.admin_required(),
        "auth_mode": access.mode(),
        "auth_warnings": access.warnings(),
        "preset": _preset(),
    }


def _preset() -> dict:
    repo, filename = config.default_repo(), config.default_file()
    factory = repo == config.DEFAULT_REPO and filename == config.DEFAULT_FILE
    size = config.DEFAULT_SIZE if factory else None
    if size is None:
        try:
            size = hfclient.file_size(repo, filename)
        except Exception:  # noqa: BLE001 - el panel funciona sin red
            size = None
    label = config.DEFAULT_LABEL if factory else f"{filename}"
    note = (
        "Qwen 3.5 2B en Q4_K_M: 1.40 GB, ~1.5 GB de RAM y 10-25 tokens/s en CPU. "
        "Responde bien en español y se comprueba con sha256 al descargar."
        if factory
        else f"Modelo definido con DEFAULT_REPO/DEFAULT_FILE ({repo})."
    )
    return {
        "repo_id": repo,
        "filename": filename,
        "label": label,
        "size_bytes": size,
        "size_label": format_bytes(size),
        "quant": quant_label(filename),
        "note": note,
    }


@app.get("/api/catalog/search")
def catalog_search(request: Request, q: str = "", limit: int = 12) -> dict:
    _require_admin(request)
    return {"results": hfclient.search_models(q, limit=limit)}


def _catalog_files(repo_id: str) -> list[dict]:
    files = []
    for entry in group_gguf_entries(hfclient.repo_gguf_files(repo_id)):
        if entry.get("split") and (entry.get("missing_parts") or not entry.get("parts")):
            # Una parte suelta no es un modelo. Solo se ofrece el conjunto completo.
            continue
        accessory = non_model_reason(entry["filename"])
        if accessory:
            allowed, reason = False, accessory
        else:
            allowed, reason = evaluate(entry.get("size_bytes"))
        if entry.get("split"):
            label = f"{entry.get('stem')} · modelo completo, {entry['part_total']} partes"
        else:
            label = entry["filename"]
        files.append(
            {
                "filename": entry["filename"],
                "label": label,
                "size_bytes": entry.get("size_bytes"),
                "size_label": format_bytes(entry.get("size_bytes")),
                "quant": entry.get("quant") or quant_label(entry["filename"]),
                "allowed": allowed,
                "reason": reason or None,
                "split": bool(entry.get("split")),
                "part_total": entry.get("part_total"),
                "parts": entry.get("parts") or [entry["filename"]],
            }
        )
    files.sort(key=lambda item: (item["size_bytes"] is None, item["size_bytes"] or 0, item["filename"]))
    return files


@app.get("/api/catalog/files")
def catalog_files(request: Request, repo_id: str) -> dict:
    _require_admin(request)
    repo_id = validate_repo(repo_id)
    return {"repo_id": repo_id, "files": _catalog_files(repo_id)}


@app.get("/api/catalog/model")
def catalog_model(request: Request, repo_id: str) -> dict:
    _require_admin(request)
    repo_id = validate_repo(repo_id)
    overview = hfclient.model_overview(repo_id)
    files = _catalog_files(repo_id)
    return {
        **overview,
        "files": files,
        "limit_bytes": max_model_bytes(),
        "limit_label": format_bytes(max_model_bytes()),
    }


@app.get("/api/models")
def list_installed(request: Request) -> dict:
    _require_admin(request)
    return {"models": [_present(model, request) for model in db.list_models()]}


@app.get("/api/models/{slug}")
def get_installed(slug: str, request: Request) -> dict:
    _require_admin(request)
    model = db.get(slug)
    if model is None:
        raise HTTPException(status_code=404, detail="Modelo no encontrado.")
    return _present(model, request)


@app.post("/api/models/download")
def download_model(body: DownloadRequest, request: Request) -> dict:
    _require_admin(request)
    repo_id = validate_repo(body.repo_id)
    filename, parts, size = _bundle(repo_id, body.filename)
    record = db.upsert_download(repo_id, filename, size, parts)
    if record["status"] == "queued":
        service.enqueue(record["slug"])
    return _present(db.get(record["slug"]) or record, request)


async def _erase(slug: str) -> bool:
    await runner.stop(slug)
    service.cancel(slug)
    model = db.delete(slug)
    if model is None:
        return False
    directory = config.models_dir() / slug
    if directory.exists():
        shutil.rmtree(directory, ignore_errors=True)
    return True


@app.post("/api/downloads/stop")
async def stop_downloads(request: Request) -> dict:
    _require_admin(request)
    stopped = []
    for model in list(db.list_models()):
        if model["status"] in ("queued", "downloading"):
            if await _erase(model["slug"]):
                stopped.append(model["slug"])
    return {"stopped": stopped}


@app.post("/api/models/{slug}/start")
async def start_model(slug: str, request: Request) -> dict:
    _require_admin(request)
    model = db.get(slug)
    if model is None:
        raise HTTPException(status_code=404, detail="Modelo no encontrado.")
    if model["status"] != "ready":
        raise HTTPException(status_code=409, detail="El modelo todavía no está descargado.")
    blocked = _load_block_reason(model) or _ram_reason(model)
    if blocked:
        raise HTTPException(status_code=409, detail=blocked)
    runner.schedule(slug)
    return _present(model, request)


@app.post("/api/models/{slug}/stop")
async def stop_model(slug: str, request: Request) -> dict:
    _require_admin(request)
    model = db.get(slug)
    if model is None:
        raise HTTPException(status_code=404, detail="Modelo no encontrado.")
    await runner.stop(slug)
    return _present(model, request)


@app.delete("/api/models/{slug}")
async def delete_model(slug: str, request: Request) -> dict:
    _require_admin(request)
    if not await _erase(slug):
        raise HTTPException(status_code=404, detail="Modelo no encontrado.")
    return {"deleted": slug}


@app.get("/v1/models")
def openai_models(request: Request) -> Response:
    denied = _require_api_key(request)
    if denied is not None:
        return denied
    data = []
    for model in db.list_models():
        if model["status"] != "ready":
            continue
        runtime = runner.public_status(model["slug"])
        data.append(
            {
                "id": model["slug"],
                "object": "model",
                "owned_by": "obrador",
                "status": runtime["status"],
            }
        )
    return JSONResponse({"object": "list", "data": data})


async def _run_inference(slug: str, path: str, body: dict) -> Response:
    model = db.get(slug)
    if model is None:
        return _oai_error(404, f"No existe el modelo '{slug}'. Descárgalo desde el panel.")
    if model["status"] in ("queued", "downloading"):
        return _oai_error(409, "El modelo se está descargando. Espera a que termine.")
    if model["status"] != "ready":
        return _oai_error(409, model.get("error") or "El modelo no está listo.")
    blocked = _load_block_reason(model)
    if blocked:
        return _oai_error(409, blocked)
    too_big = _ram_reason(model)
    if too_big:
        return _oai_error(503, too_big)
    try:
        running = await runner.ensure(slug)
    except Exception as exc:  # noqa: BLE001 - se devuelve al cliente de la API
        return _oai_error(503, str(exc))
    payload = {key: value for key, value in body.items() if key in _OPENAI_FIELDS and value is not None}
    payload["model"] = slug
    if "chat_template_kwargs" not in payload and path.endswith("/chat/completions"):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return await _proxy(running.port, path, payload)


def _timeout_message(seconds: int) -> str:
    return (
        f"El modelo no terminó de responder en {seconds} s y se canceló. "
        "Prueba con menos tokens, vacía el chat o usa un modelo más pequeño "
        "(GENERATION_TIMEOUT sube el límite)."
    )


def _upstream_error(raw: bytes, status: int) -> Response:
    """Traduce los errores de llama-server a mensajes que se entienden en el chat."""
    try:
        data = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        data = None
    message = None
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            message = err.get("message")
        elif isinstance(err, str):
            message = err
    if status == 503 and message and "loading" in message.lower():
        return _oai_error(503, "El modelo todavía se está cargando en memoria. Espera unos segundos.")
    if message and ("context" in message.lower() and ("exceed" in message.lower() or "size" in message.lower())):
        return _oai_error(
            400,
            f"La conversación no cabe en el contexto ({config.ctx_size()} tokens). Vacía el chat. ({message})",
        )
    if message:
        return _oai_error(status, f"llama-server: {message}")
    return Response(content=raw, status_code=status, media_type="application/json")


def _sse_error(message: str, code: int) -> bytes:
    payload = {"error": {"message": message, "type": "server_error", "code": code}}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode()


async def _proxy(port: int, path: str, body: dict) -> Response:
    total = config.generation_timeout()
    idle = min(config.stream_idle_timeout(), total)
    deadline = time.monotonic() + total
    streaming = bool(body.get("stream"))
    # Sin stream, llama-server no manda nada hasta terminar: el tope es el total.
    read = idle if streaming else total
    client = httpx.AsyncClient(timeout=httpx.Timeout(total, connect=10.0, read=read, write=30.0))
    try:
        request = client.build_request(
            "POST",
            f"http://127.0.0.1:{port}{path}",
            json=body,
        )
        response = await asyncio.wait_for(client.send(request, stream=True), timeout=total)
    except (asyncio.TimeoutError, httpx.TimeoutException):
        await client.aclose()
        log.warning("generación cancelada por tiempo (%ss) en %s", total, path)
        return _oai_error(504, _timeout_message(total))
    except httpx.HTTPError as exc:
        await client.aclose()
        return _oai_error(502, f"El modelo no respondió: {exc}")
    if response.status_code >= 400:
        raw = await response.aread()
        await response.aclose()
        await client.aclose()
        text = raw.decode("utf-8", "replace").lower()
        if "chat_template_kwargs" in body and (
            "chat_template" in text or "unknown" in text or "extra" in text
        ):
            reduced = dict(body)
            reduced.pop("chat_template_kwargs", None)
            return await _proxy(port, path, reduced)
        return _upstream_error(raw, response.status_code)
    if streaming:
        media = response.headers.get("content-type", "text/event-stream")

        async def generate():
            iterator = response.aiter_bytes()
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        yield _sse_error(_timeout_message(total), 504)
                        return
                    try:
                        chunk = await asyncio.wait_for(iterator.__anext__(), timeout=min(idle, remaining))
                    except StopAsyncIteration:
                        return
                    except (asyncio.TimeoutError, httpx.TimeoutException):
                        if time.monotonic() >= deadline:
                            message = _timeout_message(total)
                        else:
                            message = (
                                f"El modelo pasó {idle} s sin generar nada y se canceló. "
                                "Puede faltar RAM o CPU para este modelo."
                            )
                        log.warning("stream cancelado: %s", message)
                        yield _sse_error(message, 504)
                        return
                    except httpx.HTTPError as exc:
                        yield _sse_error(f"llama-server cortó la respuesta: {exc}", 502)
                        return
                    yield chunk
            finally:
                # Cerrar la conexión hace que llama-server deje de generar.
                await response.aclose()
                await client.aclose()

        return StreamingResponse(
            generate(),
            media_type=media,
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    try:
        raw = await asyncio.wait_for(response.aread(), timeout=max(1.0, deadline - time.monotonic()))
    except (asyncio.TimeoutError, httpx.TimeoutException):
        log.warning("generación cancelada por tiempo (%ss) en %s", total, path)
        return _oai_error(504, _timeout_message(total))
    except httpx.HTTPError as exc:
        return _oai_error(502, f"llama-server cortó la respuesta: {exc}")
    finally:
        await response.aclose()
        await client.aclose()
    return Response(content=raw, status_code=response.status_code, media_type="application/json")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    denied = _require_api_key(request)
    if denied is not None:
        return denied
    body = await request.json()
    slug = (body.get("model") or "").strip()
    if not slug:
        return _oai_error(400, "Falta el campo model.")
    return await _run_inference(slug, "/v1/chat/completions", body)


@app.post("/v1/completions")
async def completions(request: Request) -> Response:
    denied = _require_api_key(request)
    if denied is not None:
        return denied
    body = await request.json()
    slug = (body.get("model") or "").strip()
    if not slug:
        return _oai_error(400, "Falta el campo model.")
    return await _run_inference(slug, "/v1/completions", body)


@app.post("/v1/models/{slug}/chat/completions")
async def chat_for_model(slug: str, request: Request) -> Response:
    denied = _require_api_key(request)
    if denied is not None:
        return denied
    body = await request.json()
    body["model"] = slug
    return await _run_inference(slug, "/v1/chat/completions", body)


@app.post("/v1/models/{slug}/completions")
async def completions_for_model(slug: str, request: Request) -> Response:
    denied = _require_api_key(request)
    if denied is not None:
        return denied
    body = await request.json()
    body["model"] = slug
    return await _run_inference(slug, "/v1/completions", body)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")
