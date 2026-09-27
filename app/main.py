"""API de administración y API compatible con OpenAI, una ruta por modelo."""

from __future__ import annotations

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import config, db, hfclient
from app.downloader import service
from app.policy import (
    PolicyError,
    evaluate,
    format_bytes,
    max_model_bytes,
    quant_label,
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


def _bootstrap() -> None:
    service.resume_queued()
    if not config.auto_download():
        return
    if db.meta_get("bootstrap") == "1":
        return
    repo = config.default_repo()
    filename = config.default_file()
    if db.find(repo, filename):
        db.meta_set("bootstrap", "1")
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
            db.meta_set("bootstrap", "1")
            return
    except Exception as exc:  # noqa: BLE001 - se reintenta en el próximo arranque
        log.warning("el modelo inicial no se encoló todavía: %s", exc)
        return
    record = db.upsert_download(repo, filename, int(size))
    if record["status"] == "queued":
        service.enqueue(record["slug"])
    db.meta_set("bootstrap", "1")
    log.info("descarga inicial encolada: %s", record["slug"])


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config.ensure_dirs()
    db.init()
    service.start()
    _bootstrap()
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


@app.exception_handler(hfclient.HFError)
async def _hf_error(_request: Request, exc: hfclient.HFError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


def _bearer(request: Request) -> str:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return ""


def _require_admin(request: Request) -> None:
    expected = config.admin_token()
    if not expected:
        return
    sent = request.headers.get("x-admin-token", "").strip() or _bearer(request)
    if sent != expected:
        raise HTTPException(status_code=401, detail="Token de administración inválido.")


def _require_api_key(request: Request) -> None:
    expected = config.api_key()
    if not expected:
        return
    if _bearer(request) != expected:
        return JSONResponse(  # type: ignore[return-value]
            status_code=401,
            content={"error": {"message": "API key inválida.", "type": "invalid_request_error"}},
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
    return {
        **model,
        "quant": quant_label(model["filename"]),
        "size_label": format_bytes(size),
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


def _known_size(repo_id: str, filename: str) -> int:
    size = hfclient.file_size(repo_id, filename)
    if size is None:
        size = hfclient.head_size(repo_id, filename)
    allowed, reason = evaluate(size)
    if not allowed:
        raise PolicyError(reason)
    assert size is not None
    free = shutil.disk_usage(config.data_dir()).free
    if free < size + 512 * 1024 * 1024:
        raise PolicyError(
            f"Quedan {format_bytes(free)} libres y el archivo pesa {format_bytes(size)}."
        )
    return size


@app.get("/health")
def health() -> dict:
    models = db.list_models()
    running = 0
    for model in models:
        if runner.public_status(model["slug"])["status"] == "running":
            running += 1
    return {
        "status": "ok",
        "llama_server": runner.resolve() is not None,
        "models_ready": sum(1 for model in models if model["status"] == "ready"),
        "models_running": running,
        "max_model_bytes": max_model_bytes(),
    }


@app.get("/api/system")
def system(request: Request) -> dict:
    _require_admin(request)
    usage = shutil.disk_usage(config.data_dir())
    return {
        "max_model_bytes": max_model_bytes(),
        "max_model_label": format_bytes(max_model_bytes()),
        "disk_free": usage.free,
        "disk_free_label": format_bytes(usage.free),
        "llama_server": runner.resolve() is not None,
        "ctx_size": config.ctx_size(),
        "max_loaded_models": config.max_loaded_models(),
        "api_key_required": bool(config.api_key()),
        "admin_token_required": bool(config.admin_token()),
        "preset": {
            "repo_id": config.default_repo(),
            "filename": config.default_file(),
            "size_bytes": config.DEFAULT_SIZE,
            "size_label": format_bytes(config.DEFAULT_SIZE),
            "quant": "Q4_K_M",
            "note": "Qwen 3.5 9B en Q4_K_M. El archivo publicado pesa 6.17 GB.",
        },
    }


@app.get("/api/catalog/search")
def catalog_search(request: Request, q: str = "", limit: int = 12) -> dict:
    _require_admin(request)
    return {"results": hfclient.search_models(q, limit=limit)}


def _catalog_files(repo_id: str) -> list[dict]:
    files = []
    for entry in hfclient.repo_gguf_files(repo_id):
        allowed, reason = evaluate(entry["size_bytes"])
        files.append(
            {
                "filename": entry["filename"],
                "size_bytes": entry["size_bytes"],
                "size_label": format_bytes(entry["size_bytes"]),
                "quant": quant_label(entry["filename"]),
                "allowed": allowed,
                "reason": reason or None,
            }
        )
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
    filename = validate_filename(body.filename)
    size = _known_size(repo_id, filename)
    record = db.upsert_download(repo_id, filename, size)
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
    try:
        running = await runner.ensure(slug)
    except Exception as exc:  # noqa: BLE001 - se devuelve al cliente de la API
        return _oai_error(503, str(exc))
    payload = {key: value for key, value in body.items() if key in _OPENAI_FIELDS and value is not None}
    payload["model"] = slug
    if "chat_template_kwargs" not in payload and path.endswith("/chat/completions"):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return await _proxy(running.port, path, payload)


async def _proxy(port: int, path: str, body: dict) -> Response:
    client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=None))
    try:
        request = client.build_request(
            "POST",
            f"http://127.0.0.1:{port}{path}",
            json=body,
        )
        response = await client.send(request, stream=True)
    except httpx.HTTPError as exc:
        await client.aclose()
        return _oai_error(502, f"El modelo no respondió: {exc}")
    if response.status_code >= 400 and "chat_template_kwargs" in body:
        raw = await response.aread()
        await response.aclose()
        await client.aclose()
        text = raw.decode("utf-8", "replace").lower()
        if "chat_template" in text or "unknown" in text or "extra" in text:
            reduced = dict(body)
            reduced.pop("chat_template_kwargs", None)
            return await _proxy(port, path, reduced)
        return Response(content=raw, status_code=response.status_code, media_type="application/json")
    if response.status_code >= 400:
        raw = await response.aread()
        await response.aclose()
        await client.aclose()
        return Response(content=raw, status_code=response.status_code, media_type="application/json")
    if body.get("stream"):
        media = response.headers.get("content-type", "text/event-stream")

        async def generate():
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()
                await client.aclose()

        return StreamingResponse(generate(), media_type=media)
    raw = await response.aread()
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
