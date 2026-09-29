"""Chat que nunca se queda mudo: modelo ligero, RAM, tiempos máximos y sha256."""

import hashlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from app import config, db, resources
from app.runner import Running, runner


def test_el_modelo_por_defecto_es_ligero_y_tiene_sha256():
    assert config.DEFAULT_REPO == "bartowski/Qwen_Qwen3.5-2B-GGUF"
    assert config.DEFAULT_FILE == "Qwen_Qwen3.5-2B-Q4_K_M.gguf"
    assert config.DEFAULT_SIZE < 1_500_000_000
    assert len(config.default_sha256()) == 64


def test_volumen_viejo_con_el_9b_recibe_el_modelo_nuevo(client, monkeypatch):
    from app import main

    legacy = db.upsert_download("bartowski/Qwen_Qwen3.5-9B-GGUF", "Qwen_Qwen3.5-9B-Q4_K_M.gguf", 6_169_341_984)
    db.mark_downloading(legacy["slug"])
    db.mark_ready(legacy["slug"], "/data/models/x.gguf", 6_169_341_984)
    db.meta_set("bootstrap", "1")  # marca antigua, de cuando el defecto era el 9B
    queued = []
    monkeypatch.setenv("AUTO_DOWNLOAD_DEFAULT", "1")
    monkeypatch.setattr("app.hfclient.file_size", lambda repo, filename: config.DEFAULT_SIZE)
    monkeypatch.setattr("app.main.service.enqueue", lambda slug: queued.append(slug))
    monkeypatch.setattr("app.main.service.resume_queued", lambda: None)
    main._bootstrap()
    assert queued == ["qwen-qwen3.5-2b-q4-k-m"]
    main._bootstrap()  # la segunda vez no vuelve a encolar
    assert queued == ["qwen-qwen3.5-2b-q4-k-m"]


def _ready(tmp_path: Path, size: int) -> dict:
    path = tmp_path / "grande.gguf"
    path.write_bytes(b"GGUF")
    record = db.upsert_download("alguien/grande-GGUF", "grande-Q4_K_M.gguf", size)
    db.mark_downloading(record["slug"])
    db.mark_ready(record["slug"], str(path), size)
    return db.get(record["slug"])


def test_un_modelo_que_no_cabe_en_ram_da_error_claro(client, monkeypatch, tmp_path):
    monkeypatch.setattr(resources, "memory_limit", lambda: 4 * 1024**3)
    model = _ready(tmp_path, 6_169_341_984)
    listed = client.get("/api/models").json()["models"]
    assert "RAM" in listed[0]["ram_warning"]
    start = client.post(f"/api/models/{model['slug']}/start")
    assert start.status_code == 409
    assert "más pequeño" in start.json()["detail"]
    chat = client.post(f"/v1/models/{model['slug']}/chat/completions", json={"messages": []})
    assert chat.status_code == 503
    assert "RAM" in chat.json()["error"]["message"]


def test_el_2b_cabe_en_un_contenedor_de_2gb():
    assert resources.estimate_model_ram(config.DEFAULT_SIZE) < 2 * 1024**3


class _Slow(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length)
        if b'"stream": true' in body or b'"stream":true' in body:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"Hola"}}]}\n\n')
            self.wfile.flush()
            time.sleep(4)
            return
        time.sleep(4)
        self.send_response(200)
        self.end_headers()


@pytest.fixture
def slow_model(monkeypatch, tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Slow)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    model = _ready(tmp_path, 1_000_000)
    fake = Running(slug=model["slug"], port=server.server_address[1], proc=None, started=time.time(), healthy=True)

    async def ensure(slug):
        return fake

    monkeypatch.setattr(runner, "ensure", ensure)
    monkeypatch.setenv("GENERATION_TIMEOUT", "2")
    monkeypatch.setenv("STREAM_IDLE_TIMEOUT", "1")
    yield model
    server.shutdown()


def test_sin_stream_corta_con_504_y_mensaje(client, slow_model):
    started = time.time()
    response = client.post(
        f"/v1/models/{slow_model['slug']}/chat/completions",
        json={"messages": [{"role": "user", "content": "hola"}]},
    )
    assert time.time() - started < 3.5
    assert response.status_code == 504
    assert "no terminó de responder" in response.json()["error"]["message"]


def test_stream_colgado_manda_un_error_visible(client, slow_model):
    started = time.time()
    response = client.post(
        f"/v1/models/{slow_model['slug']}/chat/completions",
        json={"messages": [{"role": "user", "content": "hola"}], "stream": True},
    )
    assert time.time() - started < 3.5
    assert response.status_code == 200
    text = response.text
    assert "Hola" in text
    assert '"error"' in text
    assert "se canceló" in text
    assert text.rstrip().endswith("data: [DONE]")


def test_sha256_distinto_borra_el_archivo(monkeypatch, tmp_path):
    from app.downloader import Downloader

    path = tmp_path / "m.gguf.partial"
    path.write_bytes(b"GGUF" + b"x" * 100)
    good = hashlib.sha256(path.read_bytes()).hexdigest()
    loader = Downloader()
    monkeypatch.setattr("app.downloader._expected_sha256", lambda repo, name: good)
    loader._verify_sha256({"repo_id": "a/b"}, "m.gguf", path, threading.Event())
    assert path.exists()
    monkeypatch.setattr("app.downloader._expected_sha256", lambda repo, name: "0" * 64)
    with pytest.raises(RuntimeError, match="sha256"):
        loader._verify_sha256({"repo_id": "a/b"}, "m.gguf", path, threading.Event())
    assert not path.exists()


def test_llama_server_sin_repack_sin_cache_y_con_hilos_del_contenedor(monkeypatch, tmp_path):
    help_text = "--ctx-size --alias --jinja --no-webui --no-mmproj --parallel --n-gpu-layers --fit --batch-size --cache-type-k --repack, -nr, --no-repack --cache-ram --threads"
    monkeypatch.setattr(runner, "_help", lambda binary: help_text)
    monkeypatch.setattr(resources, "cpu_limit", lambda: 2)
    monkeypatch.setattr("app.runner.os.cpu_count", lambda: 32)
    cmd = runner._command("llama-server", tmp_path / "m.gguf", 18080, "m")
    assert "--no-repack" in cmd
    assert cmd[cmd.index("--cache-ram") + 1] == "0"
    assert cmd[cmd.index("-t") + 1] == "2"
    monkeypatch.setenv("LLAMA_REPACK", "1")
    monkeypatch.setattr(resources, "cpu_limit", lambda: 32)
    cmd = runner._command("llama-server", tmp_path / "m.gguf", 18080, "m")
    assert "--no-repack" not in cmd
    assert "-t" not in cmd
