"""Acceso solo con MASTER_SECRET: sesiones firmadas, API keys derivadas y fail-closed."""

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app import access, db
from app.runner import Running, runner
from tests.conftest import TEST_SECRET

SECRET = TEST_SECRET
LEGACY_API_KEY = "api-vieja-" + "a" * 30
LEGACY_ADMIN = "admin-viejo-" + "b" * 30


def _login(client, secret=SECRET):
    return client.post("/api/auth/login", json={"secret": secret})


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _new_key(anon, **extra):
    session = _login(anon).json()["token"]
    created = anon.post("/api/keys", json={"label": "mi-app", **extra}, headers=_auth(session))
    assert created.status_code == 200, created.text
    return session, created.json()["key"]


# ---------------------------------------------------------------- sin secreto: cerrado

ADMIN_ROUTES = [
    ("get", "/api/models"),
    ("get", "/api/system"),
    ("get", "/api/keys"),
    ("post", "/api/keys"),
    ("get", "/api/catalog/search?q=qwen"),
    ("post", "/api/downloads/stop"),
]
V1_ROUTES = [
    ("get", "/v1/models"),
    ("get", "/v1/models/algo"),
    ("post", "/v1/chat/completions"),
    ("post", "/v1/completions"),
    ("post", "/v1/models/algo/chat/completions"),
    ("post", "/v1/models/algo/completions"),
]


def test_sin_master_secret_todo_cerrado_salvo_health(anon, monkeypatch):
    monkeypatch.delenv("MASTER_SECRET", raising=False)
    health = anon.get("/health")
    assert health.status_code == 200
    auth = health.json()["auth"]
    assert auth["mode"] == "unconfigured"
    assert auth["master_secret"] is False
    assert "MASTER_SECRET no configurado" in auth["warnings"][0]
    assert anon.get("/").status_code == 200  # el panel carga y enseña el aviso
    login = _login(anon, "loquesea")
    assert login.status_code == 503
    assert login.json()["code"] == "master_secret_not_configured"
    for method, path in ADMIN_ROUTES:
        response = getattr(anon, method)(path, **({"json": {}} if method == "post" else {}))
        assert response.status_code == 503, path
        assert response.json()["code"] == "master_secret_not_configured"
        assert "MASTER_SECRET no configurado" in response.json()["detail"]
    for method, path in V1_ROUTES:
        kwargs = {"json": {"model": "algo", "messages": []}} if method == "post" else {}
        response = getattr(anon, method)(path, headers=_auth("obk1.x.0.y"), **kwargs)
        assert response.status_code == 503, path
        assert response.json()["error"]["code"] == "master_secret_not_configured"
    status = anon.get("/api/auth/status").json()
    assert status["mode"] == "unconfigured" and status["session"] is None


def test_sin_master_secret_las_sesiones_y_keys_anteriores_dejan_de_valer(anon, monkeypatch):
    session, key = _new_key(anon)
    monkeypatch.delenv("MASTER_SECRET", raising=False)
    assert anon.get("/api/models", headers=_auth(session)).status_code == 503
    assert anon.get("/v1/models", headers=_auth(key)).status_code == 503


# ---------------------------------------------------------------- variables antiguas

@pytest.mark.parametrize("with_master", [False, True])
def test_admin_token_y_api_key_antiguos_ya_no_dan_acceso(anon, monkeypatch, with_master):
    monkeypatch.setenv("ADMIN_TOKEN", LEGACY_ADMIN)
    monkeypatch.setenv("API_KEY", LEGACY_API_KEY)
    if not with_master:
        monkeypatch.delenv("MASTER_SECRET", raising=False)
    expected = 401 if with_master else 503
    for token in (LEGACY_ADMIN, LEGACY_API_KEY):
        assert anon.get("/api/models", headers=_auth(token)).status_code == expected
        assert anon.get("/api/models", headers={"X-Admin-Token": token}).status_code == expected
        assert anon.get("/v1/models", headers=_auth(token)).status_code == expected
        chat = anon.post("/v1/chat/completions", headers=_auth(token), json={"model": "x", "messages": []})
        assert chat.status_code == expected
    login = _login(anon, LEGACY_ADMIN)
    assert login.status_code == expected
    assert anon.get("/health").json()["auth"]["mode"] == ("master" if with_master else "unconfigured")


def test_claves_antiguas_de_la_base_no_valen_y_se_borran(anon, monkeypatch):
    db.meta_set("admin_token", "generada-antes")
    db.meta_set("api_key", "api-generada-antes")
    for token in ("generada-antes", "api-generada-antes"):
        assert anon.get("/api/models", headers=_auth(token)).status_code == 401
        assert anon.get("/v1/models", headers=_auth(token)).status_code == 401
    monkeypatch.delenv("MASTER_SECRET", raising=False)
    assert anon.get("/api/models", headers=_auth("generada-antes")).status_code == 503
    access.purge_legacy()  # se llama al arrancar
    assert db.meta_get("admin_token") is None
    assert db.meta_get("api_key") is None


def test_ruta_antigua_de_generar_claves_ya_no_existe(client):
    assert client.post("/api/access/generate", json={"which": "api"}).status_code in (404, 405)
    assert client.get("/api/access").status_code in (404, 405)


# ---------------------------------------------------------------- sesiones

def test_login_correcto_da_una_sesion_que_abre_el_panel(anon):
    assert anon.get("/api/models").status_code == 401
    response = _login(anon)
    assert response.status_code == 200
    body = response.json()
    assert body["token"].startswith("obs1.")
    assert body["expires_in"] == 12 * 3600
    assert SECRET not in response.text
    assert anon.get("/api/models", headers=_auth(body["token"])).status_code == 200
    status = anon.get("/api/auth/status", headers=_auth(body["token"])).json()
    assert status["mode"] == "master"
    assert status["session"]["kind"] == "session"
    assert SECRET not in json.dumps(status)
    assert SECRET not in anon.get("/health").text


def test_login_incorrecto_y_limite_de_intentos(anon):
    for _ in range(access.limiter.FREE_ATTEMPTS):
        assert _login(anon, "no-es").status_code == 401
    blocked = _login(anon)
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) > 0


def test_token_manipulado_caducado_o_de_otro_secreto_se_rechaza(anon, monkeypatch):
    token = _login(anon).json()["token"]
    prefix, body, sig = token.split(".")
    payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    payload["exp"] += 10 * 365 * 86400
    forged = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    assert anon.get("/api/models", headers=_auth(f"{prefix}.{forged}.{sig}")).status_code == 401
    assert anon.get("/api/models", headers=_auth(token[:-2] + "AA")).status_code == 401
    expired = access.issue_session(ttl=60, now=int(time.time()) - 3600)["token"]
    assert anon.get("/api/models", headers=_auth(expired)).status_code == 401
    monkeypatch.setenv("MASTER_SECRET", SECRET + "-rotado")
    assert anon.get("/api/models", headers=_auth(token)).status_code == 401


def test_logout_revoca_la_sesion(anon):
    token = _login(anon).json()["token"]
    assert anon.post("/api/auth/logout", headers=_auth(token)).json()["logged_out"] is True
    assert anon.get("/api/models", headers=_auth(token)).status_code == 401


# ---------------------------------------------------------------- API keys

class _FakeLlama(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        _FakeLlama.seen.append((self.path, body, self.headers.get("authorization")))
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"Hola"}}]}\n\n')
            self.wfile.write(b"data: [DONE]\n\n")
            return
        raw = json.dumps({"object": "ok", "path": self.path, "model": body.get("model")}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def served(monkeypatch, tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeLlama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    path = tmp_path / "m.gguf"
    path.write_bytes(b"GGUF")
    record = db.upsert_download("alguien/mini-GGUF", "mini-Q4_K_M.gguf", 1_000_000)
    db.mark_downloading(record["slug"])
    db.mark_ready(record["slug"], str(path), 1_000_000)
    fake = Running(slug=record["slug"], port=server.server_address[1], proc=None, started=time.time(), healthy=True)

    async def ensure(slug):
        return fake

    monkeypatch.setattr(runner, "ensure", ensure)
    _FakeLlama.seen = []
    yield record["slug"]
    server.shutdown()


def test_api_key_autentica_todas_las_rutas_v1(anon, served):
    slug = served
    _session, key = _new_key(anon)
    assert key.startswith("obk1.")
    chat = {"messages": [{"role": "user", "content": "Hola"}]}
    # Sin clave o con una clave falsa: 401 en todas.
    for method, path in V1_ROUTES:
        path = path.replace("algo", slug)
        kwargs = {"json": {"model": slug, **chat}} if method == "post" else {}
        assert getattr(anon, method)(path, **kwargs).status_code == 401, path
        assert getattr(anon, method)(path, headers=_auth(key[:-3] + "AAA"), **kwargs).status_code == 401, path

    listed = anon.get("/v1/models", headers=_auth(key))
    assert listed.status_code == 200
    assert [item["id"] for item in listed.json()["data"]] == [slug]
    one = anon.get(f"/v1/models/{slug}", headers=_auth(key))
    assert one.status_code == 200 and one.json()["id"] == slug

    response = anon.post("/v1/chat/completions", headers=_auth(key), json={"model": slug, **chat})
    assert response.status_code == 200 and response.json()["path"] == "/v1/chat/completions"
    response = anon.post("/v1/completions", headers=_auth(key), json={"model": slug, "prompt": "Hola"})
    assert response.status_code == 200 and response.json()["path"] == "/v1/completions"
    response = anon.post(f"/v1/models/{slug}/chat/completions", headers=_auth(key), json=chat)
    assert response.status_code == 200 and response.json()["model"] == slug
    response = anon.post(f"/v1/models/{slug}/completions", headers=_auth(key), json={"prompt": "Hola"})
    assert response.status_code == 200 and response.json()["path"] == "/v1/completions"

    for path in ("/v1/chat/completions", f"/v1/models/{slug}/chat/completions"):
        stream = anon.post(path, headers=_auth(key), json={"model": slug, "stream": True, **chat})
        assert stream.status_code == 200
        assert stream.headers["content-type"].startswith("text/event-stream")
        assert "Hola" in stream.text and "[DONE]" in stream.text

    # La API key nunca llega a llama-server.
    assert all(auth is None for _path, _body, auth in _FakeLlama.seen)


def test_la_sesion_del_panel_sirve_para_su_chat(client, served):
    response = client.post(
        f"/v1/models/{served}/chat/completions",
        json={"messages": [{"role": "user", "content": "Hola"}], "stream": True},
    )
    assert response.status_code == 200
    assert "Hola" in response.text


def test_una_api_key_no_sirve_para_administrar(anon, served):
    session, key = _new_key(anon)
    headers = _auth(key)
    assert anon.get("/api/models", headers=headers).status_code == 401
    assert anon.get(f"/api/models/{served}", headers=headers).status_code == 401
    assert anon.get("/api/system", headers=headers).status_code == 401
    assert anon.get("/api/keys", headers=headers).status_code == 401
    assert anon.post("/api/keys", json={"label": "otra"}, headers=headers).status_code == 401
    assert anon.post(f"/api/models/{served}/start", headers=headers).status_code == 401
    assert anon.post(f"/api/models/{served}/stop", headers=headers).status_code == 401
    assert anon.delete(f"/api/models/{served}", headers=headers).status_code == 401
    assert anon.post("/api/downloads/stop", headers=headers).status_code == 401
    assert anon.post("/api/models/download", json={"repo_id": "a/b", "filename": "c.gguf"}, headers=headers).status_code == 401
    listed = anon.get("/api/keys", headers=_auth(session)).json()["keys"]
    assert anon.delete(f"/api/keys/{listed[0]['id']}", headers=headers).status_code == 401
    assert anon.get("/api/auth/status", headers=headers).json()["session"] is None
    assert anon.post("/api/auth/logout", headers=headers).json()["logged_out"] is False
    assert db.get(served) is not None


def test_api_key_listado_revocacion_y_firma(anon):
    session, key = _new_key(anon)
    assert anon.get("/v1/models", headers=_auth(key)).status_code == 200
    listed = anon.get("/api/keys", headers=_auth(session)).json()["keys"]
    assert listed[0]["label"] == "mi-app"
    assert "key" not in listed[0]
    assert key not in json.dumps(listed)
    parts = key.split(".")
    parts[2] = "9999999999"
    assert anon.get("/v1/models", headers=_auth(".".join(parts))).status_code == 401
    assert anon.delete(f"/api/keys/{listed[0]['id']}", headers=_auth(session)).status_code == 200
    assert anon.get("/v1/models", headers=_auth(key)).status_code == 401
    assert anon.delete(f"/api/keys/{listed[0]['id']}", headers=_auth(session)).status_code == 404


def test_api_key_caducada_y_rotacion(anon, monkeypatch):
    _session, key = _new_key(anon, expires_in_days=1)
    assert anon.get("/v1/models", headers=_auth(key)).status_code == 200
    monkeypatch.setattr(access.time, "time", lambda: time.time_ns() / 1e9 + 2 * 86400)
    assert anon.get("/v1/models", headers=_auth(key)).status_code == 401
    monkeypatch.undo()
    monkeypatch.setenv("MASTER_SECRET", SECRET)
    assert anon.get("/v1/models", headers=_auth(key)).status_code == 200
    monkeypatch.setenv("MASTER_SECRET", SECRET + "-rotado")
    assert anon.get("/v1/models", headers=_auth(key)).status_code == 401


def test_crear_keys_exige_sesion(anon):
    assert anon.post("/api/keys", json={"label": "x"}).status_code == 401
    assert anon.get("/api/keys").status_code == 401
    session = _login(anon).json()["token"]
    bad = anon.post("/api/keys", json={"label": "x", "expires_in_days": 0}, headers=_auth(session))
    assert bad.status_code == 400
