"""Secreto maestro: login, sesiones firmadas, API keys derivadas y compatibilidad."""

import base64
import json
import time

import pytest

from app import access, db

SECRET = "prueba-" + "x" * 40


@pytest.fixture(autouse=True)
def limpio(monkeypatch):
    monkeypatch.delenv("MASTER_SECRET", raising=False)
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("ADMIN_TOKEN", "")
    access.limiter.reset()
    yield
    access.limiter.reset()


@pytest.fixture
def master(monkeypatch):
    monkeypatch.setenv("MASTER_SECRET", SECRET)
    return SECRET


def _login(client, secret=SECRET):
    return client.post("/api/auth/login", json={"secret": secret})


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_sin_secreto_maestro_el_login_dice_no_configurado(client):
    response = _login(client)
    assert response.status_code == 409
    assert response.json()["code"] == "master_secret_not_configured"
    health = client.get("/health").json()
    assert health["auth"]["mode"] == "open"
    assert health["auth"]["master_secret"] is False
    assert health["auth"]["warnings"]


def test_login_correcto_da_una_sesion_que_abre_el_panel(client, master):
    assert client.get("/api/models").status_code == 401
    response = _login(client)
    assert response.status_code == 200
    body = response.json()
    assert body["token"].startswith("obs1.")
    assert body["expires_in"] == 12 * 3600
    assert SECRET not in response.text
    assert client.get("/api/models", headers=_auth(body["token"])).status_code == 200
    assert client.get("/api/models", headers={"X-Admin-Token": body["token"]}).status_code == 200
    status = client.get("/api/auth/status", headers=_auth(body["token"])).json()
    assert status["mode"] == "master"
    assert status["session"]["kind"] == "session"
    assert SECRET not in json.dumps(status)
    assert SECRET not in client.get("/health").text


def test_login_incorrecto_y_limite_de_intentos(client, master):
    for _ in range(access.limiter.FREE_ATTEMPTS):
        response = _login(client, "no-es")
        assert response.status_code == 401
    blocked = _login(client)
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) > 0


def test_token_manipulado_caducado_o_de_otro_secreto_se_rechaza(client, master, monkeypatch):
    token = _login(client).json()["token"]
    prefix, body, sig = token.split(".")
    payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    payload["exp"] += 10 * 365 * 86400
    forged = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    assert client.get("/api/models", headers=_auth(f"{prefix}.{forged}.{sig}")).status_code == 401
    assert client.get("/api/models", headers=_auth(token[:-2] + "AA")).status_code == 401
    expired = access.issue_session(ttl=60, now=int(time.time()) - 3600)["token"]
    assert client.get("/api/models", headers=_auth(expired)).status_code == 401
    # Rotar MASTER_SECRET invalida todas las sesiones.
    monkeypatch.setenv("MASTER_SECRET", SECRET + "-rotado")
    assert client.get("/api/models", headers=_auth(token)).status_code == 401


def test_logout_revoca_la_sesion(client, master):
    token = _login(client).json()["token"]
    assert client.post("/api/auth/logout", headers=_auth(token)).json()["logged_out"] is True
    assert client.get("/api/models", headers=_auth(token)).status_code == 401


def test_api_key_derivada_autentica_v1_y_se_revoca(client, master, monkeypatch):
    session = _login(client).json()["token"]
    assert client.get("/v1/models").status_code == 401
    created = client.post("/api/keys", json={"label": "otro proyecto"}, headers=_auth(session))
    assert created.status_code == 200
    key = created.json()["key"]
    assert key.startswith("obk1.")
    assert client.get("/v1/models", headers=_auth(key)).status_code == 200
    # Una API key no sirve para administrar.
    assert client.get("/api/models", headers=_auth(key)).status_code == 401
    listed = client.get("/api/keys", headers=_auth(session)).json()["keys"]
    assert listed[0]["label"] == "otro proyecto"
    assert "key" not in listed[0]
    assert key not in json.dumps(listed)
    # Firma alterada
    assert client.get("/v1/models", headers=_auth(key[:-3] + "AAA")).status_code == 401
    # Caducidad alterada: la firma ya no cuadra
    parts = key.split(".")
    parts[2] = "9999999999"
    assert client.get("/v1/models", headers=_auth(".".join(parts))).status_code == 401
    assert client.delete(f"/api/keys/{listed[0]['id']}", headers=_auth(session)).status_code == 200
    assert client.get("/v1/models", headers=_auth(key)).status_code == 401


def test_api_key_caducada_y_rotacion(client, master, monkeypatch):
    session = _login(client).json()["token"]
    key = client.post("/api/keys", json={"label": "a", "expires_in_days": 1}, headers=_auth(session)).json()["key"]
    assert client.get("/v1/models", headers=_auth(key)).status_code == 200
    monkeypatch.setattr(access.time, "time", lambda: time.time_ns() / 1e9 + 2 * 86400)
    assert client.get("/v1/models", headers=_auth(key)).status_code == 401
    monkeypatch.undo()
    monkeypatch.setenv("MASTER_SECRET", SECRET + "-rotado")
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("ADMIN_TOKEN", "")
    key2 = access.create_api_key("b", None)["key"]
    monkeypatch.setenv("MASTER_SECRET", SECRET)
    assert client.get("/v1/models", headers=_auth(key2)).status_code == 401


def test_generar_claves_exige_secreto_o_sesion(client, master):
    assert client.post("/api/access/generate", json={"which": "api"}).status_code == 401
    assert client.post("/api/keys", json={"label": "x"}).status_code == 401
    session = _login(client).json()["token"]
    created = client.post("/api/access/generate", json={"which": "api"}, headers=_auth(session))
    assert created.status_code == 200
    assert client.get("/v1/models", headers=_auth(created.json()["api_key"])).status_code == 200
    admin = client.post("/api/access/generate", json={"which": "admin"}, headers=_auth(session))
    assert admin.status_code == 400


def test_sin_secreto_nadie_puede_generar_claves(client):
    response = client.post("/api/access/generate", json={"which": "both"})
    assert response.status_code == 409
    assert db.meta_get("admin_token") is None
    assert client.post("/api/keys", json={"label": "x"}).status_code == 409


def test_admin_token_y_api_key_antiguos_siguen_funcionando(client, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "admin-viejo")
    monkeypatch.setenv("API_KEY", "api-vieja")
    assert client.get("/api/models").status_code == 401
    assert client.get("/api/models", headers={"X-Admin-Token": "admin-viejo"}).status_code == 200
    assert client.get("/api/models", headers=_auth("admin-viejo")).status_code == 200
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers=_auth("api-vieja")).status_code == 200
    assert client.get("/v1/models", headers=_auth("otra")).status_code == 401
    assert client.get("/health").json()["auth"]["mode"] == "legacy"
    assert _login(client, "admin-viejo").json()["code"] == "master_secret_not_configured"
    # Con MASTER_SECRET además, las variables antiguas siguen valiendo.
    monkeypatch.setenv("MASTER_SECRET", SECRET)
    assert client.get("/api/models", headers={"X-Admin-Token": "admin-viejo"}).status_code == 200
    assert client.get("/v1/models", headers=_auth("api-vieja")).status_code == 200
    assert _login(client).status_code == 200


def test_claves_viejas_de_la_base_no_valen_con_secreto_maestro(client, monkeypatch):
    db.meta_set("admin_token", "generada-antes")
    assert client.get("/api/models", headers={"X-Admin-Token": "generada-antes"}).status_code == 200
    monkeypatch.setenv("MASTER_SECRET", SECRET)
    assert client.get("/api/models", headers={"X-Admin-Token": "generada-antes"}).status_code == 401
