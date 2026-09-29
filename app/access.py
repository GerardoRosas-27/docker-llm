"""Acceso al panel y a /v1 a partir de un único secreto maestro (MASTER_SECRET).

- Con MASTER_SECRET se entra al panel con POST /api/auth/login y se recibe una
  sesión firmada (HMAC-SHA256). Las API keys también se firman con claves
  derivadas del secreto (HKDF), así que sin él nadie puede crearlas. Cambiar
  MASTER_SECRET invalida todas las sesiones y todas las API keys de golpe.
- Las API keys (`obk1.…`) sirven para todas las rutas /v1 y nunca para administrar.
- Sin MASTER_SECRET todo queda cerrado (fail-closed): el panel, /api y /v1
  responden 503 `master_secret_not_configured` y /health lo avisa.
  ADMIN_TOKEN, API_KEY y las claves antiguas de la base ya no dan acceso.

El secreto nunca se registra ni se devuelve.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time

from app import db

SESSION_PREFIX = "obs1"
API_KEY_PREFIX = "obk1"
_MIN_RECOMMENDED = 32
NOT_CONFIGURED_CODE = "master_secret_not_configured"
NOT_CONFIGURED_MESSAGE = (
    "MASTER_SECRET no configurado: el panel y la API están cerrados hasta que se defina "
    "MASTER_SECRET en las variables de entorno del servidor."
)


class AuthError(Exception):
    def __init__(self, status: int, message: str, code: str, retry_after: int | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        self.retry_after = retry_after


# --------------------------------------------------------------------------
# Secretos y derivación

def master_secret() -> str:
    return os.environ.get("MASTER_SECRET", "").strip()


def master_configured() -> bool:
    return bool(master_secret())


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _hkdf(secret: bytes, info: bytes, length: int = 32) -> bytes:
    """HKDF-SHA256 (RFC 5869) con sal fija de la aplicación."""
    prk = hmac.new(b"obrador-hkdf-salt-v1", secret, hashlib.sha256).digest()
    out, block, counter = b"", b"", 1
    while len(out) < length:
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        out += block
        counter += 1
    return out[:length]


def _key(purpose: str) -> bytes:
    secret = master_secret()
    if not secret:
        raise not_configured()
    return _hkdf(secret.encode("utf-8"), f"obrador/{purpose}/v1".encode())


def _sign(purpose: str, message: str) -> str:
    return _b64e(hmac.new(_key(purpose), message.encode("ascii"), hashlib.sha256).digest())


def _same(a: str, b: str) -> bool:
    """Comparación en tiempo constante aunque las longitudes no coincidan."""
    return hmac.compare_digest(
        hashlib.sha256(a.encode("utf-8")).digest(),
        hashlib.sha256(b.encode("utf-8")).digest(),
    )


def not_configured() -> AuthError:
    return AuthError(503, NOT_CONFIGURED_MESSAGE, NOT_CONFIGURED_CODE)


def require_configured() -> None:
    if not master_configured():
        raise not_configured()


def mode() -> str:
    return "master" if master_configured() else "unconfigured"


def warnings() -> list[str]:
    if not master_configured():
        return [NOT_CONFIGURED_MESSAGE]
    if len(master_secret()) < _MIN_RECOMMENDED:
        return [f"MASTER_SECRET es corto; usa al menos {_MIN_RECOMMENDED} caracteres aleatorios."]
    return []


_LEGACY_META = ("admin_token", "api_key")


def purge_legacy() -> None:
    """Borra las claves que guardaba la versión antigua del panel: ya no valen."""
    for name in _LEGACY_META:
        db.meta_delete(name)


# --------------------------------------------------------------------------
# Límite de intentos de login

class _Limiter:
    FREE_ATTEMPTS = 5
    WINDOW = 15 * 60
    MAX_LOCK = 15 * 60
    GLOBAL_PER_MINUTE = 30

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fails: dict[str, tuple[int, float, float]] = {}  # ip -> (fallos, primero, bloqueado_hasta)
        self._global: list[float] = []

    def reset(self) -> None:
        with self._lock:
            self._fails.clear()
            self._global.clear()

    def check(self, client: str) -> None:
        now = time.time()
        with self._lock:
            self._global = [t for t in self._global if now - t < 60]
            if len(self._global) >= self.GLOBAL_PER_MINUTE:
                raise AuthError(429, "Demasiados intentos fallidos. Espera un minuto.", "rate_limited", 60)
            fails, _first, until = self._fails.get(client, (0, now, 0.0))
            if until > now:
                wait = int(until - now) + 1
                raise AuthError(429, f"Demasiados intentos fallidos. Espera {wait} s.", "rate_limited", wait)

    def fail(self, client: str) -> None:
        now = time.time()
        with self._lock:
            self._global.append(now)
            fails, first, _until = self._fails.get(client, (0, now, 0.0))
            if now - first > self.WINDOW:
                fails, first = 0, now
            fails += 1
            until = 0.0
            if fails >= self.FREE_ATTEMPTS:
                # 30 s, 60 s, 120 s… hasta 15 min.
                until = now + min(self.MAX_LOCK, 30 * 2 ** (fails - self.FREE_ATTEMPTS))
            self._fails[client] = (fails, first, until)

    def success(self, client: str) -> None:
        with self._lock:
            self._fails.pop(client, None)


limiter = _Limiter()


# --------------------------------------------------------------------------
# Sesiones

def session_ttl() -> int:
    hours = float(os.environ.get("SESSION_TTL_HOURS", "12"))
    return max(60, int(hours * 3600))


def login(secret: str, client: str) -> dict:
    require_configured()
    limiter.check(client)
    if not isinstance(secret, str) or not _same(secret.strip(), master_secret()):
        limiter.fail(client)
        # Un pequeño retardo también frena los intentos en paralelo.
        time.sleep(0.25)
        raise AuthError(401, "Secreto maestro incorrecto.", "invalid_secret")
    limiter.success(client)
    return issue_session()


def issue_session(ttl: int | None = None, now: int | None = None) -> dict:
    now = int(time.time()) if now is None else now
    exp = now + (session_ttl() if ttl is None else ttl)
    payload = {"role": "admin", "iat": now, "exp": exp, "jti": secrets.token_urlsafe(12)}
    body = _b64e(json.dumps(payload, separators=(",", ":")).encode())
    message = f"{SESSION_PREFIX}.{body}"
    token = f"{message}.{_sign('session', message)}"
    return {"token": token, "role": "admin", "expires_at": exp, "expires_in": exp - now}


def verify_session(token: str) -> dict | None:
    if not token or not token.startswith(SESSION_PREFIX + ".") or not master_configured():
        return None
    parts = token.split(".")
    if len(parts) != 3:
        return None
    message = f"{parts[0]}.{parts[1]}"
    try:
        expected = _sign("session", message)
    except AuthError:
        return None
    if not hmac.compare_digest(expected, parts[2]):
        return None
    try:
        payload = json.loads(_b64d(parts[1]))
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict) or payload.get("role") != "admin":
        return None
    exp = payload.get("exp")
    if not isinstance(exp, int) or exp <= time.time():
        return None
    if db.session_revoked(str(payload.get("jti"))):
        return None
    return payload


def logout(token: str) -> bool:
    payload = verify_session(token)
    if payload is None:
        return False
    db.revoke_session(str(payload["jti"]), int(payload["exp"]))
    return True


# --------------------------------------------------------------------------
# API keys derivadas

def create_api_key(label: str, expires_in_days: float | None) -> dict:
    _key("api-key")  # falla con 503 si no hay secreto maestro
    label = (label or "").strip()[:80] or "sin nombre"
    key_id = secrets.token_hex(8)
    exp = 0
    if expires_in_days:
        exp = int(time.time() + float(expires_in_days) * 86400)
    message = f"{API_KEY_PREFIX}.{key_id}.{exp}"
    token = f"{message}.{_sign('api-key', message)}"
    record = db.add_api_key(key_id, label, exp or None)
    return {**_public_key(record), "key": token}


def verify_api_key(token: str) -> dict | None:
    if not token or not token.startswith(API_KEY_PREFIX + ".") or not master_configured():
        return None
    parts = token.split(".")
    if len(parts) != 4 or not parts[2].isdigit():
        return None
    message = ".".join(parts[:3])
    try:
        expected = _sign("api-key", message)
    except AuthError:
        return None
    if not hmac.compare_digest(expected, parts[3]):
        return None
    exp = int(parts[2])
    if exp and exp <= time.time():
        return None
    record = db.get_api_key(parts[1])
    if record is None or record.get("revoked_at"):
        return None
    _touch(record["id"])
    return record


_last_touch: dict[str, float] = {}


def _touch(key_id: str) -> None:
    now = time.time()
    if now - _last_touch.get(key_id, 0) > 60:
        _last_touch[key_id] = now
        db.touch_api_key(key_id)


def _public_key(record: dict) -> dict:
    exp = record.get("expires_at")
    return {
        "id": record["id"],
        "label": record["label"],
        "created_at": record["created_at"],
        "expires_at": exp,
        "expired": bool(exp and exp <= time.time()),
        "revoked": bool(record.get("revoked_at")),
        "last_used_at": record.get("last_used_at"),
    }


def list_api_keys() -> list[dict]:
    return [_public_key(record) for record in db.list_api_keys()]


def revoke_api_key(key_id: str) -> bool:
    return db.revoke_api_key(key_id)


# --------------------------------------------------------------------------
# Comprobaciones por petición

def admin_ok(candidates: list[str]) -> bool:
    """Solo una sesión firmada administra. Una API key nunca."""
    return any(value and verify_session(value) is not None for value in candidates)


def api_ok(candidates: list[str]) -> bool:
    """Las rutas /v1 aceptan una API key derivada o la sesión del panel (su chat)."""
    for value in candidates:
        if not value:
            continue
        if value.startswith(API_KEY_PREFIX + ".") and verify_api_key(value) is not None:
            return True
        if value.startswith(SESSION_PREFIX + ".") and verify_session(value) is not None:
            return True
    return False


def session_info(candidates: list[str]) -> dict | None:
    for value in candidates:
        payload = verify_session(value) if value else None
        if payload:
            return {"role": payload["role"], "expires_at": payload["exp"], "kind": "session"}
    return None


def status(candidates: list[str] | None = None) -> dict:
    return {
        "mode": mode(),
        "master_configured": master_configured(),
        "session_ttl": session_ttl(),
        "session": session_info(candidates or []),
        "warnings": warnings(),
    }
