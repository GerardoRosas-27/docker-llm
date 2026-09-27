"""Claves del panel. Si no hay variables de entorno, se pueden generar y guardar aquí."""

from __future__ import annotations

import secrets

from app import config, db


def admin_token() -> str:
    env = config.admin_token()
    if env:
        return env
    return db.meta_get("admin_token") or ""


def api_key() -> str:
    env = config.api_key()
    if env:
        return env
    return db.meta_get("api_key") or ""


def status() -> dict:
    return {
        "admin_required": bool(admin_token()),
        "api_required": bool(api_key()),
        "admin_from_env": bool(config.admin_token()),
        "api_from_env": bool(config.api_key()),
    }


def generate(which: str) -> dict:
    if which not in ("admin", "api", "both"):
        raise ValueError("Indica admin, api o both.")
    created: dict[str, str] = {}
    if which in ("admin", "both"):
        if config.admin_token():
            raise ValueError("ADMIN_TOKEN ya está fijado al arrancar el contenedor.")
        token = secrets.token_urlsafe(24)
        db.meta_set("admin_token", token)
        created["admin_token"] = token
    if which in ("api", "both"):
        if config.api_key():
            raise ValueError("API_KEY ya está fijada al arrancar el contenedor.")
        token = secrets.token_urlsafe(24)
        db.meta_set("api_key", token)
        created["api_key"] = token
    return created
