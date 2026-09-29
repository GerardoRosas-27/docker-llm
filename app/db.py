"""Registro SQLite de los modelos descargados en el volumen."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from app import config
from app.policy import slug_source, slugify

_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    slug TEXT PRIMARY KEY,
    repo_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    size_bytes INTEGER,
    local_path TEXT,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    bytes_downloaded INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    parts TEXT,
    UNIQUE (repo_id, filename)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at INTEGER,
    revoked_at TEXT,
    last_used_at TEXT
);
CREATE TABLE IF NOT EXISTS revoked_sessions (
    jti TEXT PRIMARY KEY,
    expires_at INTEGER NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    config.ensure_dirs()
    conn = sqlite3.connect(config.db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init() -> None:
    with _lock:
        conn = _connect()
        try:
            conn.executescript(_SCHEMA)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(models)")}
            if "parts" not in columns:
                conn.execute("ALTER TABLE models ADD COLUMN parts TEXT")
            conn.commit()
        finally:
            conn.close()


def reset() -> None:
    def op(conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM models")
        conn.execute("DELETE FROM meta")
        conn.execute("DELETE FROM api_keys")
        conn.execute("DELETE FROM revoked_sessions")

    _write(op)


def _write(fn) -> Any:
    with _lock:
        conn = _connect()
        try:
            result = fn(conn)
            conn.commit()
            return result
        finally:
            conn.close()


def _read(fn) -> Any:
    with _lock:
        conn = _connect()
        try:
            return fn(conn)
        finally:
            conn.close()


def _row(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def meta_get(key: str) -> str | None:
    def op(conn: sqlite3.Connection) -> str | None:
        found = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return found["value"] if found else None

    return _read(op)


def meta_set(key: str, value: str) -> None:
    def op(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    _write(op)


def get(slug: str) -> dict | None:
    def op(conn: sqlite3.Connection) -> dict | None:
        return _row(conn.execute("SELECT * FROM models WHERE slug = ?", (slug,)).fetchone())

    return _read(op)


def find(repo_id: str, filename: str) -> dict | None:
    def op(conn: sqlite3.Connection) -> dict | None:
        return _row(
            conn.execute(
                "SELECT * FROM models WHERE repo_id = ? AND filename = ?",
                (repo_id, filename),
            ).fetchone()
        )

    return _read(op)


def list_models() -> list[dict]:
    def op(conn: sqlite3.Connection) -> list[dict]:
        rows = conn.execute("SELECT * FROM models ORDER BY created_at").fetchall()
        return [dict(row) for row in rows]

    return _read(op)


def _unique_slug(conn: sqlite3.Connection, repo_id: str, filename: str) -> str:
    base = slugify(slug_source(filename))
    current = conn.execute("SELECT slug, repo_id, filename FROM models WHERE slug = ?", (base,)).fetchone()
    if current is None:
        return base
    if current["repo_id"] == repo_id and current["filename"] == filename:
        return base
    author = slugify(repo_id.split("/")[0])[:24]
    candidate = f"{base}-{author}"[:80]
    taken = conn.execute("SELECT slug, repo_id, filename FROM models WHERE slug = ?", (candidate,)).fetchone()
    if taken is None or (taken["repo_id"] == repo_id and taken["filename"] == filename):
        return candidate
    n = 2
    while True:
        suffix = f"-{n}"
        candidate = f"{base[: 80 - len(suffix)]}{suffix}"
        taken = conn.execute("SELECT slug FROM models WHERE slug = ?", (candidate,)).fetchone()
        if taken is None:
            return candidate
        n += 1


def upsert_download(
    repo_id: str,
    filename: str,
    size_bytes: int,
    parts: list[str] | None = None,
) -> dict:
    """Crea o reencola un modelo para descarga. Devuelve la fila."""

    def op(conn: sqlite3.Connection) -> dict:
        existing = conn.execute(
            "SELECT * FROM models WHERE repo_id = ? AND filename = ?",
            (repo_id, filename),
        ).fetchone()
        stamp = _now()
        part_list = parts or [filename]
        encoded = json.dumps(part_list)
        if existing is None:
            slug = _unique_slug(conn, repo_id, filename)
            conn.execute(
                """
                INSERT INTO models (
                    slug, repo_id, filename, size_bytes, local_path, status,
                    progress, bytes_downloaded, error, created_at, updated_at, parts
                ) VALUES (?, ?, ?, ?, NULL, 'queued', 0, 0, NULL, ?, ?, ?)
                """,
                (slug, repo_id, filename, size_bytes, stamp, stamp, encoded),
            )
        else:
            slug = existing["slug"]
            if existing["status"] in ("ready", "downloading", "queued") and (
                existing["status"] != "ready" or existing["local_path"]
            ):
                return dict(existing)
            conn.execute(
                """
                UPDATE models
                   SET size_bytes = ?, status = 'queued', progress = 0,
                       bytes_downloaded = 0, error = NULL, parts = ?, updated_at = ?
                 WHERE slug = ?
                """,
                (size_bytes, encoded, stamp, slug),
            )
        row = conn.execute("SELECT * FROM models WHERE slug = ?", (slug,)).fetchone()
        return dict(row)

    return _write(op)


def mark_downloading(slug: str) -> None:
    def op(conn: sqlite3.Connection) -> None:
        conn.execute(
            "UPDATE models SET status = 'downloading', error = NULL, updated_at = ? WHERE slug = ?",
            (_now(), slug),
        )

    _write(op)


def mark_progress(slug: str, downloaded: int, total: int) -> None:
    progress = (downloaded / total) if total else 0.0

    def op(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            UPDATE models
               SET bytes_downloaded = ?, progress = ?, updated_at = ?
             WHERE slug = ? AND status = 'downloading'
            """,
            (downloaded, progress, _now(), slug),
        )

    _write(op)


def mark_ready(slug: str, local_path: str, size_bytes: int) -> None:
    def op(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            UPDATE models
               SET status = 'ready', local_path = ?, size_bytes = ?,
                   bytes_downloaded = ?, progress = 1, error = NULL, updated_at = ?
             WHERE slug = ?
            """,
            (local_path, size_bytes, size_bytes, _now(), slug),
        )

    _write(op)


def mark_error(slug: str, message: str) -> None:
    def op(conn: sqlite3.Connection) -> None:
        conn.execute(
            "UPDATE models SET status = 'error', error = ?, updated_at = ? WHERE slug = ?",
            (message[:800], _now(), slug),
        )

    _write(op)


def mark_queued(slug: str) -> None:
    def op(conn: sqlite3.Connection) -> None:
        conn.execute(
            "UPDATE models SET status = 'queued', error = NULL, updated_at = ? WHERE slug = ?",
            (_now(), slug),
        )

    _write(op)


def delete(slug: str) -> dict | None:
    def op(conn: sqlite3.Connection) -> dict | None:
        row = conn.execute("SELECT * FROM models WHERE slug = ?", (slug,)).fetchone()
        if row is None:
            return None
        conn.execute("DELETE FROM models WHERE slug = ?", (slug,))
        return dict(row)

    return _write(op)


# --------------------------------------------------------------------------
# API keys derivadas del secreto maestro. La clave en sí no se guarda: solo el
# id, para poder listarla y revocarla.

def add_api_key(key_id: str, label: str, expires_at: int | None) -> dict:
    def op(conn: sqlite3.Connection) -> dict:
        conn.execute(
            "INSERT INTO api_keys(id, label, created_at, expires_at) VALUES(?, ?, ?, ?)",
            (key_id, label, _now(), expires_at),
        )
        return dict(conn.execute("SELECT * FROM api_keys WHERE id = ?", (key_id,)).fetchone())

    return _write(op)


def get_api_key(key_id: str) -> dict | None:
    def op(conn: sqlite3.Connection) -> dict | None:
        return _row(conn.execute("SELECT * FROM api_keys WHERE id = ?", (key_id,)).fetchone())

    return _read(op)


def list_api_keys() -> list[dict]:
    def op(conn: sqlite3.Connection) -> list[dict]:
        rows = conn.execute("SELECT * FROM api_keys ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    return _read(op)


def revoke_api_key(key_id: str) -> bool:
    def op(conn: sqlite3.Connection) -> bool:
        cur = conn.execute(
            "UPDATE api_keys SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
            (_now(), key_id),
        )
        return cur.rowcount > 0

    return _write(op)


def touch_api_key(key_id: str) -> None:
    def op(conn: sqlite3.Connection) -> None:
        conn.execute("UPDATE api_keys SET last_used_at = ? WHERE id = ?", (_now(), key_id))

    _write(op)


def revoke_session(jti: str, expires_at: int) -> None:
    def op(conn: sqlite3.Connection) -> None:
        now = int(datetime.now(timezone.utc).timestamp())
        conn.execute("DELETE FROM revoked_sessions WHERE expires_at < ?", (now,))
        conn.execute(
            "INSERT OR IGNORE INTO revoked_sessions(jti, expires_at) VALUES(?, ?)",
            (jti, expires_at),
        )

    _write(op)


def session_revoked(jti: str) -> bool:
    def op(conn: sqlite3.Connection) -> bool:
        return conn.execute("SELECT 1 FROM revoked_sessions WHERE jti = ?", (jti,)).fetchone() is not None

    return _read(op)
