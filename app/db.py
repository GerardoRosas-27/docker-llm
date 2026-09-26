"""Registro SQLite de los modelos descargados en el volumen."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from app import config
from app.policy import slugify

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
    UNIQUE (repo_id, filename)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
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
            conn.commit()
        finally:
            conn.close()


def reset() -> None:
    def op(conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM models")

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
    base = slugify(filename)
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


def upsert_download(repo_id: str, filename: str, size_bytes: int) -> dict:
    """Crea o reencola un modelo para descarga. Devuelve la fila."""

    def op(conn: sqlite3.Connection) -> dict:
        existing = conn.execute(
            "SELECT * FROM models WHERE repo_id = ? AND filename = ?",
            (repo_id, filename),
        ).fetchone()
        stamp = _now()
        if existing is None:
            slug = _unique_slug(conn, repo_id, filename)
            conn.execute(
                """
                INSERT INTO models (
                    slug, repo_id, filename, size_bytes, local_path, status,
                    progress, bytes_downloaded, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, NULL, 'queued', 0, 0, NULL, ?, ?)
                """,
                (slug, repo_id, filename, size_bytes, stamp, stamp),
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
                       bytes_downloaded = 0, error = NULL, updated_at = ?
                 WHERE slug = ?
                """,
                (size_bytes, stamp, slug),
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
