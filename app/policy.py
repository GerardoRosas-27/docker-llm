"""Reglas de descarga: solo GGUF y nunca por encima de 8 GB."""

from __future__ import annotations

import os
import re

MAX_MODEL_BYTES_DEFAULT = 8_000_000_000

_REPO = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,96}/[A-Za-z0-9][A-Za-z0-9._-]{0,128}$"
)
_QUANT = re.compile(r"(IQ\d[A-Z0-9_]*|Q\d[A-Z0-9_]*|BF16|F16|F32|FP16)", re.I)


class PolicyError(ValueError):
    """El archivo no se puede descargar."""


def max_model_bytes() -> int:
    return int(os.environ.get("MAX_MODEL_BYTES", str(MAX_MODEL_BYTES_DEFAULT)))


def format_bytes(n: int | None) -> str:
    if n is None:
        return "tamaño desconocido"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f} GB"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.1f} KB"
    return f"{n} B"


def evaluate(size_bytes: int | None) -> tuple[bool, str]:
    """Devuelve (permitido, motivo). El motivo está vacío cuando se permite."""
    limit = max_model_bytes()
    if size_bytes is None:
        return False, "No se pudo conocer el tamaño del archivo. No se descarga."
    if size_bytes <= 0:
        return False, "Tamaño de archivo inválido."
    if size_bytes > limit:
        return (
            False,
            f"El archivo pesa {format_bytes(size_bytes)} y el límite es {format_bytes(limit)}.",
        )
    return True, ""


def require_allowed(size_bytes: int | None) -> None:
    allowed, reason = evaluate(size_bytes)
    if not allowed:
        raise PolicyError(reason)


def validate_repo(repo_id: str) -> str:
    repo_id = (repo_id or "").strip()
    if not _REPO.match(repo_id):
        raise PolicyError("Repositorio de Hugging Face inválido.")
    return repo_id


def validate_filename(filename: str) -> str:
    name = (filename or "").strip()
    if not name or "\\" in name or name.startswith("/"):
        raise PolicyError("Nombre de archivo inválido.")
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise PolicyError("Nombre de archivo inválido.")
    if not parts[-1].lower().endswith(".gguf"):
        raise PolicyError("Solo se pueden descargar archivos GGUF.")
    return name


def slugify(filename: str) -> str:
    base = filename.split("/")[-1]
    if base.lower().endswith(".gguf"):
        base = base[:-5]
    slug = re.sub(r"[^a-z0-9.]+", "-", base.lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-.")
    return (slug or "modelo")[:80]


def quant_label(filename: str) -> str | None:
    match = _QUANT.search(filename.split("/")[-1])
    return match.group(1).upper() if match else None
