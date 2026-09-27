"""Reglas de descarga: solo GGUF y nunca por encima de 9 GB."""

from __future__ import annotations

import os
import re

MAX_MODEL_BYTES_DEFAULT = 9_000_000_000

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


_SPLIT = re.compile(
    r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$",
    re.IGNORECASE,
)


def split_info(filename: str) -> dict | None:
    """Parte de un GGUF dividido, por ejemplo modelo-00002-of-00008.gguf."""
    base = (filename or "").replace("\\", "/").split("/")[-1]
    match = _SPLIT.match(base)
    if not match:
        return None
    index = int(match.group("index"))
    total = int(match.group("total"))
    if total < 2 or index < 1 or index > total:
        return None
    return {"stem": match.group("stem"), "index": index, "total": total, "name": base}


def part_name(stem: str, index: int, total: int) -> str:
    return f"{stem}-{index:05d}-of-{total:05d}.gguf"


def slug_source(filename: str) -> str:
    info = split_info(filename)
    if info:
        return info["stem"] + ".gguf"
    return filename


def non_model_reason(filename: str) -> str | None:
    """Archivos GGUF que no son pesos de un modelo de chat."""
    base = (filename or "").replace("\\", "/").split("/")[-1].lower()
    if "imatrix" in base:
        return (
            "Este archivo es una matriz imatrix, no un modelo. "
            "Sirve para cuantizar pesos, no para chatear. "
            "Elimínalo y descarga un GGUF de pesos, por ejemplo uno Q4_K_M."
        )
    if "mmproj" in base:
        return (
            "Este archivo es el proyector de visión (mmproj), no el modelo de texto. "
            "Descarga el GGUF principal."
        )
    return None


def split_load_reason(filename: str, sibling_names: list[str] | None = None) -> str | None:
    accessory = non_model_reason(filename)
    if accessory:
        return accessory
    info = split_info(filename)
    if not info:
        return None
    if info["index"] != 1:
        return (
            f"Esta descarga es solo la parte {info['index']} de {info['total']}. "
            "Un modelo partido no arranca con una parte suelta: hacen falta todas, "
            "y llama.cpp abre la primera. Elimínala. Si el conjunto completo pesa "
            "9 GB o menos, descárgalo de nuevo desde el catálogo."
        )
    if sibling_names is None:
        return None
    have = {name.replace("\\", "/").split("/")[-1].lower() for name in sibling_names}
    missing = [
        index
        for index in range(1, info["total"] + 1)
        if part_name(info["stem"], index, info["total"]).lower() not in have
    ]
    if missing:
        listed = ", ".join(str(index) for index in missing)
        return f"Faltan las partes {listed} de {info['total']}."
    return None


def group_gguf_entries(entries: list[dict]) -> list[dict]:
    """Junta las partes de un mismo GGUF y deja el resto como archivos sueltos."""
    buckets: dict[tuple[str, int], list[dict]] = {}
    singles: list[dict] = []
    for entry in entries:
        info = split_info(entry["filename"])
        if not info:
            singles.append({**entry, "split": False, "parts": [entry["filename"]]})
            continue
        buckets.setdefault((info["stem"].lower(), info["total"]), []).append(entry)
    grouped: list[dict] = []
    for (_stem, total), parts in buckets.items():
        by_index: dict[int, dict] = {}
        for part in parts:
            info = split_info(part["filename"])
            if info:
                by_index[info["index"]] = part
        sample = by_index.get(1) or next(iter(by_index.values()))
        info = split_info(sample["filename"])
        missing = [index for index in range(1, total + 1) if index not in by_index]
        known = [by_index[index].get("size_bytes") for index in range(1, total + 1) if index in by_index]
        complete = not missing and all(size is not None for size in known)
        ordered = [by_index[index]["filename"] for index in range(1, total + 1) if index in by_index]
        first = by_index.get(1)
        grouped.append(
            {
                "filename": first["filename"] if first else sample["filename"],
                "size_bytes": sum(known) if complete else None,
                "quant": quant_label(sample["filename"]),
                "split": True,
                "part_total": total,
                "part_count": len(by_index),
                "missing_parts": missing,
                "parts": ordered if first else [],
                "stem": info["stem"] if info else sample["filename"],
            }
        )
    return singles + grouped
