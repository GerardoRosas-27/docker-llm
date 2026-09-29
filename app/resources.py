"""RAM y CPU que de verdad tiene el contenedor (cgroups), para no colgar el chat."""

from __future__ import annotations

import math
import os
from pathlib import Path

_UNLIMITED = 1 << 60


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _meminfo(key: str) -> int | None:
    text = _read("/proc/meminfo")
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith(key + ":"):
            parts = line.split()
            try:
                return int(parts[1]) * 1024
            except (IndexError, ValueError):
                return None
    return None


def memory_limit() -> int | None:
    """Bytes de RAM que puede usar el contenedor: límite del cgroup o RAM total."""
    candidates: list[int] = []
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        raw = _read(path)
        if raw and raw.isdigit() and 0 < int(raw) < _UNLIMITED:
            candidates.append(int(raw))
    total = _meminfo("MemTotal")
    if total:
        candidates.append(total)
    return min(candidates) if candidates else None


def memory_used() -> int | None:
    raw = _read("/sys/fs/cgroup/memory.current") or _read("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if raw and raw.isdigit():
        return int(raw)
    total = _meminfo("MemTotal")
    available = _meminfo("MemAvailable")
    if total and available is not None:
        return total - available
    return None


def cpu_limit() -> int:
    """CPUs utilizables: afinidad y cuota del cgroup (Railway limita por cuota)."""
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    raw = _read("/sys/fs/cgroup/cpu.max")
    quota = period = None
    if raw:
        parts = raw.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                quota, period = int(parts[0]), int(parts[1])
            except ValueError:
                quota = period = None
    else:
        q = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        p = _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if q and p and q.lstrip("-").isdigit() and p.isdigit() and int(q) > 0:
            quota, period = int(q), int(p)
    if quota and period:
        count = min(count, max(1, math.ceil(quota / period)))
    return max(1, count)


def estimate_model_ram(size_bytes: int | None, repack: bool = False) -> int:
    """RAM aproximada para servir un GGUF en CPU con contexto corto."""
    size = int(size_bytes or 0)
    overhead = 450 * 1024 * 1024  # contexto, buffers de cómputo y proceso
    factor = 2.0 if repack else 1.05
    return int(size * factor) + overhead


def ram_block_reason(size_bytes: int | None, repack: bool = False) -> str | None:
    """Mensaje claro si el modelo no cabe en la RAM del contenedor."""
    from app.policy import format_bytes

    limit = memory_limit()
    if not limit or not size_bytes:
        return None
    need = estimate_model_ram(size_bytes, repack)
    if need <= limit:
        return None
    return (
        f"Este modelo necesita unos {format_bytes(need)} de RAM y el contenedor tiene "
        f"{format_bytes(limit)}. Se quedaría sin memoria y el chat no respondería. "
        "Usa un modelo más pequeño (por ejemplo Qwen 3.5 2B Q4_K_M, 1.40 GB) "
        "o da más RAM a Docker. Para forzarlo, RAM_CHECK=0."
    )
