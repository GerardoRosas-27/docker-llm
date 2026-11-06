"""Arnés adaptativo: ~50 especialistas mono-tarea sobre UNA base Qwen2.5-0.5B.

No carga 50 modelos: un solo modelo + deltas de sesgo rank-1. Activación por
snapshot exacto de capas tocadas (no suma/resta acumulativa). Memoria = grafo
asociativo SQLite (NO red neuronal). Router heurístico por keywords/id.
Auto-mejora: versiona deltas exitosos, propone candidatos, valida y promueve
atómicamente solo si superan al incumbente (heldout final intacto).
"""
from __future__ import annotations

def AdaptiveHarness(*args, **kwargs):
    from .orchestrator import AdaptiveHarness as _H
    return _H(*args, **kwargs)

__all__ = ["AdaptiveHarness"]
