"""Activación de deltas con snapshot exacto de capas tocadas + lock.

Restaura copiando tensores guardados (no suma/resta acumulativa de Δ).
En fp32 el restore es bit-a-bit respecto al snapshot; el lock serializa
activaciones concurrentes sobre el único modelo en RAM.
"""
from __future__ import annotations
import threading
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn

from enjambre.edits import BiasEdit, EditDelta


class SnapshotActivator:
    """Aplica un EditDelta (sesgos) tras snapshot; restore() copia el backup."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._lock = threading.RLock()
        self._bias_snap: Dict[int, Optional[torch.Tensor]] = {}
        self._active: Optional[str] = None
        self._depth = 0  # nested activate depth within same lock holder

    def layers(self):
        return self.model.model.layers

    def _down(self, layer: int) -> nn.Linear:
        return self.layers()[layer].mlp.down_proj

    def active(self) -> Optional[str]:
        return self._active

    def _snapshot_layer(self, layer: int) -> None:
        if layer in self._bias_snap:
            return
        lin = self._down(layer)
        self._bias_snap[layer] = None if lin.bias is None else lin.bias.detach().clone()

    def _apply_bias(self, edit: BiasEdit) -> None:
        self._snapshot_layer(edit.layer)
        lin = self._down(edit.layer)
        vec = (edit.scale * edit.vec).to(dtype=lin.weight.dtype, device=lin.weight.device)
        with torch.no_grad():
            if lin.bias is None:
                lin.bias = nn.Parameter(vec.clone())
            else:
                base = self._bias_snap[edit.layer]
                if base is None:
                    lin.bias.copy_(vec)
                else:
                    lin.bias.copy_(base.to(lin.bias.dtype) + vec)

    def restore(self) -> None:
        """Restaura exactamente los bias snapshoteados."""
        for layer, bak in list(self._bias_snap.items()):
            lin = self._down(layer)
            with torch.no_grad():
                if bak is None:
                    lin.bias = None
                else:
                    if lin.bias is None:
                        lin.bias = nn.Parameter(bak.clone())
                    else:
                        lin.bias.copy_(bak)
        self._bias_snap.clear()
        self._active = None

    def activate(self, delta: EditDelta) -> None:
        """Activa delta; si ya había otro, restaura antes."""
        if self._bias_snap:
            self.restore()
        for b in delta.bias_edits:
            self._apply_bias(b)
        self._active = delta.name

    @contextmanager
    def session(self, delta: Optional[EditDelta]) -> Iterator[Optional[str]]:
        """Context manager thread-safe: activate → yield → restore even on exception."""
        with self._lock:
            try:
                if delta is None:
                    if self._bias_snap:
                        self.restore()
                    yield None
                else:
                    self.activate(delta)
                    yield delta.name
            finally:
                if self._bias_snap:
                    self.restore()

    def lock(self) -> threading.RLock:
        return self._lock
