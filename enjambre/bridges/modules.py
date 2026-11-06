"""Puentes residuales prev→later con gate input-dependent y proyección final zero-init.

Identidad en init: down.weight=0 ⇒ delta=0 ⇒ h_tgt sin cambio.
Gradiente no nulo en down (y luego en up/gate) desde el primer paso.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


BRIDGE_PAIRS = [(12, 16), (16, 20), (20, 23)]  # forward-only late bank


class ResidualBridge(nn.Module):
    def __init__(self, hidden: int, rank: int = 16, gate_bias: float = -4.0):
        super().__init__()
        self.src_layer: int = -1
        self.tgt_layer: int = -1
        self.up = nn.Linear(hidden, rank, bias=False)
        self.down = nn.Linear(rank, hidden, bias=False)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.zeros_(self.down.weight)  # identity path
        self.gate = nn.Linear(hidden, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, h_src: torch.Tensor, h_tgt: torch.Tensor) -> torch.Tensor:
        # h_*: [B, T, D] — mismo token index; no mezcla temporal
        g = torch.sigmoid(self.gate(h_tgt))  # [B,T,1]
        delta = self.down(F.gelu(self.up(h_src)))
        return h_tgt + g * delta


class BridgeBank(nn.Module):
    """Banco de puentes + buffers de estados por forward (limpiados al inicio)."""

    def __init__(self, hidden: int, rank: int = 16, pairs: Optional[List[Tuple[int, int]]] = None):
        super().__init__()
        self.pairs = list(pairs or BRIDGE_PAIRS)
        self.bridges = nn.ModuleDict()
        for s, t in self.pairs:
            b = ResidualBridge(hidden, rank=rank)
            b.src_layer, b.tgt_layer = s, t
            self.bridges[f"{s}_{t}"] = b
        self._buf: Dict[int, torch.Tensor] = {}
        self._hooks: List = []
        self.enabled: bool = True

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def clear_buf(self) -> None:
        self._buf.clear()

    def detach_all(self) -> None:
        self.clear_buf()

    def attach(self, model: nn.Module) -> None:
        """Registra pre-hook (clear) + forward hooks en capas source/target."""
        self.remove_hooks()
        layers = model.model.layers
        src_set = {s for s, _ in self.pairs}
        tgt_map = {t: s for s, t in self.pairs}

        def pre_clear(_mod, _inp):
            # un buffer limpio por forward (prefill o token-step); evita stale graphs
            self.clear_buf()

        self._hooks.append(layers[0].register_forward_pre_hook(pre_clear))

        def make_hook(idx: int):
            def hook(_mod, _inp, out):
                if not self.enabled:
                    return out
                h = out
                # 1) si es target: aplicar puente con src previo del MISMO forward
                if idx in tgt_map:
                    src = tgt_map[idx]
                    if src in self._buf:
                        h_src = self._buf[src]
                        if h_src.shape[1] != h.shape[1]:
                            T = min(h_src.shape[1], h.shape[1])
                            h_src = h_src[:, -T:]
                            h = h[:, -T:]
                        key = f"{src}_{idx}"
                        h_new = self.bridges[key](h_src, h)
                        if h_new.shape[1] == out.shape[1]:
                            h = h_new
                        else:
                            out2 = out.clone()
                            out2[:, -h_new.shape[1]:, :] = h_new
                            h = out2
                # 2) si es source: guardar estado post-block (post-puente entrante si hubo)
                if idx in src_set:
                    self._buf[idx] = h
                return h if idx in tgt_map else out
            return hook

        needed = sorted(src_set | set(tgt_map))
        for idx in needed:
            self._hooks.append(layers[idx].register_forward_hook(make_hook(idx)))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.clear_buf()
