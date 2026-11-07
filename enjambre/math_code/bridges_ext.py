"""Extensión de conectores residuales entre capas partiendo de `bridges_best.pt` (K12 + Puentes Entrenados).

Cuando se agregan más capas editadas (ej. K16: 8..23, K20: 4..23, K24: 0..23),
se agregan puentes residuales que conectan las nuevas capas hacia el bloque superior:
- PAIRS_K12 (3 puentes): 12→16, 16→20, 20→23
- PAIRS_K16 (4 puentes): 8→12, 12→16, 16→20, 20→23
- PAIRS_K20 (5 puentes): 4→8, 8→12, 12→16, 16→20, 20→23
- PAIRS_K24 (6 puentes): 0→4, 4→8, 8→12, 12→16, 16→20, 20→23

Los puentes 12→16, 16→20, 20→23 se inicializan con los pesos de `bridges_best.pt`
y los nuevos puentes se inicializan con `down.weight = 0` (identidad exacta en t=0 respecto
a los 3 puentes entrenados).
"""
from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple

import torch

from enjambre.bridges.modules import BRIDGE_PAIRS, BridgeBank

DEFAULT_PRETRAINED_BRIDGES = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "bridges-k12", "bridges_best.pt"
)

PAIRS_BY_CONFIG: Dict[str, List[Tuple[int, int]]] = {
    "k12": [(12, 16), (16, 20), (20, 23)],
    "k16": [(8, 12), (12, 16), (16, 20), (20, 23)],
    "k20": [(4, 8), (8, 12), (12, 16), (16, 20), (20, 23)],
    "k24": [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 23)],
}


def load_pretrained_k12_bridges_state(path: str = DEFAULT_PRETRAINED_BRIDGES) -> Dict[str, torch.Tensor]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck["state"] if isinstance(ck, dict) and "state" in ck else ck
    return {k: v.detach().cpu().float().clone() for k, v in state.items()}


def build_warm_bridge_bank(
    hidden: int,
    pairs: List[Tuple[int, int]],
    *,
    rank: int = 16,
    pretrained_state: Optional[Dict[str, torch.Tensor]] = None,
    pretrained_path: str = DEFAULT_PRETRAINED_BRIDGES,
) -> BridgeBank:
    """Construye un BridgeBank con `pairs`, copiando los pesos entrenados de `12_16`, `16_20`, `20_23`
    y manteniendo zero-init (identidad) en cualquier puente nuevo (`0_4`, `4_8`, `8_12`)."""
    bank = BridgeBank(hidden=hidden, rank=rank, pairs=pairs)
    if pretrained_state is None and os.path.isfile(pretrained_path):
        pretrained_state = load_pretrained_k12_bridges_state(pretrained_path)
    if pretrained_state:
        own = bank.state_dict()
        with torch.no_grad():
            for k, v in pretrained_state.items():
                if k in own and own[k].shape == v.shape:
                    own[k].copy_(v)
        bank.load_state_dict(own)
    # Verificar que los puentes nuevos (no presentes en pretrained_state) tengan down.weight == 0
    pretrained_keys = set(pretrained_state.keys()) if pretrained_state else set()
    for s, t in pairs:
        bkey = f"{s}_{t}"
        wkey = f"bridges.{bkey}.down.weight"
        if wkey not in pretrained_keys:
            assert torch.allclose(
                bank.bridges[bkey].down.weight,
                torch.zeros_like(bank.bridges[bkey].down.weight),
            ), f"New bridge {bkey} must start at identity (down.weight=0)"
    return bank
