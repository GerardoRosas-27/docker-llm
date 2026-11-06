"""Construye EditDelta fusionado en K capas con presupuesto L2 comparable.

Presupuesto: suma_i ||bias_i||_2^2 = B^2  => cada capa (si unit vec) scale = B/sqrt(K).
Así más capas no implica automáticamente más edición total.
Variante redistribute_prior: reubica/promedia los scale*vec de deltas .pt previos
(no estima dirección nueva) — marcada explícitamente como hipótesis distinta.
"""
from __future__ import annotations
from typing import Dict, List, Sequence

import torch

from enjambre.edits import BiasEdit, EditDelta


def budget_B_from_prior_fusion(fused: EditDelta) -> float:
    """L2 total efectivo del multi_avg previo: sqrt(sum scale^2) si vec unitarios."""
    s2 = 0.0
    for b in fused.bias_edits:
        s2 += float(b.scale) ** 2
    return s2 ** 0.5


def fuse_directions_on_layers(
    directions: Dict[str, torch.Tensor],
    layers: Sequence[int],
    *,
    B: float,
    name: str = "sweep",
) -> EditDelta:
    """En cada capa L: promedio de unit dirs de expertos en hidden_states[L+1]; scale=B/sqrt(K)."""
    K = len(layers)
    scale = float(B) / (K ** 0.5)
    edits = []
    for L in layers:
        vecs = []
        for eid, d in directions.items():
            # d[0]=embed, d[L+1]=after block L
            v = d[L + 1].float()
            n = float(v.norm()) + 1e-8
            vecs.append(v / n)
        mean_v = torch.stack(vecs, 0).mean(0)
        n = float(mean_v.norm()) + 1e-8
        edits.append(BiasEdit(layer=int(L), vec=(mean_v / n), scale=scale))
    return EditDelta(name=name, bias_edits=edits, meta={
        "kind": "per_layer_direction_mean",
        "layers": list(layers),
        "K": K,
        "B": B,
        "scale_per_layer": scale,
        "budget": "equal_L2_per_layer_sum_sq_equals_B2",
    })


def redistribute_prior_biases(
    prior_deltas: Sequence[EditDelta],
    layers: Sequence[int],
    *,
    B: float,
    name: str = "redistribute",
) -> EditDelta:
    """Promedia todos los scale*vec previos y los coloca (misma dirección media) en cada capa target.

    NO estima activaciones por capa; solo redistribuye el sesgo ya horneado.
    """
    vecs = []
    for d in prior_deltas:
        for b in d.bias_edits:
            vecs.append((b.scale * b.vec.float()).detach())
    mean_v = torch.stack(vecs, 0).mean(0)
    n = float(mean_v.norm()) + 1e-8
    unit = mean_v / n
    K = len(layers)
    scale = float(B) / (K ** 0.5)
    edits = [BiasEdit(layer=int(L), vec=unit.clone(), scale=scale) for L in layers]
    return EditDelta(name=name, bias_edits=edits, meta={
        "kind": "redistribute_prior_bias",
        "layers": list(layers),
        "K": K,
        "B": B,
        "scale_per_layer": scale,
        "note": "NOT per-layer direction estimate; relocation of prior fused direction",
    })


def total_l2(delta: EditDelta) -> float:
    return sum(float(b.scale) ** 2 for b in delta.bias_edits) ** 0.5
