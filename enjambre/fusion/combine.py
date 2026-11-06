"""Combinación de deltas de SESGO (steering). No es fusión de ΔW rank-1.

Modos:
- multi_sum / multi_avg: aplicar cada bias en su capa de origen (suma o promedio por capa).
- single_sum / single_avg: REUBICAR todos los vectores a una capa común L* (hipótesis;
  no equivalente a origen) y sumar/promediar allí, con escala global g.
- Ablation multi_sum_raw: igual que multi_sum (sin renormalizar vectores).

Si varias ediciones caen en la misma capa, se suman (o promedian) los scale*vec.
"""
from __future__ import annotations
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import torch

from enjambre.edits import BiasEdit, EditDelta


Mode = Literal["multi_sum", "multi_avg", "single_sum", "single_avg"]


def _accum_by_layer(deltas: Sequence[EditDelta], *, average: bool) -> Dict[int, torch.Tensor]:
    """layer -> vector efectivo (ya incluye scale de cada edit)."""
    buckets: Dict[int, List[torch.Tensor]] = {}
    for d in deltas:
        for b in d.bias_edits:
            buckets.setdefault(b.layer, []).append((b.scale * b.vec.float()).detach().clone())
    out: Dict[int, torch.Tensor] = {}
    for L, vecs in buckets.items():
        s = torch.stack(vecs, 0).sum(0)
        if average:
            s = s / len(vecs)
        out[L] = s
    return out


def fuse_deltas(
    deltas: Sequence[EditDelta],
    *,
    mode: Mode,
    target_layer: Optional[int] = None,
    global_scale: float = 1.0,
    name: str = "fusion",
) -> EditDelta:
    """Devuelve un EditDelta fusionado (solo bias_edits)."""
    if mode in ("multi_sum", "multi_avg"):
        acc = _accum_by_layer(deltas, average=(mode == "multi_avg"))
        edits = []
        for L, vec in sorted(acc.items()):
            v = vec * global_scale
            n = float(v.norm()) + 1e-8
            edits.append(BiasEdit(layer=L, vec=(v / n), scale=n))  # store unit*scale form
        return EditDelta(name=name, bias_edits=edits, meta={
            "mode": mode, "global_scale": global_scale,
            "layers": [e.layer for e in edits],
            "note": "biases applied on original layers",
        })

    if mode in ("single_sum", "single_avg"):
        if target_layer is None:
            raise ValueError("target_layer required for single_* modes")
        # relocate: ignore original layer, treat each scale*vec as contribution at L*
        vecs = []
        for d in deltas:
            for b in d.bias_edits:
                vecs.append((b.scale * b.vec.float()).detach().clone())
        if not vecs:
            return EditDelta(name=name, bias_edits=[], meta={"mode": mode, "empty": True})
        s = torch.stack(vecs, 0).sum(0)
        if mode == "single_avg":
            s = s / len(vecs)
        s = s * global_scale
        n = float(s.norm()) + 1e-8
        return EditDelta(
            name=name,
            bias_edits=[BiasEdit(layer=int(target_layer), vec=(s / n), scale=n)],
            meta={
                "mode": mode, "target_layer": int(target_layer), "global_scale": global_scale,
                "n_contributions": len(vecs),
                "note": "RELOCATION hypothesis: vectors moved to a common layer; not equivalent to origin",
            },
        )
    raise ValueError(mode)


def count_bias_params_materialized(delta: EditDelta, existing_bias_layers: set) -> Dict[str, int]:
    """Cuántos parámetros de bias nuevos se materializan al aplicar (si Linear.bias era None)."""
    new_layers = {b.layer for b in delta.bias_edits if b.layer not in existing_bias_layers}
    dim = int(delta.bias_edits[0].vec.numel()) if delta.bias_edits else 0
    return {"new_bias_layers": len(new_layers), "new_bias_params": len(new_layers) * dim, "dim": dim}
