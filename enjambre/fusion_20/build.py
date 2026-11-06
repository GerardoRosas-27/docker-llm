"""Construye deltas K20-A (literal) y K20-B (L2 global = K12) desde K12 congelado.

A: capas 12..23 bit-idénticas a K12; capas 4..11 nuevas con misma scale/capa que K12
   (más presupuesto total L2 = scale*√20).
B: mismas 20 capas/direcciones unitarias; reescala TODAS para total_l2 == L2(K12).
Siempre se aplica sobre base fresca vía SnapshotActivator (nunca sumar a base ya editada).
"""
from __future__ import annotations
import hashlib
import json
import os
from typing import Any, Dict, List, Tuple

import torch

from enjambre.edits import BiasEdit, EditDelta
from enjambre.fusion_confirm12.frozen import (
    DEFAULT_DIRS,
    DEFAULT_PRIOR,
    FROZEN_K12,
    delta_content_sha256,
    rebuild_k12_delta,
)
from enjambre.fusion_sweep.build_delta import total_l2
from enjambre.fusion_sweep.directions import load_directions
from enjambre.scaling import save_delta

# Diseños CONGELADOS antes de evaluar
K20_LAYERS = list(range(4, 24))  # 20 capas existentes: 4..23
K20_NEW_LAYERS = list(range(4, 12))  # 8 nuevas: 4..11
K20_KEEP_LAYERS = list(range(12, 24))  # las 12 de K12
assert K20_LAYERS == K20_NEW_LAYERS + K20_KEEP_LAYERS
assert len(K20_LAYERS) == 20

FROZEN_DESIGNS = {
    "k12": {
        "name": "k12_exact",
        "layers": list(FROZEN_K12["layers"]),
        "note": "exact rebuild from confirm12 frozen",
    },
    "k20_A_literal": {
        "name": "k20_A_literal",
        "layers": K20_LAYERS,
        "new_layers": K20_NEW_LAYERS,
        "keep_layers": K20_KEEP_LAYERS,
        "budget": "same_per_layer_scale_as_k12_MORE_total_L2",
    },
    "k20_B_normalized": {
        "name": "k20_B_normalized",
        "layers": K20_LAYERS,
        "budget": "same_total_L2_as_k12_less_per_layer",
    },
}


def _unit_dir_at_layer(directions: Dict[str, torch.Tensor], layer: int) -> torch.Tensor:
    vecs = []
    for d in directions.values():
        v = d[layer + 1].float()
        n = float(v.norm()) + 1e-8
        vecs.append(v / n)
    mean_v = torch.stack(vecs, 0).mean(0)
    n = float(mean_v.norm()) + 1e-8
    return (mean_v / n).clone()


def build_all_deltas(
    directions_path: str = DEFAULT_DIRS,
    prior_path: str = DEFAULT_PRIOR,
) -> Dict[str, Any]:
    k12, k12_meta = rebuild_k12_delta(directions_path, prior_path)
    assert [b.layer for b in k12.bias_edits] == K20_KEEP_LAYERS
    scale_k12 = float(k12.bias_edits[0].scale)
    l2_k12 = total_l2(k12)
    # map layer -> BiasEdit from k12 (for bit-identical copy)
    by_L = {b.layer: b for b in k12.bias_edits}
    dirs = load_directions(directions_path)

    # --- A literal ---
    edits_a: List[BiasEdit] = []
    for L in K20_NEW_LAYERS:
        vec = _unit_dir_at_layer(dirs, L)
        edits_a.append(BiasEdit(layer=L, vec=vec, scale=scale_k12))
    for L in K20_KEEP_LAYERS:
        b = by_L[L]
        # clone tensors for safety but values bit-identical
        edits_a.append(BiasEdit(
            layer=L,
            vec=b.vec.detach().float().clone(),
            scale=float(b.scale),
        ))
    delta_a = EditDelta(
        name="k20_A_literal",
        bias_edits=edits_a,
        meta={
            **FROZEN_DESIGNS["k20_A_literal"],
            "scale_per_layer": scale_k12,
            "total_l2": None,  # fill below
            "k12_delta_sha256": k12_meta["delta_sha256"],
            "kind": "extend_k12_literal_plus_new_layers",
        },
    )
    delta_a.meta["total_l2"] = total_l2(delta_a)

    # verify bit-identical keep layers vs k12
    for b_a, b_k in zip(
        [e for e in delta_a.bias_edits if e.layer >= 12],
        k12.bias_edits,
    ):
        assert b_a.layer == b_k.layer
        assert abs(b_a.scale - b_k.scale) < 1e-12
        assert torch.equal(b_a.vec.cpu(), b_k.vec.cpu()), b_a.layer

    # --- B normalized: same unit vecs as A, rescale all to L2(K12) ---
    unit_by_L = {}
    for b in delta_a.bias_edits:
        v = b.vec.float()
        unit_by_L[b.layer] = v / (float(v.norm()) + 1e-8)
    scale_b = l2_k12 / (20 ** 0.5)
    edits_b = [
        BiasEdit(layer=L, vec=unit_by_L[L].clone(), scale=scale_b)
        for L in K20_LAYERS
    ]
    delta_b = EditDelta(
        name="k20_B_normalized",
        bias_edits=edits_b,
        meta={
            **FROZEN_DESIGNS["k20_B_normalized"],
            "scale_per_layer": scale_b,
            "total_l2": None,
            "target_l2": l2_k12,
            "k12_delta_sha256": k12_meta["delta_sha256"],
            "kind": "k20_same_directions_constant_total_L2",
        },
    )
    delta_b.meta["total_l2"] = total_l2(delta_b)
    assert abs(total_l2(delta_b) - l2_k12) < 1e-5

    def pack(delta: EditDelta, extra: Dict = None) -> Dict[str, Any]:
        meta = {
            "name": delta.name,
            "layers": [b.layer for b in delta.bias_edits],
            "scales": [float(b.scale) for b in delta.bias_edits],
            "total_l2": total_l2(delta),
            "delta_sha256": delta_content_sha256(delta),
            "n_layers": len(delta.bias_edits),
            **(delta.meta or {}),
            **(extra or {}),
        }
        return {"delta": delta, "meta": meta}

    return {
        "k12": pack(k12, {"source_meta": k12_meta}),
        "k20_A": pack(delta_a),
        "k20_B": pack(delta_b),
        "designs_frozen": FROZEN_DESIGNS,
        "scale_k12": scale_k12,
        "l2_k12": l2_k12,
    }


def export_deltas(out_dir: str, bundle: Dict[str, Any]) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for key, fname in [("k12", "k12_exact.pt"), ("k20_A", "k20_A_literal.pt"), ("k20_B", "k20_B_normalized.pt")]:
        path = os.path.join(out_dir, fname)
        save_delta(path, bundle[key]["delta"])
        paths[key] = path
        with open(os.path.join(out_dir, fname.replace(".pt", "_meta.json")), "w", encoding="utf-8") as f:
            json.dump(bundle[key]["meta"], f, indent=2, ensure_ascii=False, default=str)
    with open(os.path.join(out_dir, "designs_frozen.json"), "w", encoding="utf-8") as f:
        json.dump({
            "designs": FROZEN_DESIGNS,
            "K20_LAYERS": K20_LAYERS,
            "scale_k12": bundle["scale_k12"],
            "l2_k12": bundle["l2_k12"],
            "l2_A": bundle["k20_A"]["meta"]["total_l2"],
            "l2_B": bundle["k20_B"]["meta"]["total_l2"],
            "hashes": {k: bundle[k]["meta"]["delta_sha256"] for k in ("k12", "k20_A", "k20_B")},
        }, f, indent=2, ensure_ascii=False)
    return paths
