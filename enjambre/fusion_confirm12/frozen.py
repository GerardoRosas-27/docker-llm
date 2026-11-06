"""Parámetros CONGELADOS del barrido (curve_by_K K=12). No reoptimizar."""
from __future__ import annotations
import hashlib
import json
import os
from typing import Any, Dict, Tuple

import torch

from enjambre.build_experts import load_delta
from enjambre.edits import EditDelta
from enjambre.fusion_sweep.build_delta import (
    budget_B_from_prior_fusion,
    fuse_directions_on_layers,
    total_l2,
)
from enjambre.fusion_sweep.directions import load_directions
from enjambre.scaling import save_delta

# Exactos del artifact curve_by_K.json / winners.json @ a3b7a16
FROZEN_K12 = {
    "K": 12,
    "layout": "late",
    "layers": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    "scale_mult": 0.5,
    "sweep_val_mean": 0.65,
    "sweep_final_mean": 0.77,  # holdout n=20 previo; NO comparar contra base nuevo
    "sweep_val_key": "K12_late_L12-13-14-15-16-17-18-19-20-21-22-23_s0.5",
    "sweep_commit": "a3b7a16d7259991f2b5d9550ca5720e8f170fcc3",
}

DEFAULT_DIRS = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "fusion-sweep-0.5b", "directions.pt"
)
DEFAULT_PRIOR = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "fusion-0.5b", "fused_selected.pt"
)


def delta_content_sha256(delta: EditDelta) -> str:
    h = hashlib.sha256()
    for b in sorted(delta.bias_edits, key=lambda x: x.layer):
        h.update(int(b.layer).to_bytes(2, "little"))
        h.update(f"{float(b.scale):.12f}".encode())
        h.update(b.vec.detach().cpu().float().contiguous().numpy().tobytes())
    return h.hexdigest()


def rebuild_k12_delta(
    directions_path: str = DEFAULT_DIRS,
    prior_path: str = DEFAULT_PRIOR,
) -> Tuple[EditDelta, Dict[str, Any]]:
    """Reconstruye delta K12 de forma determinista desde directions + params congelados."""
    dirs = load_directions(directions_path)
    prior = load_delta(prior_path)
    B = budget_B_from_prior_fusion(prior)
    sm = float(FROZEN_K12["scale_mult"])
    B_eff = B * sm
    layers = list(FROZEN_K12["layers"])
    delta = fuse_directions_on_layers(
        dirs, layers, B=B_eff, name="confirm_K12_late_s0.5"
    )
    meta = {
        **FROZEN_K12,
        "B": B,
        "B_eff": B_eff,
        "total_l2": total_l2(delta),
        "scale_per_layer": delta.meta.get("scale_per_layer"),
        "delta_sha256": delta_content_sha256(delta),
        "directions_path": os.path.abspath(directions_path),
        "directions_sha256": hashlib.sha256(open(directions_path, "rb").read()).hexdigest(),
        "prior_path": os.path.abspath(prior_path),
        "kind": "per_layer_direction_mean",
        "note": "Rebuilt from frozen sweep params; NOT re-selected on confirm data",
    }
    delta.meta = {**delta.meta, **meta}
    # sanity vs sweep reported total_l2
    assert abs(total_l2(delta) - 1.6404) < 1e-3, total_l2(delta)
    assert [b.layer for b in delta.bias_edits] == layers
    return delta, meta


def export_k12(out_dir: str, delta: EditDelta, meta: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "k12_frozen_delta.pt")
    save_delta(path, delta)
    with open(os.path.join(out_dir, "k12_frozen_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return path
