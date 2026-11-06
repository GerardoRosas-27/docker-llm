"""Manifest auditable: identidad base, hashes y tipo real de edición."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch

from enjambre.build_experts import load_delta
from enjambre.experts import all_expert_specs


def _file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_ten_deltas(deltas_dir: str) -> Dict[str, Any]:
    """Carga exactamente los 10 expertos de all_expert_specs(); falla si falta alguno."""
    specs = all_expert_specs()
    assert len(specs) == 10
    experts = []
    missing = []
    for s in specs:
        path = os.path.join(deltas_dir, f"delta_{s.id}.pt")
        if not os.path.isfile(path):
            missing.append(s.id)
            continue
        d = load_delta(path)
        kind = "steering_bias"
        if d.rank1_edits:
            kind = "mixed_bias_and_rank1_weight"
        layers = [b.layer for b in d.bias_edits]
        scales = [float(b.scale) for b in d.bias_edits]
        norms = [float(b.vec.float().norm()) for b in d.bias_edits]
        experts.append({
            "id": s.id,
            "path": os.path.abspath(path),
            "sha256": _file_sha256(path),
            "edit_kind": kind,
            "n_bias_edits": len(d.bias_edits),
            "n_rank1_edits": len(d.rank1_edits),
            "layers": layers,
            "scales": scales,
            "vec_norms": norms,
            "vec_dim": int(d.bias_edits[0].vec.numel()) if d.bias_edits else None,
            "meta": dict(d.meta),
            "n_tune": len(s.tune_items),
            "n_eval": len(s.eval_items),
            "note": (
                "Bias steering: se suma scale*vec al bias de mlp.down_proj. "
                "Si el Linear no tenía bias, se crea un parámetro nuevo de sesgo "
                "(+hidden_size params por capa tocada). NO es ΔW rank-1."
            ),
        })
    if missing:
        raise FileNotFoundError(f"faltan deltas: {missing} en {deltas_dir}")
    return {
        "n_experts": len(experts),
        "deltas_dir": os.path.abspath(deltas_dir),
        "experts": experts,
        "edit_kind_summary": "all_steering_bias" if all(e["edit_kind"] == "steering_bias" for e in experts) else "mixed",
    }


def build_manifest(model_key: str, hf_id: str, deltas_dir: str, extra: Dict | None = None) -> Dict[str, Any]:
    ten = load_ten_deltas(deltas_dir)
    man = {
        "experiment": "fusion-expertos-10",
        "base_model_key": model_key,
        "base_hf_id": hf_id,
        "revision_note": "pesos base HuggingFace tal cual; deltas locales versionados por sha256",
        "deltas": ten,
        "hypotheses": {
            "H1_multi_layer_sum": "Sumar sesgos en sus capas de origen puede interferir; no garantiza mejora vs selector.",
            "H2_single_layer_fuse": "Reubicar todos los vectores a UNA capa común NO es equivalente a aplicarlos en origen; es hipótesis exploratoria.",
            "H3_growth": "Fusionar deltas bias en la misma arquitectura NO crece parámetros del stack Transformer; como mucho materializa bias ausente.",
            "H4_parallel_mlp": "MLP paralelo con N ramas y average de salidas, si solo difieren en bias de down_proj, es linealmente equivalente a un solo bias promedio (demostrable).",
        },
    }
    if extra:
        man.update(extra)
    return man
