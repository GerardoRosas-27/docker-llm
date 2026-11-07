"""Edición directa de pesos por capas (BiasEdit en mlp.down_proj) enfocada en Matemáticas y Programación.

Implementa:
1. Extracción de direcciones por capa (0..23) específicas para Matemáticas y Programación,
   contrastando activaciones de respuestas limpias vs respuestas contaminadas con prosa/viñetas.
2. Ortogonalización Gram-Schmidt por capa respecto a direcciones de interferencia (bullets/prosa).
3. Ensamblaje progresivo de modelos por capas editadas sobre la base K12:
   - K12_orig: capas 12..23 originales (hash b8cf061d)
   - K12_mc:   12 capas (12..23) editadas para Math + Code
   - K16_mc:   16 capas (8..23)  editadas (+4 capas intermedias-altas)
   - K20_mc:   20 capas (4..23)  editadas (+8 capas intermedias)
   - K24_mc:   24 capas (0..23)  editadas (+12 capas, modelo completo 0..23)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence

import torch

from enjambre.edits import BiasEdit, EditDelta, mean_activation_diff, unit
from enjambre.fusion_confirm12.frozen import (
    DEFAULT_DIRS,
    DEFAULT_PRIOR,
    delta_content_sha256,
    rebuild_k12_delta,
)
from enjambre.fusion_sweep.build_delta import total_l2
from enjambre.fusion_sweep.directions import load_directions
from enjambre.generate import chat_text

from .tasks import CODE_SYSTEM_PROMPT, CODE_TASKS, MATH_SYSTEM_PROMPT, MATH_TASKS


def project_out(v: torch.Tensor, u: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Elimina la proyección de `v` sobre el vector `u` (Gram-Schmidt parcial/total)."""
    un = unit(u.float())
    vf = v.float()
    dot = torch.dot(vf, un)
    return vf - strength * dot * un


def orthogonalize_against_subspace(
    v: torch.Tensor, interference_vecs: Sequence[torch.Tensor], strength: float = 0.75
) -> torch.Tensor:
    """Proyecta `v` fuera del subespacio generado por `interference_vecs`."""
    out = v.float().clone()
    basis: List[torch.Tensor] = []
    for iv in interference_vecs:
        u = iv.float().clone()
        for b in basis:
            u = u - torch.dot(u, b) * b
        if float(u.norm()) > 1e-6:
            un = unit(u)
            basis.append(un)
            out = out - strength * torch.dot(out, un) * un
    return unit(out)


@torch.no_grad()
def compute_mc_contrast_directions(
    model, tok, train_rows: Dict[str, List], batch_size: int = 8
) -> Dict[str, torch.Tensor]:
    """Calcula direcciones de activación por capa [25, D] en el modelo para Math y Code
    usando exclusivamente ejemplos del split `train`."""
    math_pos, math_neg = [], []
    for tid in MATH_TASKS:
        for p, tgt, _ in train_rows[tid][:4]:
            math_pos.append(chat_text(tok, p, MATH_SYSTEM_PROMPT) + tgt)
            math_neg.append(chat_text(tok, p, None) + tgt + ".\n- The result is ")

    code_pos, code_neg = [], []
    for tid in CODE_TASKS:
        for p, tgt, _ in train_rows[tid][:4]:
            code_pos.append(chat_text(tok, p, CODE_SYSTEM_PROMPT) + tgt)
            code_neg.append(chat_text(tok, p, None) + tgt + "\n- The function takes ")

    d_math_raw = mean_activation_diff(model, tok, math_pos, math_neg, batch_size=batch_size)
    d_code_raw = mean_activation_diff(model, tok, code_pos, code_neg, batch_size=batch_size)

    def normalize_rows(t: torch.Tensor) -> torch.Tensor:
        return torch.stack([unit(t[i].float()) if float(t[i].norm()) > 1e-6 else t[i].float()
                            for i in range(t.shape[0])]).cpu()

    return {
        "math_fresh": normalize_rows(d_math_raw),
        "code_fresh": normalize_rows(d_code_raw),
    }


def build_layer_specialist_vectors(
    directions_path: str = DEFAULT_DIRS,
    fresh_dirs: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Para cada capa L in 0..23 construye las direcciones unitarias limpias:
    - math_dir[L]: dirección especializada en matemáticas en capa L
    - code_dir[L]: dirección especializada en programación en capa L
    - combined_dir[L]: dirección fusionada ponderada por el rol funcional de la capa L
    """
    dirs = load_directions(directions_path)
    out: Dict[int, Dict[str, torch.Tensor]] = {}

    for L in range(24):
        idx = L + 1  # hidden_states[L+1] es la salida del bloque L
        d_arith = unit(dirs["arithmetic"][idx].float())
        d_code = unit(dirs["code_python"][idx].float())
        d_bullets = unit(dirs["bullets"][idx].float())
        d_spanish = unit(dirs["spanish"][idx].float())
        d_sum = unit(dirs["summarize"][idx].float())

        if fresh_dirs is not None:
            d_m = unit(0.50 * d_arith + 0.50 * unit(fresh_dirs["math_fresh"][idx].float()))
            d_c = unit(0.50 * d_code + 0.50 * unit(fresh_dirs["code_fresh"][idx].float()))
        else:
            d_m = d_arith
            d_c = d_code

        # Ortogonalizar contra bullets y prosa para evitar '- The function...' y puntos finales
        d_m_clean = orthogonalize_against_subspace(d_m, [d_bullets, d_spanish, d_sum], strength=0.80)
        d_c_clean = orthogonalize_against_subspace(d_c, [d_bullets, d_spanish, d_sum], strength=0.85)

        # Ponderación por profundidad de capa (rol funcional):
        if L < 4:
            w_m, w_c = 0.55, 0.45
        elif L < 8:
            w_m, w_c = 0.62, 0.38
        elif L < 12:
            w_m, w_c = 0.55, 0.45
        else:
            w_m, w_c = 0.50, 0.50

        comb = unit(w_m * d_m_clean + w_c * d_c_clean)
        out[L] = {
            "math": d_m_clean,
            "code": d_c_clean,
            "combined": comb,
            "bullets_interf": d_bullets,
        }
    return out


def build_math_code_layer_deltas(
    directions_path: str = DEFAULT_DIRS,
    prior_path: str = DEFAULT_PRIOR,
    fresh_dirs: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, EditDelta]:
    """Construye el catálogo de deltas por capas editadas partiendo de K12:
    - k12_orig:  12 capas (12..23) exactas de la base K12
    - k12_mc:    12 capas (12..23) con edición Math+Code sobre K12
    - k16_mc:    16 capas (8..23)  agregando 4 capas intermedias editadas (8..11)
    - k20_mc:    20 capas (4..23)  agregando 8 capas intermedias editadas (4..11)
    - k24_mc:    24 capas (0..23)  agregando 12 capas (0..11) para cubrir las 24 capas
    """
    k12, k12_meta = rebuild_k12_delta(directions_path, prior_path)
    by_L_k12 = {b.layer: b for b in k12.bias_edits}
    scale_k12 = float(k12.bias_edits[0].scale)  # ~0.473529
    spec_vecs = build_layer_specialist_vectors(directions_path, fresh_dirs=fresh_dirs)

    # 1. K12_mc: edita las capas 12..23 de K12 proyectando fuera bullets e inyectando Math+Code
    edits_k12_mc: List[BiasEdit] = []
    for L in range(12, 24):
        base_b = by_L_k12[L]
        v_k12 = unit(base_b.vec.float())
        v_k12_clean = project_out(v_k12, spec_vecs[L]["bullets_interf"], strength=0.65)
        v_mc = spec_vecs[L]["combined"]
        v_fused = unit(0.50 * unit(v_k12_clean) + 0.50 * v_mc)
        edits_k12_mc.append(BiasEdit(layer=L, vec=v_fused, scale=scale_k12))

    delta_k12_mc = EditDelta(
        name="k12_mc",
        bias_edits=edits_k12_mc,
        meta={
            "K": 12,
            "layers": list(range(12, 24)),
            "added_layers": [],
            "focus": "math+code",
            "total_l2": None,
            "base_k12_sha": k12_meta["delta_sha256"],
        },
    )
    delta_k12_mc.meta["total_l2"] = total_l2(delta_k12_mc)
    delta_k12_mc.meta["delta_sha256"] = delta_content_sha256(delta_k12_mc)

    def _extend_layers(name: str, new_layers: List[int], new_scale_factor: float) -> EditDelta:
        edits: List[BiasEdit] = []
        for L in sorted(new_layers):
            depth_ramp = 0.60 + 0.40 * (L / 12.0)
            sc = scale_k12 * new_scale_factor * depth_ramp
            edits.append(BiasEdit(layer=L, vec=spec_vecs[L]["combined"].clone(), scale=float(sc)))
        for b in edits_k12_mc:
            edits.append(BiasEdit(layer=b.layer, vec=b.vec.clone(), scale=float(b.scale)))
        d = EditDelta(
            name=name,
            bias_edits=edits,
            meta={
                "K": len(edits),
                "layers": [e.layer for e in edits],
                "added_layers": list(new_layers),
                "focus": "math+code",
                "new_scale_factor": new_scale_factor,
                "base_k12_sha": k12_meta["delta_sha256"],
            },
        )
        d.meta["total_l2"] = total_l2(d)
        d.meta["delta_sha256"] = delta_content_sha256(d)
        return d

    # 2. K16_mc: 16 capas (8..23) -> agrega 4 capas intermedias-altas (8, 9, 10, 11)
    delta_k16_mc = _extend_layers("k16_mc", list(range(8, 12)), new_scale_factor=0.55)

    # 3. K20_mc: 20 capas (4..23) -> agrega 8 capas intermedias (4..11)
    delta_k20_mc = _extend_layers("k20_mc", list(range(4, 12)), new_scale_factor=0.50)

    # 4. K24_mc: 24 capas (0..23) -> agrega las 12 capas inferiores (0..11)
    delta_k24_mc = _extend_layers("k24_mc", list(range(0, 12)), new_scale_factor=0.42)

    return {
        "k12_orig": k12,
        "k12_mc": delta_k12_mc,
        "k16_mc": delta_k16_mc,
        "k20_mc": delta_k20_mc,
        "k24_mc": delta_k24_mc,
    }
