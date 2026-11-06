"""Direcciones por experto y capa (contraste system+tune vs tune sin system).

Se estiman en cada profundidad; NO es copiar un único residual a otras capas.
Cache en artifacts para reanudar.
"""
from __future__ import annotations
import os
from typing import Dict

import torch

from enjambre.edits import mean_activation_diff, unit
from enjambre.experts import all_expert_specs
from enjambre.generate import chat_text


def compute_expert_directions(model, tok, batch_size: int = 4) -> Dict[str, torch.Tensor]:
    """id -> Tensor [n_layers+1, hidden] unitarios por capa (capa 0 puede ser ~0)."""
    out = {}
    for spec in all_expert_specs():
        pos = [chat_text(tok, it.prompt, spec.system_prompt) for it in spec.tune_items]
        neg = [chat_text(tok, it.prompt, None) for it in spec.tune_items]
        d = mean_activation_diff(model, tok, pos, neg, batch_size=batch_size)
        rows = []
        for i in range(d.shape[0]):
            v = d[i].float()
            rows.append(unit(v) if float(v.norm()) > 1e-6 else v)
        out[spec.id] = torch.stack(rows).cpu()
        print(f"[dir] {spec.id} shape={tuple(out[spec.id].shape)}", flush=True)
    return out


def save_directions(path: str, dirs: Dict[str, torch.Tensor]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({k: v.cpu() for k, v in dirs.items()}, path)


def load_directions(path: str) -> Dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return {k: v.float() for k, v in raw.items()}
