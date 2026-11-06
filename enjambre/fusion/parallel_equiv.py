"""Equivalencia estructural: MLP paralelo N ramas vs bias promedio.

Si cada rama comparte W_gate, W_up, W_down y solo difiere en bias_down b_i,
la salida promedio de N ramas en down_proj es:

  y_i = W_down x + b_i
  (1/N) sum y_i = W_down x + (1/N) sum b_i

Es decir, equivale a UN Linear con el mismo W_down y bias promedio.
Esto AUMENTA parámetros si se materializan N copias (N * |W| extras),
pero la función es idéntica al modelo con un solo bias fusionado (avg).
No hay crecimiento de capacidad expresiva por esa vía cuando solo hay bias edits.
"""
from __future__ import annotations
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn

from enjambre.edits import EditDelta


def bias_vectors_from_deltas(deltas: Sequence[EditDelta], layer: int) -> List[torch.Tensor]:
    """Extrae scale*vec de cada delta que toque `layer` (si no, vector cero del primero)."""
    out = []
    ref_dim = None
    for d in deltas:
        hit = None
        for b in d.bias_edits:
            ref_dim = b.vec.numel()
            if b.layer == layer:
                hit = (b.scale * b.vec.float()).detach().clone()
                break
        if hit is None:
            if ref_dim is None:
                continue
            hit = torch.zeros(ref_dim)
        out.append(hit)
    return out


@torch.no_grad()
def prove_average_bias_equivalence(
    W: torch.Tensor,
    biases: Sequence[torch.Tensor],
    x: torch.Tensor,
    atol: float = 1e-5,
) -> dict:
    """Demostración numérica: mean_i(W x + b_i) == W x + mean(b)."""
    assert biases, "need biases"
    ys = [x @ W.T + b.to(x.dtype) for b in biases]  # W is [out,in] like nn.Linear
    y_mean_branches = torch.stack(ys, 0).mean(0)
    b_mean = torch.stack([b.float() for b in biases], 0).mean(0)
    y_fused = x @ W.T + b_mean.to(x.dtype)
    err = float((y_mean_branches - y_fused).abs().max())
    # param growth if materializing N full down_proj copies vs one
    n = len(biases)
    params_one = W.numel() + b_mean.numel()
    params_parallel_naive = n * (W.numel() + biases[0].numel())
    return {
        "n_branches": n,
        "max_abs_err": err,
        "equivalent": err <= atol,
        "params_single_down_with_bias": params_one,
        "params_naive_parallel_n_copies": params_parallel_naive,
        "extra_params_if_materialized": params_parallel_naive - params_one,
        "conclusion": (
            "Parallel average of bias-only branches ≡ single Linear with average bias. "
            "Materializing N copies grows storage/params but NOT function class for this edit type."
        ),
    }


class ParallelDownAvg(nn.Module):
    """Toy módulo: N down_proj compartiendo W, biases distintos, salida promedio."""

    def __init__(self, W: torch.Tensor, biases: Sequence[torch.Tensor]):
        super().__init__()
        out, inn = W.shape
        self.shared = nn.Linear(inn, out, bias=False)
        with torch.no_grad():
            self.shared.weight.copy_(W)
        self.biases = nn.ParameterList([nn.Parameter(b.float().clone()) for b in biases])

    def forward(self, x):
        base = self.shared(x)
        return base + torch.stack([b for b in self.biases], 0).mean(0)
