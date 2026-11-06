"""Layouts de K capas entre 0..nL-1 (bloques existentes, no se añaden capas)."""
from __future__ import annotations
from typing import Dict, List


def clamp_unique(idxs: List[int], nL: int) -> List[int]:
    out = []
    for i in idxs:
        j = min(nL - 1, max(0, int(i)))
        if j not in out:
            out.append(j)
    # if duplicates removed, pad forward
    k = 0
    while len(out) < len(idxs) and k < nL:
        if k not in out:
            out.append(k)
        k += 1
    return sorted(out)[: len(idxs)]


def layout_layers(nL: int, K: int, kind: str) -> List[int]:
    K = min(K, nL)
    if kind == "early":
        return list(range(K))
    if kind == "late":
        return list(range(nL - K, nL))
    if kind == "mid":
        start = max(0, (nL // 2) - (K // 2))
        return list(range(start, start + K))
    if kind == "uniform":
        if K == 1:
            return [nL // 2]
        # inclusive endpoints
        return clamp_unique([round(i * (nL - 1) / (K - 1)) for i in range(K)], nL)
    raise ValueError(kind)


def default_Ks(nL: int) -> List[int]:
    want = [1, 2, 3, 4, 6, 8, 12, 24]
    return [k for k in want if k <= nL]


def all_configs(nL: int) -> List[Dict]:
    layouts = ["early", "mid", "late", "uniform"]
    out = []
    for K in default_Ks(nL):
        for kind in layouts:
            layers = layout_layers(nL, K, kind)
            out.append({"K": K, "layout": kind, "layers": layers})
    return out
