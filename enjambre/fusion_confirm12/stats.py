"""Bootstrap pareado CI95 y McNemar sobre aciertos por ítem."""
from __future__ import annotations
from typing import Any, Dict, List, Sequence, Tuple
import math
import random


def paired_flags(base_ok: Sequence[bool], other_ok: Sequence[bool]) -> Dict[str, Any]:
    assert len(base_ok) == len(other_ok)
    n = len(base_ok)
    b01 = b10 = both = neither = 0
    for a, b in zip(base_ok, other_ok):
        if (not a) and b:
            b01 += 1  # other gains
        elif a and (not b):
            b10 += 1  # other loses
        elif a and b:
            both += 1
        else:
            neither += 1
    return {
        "n": n,
        "n_other_better": b01,
        "n_other_worse": b10,
        "n_both_ok": both,
        "n_both_fail": neither,
        "acc_base": round(sum(base_ok) / max(1, n), 4),
        "acc_other": round(sum(other_ok) / max(1, n), 4),
        "delta": round(sum(other_ok) / max(1, n) - sum(base_ok) / max(1, n), 4),
    }


def mcnemar_exact(n01: int, n10: int) -> Dict[str, Any]:
    """McNemar exact (binomial mid-p style approx via two-sided binomial)."""
    n = n01 + n10
    if n == 0:
        return {"n_discordant": 0, "p_two_sided": 1.0, "stat": "no_discordant"}
    # two-sided exact: sum binom probs of k>=max(n01,n10) under p=0.5, *2 capped
    k = max(n01, n10)
    # P(X>=k) + P(X<=n-k) under Bin(n,0.5)
    # compute with recursive probs
    # C(n,i)/2^n
    from math import comb
    p_tail = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    p = min(1.0, 2.0 * p_tail)
    return {"n_discordant": n, "n01": n01, "n10": n10, "p_two_sided": round(p, 6)}


def bootstrap_delta_ci(
    base_ok: Sequence[bool],
    other_ok: Sequence[bool],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    assert len(base_ok) == len(other_ok)
    n = len(base_ok)
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        b = sum(1 for i in idx if base_ok[i]) / n
        o = sum(1 for i in idx if other_ok[i]) / n
        deltas.append(o - b)
    deltas.sort()
    lo = deltas[int((alpha / 2) * n_boot)]
    hi = deltas[int((1 - alpha / 2) * n_boot) - 1]
    point = sum(other_ok) / n - sum(base_ok) / n
    return {
        "delta": round(point, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "n_boot": n_boot,
        "seed": seed,
    }
