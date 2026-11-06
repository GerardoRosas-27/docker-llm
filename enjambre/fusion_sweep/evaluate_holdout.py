"""Eval en holdout nuevo (confirmación final) y utilidades de scoring."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence

from enjambre.adaptive.snapshot import SnapshotActivator
from enjambre.edits import EditDelta
from enjambre.experts import ExpertSpec, Item
from enjambre.generate import generate, perplexity


def score_items(model, tok, items: Sequence[Item], max_new: int = 64) -> Dict[str, Any]:
    if not items:
        return {"n": 0, "acc": None, "n_ok": 0}
    outs = generate(model, tok, [it.prompt for it in items], max_new_tokens=max_new, batch_size=4)
    flags = [bool(it.check(o)) for it, o in zip(items, outs)]
    return {
        "n": len(items),
        "acc": round(sum(flags) / len(flags), 4),
        "n_ok": int(sum(flags)),
        "samples": [
            {"prompt": it.prompt[:70], "out": o[:100], "ok": bool(f)}
            for it, o, f in zip(items[:2], outs[:2], flags[:2])
        ],
    }


def eval_holdout(
    model, tok, act: SnapshotActivator,
    specs: List[ExpertSpec],
    holdout: Dict[str, List[Item]],
    delta: Optional[EditDelta],
    *,
    max_new: int = 64,
    measure_ppl: bool = True,
    label: str = "cond",
) -> Dict[str, Any]:
    rows = []
    for spec in specs:
        items = holdout[spec.id]
        with act.session(delta):
            sc = score_items(model, tok, items, max_new=max_new)
            ppl = perplexity(model, tok) if measure_ppl else None
        rows.append({
            "task": spec.id, **sc,
            "ppl": None if ppl is None else round(ppl, 3),
            "delta_active": None if delta is None else delta.name,
        })
    accs = [r["acc"] for r in rows if r["acc"] is not None]
    mean = round(sum(accs) / max(1, len(accs)), 4)
    return {"label": label, "mean_acc": mean, "tasks": rows, "n_tasks": len(rows), "n_per_task": rows[0]["n"] if rows else 0}


def paired_vs_base(base_tasks: List[Dict], other_tasks: List[Dict]) -> Dict[str, Any]:
    by_b = {t["task"]: t["acc"] for t in base_tasks}
    per = []
    gain = loss = flat = 0
    for t in other_tasks:
        b = by_b[t["task"]]
        o = t["acc"]
        d = round(o - b, 4)
        if d > 1e-9:
            gain += 1
        elif d < -1e-9:
            loss += 1
        else:
            flat += 1
        per.append({"task": t["task"], "base": b, "other": o, "delta": d})
    return {"n_gain": gain, "n_loss": loss, "n_flat": flat, "per_task": per}
