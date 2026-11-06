"""Evaluación por tarea con splits tune / fusion_val / fusion_confirm / exploratory_full."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from enjambre.adaptive.snapshot import SnapshotActivator
from enjambre.edits import EditDelta
from enjambre.experts import ExpertSpec, Item
from enjambre.generate import generate, perplexity


def split_eval(spec: ExpertSpec) -> Tuple[List[Item], List[Item]]:
    """Primera mitad eval → fusion_val (hiperparams); segunda → fusion_confirm (reporte final)."""
    ev = list(spec.eval_items)
    mid = max(1, len(ev) // 2)
    return ev[:mid], ev[mid:]


def score_items(model, tok, items: Sequence[Item], max_new: int = 64) -> Dict[str, Any]:
    if not items:
        return {"n": 0, "acc": None, "n_ok": 0}
    outs = generate(model, tok, [it.prompt for it in items], max_new_tokens=max_new, batch_size=4)
    flags = [bool(it.check(o)) for it, o in zip(items, outs)]
    return {
        "n": len(items),
        "acc": round(sum(flags) / len(flags), 4),
        "n_ok": int(sum(flags)),
        "samples": [{"prompt": it.prompt[:70], "out": o[:100], "ok": bool(f)} for it, o, f in zip(items[:2], outs[:2], flags[:2])],
    }


def eval_condition(
    model, tok, act: SnapshotActivator, specs: List[ExpertSpec],
    delta_for_task,  # callable task_id -> Optional[EditDelta]
    *,
    split: str = "confirm",
    max_new: int = 64,
    measure_ppl: bool = True,
) -> Dict[str, Any]:
    """Evalúa cada tarea con el delta que indique delta_for_task(task_id)."""
    rows = []
    for spec in specs:
        val, confirm = split_eval(spec)
        if split == "tune":
            items = spec.tune_items
        elif split == "val":
            items = val
        elif split == "confirm":
            items = confirm
        elif split == "full":
            items = spec.eval_items
        else:
            raise ValueError(split)
        delta = delta_for_task(spec.id)
        with act.session(delta):
            sc = score_items(model, tok, items, max_new=max_new)
            ppl = perplexity(model, tok) if measure_ppl else None
        rows.append({
            "task": spec.id, "split": split, **sc,
            "ppl": None if ppl is None else round(ppl, 3),
            "delta_active": None if delta is None else delta.name,
        })
    accs = [r["acc"] for r in rows if r["acc"] is not None]
    mean = round(sum(accs) / max(1, len(accs)), 4)
    return {"split": split, "mean_acc": mean, "tasks": rows, "n_tasks": len(rows)}
