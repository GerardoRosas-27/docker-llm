"""Benchmark heldout: base vs delta activo. Checkpoint reanudable. Heldout intacto (solo eval)."""
from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, TYPE_CHECKING

from .catalog import all_tasks
from .improve import _acc

if TYPE_CHECKING:
    from .orchestrator import AdaptiveHarness


def run_heldout_bench(harness: "AdaptiveHarness", *, resume: bool = True, max_new: int = 32) -> Dict[str, Any]:
    path = os.path.join(harness.root, "heldout_bench.json")
    done = {}
    if resume and os.path.isfile(path):
        prev = json.loads(open(path, encoding="utf-8").read())
        done = {r["task_id"]: r for r in prev.get("tasks", [])}
        print(f"[bench] resume {len(done)}", flush=True)
    t0 = time.time()
    rows = list(done.values())
    for task in all_tasks():
        if task.id in done:
            continue
        print(f"[bench] {task.id}", flush=True)
        items = task.heldout_items
        with harness.act.session(None):
            b_acc, b_outs, b_flags = _acc(harness.model, harness.tok, items, max_new)
        active = harness.bank.get_active(task.id)
        if active:
            delta = harness.bank.load_delta(active.version_id)
            with harness.act.session(delta):
                e_acc, e_outs, e_flags = _acc(harness.model, harness.tok, items, max_new)
            vid = active.version_id
            status = active.status
        else:
            e_acc, e_outs, e_flags = b_acc, b_outs, b_flags
            vid, status = None, None
        row = {
            "task_id": task.id, "family": task.family, "n": len(items),
            "base_acc": round(b_acc, 4), "expert_acc": round(e_acc, 4),
            "delta": round(e_acc - b_acc, 4), "active_version": vid, "status": status,
            "base_ok": int(sum(b_flags)), "expert_ok": int(sum(e_flags)),
            "samples": [
                {"prompt": it.prompt[:60], "base": bo[:80], "expert": eo[:80], "base_ok": bool(bf), "expert_ok": bool(ef)}
                for it, bo, eo, bf, ef in zip(items[:2], b_outs[:2], e_outs[:2], b_flags[:2], e_flags[:2])
            ],
        }
        rows.append(row)
        done[task.id] = row
        payload = {"tasks": rows, "updated_at": time.time(), "elapsed_s": round(time.time() - t0, 1)}
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(path + ".tmp", path)
    n = len(rows)
    base_mean = sum(r["base_acc"] for r in rows) / max(1, n)
    exp_mean = sum(r["expert_acc"] for r in rows) / max(1, n)
    n_gain = sum(1 for r in rows if r["delta"] > 1e-9)
    n_loss = sum(1 for r in rows if r["delta"] < -1e-9)
    n_active = sum(1 for r in rows if r["status"] == "active")
    summary = {
        "n_tasks": n, "base_mean": round(base_mean, 4), "expert_mean": round(exp_mean, 4),
        "global_delta": round(exp_mean - base_mean, 4), "n_gain": n_gain, "n_loss": n_loss,
        "n_flat": n - n_gain - n_loss, "n_with_active_delta": n_active,
        "elapsed_s": round(time.time() - t0, 1), "path": path,
        "note": "Heldout only for reporting; never used for promotion. Router not scored here.",
    }
    with open(os.path.join(harness.root, "heldout_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[bench] DONE", summary, flush=True)
    return {"summary": summary, "tasks": rows}
