"""Construcción de los 50 deltas con checkpoint reanudable (CPU)."""
from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, TYPE_CHECKING

from .catalog import all_tasks

if TYPE_CHECKING:
    from .orchestrator import AdaptiveHarness


def build_all_experts(harness: "AdaptiveHarness", *, resume: bool = True, checkpoint_name: str = "build_checkpoint.json") -> Dict[str, Any]:
    ckpt_path = os.path.join(harness.root, checkpoint_name)
    done = set()
    results = []
    if resume and os.path.isfile(ckpt_path):
        prev = json.loads(open(ckpt_path, encoding="utf-8").read())
        done = set(prev.get("done_task_ids") or [])
        results = list(prev.get("results") or [])
        print(f"[build] resume {len(done)} done", flush=True)
    t0 = time.time()
    for task in all_tasks():
        if task.id in done:
            continue
        print(f"[build] {task.id} ...", flush=True)
        # skip if already has any version
        if harness.bank.list_for_task(task.id):
            detail = {"task_id": task.id, "skipped": True, "reason": "already_has_version"}
        else:
            detail = harness.create_expert(task.id)
            detail["task_id"] = task.id
        results.append(detail)
        done.add(task.id)
        tmp = {
            "done_task_ids": sorted(done),
            "results": results,
            "updated_at": time.time(),
            "elapsed_s": round(time.time() - t0, 1),
        }
        with open(ckpt_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(tmp, f, indent=2, ensure_ascii=False)
        os.replace(ckpt_path + ".tmp", ckpt_path)
        print(f"[build] {task.id} -> {detail.get('status') or detail.get('skipped')} accepted={detail.get('accepted')}", flush=True)
    n_active = sum(1 for r in results if r.get("status") == "active" or r.get("accepted") is True)
    n_exp = sum(1 for r in results if r.get("status") == "experimental" or r.get("accepted") is False)
    summary = {
        "n_tasks": len(all_tasks()),
        "n_built": len(results),
        "n_active_promoted": n_active,
        "n_experimental_or_rejected": n_exp,
        "elapsed_s": round(time.time() - t0, 1),
        "checkpoint": ckpt_path,
        "bank": harness.bank.summary(),
    }
    with open(os.path.join(harness.root, "build_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary
