#!/usr/bin/env python3
"""Pipeline completo: construir 10 expertos + bench (base/experto/swarm/memoria) + guardar artefactos."""
from __future__ import annotations
import json
import os
import sys
import time

# permitir `python -m enjambre.run_all` desde la raíz del repo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def main():
    torch.manual_seed(0)
    from enjambre.orchestrator import Swarm
    from enjambre.build_experts import build_all
    from enjambre.bench import run_full_bench

    out = os.environ.get(
        "ENJAMBRE_DELTAS",
        os.path.join(os.path.dirname(__file__), "artifacts"),
    )
    os.makedirs(out, exist_ok=True)
    T0 = time.time()
    print("== enjambre: carga modelo ==", flush=True)
    swarm = Swarm(deltas_dir=None)
    print("== enjambre: build expertos ==", flush=True)
    report, deltas = build_all(swarm.em, swarm.tok, out)
    swarm.deltas.update(deltas)
    swarm.deltas_dir = out
    print("== enjambre: bench completo ==", flush=True)
    bench = run_full_bench(swarm, os.path.join(out, "bench_report.json"))
    summary = {
        "total_seconds": round(time.time() - T0, 1),
        "base_ppl": report["base_ppl"],
        "experts": {
            k: {
                "layer": v["layer"], "scale": v["scale"],
                "tune_acc": v["tune_acc"],
                "eval_acc_base": v["eval_acc_base"],
                "eval_acc_expert": v["eval_acc_expert"],
                "ppl_expert": v["ppl_expert"],
            }
            for k, v in report["experts"].items()
        },
        "router_accuracy": bench["tasks"]["router_accuracy"],
        "swarm_rows": [
            {k: r[k] for k in ("task", "base_acc", "expert_acc", "swarm_acc", "router_match_rate")}
            for r in bench["tasks"]["rows"]
        ],
        "memory": {
            "recall_base_first_token": bench["memory"]["recall_base_first_token"],
            "recall_single_first_token": bench["memory"]["recall_single_first_token"],
            "recall_multi_first_token": bench["memory"]["recall_multi_first_token"],
            "recall_single_text": bench["memory"]["recall_single_text"],
            "recall_multi_text": bench["memory"]["recall_multi_text"],
            "unrelated_ok_with_single_avg": bench["memory"]["unrelated_ok_with_single_avg"],
            "unrelated_ok_with_multi": bench["memory"]["unrelated_ok_with_multi"],
            "ppl_base": bench["memory"]["ppl_base"],
            "ppl_memory_multi": bench["memory"]["ppl_memory_multi"],
            "layers": bench["memory"]["layers"],
            "method": bench["memory"]["method"],
        },
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("DONE total_s", summary["total_seconds"], flush=True)


if __name__ == "__main__":
    main()
