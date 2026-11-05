#!/usr/bin/env python3
"""CLI del enjambre: list/route/generate/memory/bench/build."""
from __future__ import annotations
import argparse
import json
import os
import sys


def main(argv=None):
    p = argparse.ArgumentParser(prog="enjambre", description="Enjambre de micro-expertos SmolLM2")
    p.add_argument("--deltas", default=os.path.join(os.path.dirname(__file__), "artifacts"))
    p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status")
    sub.add_parser("list")

    pr = sub.add_parser("route")
    pr.add_argument("prompt")
    pr.add_argument("--strategy", default="hybrid")

    pg = sub.add_parser("generate")
    pg.add_argument("prompt")
    pg.add_argument("--expert")
    pg.add_argument("--max-new", type=int, default=80)

    pw = sub.add_parser("memory-write")
    pw.add_argument("subject")
    pw.add_argument("object")
    pw.add_argument("--id")

    prd = sub.add_parser("memory-read")
    prd.add_argument("subject")

    sub.add_parser("memory-list")

    pb = sub.add_parser("build")
    pb.add_argument("--out", default=None)

    pe = sub.add_parser("bench")
    pe.add_argument("--out", default=None)

    pa = sub.add_parser("serve")
    pa.add_argument("--host", default="0.0.0.0")
    pa.add_argument("--port", type=int, default=8090)

    args = p.parse_args(argv)

    if args.cmd == "serve":
        os.environ["ENJAMBRE_DELTAS"] = args.deltas
        os.environ["ENJAMBRE_MODEL"] = args.model
        os.environ["ENJAMBRE_HOST"] = args.host
        os.environ["ENJAMBRE_PORT"] = str(args.port)
        from .api import main as api_main
        api_main()
        return

    from .orchestrator import Swarm
    swarm = Swarm(model_name=args.model, deltas_dir=args.deltas if os.path.isdir(args.deltas) else None)

    if args.cmd == "status":
        print(json.dumps(swarm.status(), indent=2, ensure_ascii=False))
    elif args.cmd == "list":
        print(json.dumps(swarm.list_experts(), indent=2, ensure_ascii=False))
    elif args.cmd == "route":
        print(json.dumps(swarm.route_and_run(args.prompt, strategy=args.strategy), indent=2, ensure_ascii=False))
    elif args.cmd == "generate":
        print(swarm.generate(args.prompt, expert_id=args.expert, max_new_tokens=args.max_new))
    elif args.cmd == "memory-write":
        print(json.dumps(swarm.memory_write(args.subject, args.object, fact_id=args.id), indent=2))
    elif args.cmd == "memory-read":
        print(json.dumps(swarm.memory_read(args.subject), indent=2, ensure_ascii=False))
    elif args.cmd == "memory-list":
        print(json.dumps(swarm.memory_list(), indent=2, ensure_ascii=False))
    elif args.cmd == "build":
        from .build_experts import build_all
        out = args.out or args.deltas
        os.makedirs(out, exist_ok=True)
        report, deltas = build_all(swarm.em, swarm.tok, out)
        swarm.deltas.update(deltas)
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("samples_expert", "samples_base", "grid")}
                          for k, v in report["experts"].items()}, indent=2, ensure_ascii=False))
        print("total_s", report["total_seconds"], "base_ppl", report["base_ppl"])
    elif args.cmd == "bench":
        from .bench import run_full_bench
        out = args.out or os.path.join(args.deltas, "bench_report.json")
        if not swarm.deltas:
            swarm.load_deltas(args.deltas)
        rep = run_full_bench(swarm, out)
        summary = {
            "router_accuracy": rep["tasks"]["router_accuracy"],
            "rows": [{k: r[k] for k in ("task", "base_acc", "expert_acc", "swarm_acc", "ppl_expert", "layer", "scale")}
                     for r in rep["tasks"]["rows"]],
            "memory": {k: rep["memory"][k] for k in (
                "recall_single_first_token", "recall_multi_first_token",
                "unrelated_ok_with_single_avg", "unrelated_ok_with_multi",
                "ppl_base", "ppl_memory_multi", "layers"
            ) if k in rep["memory"]},
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        p.error("comando desconocido")


if __name__ == "__main__":
    main()
