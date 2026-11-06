"""CLI: python -m enjambre.fusion {manifest,run,generate,export-info}"""
from __future__ import annotations
import argparse
import json
import os

from .orchestrator import FusionHarness


def main(argv=None):
    p = argparse.ArgumentParser(prog="enjambre.fusion", description="Fusion de 10 expertos 0.5B (bias steering)")
    p.add_argument("--model", default="qwen2.5-0.5b")
    p.add_argument("--deltas", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--max-new", type=int, default=64)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest")
    pr = sub.add_parser("run")
    pr.add_argument("--no-resume", action="store_true")
    pg = sub.add_parser("generate")
    pg.add_argument("prompt")
    pg.add_argument("--mode", default="selected", choices=["base", "oracle", "selected", "multi_avg", "single_avg"])
    pg.add_argument("--task", default=None)
    sub.add_parser("export-info")
    args = p.parse_args(argv)

    if args.cmd == "manifest":
        h = FusionHarness(model_key=args.model, deltas_dir=args.deltas, out_dir=args.out, load_model=False)
        print(json.dumps(h.manifest(), indent=2, ensure_ascii=False))
        return

    h = FusionHarness(model_key=args.model, deltas_dir=args.deltas, out_dir=args.out, max_new=args.max_new, load_model=True)
    if args.cmd == "run":
        out = h.run_experiment(resume=not args.no_resume)
        print(json.dumps(out["summary"], indent=2, ensure_ascii=False, default=str))
    elif args.cmd == "generate":
        from .combine import fuse_deltas
        from enjambre.generate import generate_one
        from enjambre.edits import BiasEdit, EditDelta
        import torch
        if args.mode == "base":
            delta = None
        elif args.mode == "oracle":
            if not args.task:
                raise SystemExit("--task required for oracle")
            delta = h.deltas[args.task]
        elif args.mode == "multi_avg":
            delta = fuse_deltas(list(h.deltas.values()), mode="multi_avg", global_scale=1.0)
        elif args.mode == "single_avg":
            used = sorted({b.layer for d in h.deltas.values() for b in d.bias_edits})
            delta = fuse_deltas(list(h.deltas.values()), mode="single_avg", target_layer=used[len(used)//2], global_scale=1.0)
        else:
            path = os.path.join(h.out_dir, "fused_selected.pt")
            raw = torch.load(path, map_location="cpu", weights_only=False)
            delta = EditDelta(raw["name"], [BiasEdit(e["layer"], e["vec"].float(), e["scale"]) for e in raw["edits"]], raw.get("meta") or {})
        with h.act.session(delta):
            print(generate_one(h.model, h.tok, args.prompt, max_new_tokens=args.max_new))
    elif args.cmd == "export-info":
        path = os.path.join(h.out_dir, "fused_selected_export.json")
        print(open(path, encoding="utf-8").read() if os.path.isfile(path) else json.dumps({"error": "run experiment first"}))


if __name__ == "__main__":
    main()
