"""CLI: python -m enjambre.fusion_sweep {manifest,run,generate,export-info,curve}"""
from __future__ import annotations
import argparse
import json
import os

from enjambre.build_experts import load_delta
from enjambre.generate import generate_one

from .sweep import SweepHarness, DEFAULT_OUT


def main(argv=None):
    p = argparse.ArgumentParser(prog="enjambre.fusion_sweep", description="Barrido capas fusión 0.5B")
    p.add_argument("--model", default="qwen2.5-0.5b")
    p.add_argument("--out", default=None)
    p.add_argument("--prior", default=None)
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--holdout-n", type=int, default=20)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest")
    pr = sub.add_parser("run")
    pr.add_argument("--no-resume", action="store_true")
    pg = sub.add_parser("generate")
    pg.add_argument("prompt")
    pg.add_argument("--mode", default="selected", choices=["base", "selected", "prior"])
    sub.add_parser("export-info")
    sub.add_parser("curve")
    args = p.parse_args(argv)

    out = args.out
    if args.cmd == "manifest":
        h = SweepHarness(model_key=args.model, out_dir=out, prior_path=args.prior, load_model=False)
        # minimal without model
        from enjambre.config import resolve
        from .layouts import all_configs, default_Ks
        cfg = resolve(args.model)
        print(json.dumps({
            "model": cfg.key, "hf_id": cfg.hf_id,
            "note": "24 capas existentes; no se añaden bloques",
            "Ks": default_Ks(24),
            "n_configs": len(all_configs(24)),
            "out_dir": os.path.abspath(out or DEFAULT_OUT),
            "selection": "tune+val; final=new holdout",
        }, indent=2, ensure_ascii=False))
        return

    h = SweepHarness(
        model_key=args.model, out_dir=out, prior_path=args.prior,
        max_new=args.max_new, holdout_n=args.holdout_n, load_model=True,
    )
    if args.cmd == "run":
        summary = h.run(resume=not args.no_resume)
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    elif args.cmd == "generate":
        if args.mode == "base":
            delta = None
        elif args.mode == "prior":
            delta = h.prior
        else:
            path = os.path.join(h.out_dir, "best_delta.pt")
            if not os.path.isfile(path):
                raise SystemExit("run experiment first")
            delta = load_delta(path)
        with h.act.session(delta):
            print(generate_one(h.model, h.tok, args.prompt, max_new_tokens=args.max_new))
    elif args.cmd == "export-info":
        path = os.path.join(h.out_dir, "best_delta_export.json")
        print(open(path, encoding="utf-8").read() if os.path.isfile(path) else json.dumps({"error": "missing"}))
    elif args.cmd == "curve":
        path = os.path.join(h.out_dir, "curve_by_K.json")
        print(open(path, encoding="utf-8").read() if os.path.isfile(path) else json.dumps({"error": "missing"}))


if __name__ == "__main__":
    main()
