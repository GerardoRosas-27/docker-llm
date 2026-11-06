"""CLI: python -m enjambre.fusion_confirm12 {manifest,run,generate,export-info}"""
from __future__ import annotations
import argparse
import json
import os

from enjambre.generate import generate_one

from .frozen import FROZEN_K12, rebuild_k12_delta
from .harness import Confirm12Harness, DEFAULT_OUT


def main(argv=None):
    p = argparse.ArgumentParser(prog="enjambre.fusion_confirm12")
    p.add_argument("--model", default="qwen2.5-0.5b")
    p.add_argument("--out", default=None)
    p.add_argument("--holdout-n", type=int, default=50)
    p.add_argument("--max-new", type=int, default=64)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest")
    pr = sub.add_parser("run")
    pr.add_argument("--no-resume", action="store_true")
    pg = sub.add_parser("generate")
    pg.add_argument("prompt")
    pg.add_argument("--mode", default="k12", choices=["base", "prior", "k12"])
    sub.add_parser("export-info")
    args = p.parse_args(argv)

    if args.cmd == "manifest":
        print(json.dumps({
            "frozen": FROZEN_K12,
            "out_dir": os.path.abspath(args.out or DEFAULT_OUT),
            "holdout_n": args.holdout_n,
            "note": "Confirmación sin reoptimizar; holdout nuevo",
        }, indent=2, ensure_ascii=False))
        return

    h = Confirm12Harness(
        model_key=args.model, out_dir=args.out,
        holdout_n=args.holdout_n, max_new=args.max_new, load_model=True,
    )
    if args.cmd == "run":
        print(json.dumps(h.run(resume=not args.no_resume), indent=2, ensure_ascii=False, default=str))
    elif args.cmd == "generate":
        delta = {"base": None, "prior": h.prior, "k12": h.k12}[args.mode]
        with h.act.session(delta):
            print(generate_one(h.model, h.tok, args.prompt, max_new_tokens=args.max_new))
    elif args.cmd == "export-info":
        path = os.path.join(h.out_dir, "k12_frozen_meta.json")
        print(open(path, encoding="utf-8").read() if os.path.isfile(path) else json.dumps(FROZEN_K12, indent=2))


if __name__ == "__main__":
    main()
