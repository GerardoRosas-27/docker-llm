"""CLI: python -m enjambre.fusion_20 {manifest,run,generate,report}"""
from __future__ import annotations
import argparse
import json
import os

from enjambre.generate import generate_one

from .build import FROZEN_DESIGNS, K20_LAYERS, build_all_deltas
from .harness import DEFAULT_OUT, Fusion20Harness


def main(argv=None):
    p = argparse.ArgumentParser(prog="enjambre.fusion_20")
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
    pg.add_argument("--mode", default="k20_A", choices=["base", "k12", "k20_A", "k20_B"])
    sub.add_parser("report")
    args = p.parse_args(argv)

    if args.cmd == "manifest":
        b = build_all_deltas()
        print(json.dumps({
            "designs": FROZEN_DESIGNS,
            "K20_LAYERS": K20_LAYERS,
            "l2": {"k12": b["l2_k12"], "A": b["k20_A"]["meta"]["total_l2"], "B": b["k20_B"]["meta"]["total_l2"]},
            "hashes": {k: b[k]["meta"]["delta_sha256"] for k in ("k12", "k20_A", "k20_B")},
            "out_dir": os.path.abspath(args.out or DEFAULT_OUT),
        }, indent=2, ensure_ascii=False))
        return

    h = Fusion20Harness(model_key=args.model, out_dir=args.out,
                        holdout_n=args.holdout_n, max_new=args.max_new, load_model=True)
    if args.cmd == "run":
        print(json.dumps(h.run(resume=not args.no_resume), indent=2, ensure_ascii=False, default=str))
    elif args.cmd == "generate":
        with h.act.session(h._delta(args.mode)):
            print(generate_one(h.model, h.tok, args.prompt, max_new_tokens=args.max_new))
    elif args.cmd == "report":
        path = os.path.join(h.out_dir, "summary.json")
        print(open(path, encoding="utf-8").read() if os.path.isfile(path) else json.dumps({"error": "run first"}))


if __name__ == "__main__":
    main()
