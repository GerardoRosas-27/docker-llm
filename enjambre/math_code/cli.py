"""CLI: python -m enjambre.math_code {run,report,identity}"""
from __future__ import annotations
import argparse
import json
import os

from .harness import DEFAULT_OUT, MathCodeLayerHarness


def main(argv=None):
    p = argparse.ArgumentParser(prog="enjambre.math_code")
    p.add_argument("--out", default=None)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--train-steps", type=int, default=60)
    p.add_argument("--n-hold", type=int, default=15)
    p.add_argument("--n-legacy-hold", type=int, default=30)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("identity")
    sub.add_parser("run")
    sub.add_parser("report")
    args = p.parse_args(argv)

    if args.cmd == "report":
        path = os.path.join(os.path.abspath(args.out or DEFAULT_OUT), "summary.json")
        print(open(path).read() if os.path.isfile(path) else json.dumps({"error": "missing"}))
        return

    h = MathCodeLayerHarness(
        out_dir=args.out,
        rank=args.rank,
        train_steps=args.train_steps,
        n_hold=args.n_hold,
        n_legacy_hold=args.n_legacy_hold,
    )
    if args.cmd == "identity":
        res = h.verify_warm_extension_identity()
        print(json.dumps(res, indent=2))
        h.bank.remove_hooks()
    elif args.cmd == "run":
        summary = h.run_all()
        print(json.dumps({
            "winner": summary["winner"],
            "progression_by_layers": summary["progression_by_layers"],
            "total_wall_secs": summary["total_wall_secs"],
        }, indent=2))


if __name__ == "__main__":
    main()
