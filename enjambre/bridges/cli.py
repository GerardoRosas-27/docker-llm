"""CLI: python -m enjambre.bridges {identity,pilot,run,report}"""
from __future__ import annotations
import argparse
import json
import os
from .train_eval import BridgesHarness, DEFAULT_OUT


def main(argv=None):
    p = argparse.ArgumentParser(prog="enjambre.bridges")
    p.add_argument("--out", default=None)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--pilot-steps", type=int, default=20)
    p.add_argument("--n-hold", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("identity")
    sub.add_parser("pilot")
    sub.add_parser("run")
    sub.add_parser("report")
    args = p.parse_args(argv)

    if args.cmd == "report":
        path = os.path.join(os.path.abspath(args.out or DEFAULT_OUT), "summary.json")
        print(open(path).read() if os.path.isfile(path) else json.dumps({"error": "missing"}))
        return

    h = BridgesHarness(out_dir=args.out, rank=args.rank, max_steps=args.max_steps,
                       pilot_steps=args.pilot_steps, n_hold=args.n_hold, lr=args.lr)
    if args.cmd == "identity":
        idn = h.identity_check()
        path = os.path.join(h.out_dir, "identity.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(idn, f, indent=2)
        print(json.dumps(idn, indent=2))
        h.bank.remove_hooks()
    elif args.cmd == "pilot":
        info = h.pilot()
        print(json.dumps(info, indent=2))
        h.bank.remove_hooks()
    elif args.cmd == "run":
        print(json.dumps(h.run(), indent=2, default=str))


if __name__ == "__main__":
    main()
