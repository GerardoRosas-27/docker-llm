"""CLI: python -m enjambre.math_code {run,report,identity,neuro-k24,neuro-report}"""
from __future__ import annotations
import argparse
import json
import os

from .harness import DEFAULT_OUT, MathCodeLayerHarness
from .neuro_harness import DEFAULT_NEURO_OUT, NeuroConnectorHarness


def main(argv=None):
    p = argparse.ArgumentParser(prog="enjambre.math_code")
    p.add_argument("--out", default=None)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--train-steps", type=int, default=28)
    p.add_argument("--n-hold", type=int, default=15)
    p.add_argument("--n-ext-hold", type=int, default=10)
    p.add_argument("--n-legacy-hold", type=int, default=25)
    p.add_argument("--stage", default=None, help="Etapa opcional (1..5) para neuro-k24")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("identity")
    sub.add_parser("run")
    sub.add_parser("report")
    p_neuro = sub.add_parser("neuro-k24")
    p_neuro.add_argument("--stage", default=None)
    sub.add_parser("neuro-report")
    p_comp = sub.add_parser("complex-bench")
    p_comp.add_argument("--stage", default=None)
    sub.add_parser("complex-report")
    p_aa = sub.add_parser("aa-bench")
    p_aa.add_argument("--samples", type=int, default=28)
    sub.add_parser("aa-report")
    args = p.parse_args(argv)

    if args.cmd == "report":
        path = os.path.join(os.path.abspath(args.out or DEFAULT_OUT), "summary.json")
        print(open(path).read() if os.path.isfile(path) else json.dumps({"error": "missing"}))
        return

    if args.cmd == "neuro-report":
        path = os.path.join(os.path.abspath(args.out or DEFAULT_NEURO_OUT), "summary_neuro_k24.json")
        print(open(path).read() if os.path.isfile(path) else json.dumps({"error": "missing"}))
        return

    if args.cmd == "complex-report":
        from .complex_eval import DEFAULT_COMPLEX_OUT
        path = os.path.join(os.path.abspath(args.out or DEFAULT_COMPLEX_OUT), "summary_complex_benchmark.json")
        print(open(path).read() if os.path.isfile(path) else json.dumps({"error": "missing"}))
        return

    if args.cmd == "aa-report":
        from enjambre.artificial_analysis.bench import DEFAULT_AA_OUT
        path = os.path.join(os.path.abspath(args.out or DEFAULT_AA_OUT), "summary_artificial_analysis.json")
        print(open(path).read() if os.path.isfile(path) else json.dumps({"error": "missing"}))
        return

    if args.cmd == "aa-bench":
        from enjambre.artificial_analysis.bench import ArtificialAnalysisHarness
        samples = getattr(args, "samples", 28)
        aah = ArtificialAnalysisHarness(out_dir=args.out, rank=args.rank, sample_size=samples)
        res = aah.run_artificial_analysis_suite()
        print(json.dumps({
            "benchmark_name": res["benchmark_name"],
            "winner": res["winner"],
            "comparison_table": res["comparison_table"],
            "total_wall_secs": res["total_wall_secs"],
        }, indent=2))
        return

    if args.cmd == "complex-bench":
        from .complex_eval import ComplexBenchmarkHarness
        ch = ComplexBenchmarkHarness(out_dir=args.out, rank=args.rank)
        res = ch.run_full_complex_experiment(only_stage=args.stage)
        if args.stage and args.stage not in ("summary", "all"):
            print(json.dumps({
                "label": res["label"],
                "arch": res["arch"],
                "math_acc": res["math_acc"],
                "code_acc": res["code_acc"],
                "mean_acc": res["mean_acc"],
            }, indent=2))
        else:
            print(json.dumps({
                "winner": res["winner"],
                "comparison_table": res["comparison_table"],
                "total_wall_secs": res["total_wall_secs"],
            }, indent=2))
        return

    if args.cmd == "neuro-k24":
        nh = NeuroConnectorHarness(
            out_dir=args.out,
            rank=args.rank,
            train_steps=args.train_steps,
            n_hold=args.n_hold,
            n_ext_hold=args.n_ext_hold,
            n_legacy_hold=args.n_legacy_hold,
        )
        res = nh.run_experiment(only_stage=args.stage)
        if args.stage:
            print(json.dumps({
                "label": res["label"],
                "arch": res["arch"],
                "core_mean_acc": res["core_mean_acc"],
                "ext_mean_acc": res["ext_mean_acc"],
                "all12_mean_acc": res["all12_mean_acc"],
            }, indent=2))
        else:
            print(json.dumps({
                "winning_architecture": res["winning_architecture"],
                "comparison_table": res["comparison_table"],
                "total_wall_secs": res["total_wall_secs"],
            }, indent=2))
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
