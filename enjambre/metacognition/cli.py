"""CLI para entrenar y evaluar la Arquitectura Multi-Agente Metacognitiva Bidireccional."""
from __future__ import annotations

import argparse
import json

from .evaluator import (
    evaluate_bidirectional_part1,
    evaluate_bidirectional_part2,
    evaluate_bidirectional_system,
)
from .trainer import BidirectionalMetaTrainer, DEFAULT_META_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI de Introspección Multi-Agente Metacognitiva Bidireccional (S1 <-> S2)"
    )
    parser.add_argument(
        "action",
        choices=[
            "train",
            "stage1",
            "stage2_s2",
            "stage2_s1",
            "phase1",
            "phase2",
            "refine_s1",
            "spot_s1",
            "eval",
            "eval_p1",
            "eval_p2",
            "all",
        ],
        help="Acción a ejecutar: entrenar (completo o por etapas), refinar S1, evaluar o ambos.",
    )
    parser.add_argument("--out-dir", default=DEFAULT_META_DIR)
    parser.add_argument("--stage1-epochs", type=int, default=260)
    parser.add_argument("--stage2-steps-s2", type=int, default=18)
    parser.add_argument("--stage2-steps-s1", type=int, default=12)
    parser.add_argument("--refine-steps", type=int, default=4)
    parser.add_argument("--refine-lr", type=float, default=6.5e-4)
    parser.add_argument("--refine-seed-offset", type=int, default=911)
    parser.add_argument("--refine-grad-accum", type=int, default=3)
    args = parser.parse_args()

    if args.action in ("train", "stage1", "stage2_s2", "stage2_s1", "phase1", "phase2", "refine_s1", "spot_s1", "all"):
        trainer = BidirectionalMetaTrainer(
            out_dir=args.out_dir,
            stage1_epochs=args.stage1_epochs,
            stage2_steps_s2=args.stage2_steps_s2,
            stage2_steps_s1=args.stage2_steps_s1,
        )
        if args.action == "stage1":
            s1_res = trainer.run_stage1_only()
            print(json.dumps({"stage1_total_s": s1_res["stage1_total_s"], "stage1": s1_res["stage1"]}, indent=2, ensure_ascii=False))
        elif args.action == "stage2_s2":
            p1_res = trainer.run_stage2_s2_only()
            print(json.dumps({"phase1_elapsed_s": p1_res["phase1_elapsed_s"], "stage2_s2_to_s1": p1_res["stage2_s2_to_s1"]}, indent=2, ensure_ascii=False))
        elif args.action in ("stage2_s1", "phase2"):
            t_res = trainer.run_training_phase2()
            print(json.dumps({"total_elapsed_s": t_res["total_elapsed_s"], "stage2_s1_to_s2": t_res["stage2_s1_to_s2"]}, indent=2, ensure_ascii=False))
        elif args.action == "phase1":
            p1_res = trainer.run_training_phase1()
            print(json.dumps({"phase1_elapsed_s": p1_res["phase1_elapsed_s"], "stage1": p1_res["stage1"], "stage2_s2_to_s1": p1_res["stage2_s2_to_s1"]}, indent=2, ensure_ascii=False))
        elif args.action == "refine_s1":
            r_res = trainer.refine_s1_soft_prompt(
                steps=args.refine_steps,
                lr=args.refine_lr,
                seed_offset=args.refine_seed_offset,
                grad_accum=args.refine_grad_accum,
            )
            print(json.dumps(r_res, indent=2, ensure_ascii=False))
        elif args.action == "spot_s1":
            sp_res = trainer.spot_check_s1_val()
            print(json.dumps(sp_res, indent=2, ensure_ascii=False))
        else:
            t_res = trainer.run_training()
            if args.action == "train":
                print(json.dumps(t_res["stage1"], indent=2, ensure_ascii=False))

    if args.action == "eval_p1":
        p1 = evaluate_bidirectional_part1(meta_dir=args.out_dir)
        print(
            json.dumps(
                {
                    "part1_elapsed_s": p1["part1_elapsed_s"],
                    "memory_mb": p1["memory_mb"],
                    "s2_to_s1_holdout": p1["quantitative_introspection"]["s2_observes_s1"]["holdout_unseen"],
                    "s1_to_s2_holdout": p1["quantitative_introspection"]["s1_observes_s2"]["holdout_unseen"],
                    "generative_s2_to_s1": {
                        k: v for k, v in p1["generative_weight_descriptions_holdout"]["s2_to_s1_qwen_0_5b"].items() if k != "samples"
                    },
                    "generative_s1_to_s2": {
                        k: v for k, v in p1["generative_weight_descriptions_holdout"]["s1_to_s2_qwen_1_5b"].items() if k != "samples"
                    },
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.action in ("eval_p2", "eval", "all"):
        e_res = (
            evaluate_bidirectional_part2(meta_dir=args.out_dir)
            if args.action == "eval_p2"
            else evaluate_bidirectional_system(meta_dir=args.out_dir)
        )
        print(
            json.dumps(
                {
                    "memory_mb": e_res["memory_mb"],
                    "s2_to_s1_holdout": e_res["quantitative_introspection"]["s2_observes_s1"]["holdout_unseen"],
                    "s1_to_s2_holdout": e_res["quantitative_introspection"]["s1_observes_s2"]["holdout_unseen"],
                    "generative_s2_to_s1": {
                        k: v
                        for k, v in e_res["generative_weight_descriptions_holdout"]["s2_to_s1_qwen_0_5b"].items()
                        if k != "samples"
                    },
                    "generative_s1_to_s2": {
                        k: v
                        for k, v in e_res["generative_weight_descriptions_holdout"]["s1_to_s2_qwen_1_5b"].items()
                        if k != "samples"
                    },
                    "counterfactuals": e_res["counterfactual_live_interventions"],
                    "primary_preservation": e_res["primary_task_preservation"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
