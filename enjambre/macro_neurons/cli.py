"""CLI para la Red de Macro-Neuronas y el Campo CONSTELACIÓN v4."""
from __future__ import annotations
import argparse
import json
import os
import sys

from .train_field import FieldTrainer
from .eval_field import MacroFieldEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Red de Macro-Neuronas y Campo CONSTELACIÓN v4")
    sub = parser.add_subparsers(dest="cmd")

    # train
    p_train = sub.add_parser("train", help="Entrena el Campo CONSTELACIÓN v4")
    p_train.add_argument("--macro", type=int, default=66, help="Número de macro-neuronas (32, 36 o 66)")
    p_train.add_argument("--base", type=str, default="0.5b", help="Tamaño del modelo base (0.5b o 1.5b)")
    p_train.add_argument("--steps", type=int, default=40, help="Pasos de entrenamiento")
    p_train.add_argument("--batch-size", type=int, default=2, help="Tamaño de batch")
    p_train.add_argument("--lr", type=float, default=8e-4, help="Learning rate del campo")

    # eval-aa
    p_aa = sub.add_parser("eval-aa", help="Ejecuta exclusivamente el benchmark de Artificial Analysis")
    p_aa.add_argument("--macro", type=int, default=66, help="Número de macro-neuronas (32, 36 o 66)")
    p_aa.add_argument("--base", type=str, default="0.5b", help="Tamaño del modelo base (0.5b o 1.5b)")
    p_aa.add_argument("--samples", type=int, default=28, help="Muestra de MATH-500")
    p_aa.add_argument("--tokens", type=int, default=96, help="Tokens de generación")
    p_aa.add_argument("--limit-per-run", type=int, default=None, help="Límite de problemas por corrida")
    p_aa.add_argument("--checkpoint", type=str, default=None, help="Ruta al checkpoint del campo")

    # eval-complex
    p_comp = sub.add_parser("eval-complex", help="Ejecuta exclusivamente la suite de 10 tareas complejas")
    p_comp.add_argument("--macro", type=int, default=66, help="Número de macro-neuronas (32, 36 o 66)")
    p_comp.add_argument("--base", type=str, default="0.5b", help="Tamaño del modelo base (0.5b o 1.5b)")
    p_comp.add_argument("--hold", type=int, default=15, help="Ítems holdout por tarea compleja")
    p_comp.add_argument("--checkpoint", type=str, default=None, help="Ruta al checkpoint del campo")

    # eval (ambos)
    p_eval = sub.add_parser("eval", help="Ejecuta Artificial Analysis y Complex Benchmarks")
    p_eval.add_argument("--macro", type=int, default=66, help="Número de macro-neuronas (32, 36 o 66)")
    p_eval.add_argument("--base", type=str, default="0.5b", help="Tamaño del modelo base (0.5b o 1.5b)")
    p_eval.add_argument("--samples", type=int, default=28, help="Muestra de MATH-500")
    p_eval.add_argument("--hold", type=int, default=15, help="Ítems holdout por tarea compleja")
    p_eval.add_argument("--checkpoint", type=str, default=None, help="Ruta al checkpoint del campo")

    # all
    p_all = sub.add_parser("all", help="Entrena y evalúa completamente")
    p_all.add_argument("--macro", type=int, default=66)
    p_all.add_argument("--base", type=str, default="0.5b")
    p_all.add_argument("--steps", type=int, default=40)
    p_all.add_argument("--samples", type=int, default=28)
    p_all.add_argument("--hold", type=int, default=15)

    args = parser.parse_args()

    if args.cmd == "train":
        trainer = FieldTrainer(n_macro=args.macro, base_size=args.base, train_steps=args.steps, batch_size=args.batch_size, lr_field=args.lr)
        trainer.train()

    elif args.cmd == "eval-aa":
        evaluator = MacroFieldEvaluator(n_macro=args.macro, base_size=args.base, checkpoint_path=args.checkpoint)
        evaluator.run_artificial_analysis_eval(n_samples=args.samples, max_tokens=args.tokens, max_per_run=args.limit_per_run)

    elif args.cmd == "eval-complex":
        evaluator = MacroFieldEvaluator(n_macro=args.macro, base_size=args.base, checkpoint_path=args.checkpoint)
        evaluator.run_complex_benchmark(n_hold=args.hold)

    elif args.cmd == "eval":
        evaluator = MacroFieldEvaluator(n_macro=args.macro, base_size=args.base, checkpoint_path=args.checkpoint)
        evaluator.run_artificial_analysis_eval(n_samples=args.samples)
        evaluator.run_complex_benchmark(n_hold=args.hold)

    elif args.cmd == "all":
        trainer = FieldTrainer(n_macro=args.macro, base_size=args.base, train_steps=args.steps)
        ckpt = trainer.train()
        evaluator = MacroFieldEvaluator(n_macro=args.macro, base_size=args.base, checkpoint_path=ckpt)
        evaluator.run_artificial_analysis_eval(n_samples=args.samples)
        evaluator.run_complex_benchmark(n_hold=args.hold)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
