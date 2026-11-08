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
    p_train = sub.add_parser("train", help="Entrena el Campo CONSTELACIÓN v4 sobre K32")
    p_train.add_argument("--steps", type=int, default=30, help="Pasos de entrenamiento")
    p_train.add_argument("--batch-size", type=int, default=4, help="Tamaño de batch")
    p_train.add_argument("--lr", type=float, default=8e-4, help="Learning rate del campo")

    # eval
    p_eval = sub.add_parser("eval", help="Ejecuta Artificial Analysis y Complex Benchmarks")
    p_eval.add_argument("--samples", type=int, default=28, help="Muestra de MATH-500")
    p_eval.add_argument("--hold", type=int, default=15, help="Ítems holdout por tarea compleja")
    p_eval.add_argument("--checkpoint", type=str, default=None, help="Ruta al checkpoint del campo")

    # all
    p_all = sub.add_parser("all", help="Entrena y evalúa completamente")
    p_all.add_argument("--steps", type=int, default=30)
    p_all.add_argument("--samples", type=int, default=28)
    p_all.add_argument("--hold", type=int, default=15)

    args = parser.parse_args()

    if args.cmd == "train":
        trainer = FieldTrainer(train_steps=args.steps, batch_size=args.batch_size, lr_field=args.lr)
        trainer.train()

    elif args.cmd == "eval":
        evaluator = MacroFieldEvaluator(checkpoint_path=args.checkpoint)
        evaluator.run_artificial_analysis_eval(n_samples=args.samples)
        evaluator.run_complex_benchmark(n_hold=args.hold)

    elif args.cmd == "all":
        trainer = FieldTrainer(train_steps=args.steps)
        ckpt = trainer.train()
        evaluator = MacroFieldEvaluator(checkpoint_path=ckpt)
        evaluator.run_artificial_analysis_eval(n_samples=args.samples)
        evaluator.run_complex_benchmark(n_hold=args.hold)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
