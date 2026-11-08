"""Evaluación integral de la Red de Macro-Neuronas (K32 / K36) + Campo CONSTELACIÓN v4.

Ejecuta:
1. Benchmark Oficial de Artificial Analysis:
   - Latencia y Rendimiento: TTFT (ms), Output Speed (TPS), Latencia Mediana P50 y P95.
   - Precisión en MATH-500 (HuggingFaceH4/MATH-500).
2. Benchmark de Tareas Complejas (150 ítems de holdout):
   - 10 tareas de algoritmia y matemáticas multi-paso.
"""
from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from enjambre.artificial_analysis.math500 import (
    extract_boxed,
    format_math500_prompt,
    load_math500,
    math_equal,
    normalize_math,
)
from enjambre.artificial_analysis.perf import benchmark_generation_perf
from enjambre.generate import generate
from enjambre.math_code.complex_tasks import COMPLEX_SPECS, make_complex_mc_splits

from .network import (
    build_k32_macro_network,
    build_k36_macro_network,
    build_k66_macro_network,
    macro_param_checksum,
)
from .field_v4 import ConstelacionV4Field, FIELD_PAIRS_32, FIELD_PAIRS_36, FIELD_PAIRS_66

DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "macro-neurons-v4"
)
SEED = 42


def _jdump(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


class MacroFieldEvaluator:
    def __init__(
        self,
        n_macro: int = 36,
        checkpoint_path: Optional[str] = None,
        out_dir: Optional[str] = None,
        rank: int = 16,
    ):
        self.n_macro = n_macro
        self.out_dir = os.path.abspath(out_dir or DEFAULT_OUT_DIR)
        os.makedirs(self.out_dir, exist_ok=True)
        self.rank = rank

        torch.manual_seed(SEED)
        torch.set_num_threads(min(8, os.cpu_count() or 8))

        print(f"[macro-eval] Cargando Red de {self.n_macro} Macro-Neuronas...", flush=True)
        if self.n_macro == 66:
            self.model, self.tok, self.delta = build_k66_macro_network()
            pairs = FIELD_PAIRS_66
        elif self.n_macro == 36:
            self.model, self.tok, self.delta = build_k36_macro_network()
            pairs = FIELD_PAIRS_36
        else:
            self.model, self.tok, self.delta = build_k32_macro_network()
            pairs = FIELD_PAIRS_32

        self.eos = self.tok.eos_token or ""
        self.base_checksum = macro_param_checksum(self.model)

        hidden = self.model.config.hidden_size
        self.field = ConstelacionV4Field(hidden=hidden, rank=rank, pairs=pairs).to(self.model.dtype)

        ckpt_file = checkpoint_path or os.path.join(self.out_dir, f"field_v4_k{self.n_macro}_best.pt")
        if os.path.isfile(ckpt_file):
            print(f"[macro-eval] Cargando pesos del campo desde {ckpt_file}...", flush=True)
            data = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            state = data["field_state"] if "field_state" in data else data
            self.field.load_state_dict(state)
        else:
            print("[macro-eval] ADVERTENCIA: Checkpoint no encontrado, evaluando campo en step 0.", flush=True)

        self.field.attach(self.model)

    def run_artificial_analysis_eval(self, n_samples: int = 28, max_tokens: int = 64) -> Dict[str, Any]:
        """Ejecuta el protocolo de Artificial Analysis sobre K36 + Campo v4."""
        print(f"[macro-eval] Evaluando Artificial Analysis (MATH-500 + Inferencia, {n_samples} problemas)...", flush=True)
        problems = load_math500(sample_size=n_samples, seed=SEED)

        ttfts: List[float] = []
        tpss: List[float] = []
        latencies: List[float] = []
        correct_count = 0
        sample_results = []

        for idx, p in enumerate(problems):
            prob_text = p.get("problem", "")
            sol_text = p.get("solution", "")
            ans_text = p.get("answer", "")
            prompt = format_math500_prompt(prob_text)
            perf = benchmark_generation_perf(
                model=self.model,
                tok=self.tok,
                prompt=prompt,
                max_new_tokens=max_tokens,
            )

            ttfts.append(perf["ttft_ms"])
            tpss.append(perf["tps"])
            latencies.append(perf["total_ms"])

            gen_norm = normalize_math(perf["output_text"])
            pred_boxed = extract_boxed(gen_norm)
            is_ok = False
            if pred_boxed:
                is_ok = math_equal(pred_boxed, sol_text) or (ans_text and math_equal(pred_boxed, ans_text))
            elif ans_text and ans_text.strip() in gen_norm:
                is_ok = True

            if is_ok:
                correct_count += 1

            sample_results.append({
                "subject": p.get("subject"),
                "level": p.get("level"),
                "is_correct": is_ok,
                "ttft_ms": round(perf["ttft_ms"], 2),
                "tps": round(perf["tps"], 2),
                "latency_ms": round(perf["total_ms"], 2),
                "pred_boxed": pred_boxed,
                "gold_answer": ans_text,
            })

        def p50(v):
            s = sorted(v)
            return s[len(s) // 2] if s else 0.0

        def p95(v):
            s = sorted(v)
            return s[int(len(s) * 0.95)] if s else 0.0

        res = {
            "model": f"K{self.n_macro}_macro + Campo CONSTELACIÓN v4",
            "n_macro_neurons": self.n_macro,
            "n_field_bridges": len(self.field.bridges),
            "samples_evaluated": len(problems),
            "math500_accuracy": round(correct_count / max(1, len(problems)), 4),
            "correct": correct_count,
            "inference_performance": {
                "ttft_ms_p50": round(p50(ttfts), 2),
                "ttft_ms_mean": round(sum(ttfts) / max(1, len(ttfts)), 2),
                "tps_mean": round(sum(tpss) / max(1, len(tpss)), 2),
                "tps_p50": round(p50(tpss), 2),
                "latency_ms_p50": round(p50(latencies), 2),
                "latency_ms_p95": round(p95(latencies), 2),
            },
            "sample_results": sample_results,
        }

        out_path = os.path.join(self.out_dir, f"results_artificial_analysis_k{self.n_macro}_campo.json")
        _jdump(out_path, res)
        print(f"[macro-eval] Artificial Analysis: TPS={res['inference_performance']['tps_mean']:.2f} | TTFT P50={res['inference_performance']['ttft_ms_p50']:.1f}ms | MATH-500 Acc={res['math500_accuracy']:.2%}", flush=True)
        return res

    def run_complex_benchmark(self, n_hold: int = 15) -> Dict[str, Any]:
        """Ejecuta la suite de 10 tareas complejas (150 ítems de holdout)."""
        print(f"[macro-eval] Evaluando Suite Compleja (10 tareas x {n_hold} = 150 ítems)...", flush=True)
        _, _, hold_items, _ = make_complex_mc_splits(n_train=6, n_val=4, n_hold=n_hold, seed=SEED)

        task_scores: Dict[str, Dict[str, Any]] = {}
        all_oks: List[bool] = []
        math_oks: List[bool] = []
        code_oks: List[bool] = []

        t0 = time.time()
        for spec in COMPLEX_SPECS:
            items = hold_items[spec.id]
            toks_limit = 72 if spec.domain == "code" else 20
            prompts = [it.prompt for it in items]
            outs = generate(self.model, self.tok, prompts, max_new_tokens=toks_limit, batch_size=8)
            oks = [bool(it.check(o)) for it, o in zip(items, outs)]
            acc = sum(oks) / max(1, len(oks))
            task_scores[spec.id] = {
                "acc": round(acc, 4),
                "correct": sum(oks),
                "total": len(items),
            }
            all_oks.extend(oks)
            if spec.domain == "math":
                math_oks.extend(oks)
            else:
                code_oks.extend(oks)

        mean_acc = sum(all_oks) / max(1, len(all_oks))
        math_acc = sum(math_oks) / max(1, len(math_oks))
        code_acc = sum(code_oks) / max(1, len(code_oks))

        res = {
            "condition": f"9_k{self.n_macro}_macro_campo_v4",
            "model_desc": f"K{self.n_macro} Macro-Neuronas ({self.n_macro}L) + Campo CONSTELACIÓN v4 ({len(self.field.bridges)} Puentes)",
            "n_macro_neurons": self.n_macro,
            "n_field_bridges": len(self.field.bridges),
            "trainable_field_params": self.field.n_params(),
            "mean_acc": round(mean_acc, 4),
            "math_acc": round(math_acc, 4),
            "code_acc": round(code_acc, 4),
            "total_items": len(all_oks),
            "elapsed_seconds": round(time.time() - t0, 2),
            "tasks": task_scores,
        }

        out_path = os.path.join(self.out_dir, f"results_complex_k{self.n_macro}_campo.json")
        _jdump(out_path, res)
        print(f"[macro-eval] Complex Bench: Global={mean_acc:.2%} | Math={math_acc:.2%} | Code={code_acc:.2%}", flush=True)
        return res
