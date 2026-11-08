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
    extract_answer_fallback,
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
    build_k66_macro_network_1_5b,
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
        n_macro: int = 66,
        base_size: str = "0.5b",
        checkpoint_path: Optional[str] = None,
        out_dir: Optional[str] = None,
        rank: int = 16,
    ):
        self.n_macro = n_macro
        self.base_size = base_size.lower().strip()
        self.base_tag = "1_5b" if self.base_size in ("1.5b", "1b") else ""
        self.out_dir = os.path.abspath(out_dir or DEFAULT_OUT_DIR)
        os.makedirs(self.out_dir, exist_ok=True)
        self.rank = rank

        torch.manual_seed(SEED)
        torch.set_num_threads(min(8, os.cpu_count() or 8))

        print(f"[macro-eval] Cargando Red de {self.n_macro} Macro-Neuronas (Base: {self.base_size.upper()})...", flush=True)
        if self.n_macro in (24, 28, 0):
            from transformers import AutoModelForCausalLM
            hf_id = "Qwen/Qwen2.5-1.5B-Instruct" if self.base_tag == "1_5b" else "Qwen/Qwen2.5-0.5B-Instruct"
            self.tok = AutoTokenizer.from_pretrained(hf_id)
            self.tok.padding_side = "left"
            if self.tok.pad_token_id is None:
                self.tok.pad_token = self.tok.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.bfloat16).eval()
            self.delta = None
            pairs = []
        elif self.base_tag == "1_5b" and self.n_macro == 66:
            self.model, self.tok, self.delta = build_k66_macro_network_1_5b()
            pairs = FIELD_PAIRS_66
        elif self.n_macro == 66:
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

        tag = f"_{self.base_tag}" if self.base_tag else ""
        ckpt_file = checkpoint_path or os.path.join(self.out_dir, f"field_v4_k{self.n_macro}{tag}_best.pt")
        if pairs and os.path.isfile(ckpt_file):
            print(f"[macro-eval] Cargando pesos del campo desde {ckpt_file}...", flush=True)
            data = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            state = data["field_state"] if "field_state" in data else data
            self.field.load_state_dict(state)
        elif pairs:
            print(f"[macro-eval] ADVERTENCIA: Checkpoint ({ckpt_file}) no encontrado, evaluando campo en step 0.", flush=True)

        self.field.attach(self.model)

    def run_artificial_analysis_eval(
        self, n_samples: int = 28, max_tokens: int = 96, max_per_run: Optional[int] = None
    ) -> Dict[str, Any]:
        """Ejecuta el protocolo de Artificial Analysis sobre K36 + Campo v4."""
        print(f"[macro-eval] Evaluando Artificial Analysis (MATH-500 + Inferencia, {n_samples} problemas)...", flush=True)
        problems = load_math500(sample_size=n_samples, seed=SEED)

        tag = f"_{self.base_tag}" if self.base_tag else ""
        part_path = os.path.join(self.out_dir, f".partial_aa_k{self.n_macro}{tag}.json")

        sample_results: List[Dict[str, Any]] = []
        ttfts: List[float] = []
        tpss: List[float] = []
        latencies: List[float] = []
        correct_count = 0

        if os.path.isfile(part_path):
            try:
                with open(part_path, "r", encoding="utf-8") as f:
                    part_data = json.load(f)
                sample_results = part_data.get("sample_results", [])
                ttfts = part_data.get("ttfts", [])
                tpss = part_data.get("tpss", [])
                latencies = part_data.get("latencies", [])
                correct_count = part_data.get("correct_count", 0)
                print(f"[macro-eval] Reanudando AA desde problema {len(sample_results)}/{len(problems)} (correctos previos={correct_count})...", flush=True)
            except Exception as e:
                print(f"[macro-eval] Error leyendo cache parcial ({e}), iniciando desde 0.", flush=True)

        run_count = 0
        for idx in range(len(sample_results), len(problems)):
            if max_per_run is not None and run_count >= max_per_run:
                print(f"[macro-eval] Límite por corrida alcanzado ({run_count} ítems). Avance: {len(sample_results)}/{len(problems)}.", flush=True)
                break
            run_count += 1
            p = problems[idx]
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
            else:
                fb = extract_answer_fallback(perf["output_text"])
                if fb:
                    is_ok = math_equal(fb, sol_text) or (ans_text and math_equal(fb, ans_text))

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

            # Guardar estado parcial incremental
            _jdump(part_path, {
                "sample_results": sample_results,
                "ttfts": ttfts,
                "tpss": tpss,
                "latencies": latencies,
                "correct_count": correct_count,
            })

            if (idx + 1) % 4 == 0 or idx == len(problems) - 1:
                print(f"  [aa-eval] {idx + 1}/{len(problems)} procesados (correctos={correct_count})...", flush=True)

        if len(sample_results) < len(problems):
            print(f"[macro-eval] Lote parcial completado ({len(sample_results)}/{len(problems)}, correctos={correct_count}). Ejecuta nuevamente para procesar el siguiente lote.", flush=True)
            return {"status": "partial", "completed": len(sample_results), "total": len(problems), "correct": correct_count}

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

        tag = f"_{self.base_tag}" if self.base_tag else ""
        out_path = os.path.join(self.out_dir, f"results_artificial_analysis_k{self.n_macro}{tag}_campo.json")
        _jdump(out_path, res)
        if os.path.isfile(part_path):
            try:
                os.remove(part_path)
            except OSError:
                pass
        print(f"[macro-eval] Artificial Analysis: TPS={res['inference_performance']['tps_mean']:.2f} | TTFT P50={res['inference_performance']['ttft_ms_p50']:.1f}ms | MATH-500 Acc={res['math500_accuracy']:.2%}", flush=True)
        return res

    def run_complex_benchmark(self, n_hold: int = 15) -> Dict[str, Any]:
        """Ejecuta la suite de 10 tareas complejas (150 ítems de holdout o n_hold configurado)."""
        print(f"[macro-eval] Evaluando Suite Compleja (10 tareas x {n_hold} = {10 * n_hold} ítems)...", flush=True)
        _, _, hold_items, _ = make_complex_mc_splits(n_train=6, n_val=4, n_hold=n_hold, seed=SEED)

        tag = f"_{self.base_tag}" if self.base_tag else ""
        part_path = os.path.join(self.out_dir, f".partial_complex_k{self.n_macro}{tag}.json")
        out_path = os.path.join(self.out_dir, f"results_complex_k{self.n_macro}{tag}_campo.json")

        task_scores: Dict[str, Dict[str, Any]] = {}
        all_oks: List[bool] = []
        math_oks: List[bool] = []
        code_oks: List[bool] = []

        if os.path.isfile(part_path):
            try:
                with open(part_path, "r", encoding="utf-8") as f:
                    part_data = json.load(f)
                task_scores = part_data.get("tasks", {})
                all_oks = part_data.get("all_oks", [])
                math_oks = part_data.get("math_oks", [])
                code_oks = part_data.get("code_oks", [])
                print(f"[macro-eval] Reanudando suite compleja desde tarea {len(task_scores)}/10...", flush=True)
            except Exception as e:
                print(f"[macro-eval] Error leyendo cache parcial ({e}), iniciando desde 0.", flush=True)

        t0 = time.time()
        for spec in COMPLEX_SPECS:
            if spec.id in task_scores:
                print(f"  [task {spec.id}] (cached) acc={task_scores[spec.id]['acc']:.2%}", flush=True)
                continue

            if hasattr(self.field, "set_domain"):
                self.field.set_domain(spec.domain)

            items = hold_items[spec.id]
            toks_limit = 72
            prompts = [it.prompt for it in items]
            outs = generate(self.model, self.tok, prompts, system=spec.system_prompt, max_new_tokens=toks_limit, batch_size=8)
            oks = [bool(it.check(o)) for it, o in zip(items, outs)]
            acc = sum(oks) / max(1, len(oks))
            task_scores[spec.id] = {
                "acc": round(acc, 4),
                "correct": sum(oks),
                "total": len(items),
            }
            print(f"  [task {spec.id}] acc={acc:.2%} ({sum(oks)}/{len(items)})", flush=True)
            all_oks.extend(oks)
            if spec.domain == "math":
                math_oks.extend(oks)
            else:
                code_oks.extend(oks)

            # Persistir avance incremental
            _jdump(part_path, {
                "tasks": task_scores,
                "all_oks": all_oks,
                "math_oks": math_oks,
                "code_oks": code_oks,
            })

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

        _jdump(out_path, res)
        if os.path.isfile(part_path):
            try:
                os.remove(part_path)
            except OSError:
                pass
        print(f"[macro-eval] Complex Bench: Global={mean_acc:.2%} | Math={math_acc:.2%} | Code={code_acc:.2%}", flush=True)
        return res
