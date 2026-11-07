"""Harness de evaluación de Artificial Analysis para K24_mc y Neuro-Conectores.

Evalúa:
1. Precisión en MATH-500 (Artificial Analysis Benchmark oficial).
2. Desglose por materia: Algebra, Precalculus, Number Theory, Geometry, etc.
3. Rendimiento de inferencia (TTFT, TPS, latencia P50/P95).

Referencia: https://artificialanalysis.ai/evaluations/math-500
"""
from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from enjambre.adaptive.snapshot import SnapshotActivator
from enjambre.bridges.train_eval import freeze_model, param_checksum
from enjambre.config import resolve
from enjambre.edits import BiasEdit, EditDelta
from enjambre.math_code.neuro_connectors import (
    ARCHITECTURES,
    PAIRS_K24,
    NeuroBridgeBank,
    build_warm_neuro_bridge_bank,
)

from .math500 import (
    extract_answer_fallback,
    extract_boxed,
    format_math500_prompt,
    load_math500,
    math_equal,
)
from .perf import aggregate_perf_stats, benchmark_generation_perf

DEFAULT_AA_OUT = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "artificial-analysis"
)
K24_MC_ARTIFACTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "math-code-k12plus"
)
NEURO_ARTIFACTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "neuro-connectors-k24"
)
COMPLEX_ARTIFACTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "complex-bench-k24"
)


def _jdump(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def _load_delta(path: str) -> EditDelta:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, EditDelta):
        return raw
    edits = [
        BiasEdit(layer=int(e["layer"]), vec=e["vec"].float().clone(), scale=float(e["scale"]))
        for e in raw["edits"]
    ]
    return EditDelta(name=str(raw.get("name", "k24_mc")), bias_edits=edits, meta=dict(raw.get("meta", {})))


class ArtificialAnalysisHarness:
    def __init__(
        self,
        out_dir: Optional[str] = None,
        rank: int = 16,
        sample_size: int = 28,  # 4 problemas por cada una de las 7 materias de MATH-500
        seed: int = 42,
    ):
        self.cfg = resolve("qwen2.5-0.5b")
        self.out_dir = os.path.abspath(out_dir or DEFAULT_AA_OUT)
        os.makedirs(self.out_dir, exist_ok=True)
        self.rank = rank
        self.sample_size = sample_size
        self.seed = seed

        torch.manual_seed(seed)
        torch.set_num_threads(min(8, os.cpu_count() or 8))

        self.tok = AutoTokenizer.from_pretrained(self.cfg.hf_id)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.hf_id, dtype=torch.float32).eval()
        freeze_model(self.model)
        self.base_checksum = param_checksum(self.model)
        self.act = SnapshotActivator(self.model)

        # Cargar problemas de MATH-500
        self.problems = load_math500(sample_size=self.sample_size, seed=self.seed)

        # Cargar pesos de K24_mc
        k24_delta_path = os.path.join(K24_MC_ARTIFACTS_DIR, "deltas", "k24_mc.pt")
        k24_bridges_path = os.path.join(K24_MC_ARTIFACTS_DIR, "bridges_k24_mc_best.pt")
        self.k24_delta = _load_delta(k24_delta_path)

        ck = torch.load(k24_bridges_path, map_location="cpu", weights_only=False)
        self.k24_bridge_state = {
            k: v.detach().cpu().float().clone()
            for k, v in (ck["state"] if isinstance(ck, dict) and "state" in ck else ck).items()
        }

        # Checkpoints de conectores adaptados
        self.connector_states: Dict[str, Dict[str, torch.Tensor]] = {
            "residual_base": self.k24_bridge_state,
        }
        for arch in ("constelacion_v4", "bq10_dastr", "soft_bellman_simplicial"):
            # Buscar primero en complex-bench y luego en neuro-connectors
            p_comp = os.path.join(COMPLEX_ARTIFACTS_DIR, f"connectors_complex_{arch}_adapted_best.pt")
            p_neuro = os.path.join(NEURO_ARTIFACTS_DIR, f"connectors_{arch}_best.pt")
            target_p = p_comp if os.path.isfile(p_comp) else p_neuro
            if os.path.isfile(target_p):
                c = torch.load(target_p, map_location="cpu", weights_only=False)
                self.connector_states[arch] = {
                    k: v.detach().cpu().float().clone()
                    for k, v in (c["state"] if isinstance(c, dict) and "state" in c else c).items()
                }

        self.bank: Optional[NeuroBridgeBank] = None

    def _chat_prompt(self, problem_str: str) -> str:
        prompt_content = format_math500_prompt(problem_str)
        msgs = [{"role": "user", "content": prompt_content}]
        if hasattr(self.tok, "apply_chat_template"):
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return prompt_content

    def evaluate_condition_math500(
        self,
        label: str,
        arch: str,
        bridge_state: Optional[Dict[str, torch.Tensor]] = None,
        use_delta: bool = True,
        max_tokens: int = 120,
    ) -> Dict[str, Any]:
        """Evalúa un modelo en el subconjunto representativo de MATH-500 y mide rendimiento."""
        out_path = os.path.join(self.out_dir, f"results_math500_{label}.json")
        if os.path.isfile(out_path):
            cached = json.load(open(out_path, encoding="utf-8"))
            if cached.get("n_problems") == len(self.problems) and cached.get("max_tokens") == max_tokens:
                print(f"[aa-math500:{label}] Reutilizando caché acc={cached['accuracy']:.4f}", flush=True)
                return cached

        if self.bank is not None:
            self.bank.remove_hooks()
            self.bank = None

        if bridge_state is not None:
            self.bank = build_warm_neuro_bridge_bank(
                self.model.config.hidden_size,
                arch=arch,
                rank=self.rank,
                pairs=PAIRS_K24,
                k24_pretrained_state=bridge_state,
            )
            self.bank.attach(self.model)
            self.bank.enabled = True
            self.bank.clear_buf()

        t0 = time.time()
        results = []
        perf_runs = []
        by_subject: Dict[str, List[bool]] = {}
        by_level: Dict[int, List[bool]] = {}

        delta_ctx = self.act.session(self.k24_delta) if use_delta else nullcontext()

        with delta_ctx:
            for i, p in enumerate(self.problems):
                raw_prompt = self._chat_prompt(p["problem"])
                # Medir inferencia, TTFT y TPS
                perf = benchmark_generation_perf(
                    self.model, self.tok, raw_prompt, max_new_tokens=max_tokens
                )
                perf_runs.append(perf)

                gen_text = perf["output_text"]
                # Extraer predicción
                pred = extract_boxed(gen_text) or extract_answer_fallback(gen_text)
                target = p["answer"]
                ok = math_equal(pred, target)

                sub = p.get("subject", "General")
                lvl = int(p.get("level", 1))
                by_subject.setdefault(sub, []).append(ok)
                by_level.setdefault(lvl, []).append(ok)

                results.append({
                    "id": p.get("unique_id", f"math500_{i}"),
                    "subject": sub,
                    "level": lvl,
                    "problem": p["problem"][:160],
                    "target": target,
                    "pred": pred,
                    "generated_text": gen_text[:200],
                    "ok": ok,
                    "ttft_ms": perf["ttft_ms"],
                    "tps": perf["tps"],
                })
                if (i + 1) % 7 == 0 or (i + 1) == len(self.problems):
                    cur_acc = sum(r["ok"] for r in results) / len(results)
                    print(f"[aa-math500:{label}] {i+1}/{len(self.problems)} acc={cur_acc:.4f} (TPS={perf['tps']:.1f})", flush=True)

        if self.bank is not None:
            self.bank.clear_buf()

        total_ok = sum(r["ok"] for r in results)
        accuracy = round(total_ok / len(results), 4) if results else 0.0

        subject_acc = {
            s: round(sum(oks) / len(oks), 4) for s, oks in sorted(by_subject.items())
        }
        level_acc = {
            str(l): round(sum(oks) / len(oks), 4) for l, oks in sorted(by_level.items())
        }
        perf_agg = aggregate_perf_stats(perf_runs)

        rep = {
            "label": label,
            "arch": arch,
            "use_delta": use_delta,
            "max_tokens": max_tokens,
            "n_problems": len(self.problems),
            "n_correct": total_ok,
            "accuracy": accuracy,
            "by_subject": subject_acc,
            "by_level": level_acc,
            "performance": perf_agg,
            "wall_secs": round(time.time() - t0, 2),
            "sample_results": results[:4],
        }
        _jdump(out_path, rep)
        return rep

    def run_artificial_analysis_suite(self) -> Dict[str, Any]:
        """Ejecuta el benchmark completo comparando la Base vs K24_mc vs Conectores Neuro-Geométricos."""
        t_start = time.time()

        # 1. Base Qwen2.5-0.5B sin editar (Cero puentes, sin delta)
        rep_base = self.evaluate_condition_math500(
            "1_base_qwen_unedited",
            arch="none",
            bridge_state=None,
            use_delta=False,
        )

        # 2. K24_mc Residual Base (24 capas editadas + 6 puentes Residuales)
        rep_k24_residual = self.evaluate_condition_math500(
            "2_k24_mc_residual",
            arch="residual_base",
            bridge_state=self.connector_states["residual_base"],
            use_delta=True,
        )

        # 3. K24_mc + CONSTELACIÓN v4 (Koopman 2x2 para dinámicas matemáticas)
        st_cst = self.connector_states.get("constelacion_v4", self.k24_bridge_state)
        rep_cst = self.evaluate_condition_math500(
            "3_k24_constelacion_v4",
            arch="constelacion_v4",
            bridge_state=st_cst,
            use_delta=True,
        )

        # 4. K24_mc + BQ10 DA-STR (Innovación causal y atenuación de grado)
        st_bq10 = self.connector_states.get("bq10_dastr", self.k24_bridge_state)
        rep_bq10 = self.evaluate_condition_math500(
            "4_k24_bq10_dastr",
            arch="bq10_dastr",
            bridge_state=st_bq10,
            use_delta=True,
        )

        if self.bank is not None:
            self.bank.remove_hooks()
            self.bank = None

        after_checksum = param_checksum(self.model)
        assert after_checksum == self.base_checksum, "Base model weights must remain 100% frozen!"

        all_conditions = [rep_base, rep_k24_residual, rep_cst, rep_bq10]
        winner = max(all_conditions, key=lambda c: (c["accuracy"], c["performance"]["output_tps"]["mean"]))

        summary = {
            "benchmark_name": "Artificial Analysis MATH-500 & Performance Benchmark",
            "benchmark_url": "https://artificialanalysis.ai/evaluations/math-500",
            "eval_framework": "Artificial Analysis Evaluation Protocol",
            "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "sample_size": len(self.problems),
            "winner": {
                "label": winner["label"],
                "arch": winner["arch"],
                "math500_accuracy": winner["accuracy"],
                "output_tps": winner["performance"]["output_tps"]["mean"],
                "ttft_ms": winner["performance"]["ttft_ms"]["mean"],
            },
            "comparison_table": [
                {
                    "label": c["label"],
                    "arch": c["arch"],
                    "math500_acc": c["accuracy"],
                    "ttft_ms_p50": c["performance"]["ttft_ms"]["p50"],
                    "output_tps_mean": c["performance"]["output_tps"]["mean"],
                    "latency_p50_ms": c["performance"]["latency_ms"]["p50"],
                    "wall_secs": c["wall_secs"],
                }
                for c in all_conditions
            ],
            "by_subject_table": {
                c["label"]: c["by_subject"] for c in all_conditions
            },
            "by_level_table": {
                c["label"]: c["by_level"] for c in all_conditions
            },
            "total_wall_secs": round(time.time() - t_start, 2),
        }

        _jdump(os.path.join(self.out_dir, "summary_artificial_analysis.json"), summary)
        return summary


class nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
