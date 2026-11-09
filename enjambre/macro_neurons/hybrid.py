"""Arquitectura Híbrida Inteligente (Hybrid Base + K66 Macro-Neuronas Campo v4).

Combina lo mejor de dos mundos:
1. Especialista en Matemáticas: Modelo Base puro (28L en 1.5B o 24L en 0.5B),
   cuyos circuitos aritméticos nativos y memorias asociativas en MLPs están 100% intactos.
2. Especialista en Código: Red Profunda de Macro-Neuronas K66 (66L) + Campo CONSTELACIÓN v4 (28 Puentes),
   que actúa como una máquina de estados de ejecución con marcos de pila apilados para algoritmia.
"""
from __future__ import annotations
import gc
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from enjambre.generate import generate
from enjambre.math_code.complex_tasks import COMPLEX_SPECS, make_complex_mc_splits
from .network import build_k66_macro_network_1_5b
from .field_v4 import ConstelacionV4Field, FIELD_PAIRS_66

SEED = 42
DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "macro-neurons-v4"
)


def _jdump(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def classify_prompt_domain(prompt: str) -> str:
    """Clasificador de enrutamiento rápido y determinista (Matemáticas vs Código)."""
    p_lower = prompt.lower()
    code_keywords = [
        "python", "def ", "function", "return", "list", "dict",
        "array", "loop", "string", "recursion", "search", "matrix",
        "nested", "algorithm", "filter", "sort", "indices", "sublist",
    ]
    if any(k in p_lower for k in code_keywords):
        return "code"
    return "math"


class HybridMacroBaseModel:
    """Modelo Híbrido que enruta dinámicamente cada prompt a su subsistema óptimo:

    - Prompts de Matemáticas -> Modelo Base (28 capas nativas intactas)
    - Prompts de Código -> Red K66 Macro-Neuronas (66 capas + Campo CONSTELACIÓN v4)
    """

    def __init__(
        self,
        base_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.base_model_id = base_model_id
        self.device = device
        self.dtype = dtype

        self.tok = AutoTokenizer.from_pretrained(base_model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def generate_routed(
        self,
        prompts: List[str],
        domain: Optional[str] = None,
        system: Optional[str] = None,
        max_new_tokens: int = 72,
        batch_size: int = 8,
    ) -> List[str]:
        """Enruta los prompts al especialista correspondiente y retorna las respuestas en orden."""
        if not prompts:
            return []

        # Determinar dominios por prompt
        domains = [domain if domain is not None else classify_prompt_domain(p) for p in prompts]

        # Si todos pertenecen al mismo dominio, procesar en un solo paso
        if all(d == "math" for d in domains):
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id, torch_dtype=self.dtype
            ).eval()
            outs = generate(base_model, self.tok, prompts, system=system, max_new_tokens=max_new_tokens, batch_size=batch_size)
            del base_model
            gc.collect()
            return outs

        if all(d == "code" for d in domains):
            k66_model, _, _ = build_k66_macro_network_1_5b(model_hf_id=self.base_model_id, dtype=self.dtype)
            field = ConstelacionV4Field(hidden=k66_model.config.hidden_size, rank=16, pairs=FIELD_PAIRS_66).to(k66_model.dtype)
            ckpt = os.path.join(DEFAULT_OUT_DIR, "field_v4_k66_1_5b_best.pt")
            if os.path.isfile(ckpt):
                state = torch.load(ckpt, map_location="cpu", weights_only=False)
                field.load_state_dict(state.get("field_state", state))
            field.attach(k66_model)
            outs = generate(k66_model, self.tok, prompts, system=system, max_new_tokens=max_new_tokens, batch_size=batch_size)
            del k66_model, field
            gc.collect()
            return outs

        # Batch mixto: indexar y despachar
        math_indices = [i for i, d in enumerate(domains) if d == "math"]
        code_indices = [i for i, d in enumerate(domains) if d == "code"]

        results = [""] * len(prompts)

        if math_indices:
            math_prompts = [prompts[i] for i in math_indices]
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id, torch_dtype=self.dtype
            ).eval()
            math_outs = generate(base_model, self.tok, math_prompts, system=system, max_new_tokens=max_new_tokens, batch_size=batch_size)
            for idx, out in zip(math_indices, math_outs):
                results[idx] = out
            del base_model
            gc.collect()

        if code_indices:
            code_prompts = [prompts[i] for i in code_indices]
            k66_model, _, _ = build_k66_macro_network_1_5b(model_hf_id=self.base_model_id, dtype=self.dtype)
            field = ConstelacionV4Field(hidden=k66_model.config.hidden_size, rank=16, pairs=FIELD_PAIRS_66).to(k66_model.dtype)
            ckpt = os.path.join(DEFAULT_OUT_DIR, "field_v4_k66_1_5b_best.pt")
            if os.path.isfile(ckpt):
                state = torch.load(ckpt, map_location="cpu", weights_only=False)
                field.load_state_dict(state.get("field_state", state))
            field.attach(k66_model)
            code_outs = generate(k66_model, self.tok, code_prompts, system=system, max_new_tokens=max_new_tokens, batch_size=batch_size)
            for idx, out in zip(code_indices, code_outs):
                results[idx] = out
            del k66_model, field
            gc.collect()

        return results


def run_hybrid_complex_benchmark(
    base_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    n_hold: int = 8,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Ejecuta la suite compleja completa con la arquitectura Híbrida (Base 1.5B + K66_macro)."""
    out_dir = out_dir or DEFAULT_OUT_DIR
    part_path = os.path.join(out_dir, ".partial_complex_hybrid.json")
    out_path = os.path.join(out_dir, "results_complex_hybrid_base_k66_campo.json")

    print(f"[hybrid-eval] Evaluando Arquitectura Híbrida (Base 1.5B [Math] + K66_macro [Code])...", flush=True)
    _, _, hold_items, _ = make_complex_mc_splits(n_train=6, n_val=4, n_hold=n_hold, seed=SEED)

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
            print(f"[hybrid-eval] Reanudando suite híbrida desde tarea {len(task_scores)}/10...", flush=True)
        except Exception as e:
            print(f"[hybrid-eval] Error leyendo cache parcial ({e}), iniciando desde 0.", flush=True)

    hybrid = HybridMacroBaseModel(base_model_id=base_model_id)

    # 1. Evaluar tareas matemáticas con Base Model
    math_specs = [s for s in COMPLEX_SPECS if s.domain == "math" and s.id not in task_scores]
    if math_specs:
        print(f"[hybrid-eval] Despachando {len(math_specs)} tareas matemáticas al Especialista Base (28L)...", flush=True)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16).eval()
        for spec in math_specs:
            items = hold_items[spec.id]
            prompts = [it.prompt for it in items]
            outs = generate(base_model, hybrid.tok, prompts, system=spec.system_prompt, max_new_tokens=72, batch_size=8)
            oks = [bool(it.check(o)) for it, o in zip(items, outs)]
            acc = sum(oks) / max(1, len(oks))
            task_scores[spec.id] = {
                "acc": round(acc, 4),
                "correct": sum(oks),
                "total": len(items),
                "specialist": "Base_Qwen-1.5B (28L)",
            }
            print(f"  [task {spec.id}] (Math/Base) acc={acc:.2%} ({sum(oks)}/{len(items)})", flush=True)
            all_oks.extend(oks)
            math_oks.extend(oks)
            _jdump(part_path, {
                "tasks": task_scores,
                "all_oks": all_oks,
                "math_oks": math_oks,
                "code_oks": code_oks,
            })
        del base_model
        gc.collect()

    # 2. Evaluar tareas de código con K66 + Campo CONSTELACIÓN v4
    code_specs = [s for s in COMPLEX_SPECS if s.domain == "code" and s.id not in task_scores]
    if code_specs:
        print(f"[hybrid-eval] Despachando {len(code_specs)} tareas de código al Especialista K66 (66L + Campo v4)...", flush=True)
        k66_model, _, _ = build_k66_macro_network_1_5b(model_hf_id=base_model_id, dtype=torch.bfloat16)
        field = ConstelacionV4Field(hidden=k66_model.config.hidden_size, rank=16, pairs=FIELD_PAIRS_66).to(k66_model.dtype)
        ckpt = os.path.join(out_dir, "field_v4_k66_1_5b_best.pt")
        if os.path.isfile(ckpt):
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            field.load_state_dict(state.get("field_state", state))
        field.attach(k66_model)
        for spec in code_specs:
            items = hold_items[spec.id]
            prompts = [it.prompt for it in items]
            outs = generate(k66_model, hybrid.tok, prompts, system=spec.system_prompt, max_new_tokens=72, batch_size=8)
            oks = [bool(it.check(o)) for it, o in zip(items, outs)]
            acc = sum(oks) / max(1, len(oks))
            task_scores[spec.id] = {
                "acc": round(acc, 4),
                "correct": sum(oks),
                "total": len(items),
                "specialist": "K66_Macro_Campo_v4 (66L, 28 Puentes)",
            }
            print(f"  [task {spec.id}] (Code/K66) acc={acc:.2%} ({sum(oks)}/{len(items)})", flush=True)
            all_oks.extend(oks)
            code_oks.extend(oks)
            _jdump(part_path, {
                "tasks": task_scores,
                "all_oks": all_oks,
                "math_oks": math_oks,
                "code_oks": code_oks,
            })
        del k66_model, field
        gc.collect()

    mean_acc = sum(all_oks) / max(1, len(all_oks))
    math_acc = sum(math_oks) / max(1, len(math_oks))
    code_acc = sum(code_oks) / max(1, len(code_oks))

    res = {
        "condition": "10_hybrid_base_math_k66_code_campo_v4",
        "model_desc": "Híbrido Inteligente: Base Qwen-1.5B (Math, 28L) + K66 Macro-Neuronas (Code, 66L + Campo v4)",
        "router": "Deterministic Prompt Domain Dispatcher",
        "math_specialist": "Base Qwen-1.5B (28 capas nativas intactas)",
        "code_specialist": "K66_macro (66 macro-neuronas + Campo CONSTELACIÓN v4, 28 puentes)",
        "mean_acc": round(mean_acc, 4),
        "math_acc": round(math_acc, 4),
        "code_acc": round(code_acc, 4),
        "total_items": len(all_oks),
        "math_items": len(math_oks),
        "code_items": len(code_oks),
        "tasks": task_scores,
    }

    _jdump(out_path, res)
    if os.path.isfile(part_path):
        try:
            os.remove(part_path)
        except OSError:
            pass

    print(f"[hybrid-eval] Hybrid Bench: Global={mean_acc:.2%} | Math={math_acc:.2%} | Code={code_acc:.2%}", flush=True)
    return res
