"""Arnés de orquestación: base compartida + deltas de expertos + memoria."""
from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .build_experts import load_delta
from .edits import BiasEdit, EditDelta, EditableModel, bake_steering_bias, mean_activation_diff
from .experts import all_expert_specs, expert_by_id
from .generate import chat_text, generate, generate_one, perplexity
from .memory import SharedMemory, build_recall_prompt, recall_check
from .router import Router
from .config import MEMIT_LAYER_FRACS, ROME_LAYER_FRAC, artifacts_dir, resolve, scale_layers

INST = "HuggingFaceTB/SmolLM2-135M-Instruct"
LEGACY_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
# fracciones de las capas ROME legacy (14..24 de 30 en 135M)
LEGACY_FACT_FRACS = [0.47, 0.53, 0.6, 0.67, 0.73, 0.8]


class Swarm:
    def __init__(
        self,
        model_name: str = INST,
        deltas_dir: Optional[str] = None,
        dtype=torch.float32,
        memory_layer: int = 15,
    ):
        torch.manual_seed(0)
        self.cfg = resolve(model_name)
        model_name = self.cfg.hf_id
        if deltas_dir == "auto":
            d = artifacts_dir(self.cfg.key)
            deltas_dir = d if os.path.isdir(d) else (LEGACY_DIR if self.cfg.key == "smollm2-135m" else None)
        torch.set_num_threads(min(8, os.cpu_count() or 8))
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).eval()
        self.em = EditableModel(model)
        self.router = Router()
        self.deltas: Dict[str, EditDelta] = {}
        self.deltas_dir = deltas_dir
        if deltas_dir and os.path.isdir(deltas_dir):
            self.load_deltas(deltas_dir)
        nL = self.em.n_layers()
        self.memory = SharedMemory(self.em, self.tok, layer=memory_layer,
                                   fact_layers=scale_layers(LEGACY_FACT_FRACS, nL))
        self.model_name = model_name
        self._wmem = None  # MEMIT/ROME con covarianza (lazy)

    def weight_memory(self):
        if self._wmem is None:
            from .memit import WeightMemory
            from .membench import cov_texts
            nL = self.em.n_layers()
            self._wmem = WeightMemory(self.em.model, self.tok, cov_texts(self.tok),
                                      scale_layers([ROME_LAYER_FRAC], nL)[0], scale_layers(MEMIT_LAYER_FRACS, nL))
        return self._wmem

    def memory_write_batch(self, facts: List[Dict[str, str]], method: str = "memit") -> Dict[str, Any]:
        """Escribe varios hechos a la vez (MEMIT) o en secuencia (ROME+cov). Reemplaza la memoria previa."""
        from .memit import FactSpec
        self.memory.clear()
        wm = self.weight_memory()
        wm.clear()
        specs = [FactSpec(f["subject"], f["object"],
                          f.get("train_prompts") or ["What is {s}? Answer briefly:", "Tell me about {s}."], [],
                          f.get("target_tpl") or "{s} is {o}.") for f in facts]
        if method == "rome":
            for sp in specs:
                wm.write_rome(sp)
        else:
            wm.write_memit(specs)
        return {"method": method, "n_facts": len(specs), "layers": sorted({e.layer for e in wm.applied})}

    def load_deltas(self, directory: str) -> List[str]:
        loaded = []
        for e in all_expert_specs():
            path = os.path.join(directory, f"delta_{e.id}.pt")
            if os.path.isfile(path):
                self.deltas[e.id] = load_delta(path)
                loaded.append(e.id)
        self.deltas_dir = directory
        return loaded

    def list_experts(self) -> List[Dict[str, Any]]:
        out = []
        for e in all_expert_specs():
            d = self.deltas.get(e.id)
            out.append({
                "id": e.id,
                "name": e.name,
                "description": e.description,
                "keywords": e.keywords,
                "ready": d is not None,
                "layer": d.meta.get("layer") if d else None,
                "scale": d.meta.get("scale") if d else None,
            })
        return out

    def apply_expert(self, expert_id: str, keep_memory: bool = True) -> Dict[str, Any]:
        if expert_id not in self.deltas:
            raise KeyError(f"experto '{expert_id}' sin delta; construye o carga deltas")
        # Mantener facts de memoria (rank1) si se pide
        if keep_memory:
            facts = list(self.memory.facts.values())
            self.em.revert_all()
            self.em.apply_only_biases(self.deltas[expert_id])
            for f in facts:
                self.em.apply_rank1(f.edit)
            self.memory.facts = {f.fact_id: f for f in facts}
        else:
            self.em.apply_delta(self.deltas[expert_id], keep_memory=False)
            self.memory.facts.clear()
        return {"active": expert_id, "keep_memory": keep_memory}

    def revert(self) -> None:
        facts = list(self.memory.facts.values())
        self.em.revert_all()
        for f in facts:
            self.em.apply_rank1(f.edit)

    def create_or_modify_expert(
        self,
        expert_id: str,
        contrast_prompts: List[Dict[str, str]],
        layer: int = 10,
        scale: float = 1.0,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """API: calcula dirección desde contrast prompts y hornea sesgo.

        contrast_prompts: lista de {prompt, positive_system?} o {pos_text, neg_text}.
        """
        pos, neg = [], []
        for cp in contrast_prompts:
            if "pos_text" in cp and "neg_text" in cp:
                pos.append(cp["pos_text"])
                neg.append(cp["neg_text"])
            else:
                p = cp["prompt"]
                sys_p = cp.get("positive_system") or system_prompt or "Follow the task exactly."
                pos.append(chat_text(self.tok, p, sys_p))
                neg.append(chat_text(self.tok, p, None))
        self.em.revert_biases()
        # no tocar memoria rank1 al medir dirección: revert biases only already done
        direction = mean_activation_diff(self.em.model, self.tok, pos, neg, batch_size=4)
        vec = direction[layer + 1]
        delta = EditDelta(
            name=expert_id,
            bias_edits=[bake_steering_bias(vec, layer, scale)],
            meta={"layer": layer, "scale": scale, "n_contrast": len(contrast_prompts)},
        )
        self.deltas[expert_id] = delta
        if self.deltas_dir:
            os.makedirs(self.deltas_dir, exist_ok=True)
            torch.save(
                {"name": expert_id, "layer": layer, "scale": scale, "vec": vec, "meta": delta.meta},
                os.path.join(self.deltas_dir, f"delta_{expert_id}.pt"),
            )
        return {"id": expert_id, "layer": layer, "scale": scale, "vec_norm": float(vec.norm())}

    def generate(self, prompt: str, expert_id: Optional[str] = None, max_new_tokens: int = 80) -> str:
        if expert_id:
            self.apply_expert(expert_id, keep_memory=True)
        elif expert_id is None and self.deltas:
            # sin experto explícito: base (+ memoria)
            self.revert()
        return generate_one(self.em.model, self.tok, prompt, max_new_tokens=max_new_tokens)

    def route_and_run(self, prompt: str, strategy: str = "hybrid", max_new_tokens: int = 80) -> Dict[str, Any]:
        ranked = self.router.route(prompt, strategy=strategy, top_k=3)
        eid = ranked[0][0]
        if eid in self.deltas:
            self.apply_expert(eid, keep_memory=True)
        else:
            self.revert()
        out = generate_one(self.em.model, self.tok, prompt, max_new_tokens=max_new_tokens)
        return {"expert": eid, "ranking": ranked, "output": out}

    # ---- memoria ----
    def memory_write(self, subject: str, obj: str, fact_id: Optional[str] = None, layer: Optional[int] = None) -> Dict:
        # escritura sobre el modelo SIN sesgos de experto (solo memoria)
        self.em.revert_biases()
        fact = self.memory.write(subject, obj, fact_id=fact_id, layer=layer)
        return {"fact_id": fact.fact_id, "subject": fact.subject, "object": fact.obj, "layer": fact.layer}

    def memory_read(self, subject: str, max_new_tokens: int = 40) -> Dict:
        self.em.revert_biases()  # leer con memoria, sin experto
        # reasegurar rank1
        self.em.revert_rank1()
        for f in self.memory.facts.values():
            self.em.apply_rank1(f.edit)
        prompt = build_recall_prompt(subject)
        out = generate_one(self.em.model, self.tok, prompt, max_new_tokens=max_new_tokens)
        return {"subject": subject, "prompt": prompt, "output": out}

    def memory_list(self) -> List[Dict]:
        return self.memory.list_facts()

    def ppl(self) -> float:
        return perplexity(self.em.model, self.tok)

    def status(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "model_key": self.cfg.key,
            "n_layers": self.em.n_layers(),
            "hidden": self.em.hidden_size(),
            "experts_ready": list(self.deltas.keys()),
            "active": list(self.em._active),
            "memory_facts": len(self.memory.facts),
            "delta_bytes": sum(d.nbytes() for d in self.deltas.values()),
            "note": "Un solo modelo base en RAM; expertos = deltas de sesgo (~2KB c/u).",
        }
