"""Cirugía de Vectores de Tarea (Task Vector Surgery) y Ensamblado de Capas.

Fusión entre `Qwen/Qwen2.5-0.5B-Instruct` y `Qwen/Qwen2.5-Coder-0.5B-Instruct`:
Calcula:
    ΔW = W_coder - W_base
Aplica selectivamente a las capas centrales de cómputo algorítmico (capas 8 a 16):
    W_fused = W_base + λ * ΔW

Conserva la arquitectura de 24 capas, la edición Gram-Schmidt de K24_mc y los
conectores neuro-geométricos (CONSTELACIÓN v4 / BQ10 DA-STR).
"""
from __future__ import annotations
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

DEFAULT_TARGET_LAYERS = list(range(8, 17))  # Capas 8 a 16 (núcleo de cómputo)


class TaskVectorSurgeon:
    def __init__(
        self,
        base_model: nn.Module,
        coder_model_id: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        target_layers: Optional[List[int]] = None,
        scale: float = 0.6,
    ):
        self.base_model = base_model
        self.target_layers = target_layers or DEFAULT_TARGET_LAYERS
        self.scale = scale
        self.coder_model_id = coder_model_id
        self._backups: Dict[str, torch.Tensor] = {}

    def extract_and_apply_task_vectors(self) -> Dict[str, Any]:
        """Carga Qwen2.5-Coder y aplica el vector de tarea a las capas seleccionadas."""
        t0 = time.time()
        print(f"[task_vectors] Cargando {self.coder_model_id} para cirugía de capas...", flush=True)
        coder = AutoModelForCausalLM.from_pretrained(self.coder_model_id, dtype=torch.float32)

        n_tensors_edited = 0
        total_delta_norm = 0.0

        with torch.no_grad():
            for layer_idx in self.target_layers:
                base_layer = self.base_model.model.layers[layer_idx]
                coder_layer = coder.model.layers[layer_idx]

                # Fusión en MLP (down_proj, up_proj, gate_proj) y Atención (o_proj, q_proj, k_proj, v_proj)
                modules_to_fuse = [
                    ("mlp.down_proj.weight", base_layer.mlp.down_proj, coder_layer.mlp.down_proj),
                    ("mlp.up_proj.weight", base_layer.mlp.up_proj, coder_layer.mlp.up_proj),
                    ("mlp.gate_proj.weight", base_layer.mlp.gate_proj, coder_layer.mlp.gate_proj),
                    ("self_attn.o_proj.weight", base_layer.self_attn.o_proj, coder_layer.self_attn.o_proj),
                    ("self_attn.q_proj.weight", base_layer.self_attn.q_proj, coder_layer.self_attn.q_proj),
                    ("self_attn.k_proj.weight", base_layer.self_attn.k_proj, coder_layer.self_attn.k_proj),
                    ("self_attn.v_proj.weight", base_layer.self_attn.v_proj, coder_layer.self_attn.v_proj),
                ]

                for mod_name, b_mod, c_mod in modules_to_fuse:
                    key = f"layer_{layer_idx}.{mod_name}"
                    if key not in self._backups:
                        self._backups[key] = b_mod.weight.detach().clone()

                    delta = c_mod.weight.detach() - b_mod.weight.detach()
                    total_delta_norm += float(delta.norm())
                    b_mod.weight.add_(self.scale * delta)
                    n_tensors_edited += 1

        del coder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"[task_vectors] Cirugía completada en {time.time()-t0:.1f}s: "
            f"{n_tensors_edited} tensores fusionados en capas {self.target_layers} con λ={self.scale}",
            flush=True,
        )
        return {
            "target_layers": self.target_layers,
            "scale": self.scale,
            "n_tensors_edited": n_tensors_edited,
            "mean_delta_norm": round(total_delta_norm / max(1, n_tensors_edited), 4),
            "wall_secs": round(time.time() - t0, 1),
        }

    def restore_base_weights(self) -> None:
        """Restaura exactamente los pesos originales de la base."""
        with torch.no_grad():
            for layer_idx in self.target_layers:
                base_layer = self.base_model.model.layers[layer_idx]
                modules = [
                    ("mlp.down_proj.weight", base_layer.mlp.down_proj),
                    ("mlp.up_proj.weight", base_layer.mlp.up_proj),
                    ("mlp.gate_proj.weight", base_layer.mlp.gate_proj),
                    ("self_attn.o_proj.weight", base_layer.self_attn.o_proj),
                    ("self_attn.q_proj.weight", base_layer.self_attn.q_proj),
                    ("self_attn.k_proj.weight", base_layer.self_attn.k_proj),
                    ("self_attn.v_proj.weight", base_layer.self_attn.v_proj),
                ]
                for mod_name, b_mod in modules:
                    key = f"layer_{layer_idx}.{mod_name}"
                    if key in self._backups:
                        b_mod.weight.copy_(self._backups[key])
        self._backups.clear()
        print("[task_vectors] Pesos base restaurados con éxito.", flush=True)
