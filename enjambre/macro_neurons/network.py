"""Ensamblaje de la Red de Macro-Neuronas (K32_macro).

Cada capa del Transformer actúa como una 'macro-neurona' dotada de su propio
subespacio receptivo y de atención.
Se expande el modelo base de 24 capas a 32 macro-neuronas:
- 24 macro-neuronas base (con ediciones de bias Gram-Schmidt para Math & Code)
- +4 macro-neuronas especializadas en Matemáticas Simbólicas (capas 12..15)
- +4 macro-neuronas especializadas en Algoritmia Python (capas 22..25)
"""
from __future__ import annotations
import copy
import hashlib
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from enjambre.config import resolve
from enjambre.edits import BiasEdit, EditDelta, unit
from enjambre.math_code.layer_edits import (
    DEFAULT_DIRS,
    build_layer_specialist_vectors,
    build_math_code_layer_deltas,
)


def macro_param_checksum(model: nn.Module) -> str:
    """Checksum SHA256 de todos los pesos del modelo excluyendo puentes del campo."""
    h = hashlib.sha256()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "field" in name or "bridges" in name:
                continue
            h.update(name.encode())
            h.update(p.detach().cpu().float().contiguous().numpy().tobytes())
    return h.hexdigest()


class MacroNeuronNetwork:
    """Encapsula un modelo CausalLM configurado como una Red de 32 Macro-Neuronas."""

    def __init__(self, model: nn.Module, tok: AutoTokenizer, damping: float = 0.50):
        self.model = model
        self.tok = tok
        self.damping = damping
        self.n_macro = len(self.model.model.layers)
        self.spec_vecs = build_layer_specialist_vectors(DEFAULT_DIRS)

    def freeze(self) -> None:
        """Congela el 100% de los parámetros de las macro-neuronas."""
        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def layers(self) -> nn.ModuleList:
        return self.model.model.layers

    def __len__(self) -> int:
        return len(self.layers)


def build_k32_macro_network(
    model_hf_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    damping: float = 0.50,
    directions_path: str = DEFAULT_DIRS,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, EditDelta]:
    """Construye y calibra la Red de 32 Macro-Neuronas partiendo del modelo base:

    Distribución topológica (32 Macro-Neuronas):
    - Capas 0..11:   Macro-neuronas de percepción sintáctica y representaciones tempranas.
    - Capas 12..15:  [CLUSTER MATH] 4 macro-neuronas clonadas del núcleo con damping 0.50
                     y editadas quirúrgicamente con dirección matemática pura.
    - Capas 16..21:  Macro-neuronas intermedias originales (12..17) de razonamiento.
    - Capas 22..25:  [CLUSTER CODE] 4 macro-neuronas clonadas del núcleo con damping 0.50
                     y editadas quirúrgicamente con dirección de algoritmia Python.
    - Capas 26..31:  Macro-neuronas de síntesis superior y generación de tokens de salida (18..23).
    """
    tok = AutoTokenizer.from_pretrained(model_hf_id)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_hf_id, dtype=torch.float32).eval()
    spec_vecs = build_layer_specialist_vectors(directions_path)
    deltas = build_math_code_layer_deltas(directions_path)
    k24_delta = deltas["k24_mc"]

    # Mapeo de sesgos originales de K24_mc
    k24_by_layer = {b.layer: b for b in k24_delta.bias_edits}
    scale_ref = float(k24_delta.bias_edits[0].scale)

    new_layers: List[nn.Module] = []
    edits: List[BiasEdit] = []

    # 1. Macro-Neuronas 0..11 (Base 0..11)
    for orig_idx in range(12):
        l = base_model.model.layers[orig_idx]
        new_layers.append(l)
        new_idx = len(new_layers) - 1
        if orig_idx in k24_by_layer:
            b = k24_by_layer[orig_idx]
            edits.append(BiasEdit(layer=new_idx, vec=b.vec.clone(), scale=float(b.scale)))

    # 2. Macro-Neuronas 12..15: Cluster Math (clonadas de 10..13)
    for j in range(4):
        src_idx = 10 + j
        cloned = copy.deepcopy(base_model.model.layers[src_idx])
        with torch.no_grad():
            cloned.mlp.down_proj.weight.mul_(damping)
            cloned.self_attn.o_proj.weight.mul_(damping)
        new_layers.append(cloned)
        new_idx = len(new_layers) - 1
        # Edición especializada en matemáticas puras
        v_math = spec_vecs[src_idx]["math"]
        edits.append(BiasEdit(layer=new_idx, vec=v_math.clone(), scale=scale_ref * 0.45))

    # 3. Macro-Neuronas 16..21: Razonamiento Intermedio (Base 12..17)
    for orig_idx in range(12, 18):
        l = base_model.model.layers[orig_idx]
        new_layers.append(l)
        new_idx = len(new_layers) - 1
        if orig_idx in k24_by_layer:
            b = k24_by_layer[orig_idx]
            edits.append(BiasEdit(layer=new_idx, vec=b.vec.clone(), scale=float(b.scale)))

    # 4. Macro-Neuronas 22..25: Cluster Code (clonadas de 14..17)
    for j in range(4):
        src_idx = 14 + j
        cloned = copy.deepcopy(base_model.model.layers[src_idx])
        with torch.no_grad():
            cloned.mlp.down_proj.weight.mul_(damping)
            cloned.self_attn.o_proj.weight.mul_(damping)
        new_layers.append(cloned)
        new_idx = len(new_layers) - 1
        # Edición especializada en código Python limpio
        v_code = spec_vecs[src_idx]["code"]
        edits.append(BiasEdit(layer=new_idx, vec=v_code.clone(), scale=scale_ref * 0.45))

    # 5. Macro-Neuronas 26..31: Síntesis Superior (Base 18..23)
    for orig_idx in range(18, 24):
        l = base_model.model.layers[orig_idx]
        new_layers.append(l)
        new_idx = len(new_layers) - 1
        if orig_idx in k24_by_layer:
            b = k24_by_layer[orig_idx]
            edits.append(BiasEdit(layer=new_idx, vec=b.vec.clone(), scale=float(b.scale)))

    # Re-indexación estricta de layer_idx para KV cache y rotary embeddings
    for i, l in enumerate(new_layers):
        l.self_attn.layer_idx = i

    base_model.model.layers = nn.ModuleList(new_layers)
    base_model.config.num_hidden_layers = len(new_layers)
    base_model.config.layer_types = ["full_attention"] * len(new_layers)

    # Aplicar físicamente las ediciones a los sesgos de down_proj
    for b in edits:
        lin = base_model.model.layers[b.layer].mlp.down_proj
        vec = (b.scale * b.vec).to(dtype=lin.weight.dtype, device=lin.weight.device)
        with torch.no_grad():
            if lin.bias is None:
                lin.bias = nn.Parameter(vec.clone())
            else:
                lin.bias.add_(vec)

    # Congelar el 100% de los pesos de la red de macro-neuronas
    for p in base_model.parameters():
        p.requires_grad_(False)

    macro_delta = EditDelta(
        name="k32_macro",
        bias_edits=edits,
        meta={
            "n_macro": 32,
            "damping": damping,
            "math_cluster": [12, 13, 14, 15],
            "code_cluster": [22, 23, 24, 25],
        },
    )

    return base_model, tok, macro_delta
