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


def build_k36_macro_network(
    model_hf_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    damping: float = 0.50,
    directions_path: str = DEFAULT_DIRS,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, EditDelta]:
    """Construye y calibra la Red de 36 Macro-Neuronas partiendo del modelo base:

    Distribución topológica (36 Macro-Neuronas):
    - Capas 0..9:    Macro-neuronas de percepción sintáctica temprana (Base 0..9).
    - Capas 10..13:  [CLUSTER ÁLGEBRA Y ECUACIONES] 4 macro-neuronas clonadas de 8..11
                     con damping 0.50 y editadas para cálculo algebraico simbólico.
    - Capas 14..19:  Macro-neuronas intermedias de razonamiento semántico (Base 10..15).
    - Capas 20..23:  [CLUSTER ARITMÉTICA Y RECURRENCIAS] 4 macro-neuronas clonadas de 12..15
                     con damping 0.50 y editadas para recurrencia y exponenciación modular.
    - Capas 24..27:  [CLUSTER ALGORITMIA Y DICCIONARIOS] 4 macro-neuronas clonadas de 16..19
                     con damping 0.50 y editadas para algoritmos y estructuras de datos Python.
    - Capas 28..35:  Macro-neuronas de síntesis superior y decodificación final (Base 16..23).
    """
    tok = AutoTokenizer.from_pretrained(model_hf_id)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_hf_id, dtype=torch.float32).eval()
    spec_vecs = build_layer_specialist_vectors(directions_path)
    deltas = build_math_code_layer_deltas(directions_path)
    k24_delta = deltas["k24_mc"]

    k24_by_layer = {b.layer: b for b in k24_delta.bias_edits}
    scale_ref = float(k24_delta.bias_edits[0].scale)

    new_layers: List[nn.Module] = []
    edits: List[BiasEdit] = []

    # 1. Macro-Neuronas 0..9 (Base 0..9)
    for orig_idx in range(10):
        l = base_model.model.layers[orig_idx]
        new_layers.append(l)
        new_idx = len(new_layers) - 1
        if orig_idx in k24_by_layer:
            b = k24_by_layer[orig_idx]
            edits.append(BiasEdit(layer=new_idx, vec=b.vec.clone(), scale=float(b.scale)))

    # 2. Macro-Neuronas 10..13: Cluster Álgebra y Ecuaciones (clonadas de 8..11)
    for j in range(4):
        src_idx = 8 + j
        cloned = copy.deepcopy(base_model.model.layers[src_idx])
        with torch.no_grad():
            cloned.mlp.down_proj.weight.mul_(damping)
            cloned.self_attn.o_proj.weight.mul_(damping)
        new_layers.append(cloned)
        new_idx = len(new_layers) - 1
        v_math = spec_vecs[src_idx]["math"]
        edits.append(BiasEdit(layer=new_idx, vec=v_math.clone(), scale=scale_ref * 0.50))

    # 3. Macro-Neuronas 14..19: Razonamiento Semántico Intermedio (Base 10..15)
    for orig_idx in range(10, 16):
        l = base_model.model.layers[orig_idx]
        new_layers.append(l)
        new_idx = len(new_layers) - 1
        if orig_idx in k24_by_layer:
            b = k24_by_layer[orig_idx]
            edits.append(BiasEdit(layer=new_idx, vec=b.vec.clone(), scale=float(b.scale)))

    # 4. Macro-Neuronas 20..23: Cluster Aritmética y Recurrencias (clonadas de 12..15)
    for j in range(4):
        src_idx = 12 + j
        cloned = copy.deepcopy(base_model.model.layers[src_idx])
        with torch.no_grad():
            cloned.mlp.down_proj.weight.mul_(damping)
            cloned.self_attn.o_proj.weight.mul_(damping)
        new_layers.append(cloned)
        new_idx = len(new_layers) - 1
        v_math = spec_vecs[src_idx]["math"]
        edits.append(BiasEdit(layer=new_idx, vec=v_math.clone(), scale=scale_ref * 0.50))

    # 5. Macro-Neuronas 24..27: Cluster Algoritmia y Diccionarios (clonadas de 16..19)
    for j in range(4):
        src_idx = 16 + j
        cloned = copy.deepcopy(base_model.model.layers[src_idx])
        with torch.no_grad():
            cloned.mlp.down_proj.weight.mul_(damping)
            cloned.self_attn.o_proj.weight.mul_(damping)
        new_layers.append(cloned)
        new_idx = len(new_layers) - 1
        v_code = spec_vecs[src_idx]["code"]
        edits.append(BiasEdit(layer=new_idx, vec=v_code.clone(), scale=scale_ref * 0.50))

    # 6. Macro-Neuronas 28..35: Síntesis Superior y Salida (Base 16..23)
    for orig_idx in range(16, 24):
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
        name="k36_macro",
        bias_edits=edits,
        meta={
            "n_macro": 36,
            "damping": damping,
            "algebra_cluster": [10, 11, 12, 13],
            "arithmetic_cluster": [20, 21, 22, 23],
            "code_cluster": [24, 25, 26, 27],
        },
    )

    return base_model, tok, macro_delta


def build_k66_macro_network(
    model_hf_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    damping: float = 0.35,
    math_bias_scale: float = 0.0,
    directions_path: str = DEFAULT_DIRS,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, EditDelta]:
    """Construye y calibra la Red de 66 Macro-Neuronas (K66_macro) partiendo del modelo base:

    Distribución topológica en 11 etapas balanceadas de 6 macro-neuronas (66 Macro-Neuronas):
    - Etapa 0 (0..5):   Percepción Sintáctica Temprana (Base 0..5).
    - Etapa 1 (6..11):  [CLUSTER ÁLGEBRA SIMBÓLICA] 6 macro-neuronas clonadas de 6..11
                        con damping residual 0.35.
    - Etapa 2 (12..17): [CLUSTER TEORÍA DE NÚMEROS Y EUCLIDES] 6 macro-neuronas clonadas de 8..13
                        con damping residual 0.35.
    - Etapa 3 (18..23): Núcleo Semántico Inicial (Base 6..11).
    - Etapa 4 (24..29): [CLUSTER ARITMÉTICA MODULAR Y EXPONENCIACIÓN] 6 macro-neuronas clonadas de 10..15
                        con damping residual 0.35.
    - Etapa 5 (30..35): [CLUSTER COMBINATORIA Y RECURRENCIAS] 6 macro-neuronas clonadas de 12..17
                        con damping residual 0.35.
    - Etapa 6 (36..41): Núcleo Semántico Intermedio (Base 12..17).
    - Etapa 7 (42..47): [CLUSTER ESTRUCTURAS DE DATOS Y DICCIONARIOS] 6 macro-neuronas clonadas de 14..19
                        con damping residual 0.35 y editadas con dirección de algoritmia Python.
    - Etapa 8 (48..53): [CLUSTER BÚSQUEDA, RECURSIÓN Y GRAFOS] 6 macro-neuronas clonadas de 16..21
                        con damping residual 0.35 y editadas con dirección de algoritmia Python.
    - Etapa 9 (54..59): [CLUSTER ALGORITMIA PYTHON Y FILTRADO] 6 macro-neuronas clonadas de 18..23
                        con damping residual 0.35 y editadas con dirección de algoritmia Python.
    - Etapa 10 (60..65): Síntesis Superior y Decodificación de Salida (Base 18..23).
    """
    tok = AutoTokenizer.from_pretrained(model_hf_id)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_hf_id, dtype=dtype).eval()
    spec_vecs = build_layer_specialist_vectors(directions_path)
    deltas = build_math_code_layer_deltas(directions_path)
    k24_delta = deltas["k24_mc"]

    k24_by_layer = {b.layer: b for b in k24_delta.bias_edits}
    scale_ref = float(k24_delta.bias_edits[0].scale)

    new_layers: List[nn.Module] = []
    edits: List[BiasEdit] = []

    # Etapa 0: Macro-Neuronas 0..5 (Base 0..5)
    for orig_idx in range(6):
        l = base_model.model.layers[orig_idx]
        new_layers.append(l)
        new_idx = len(new_layers) - 1
        if orig_idx in k24_by_layer:
            b = k24_by_layer[orig_idx]
            edits.append(BiasEdit(layer=new_idx, vec=b.vec.clone(), scale=float(b.scale)))

    # Etapa 1: Macro-Neuronas 6..11 [Álgebra Simbólica: clonadas de 6..11]
    for j in range(6):
        src_idx = 6 + j
        cloned = copy.deepcopy(base_model.model.layers[src_idx])
        with torch.no_grad():
            cloned.mlp.down_proj.weight.mul_(damping)
            cloned.self_attn.o_proj.weight.mul_(damping)
        new_layers.append(cloned)
        new_idx = len(new_layers) - 1
        if math_bias_scale > 0.0:
            v_math = spec_vecs[src_idx]["math"]
            edits.append(BiasEdit(layer=new_idx, vec=v_math.clone(), scale=scale_ref * math_bias_scale))

    # Etapa 2: Macro-Neuronas 12..17 [Teoría de Números y Euclides: clonadas de 8..13]
    for j in range(6):
        src_idx = 8 + j
        cloned = copy.deepcopy(base_model.model.layers[src_idx])
        with torch.no_grad():
            cloned.mlp.down_proj.weight.mul_(damping)
            cloned.self_attn.o_proj.weight.mul_(damping)
        new_layers.append(cloned)
        new_idx = len(new_layers) - 1
        if math_bias_scale > 0.0:
            v_math = spec_vecs[src_idx]["math"]
            edits.append(BiasEdit(layer=new_idx, vec=v_math.clone(), scale=scale_ref * math_bias_scale))

    # Etapa 3: Macro-Neuronas 18..23 (Base 6..11)
    for orig_idx in range(6, 12):
        l = base_model.model.layers[orig_idx]
        new_layers.append(l)
        new_idx = len(new_layers) - 1
        if orig_idx in k24_by_layer:
            b = k24_by_layer[orig_idx]
            edits.append(BiasEdit(layer=new_idx, vec=b.vec.clone(), scale=float(b.scale)))

    # Etapa 4: Macro-Neuronas 24..29 [Aritmética Modular y Exp: clonadas de 10..15]
    for j in range(6):
        src_idx = 10 + j
        cloned = copy.deepcopy(base_model.model.layers[src_idx])
        with torch.no_grad():
            cloned.mlp.down_proj.weight.mul_(damping)
            cloned.self_attn.o_proj.weight.mul_(damping)
        new_layers.append(cloned)
        new_idx = len(new_layers) - 1
        if math_bias_scale > 0.0:
            v_math = spec_vecs[src_idx]["math"]
            edits.append(BiasEdit(layer=new_idx, vec=v_math.clone(), scale=scale_ref * math_bias_scale))

    # Etapa 5: Macro-Neuronas 30..35 [Combinatoria y Recurrencias: clonadas de 12..17]
    for j in range(6):
        src_idx = 12 + j
        cloned = copy.deepcopy(base_model.model.layers[src_idx])
        with torch.no_grad():
            cloned.mlp.down_proj.weight.mul_(damping)
            cloned.self_attn.o_proj.weight.mul_(damping)
        new_layers.append(cloned)
        new_idx = len(new_layers) - 1
        if math_bias_scale > 0.0:
            v_math = spec_vecs[src_idx]["math"]
            edits.append(BiasEdit(layer=new_idx, vec=v_math.clone(), scale=scale_ref * math_bias_scale))

    # Etapa 6: Macro-Neuronas 36..41 (Base 12..17)
    for orig_idx in range(12, 18):
        l = base_model.model.layers[orig_idx]
        new_layers.append(l)
        new_idx = len(new_layers) - 1
        if orig_idx in k24_by_layer:
            b = k24_by_layer[orig_idx]
            edits.append(BiasEdit(layer=new_idx, vec=b.vec.clone(), scale=float(b.scale)))

    # Etapa 7: Macro-Neuronas 42..47 [Estructuras y Diccionarios: clonadas de 14..19]
    for j in range(6):
        src_idx = 14 + j
        cloned = copy.deepcopy(base_model.model.layers[src_idx])
        with torch.no_grad():
            cloned.mlp.down_proj.weight.mul_(damping)
            cloned.self_attn.o_proj.weight.mul_(damping)
        new_layers.append(cloned)
        new_idx = len(new_layers) - 1
        v_code = spec_vecs[src_idx]["code"]
        edits.append(BiasEdit(layer=new_idx, vec=v_code.clone(), scale=scale_ref * 0.40))

    # Etapa 8: Macro-Neuronas 48..53 [Búsqueda y Recursión: clonadas de 16..21]
    for j in range(6):
        src_idx = 16 + j
        cloned = copy.deepcopy(base_model.model.layers[src_idx])
        with torch.no_grad():
            cloned.mlp.down_proj.weight.mul_(damping)
            cloned.self_attn.o_proj.weight.mul_(damping)
        new_layers.append(cloned)
        new_idx = len(new_layers) - 1
        v_code = spec_vecs[src_idx]["code"]
        edits.append(BiasEdit(layer=new_idx, vec=v_code.clone(), scale=scale_ref * 0.40))

    # Etapa 9: Macro-Neuronas 54..59 [Algoritmia Python y Filtrado: clonadas de 18..23]
    for j in range(6):
        src_idx = 18 + j
        cloned = copy.deepcopy(base_model.model.layers[src_idx])
        with torch.no_grad():
            cloned.mlp.down_proj.weight.mul_(damping)
            cloned.self_attn.o_proj.weight.mul_(damping)
        new_layers.append(cloned)
        new_idx = len(new_layers) - 1
        v_code = spec_vecs[src_idx]["code"]
        edits.append(BiasEdit(layer=new_idx, vec=v_code.clone(), scale=scale_ref * 0.40))

    # Etapa 10: Macro-Neuronas 60..65 (Base 18..23)
    for orig_idx in range(18, 24):
        l = base_model.model.layers[orig_idx]
        new_layers.append(l)
        new_idx = len(new_layers) - 1
        if orig_idx in k24_by_layer:
            b = k24_by_layer[orig_idx]
            edits.append(BiasEdit(layer=new_idx, vec=b.vec.clone(), scale=float(b.scale)))

    # Re-indexación estricta de layer_idx
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
        name="k66_macro",
        bias_edits=edits,
        meta={
            "n_macro": 66,
            "damping": damping,
            "stages": 11,
            "stage_size": 6,
            "algebra_cluster": list(range(6, 12)),
            "euclid_cluster": list(range(12, 18)),
            "arithmetic_cluster": list(range(24, 30)),
            "combinatorics_cluster": list(range(30, 36)),
            "dicts_cluster": list(range(42, 48)),
            "recursion_cluster": list(range(48, 54)),
            "algorithms_cluster": list(range(54, 60)),
        },
    )

    return base_model, tok, macro_delta


def build_k66_macro_network_1_5b(
    model_hf_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    damping: float = 0.25,
    bias_scale: float = 0.25,
    math_bias_scale: float = 0.0,
    directions_path: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, EditDelta]:
    """Construye la Red de 66 Macro-Neuronas (K66_macro) escalada a partir del modelo base de 1.5B (28 capas):

    Distribución topológica en 10 etapas (66 Macro-Neuronas, d=1536):
    - Etapa 0 (0..6):   Percepción Sintáctica Temprana (Base 0..6, 7 capas).
    - Etapa 1 (7..12):  [CLUSTER ÁLGEBRA SIMBÓLICA] 6 macro-neuronas clonadas de 7..12 con damping 0.25.
    - Etapa 2 (13..18): [CLUSTER TEORÍA DE NÚMEROS Y EUCLIDES] 6 macro-neuronas clonadas de 9..14 con damping 0.25.
    - Etapa 3 (19..25): Núcleo Semántico Inicial (Base 7..13, 7 capas).
    - Etapa 4 (26..31): [CLUSTER ARITMÉTICA MODULAR Y EXP] 6 macro-neuronas clonadas de 11..16 con damping 0.25.
    - Etapa 5 (32..37): [CLUSTER COMBINATORIA Y RECURRENCIAS] 6 macro-neuronas clonadas de 13..18 con damping 0.25.
    - Etapa 6 (38..44): Núcleo Semántico Intermedio (Base 14..20, 7 capas).
    - Etapa 7 (45..51): [CLUSTER ESTRUCTURAS Y DICCIONARIOS] 7 macro-neuronas clonadas de 15..21 con damping 0.25 + d_code.
    - Etapa 8 (52..58): [CLUSTER ALGORITMIA, RECURSIÓN Y BÚSQUEDA] 7 macro-neuronas clonadas de 17..23 con damping 0.25 + d_code.
    - Etapa 9 (59..65): Síntesis Superior y Salida Final (Base 21..27, 7 capas).
    """
    tok = AutoTokenizer.from_pretrained(model_hf_id)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_hf_id, dtype=dtype).eval()

    dirs_file = directions_path or os.path.join(
        os.path.dirname(__file__), "..", "artifacts", "macro-neurons-v4", "directions_qwen1.5b.pt"
    )
    if os.path.isfile(dirs_file):
        d_loaded = torch.load(dirs_file, map_location="cpu", weights_only=False)
        math_dirs = d_loaded["math_fresh"]
        code_dirs = d_loaded["code_fresh"]
    else:
        hidden_sz = base_model.config.hidden_size
        math_dirs = torch.zeros(29, hidden_sz)
        code_dirs = torch.zeros(29, hidden_sz)

    new_layers: List[nn.Module] = []
    edits: List[BiasEdit] = []

    # Etapa 0: 0..6 (Base 0..6, 7 capas)
    for orig_idx in range(7):
        new_layers.append(base_model.model.layers[orig_idx])

    # Etapa 1: 7..12 (Álgebra: clonadas de 7..12, 6 capas)
    for j in range(6):
        src_idx = 7 + j
        cloned = copy.deepcopy(base_model.model.layers[src_idx])
        with torch.no_grad():
            cloned.mlp.down_proj.weight.mul_(damping)
            cloned.self_attn.o_proj.weight.mul_(damping)
        new_layers.append(cloned)
        new_idx = len(new_layers) - 1
        if math_bias_scale > 0.0:
            v_math = math_dirs[src_idx + 1]
            edits.append(BiasEdit(layer=new_idx, vec=v_math.clone(), scale=math_bias_scale))

    # Etapa 2: 13..18 (Euclides / Números: clonadas de 9..14, 6 capas)
    for j in range(6):
        src_idx = 9 + j
        cloned = copy.deepcopy(base_model.model.layers[src_idx])
        with torch.no_grad():
            cloned.mlp.down_proj.weight.mul_(damping)
            cloned.self_attn.o_proj.weight.mul_(damping)
        new_layers.append(cloned)
        new_idx = len(new_layers) - 1
        if math_bias_scale > 0.0:
            v_math = math_dirs[src_idx + 1]
            edits.append(BiasEdit(layer=new_idx, vec=v_math.clone(), scale=math_bias_scale))

    # Etapa 3: 19..25 (Base 7..13, 7 capas)
    for orig_idx in range(7, 14):
        new_layers.append(base_model.model.layers[orig_idx])

    # Etapa 4: 26..31 (Modular Exp: clonadas de 11..16, 6 capas)
    for j in range(6):
        src_idx = 11 + j
        cloned = copy.deepcopy(base_model.model.layers[src_idx])
        with torch.no_grad():
            cloned.mlp.down_proj.weight.mul_(damping)
            cloned.self_attn.o_proj.weight.mul_(damping)
        new_layers.append(cloned)
        new_idx = len(new_layers) - 1
        if math_bias_scale > 0.0:
            v_math = math_dirs[src_idx + 1]
            edits.append(BiasEdit(layer=new_idx, vec=v_math.clone(), scale=math_bias_scale))

    # Etapa 5: 32..37 (Combinatoria: clonadas de 13..18, 6 capas)
    for j in range(6):
        src_idx = 13 + j
        cloned = copy.deepcopy(base_model.model.layers[src_idx])
        with torch.no_grad():
            cloned.mlp.down_proj.weight.mul_(damping)
            cloned.self_attn.o_proj.weight.mul_(damping)
        new_layers.append(cloned)
        new_idx = len(new_layers) - 1
        if math_bias_scale > 0.0:
            v_math = math_dirs[src_idx + 1]
            edits.append(BiasEdit(layer=new_idx, vec=v_math.clone(), scale=math_bias_scale))

    # Etapa 6: 38..44 (Base 14..20, 7 capas)
    for orig_idx in range(14, 21):
        new_layers.append(base_model.model.layers[orig_idx])

    # Etapa 7: 45..51 (Estructuras y Dicts: clonadas de 15..21, 7 capas)
    for j in range(7):
        src_idx = 15 + j
        cloned = copy.deepcopy(base_model.model.layers[src_idx])
        with torch.no_grad():
            cloned.mlp.down_proj.weight.mul_(damping)
            cloned.self_attn.o_proj.weight.mul_(damping)
        new_layers.append(cloned)
        new_idx = len(new_layers) - 1
        v_code = code_dirs[src_idx + 1]
        edits.append(BiasEdit(layer=new_idx, vec=v_code.clone(), scale=bias_scale))

    # Etapa 8: 52..58 (Algoritmos y Recursión: clonadas de 17..23, 7 capas)
    for j in range(7):
        src_idx = 17 + j
        cloned = copy.deepcopy(base_model.model.layers[src_idx])
        with torch.no_grad():
            cloned.mlp.down_proj.weight.mul_(damping)
            cloned.self_attn.o_proj.weight.mul_(damping)
        new_layers.append(cloned)
        new_idx = len(new_layers) - 1
        v_code = code_dirs[src_idx + 1]
        edits.append(BiasEdit(layer=new_idx, vec=v_code.clone(), scale=bias_scale))

    # Etapa 9: 59..65 (Base 21..27, 7 capas)
    for orig_idx in range(21, 28):
        new_layers.append(base_model.model.layers[orig_idx])

    for i, l in enumerate(new_layers):
        l.self_attn.layer_idx = i

    base_model.model.layers = nn.ModuleList(new_layers)
    base_model.config.num_hidden_layers = len(new_layers)
    base_model.config.layer_types = ["full_attention"] * len(new_layers)

    for b in edits:
        lin = base_model.model.layers[b.layer].mlp.down_proj
        vec = (b.scale * b.vec).to(dtype=lin.weight.dtype, device=lin.weight.device)
        with torch.no_grad():
            if lin.bias is None:
                lin.bias = nn.Parameter(vec.clone())
            else:
                lin.bias.add_(vec)

    for p in base_model.parameters():
        p.requires_grad_(False)

    macro_delta = EditDelta(
        name="k66_macro_1_5b",
        bias_edits=edits,
        meta={
            "n_macro": 66,
            "base_model": model_hf_id,
            "base_size": "1.5b",
            "base_layers": 28,
            "added_layers": 38,
            "damping": damping,
            "bias_scale": bias_scale,
            "algebra_cluster": list(range(7, 13)),
            "euclid_cluster": list(range(13, 19)),
            "arithmetic_cluster": list(range(26, 32)),
            "combinatorics_cluster": list(range(32, 38)),
            "dicts_cluster": list(range(45, 52)),
            "recursion_cluster": list(range(52, 59)),
        },
    )

    return base_model, tok, macro_delta

