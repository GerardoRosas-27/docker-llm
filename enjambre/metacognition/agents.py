"""Agentes Multi-Agente Metacognitivos Bidireccionales (Sistema Principal S1 <-> Observador S2).

Implementa:
1. Constructores con compartición virtual de pesos (`DampedLinear`):
   - `build_weight_shared_k66_1_5b`: Red de 66 Macro-Neuronas (1.5B) + Campo CONSTELACIÓN v4 en ~3.0 GB RAM.
   - `build_weight_shared_k32_0_5b`: Red de 32 Macro-Neuronas Metacognitivas (0.5B) en ~1.0 GB RAM.
2. `RealTimeHookCollector`: Recolector de activaciones y residuos en tiempo real mediante hooks
   de PyTorch sin bloqueo ni fugas de memoria.
3. `MetacognitiveObserverAgent` (S2): Observa pesos y dinámica de capas de S1 (en tiempo real o
   por lotes de episodios) y genera reportes cuantitativos y descripciones con su Transformer.
4. `ReflexivePrimaryAgent` (S1): Resuelve tareas de dominio (Matemáticas, Código, Conocimiento General)
   con preservación de circuitos nativos, mientras observa recíprocamente los pesos del Observador S2
   (incluidos sus módulos activos de introspección/Koopman) y los describe con su Transformer.
5. `BidirectionalMetacognitiveSystem`: Orquestador dual que ejecuta episodios en modo tiempo real
   y en lotes de episodios (`episodic batches`).
"""
from __future__ import annotations

import copy
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from enjambre.edits import BiasEdit, EditDelta
from enjambre.generate import chat_text
from enjambre.macro_neurons.field_v4 import (
    ConstelacionV4Field,
    FIELD_PAIRS_32,
    FIELD_PAIRS_66,
)
from .modules import WeightIntrospectionModule, build_metacognitive_subvocab
from .spectrometer import (
    DampedLinear,
    LayerWeightSnapshot,
    LayerWeightSpectrometer,
    ROLE_LABELS,
)

ARTIFACTS_V4_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "artifacts", "macro-neurons-v4")
)

DEFAULT_K_PREFIX = 46
S2_TO_S1_PROMPT = "Lectura sensorial de capa del Sistema Principal:"
S1_TO_S2_PROMPT = "Lectura sensorial de capa del Observador:"
INSPECTOR_SYSTEM_PROMPT = (
    "Inspector neuronal metacognitivo. Decodifica los tokens sensoriales en exactamente "
    "1 línea con este formato exacto (sin viñetas ni texto adicional):\n"
    "Rol: <rol>; Estado: <estado>; Damping: <x.xx>; NormaF: <x.x>; RangoEspectral: <x.x>; Bias: <x.xx>; Salud: <salud>."
)


def build_sensory_prompt_parts(
    model: nn.Module,
    tok: Any,
    invariant_prompt: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construye los embeddings fijos pre/post alrededor de los tokens sensoriales dentro del turno user."""
    pre_str = f"<|im_start|>system\n{INSPECTOR_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{invariant_prompt}\n"
    post_str = "\nEmite el reporte:<|im_end|>\n<|im_start|>assistant\n"
    pre_ids = tok(pre_str, return_tensors="pt", add_special_tokens=False)["input_ids"]
    post_ids = tok(post_str, return_tensors="pt", add_special_tokens=False)["input_ids"]
    emb_layer = model.get_input_embeddings()
    return emb_layer(pre_ids).detach(), emb_layer(post_ids).detach()


def _shallow_wrap_damped_layer(
    src_layer: nn.Module,
    layer_idx: int,
    damping: float = 1.0,
    bias_vec: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Clona superficialmente un bloque Transformer compartiendo matrices W grandes y aislando sesgos."""
    layer = copy.copy(src_layer)
    layer._modules = copy.copy(src_layer._modules)
    layer._parameters = copy.copy(src_layer._parameters)
    layer._buffers = copy.copy(src_layer._buffers)

    attn = copy.copy(src_layer.self_attn)
    attn._modules = copy.copy(src_layer.self_attn._modules)
    attn._parameters = copy.copy(src_layer.self_attn._parameters)
    attn._buffers = copy.copy(src_layer.self_attn._buffers)
    attn.layer_idx = layer_idx
    if damping != 1.0:
        attn.o_proj = DampedLinear(src_layer.self_attn.o_proj, damping=damping)
    layer.self_attn = attn

    mlp = copy.copy(src_layer.mlp)
    mlp._modules = copy.copy(src_layer.mlp._modules)
    mlp._parameters = copy.copy(src_layer.mlp._parameters)
    mlp._buffers = copy.copy(src_layer.mlp._buffers)
    if damping != 1.0 or bias_vec is not None:
        base_bias = src_layer.mlp.down_proj.bias
        if bias_vec is not None:
            merged_bias = bias_vec.clone() if base_bias is None else (base_bias.detach().float() + bias_vec.float())
        else:
            merged_bias = None
        mlp.down_proj = DampedLinear(src_layer.mlp.down_proj, damping=damping, bias=merged_bias)
    layer.mlp = mlp
    return layer


def build_weight_shared_k66_1_5b(
    model_hf_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    damping: float = 0.25,
    bias_scale: float = 0.25,
    dtype: torch.dtype = torch.bfloat16,
    attach_field: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, EditDelta, Optional[ConstelacionV4Field]]:
    """Construye la Red K66 del Sistema Principal (1.5B, 66 capas) compartiendo tensores W en RAM."""
    tok = AutoTokenizer.from_pretrained(model_hf_id)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_hf_id, dtype=dtype).eval()
    base_layers = list(base_model.model.layers)
    n_base = len(base_layers)
    hidden_sz = base_model.config.hidden_size

    spec_helper = LayerWeightSpectrometer()
    dom_vecs = spec_helper._get_domain_vectors(hidden_sz)

    dirs_file = os.path.join(ARTIFACTS_V4_DIR, "directions_qwen1.5b.pt")
    if os.path.isfile(dirs_file) and hidden_sz == 1536:
        d_loaded = torch.load(dirs_file, map_location="cpu", weights_only=False)
        code_dirs = d_loaded.get("code_fresh", dom_vecs["code"].unsqueeze(0).repeat(n_base + 1, 1))
        math_dirs = d_loaded.get("math_fresh", dom_vecs["math"].unsqueeze(0).repeat(n_base + 1, 1))
    else:
        code_dirs = dom_vecs["code"].unsqueeze(0).repeat(n_base + 1, 1)
        math_dirs = dom_vecs["math"].unsqueeze(0).repeat(n_base + 1, 1)

    v_ai = dom_vecs["ai_analysis"]

    new_layers: List[nn.Module] = []
    edits: List[BiasEdit] = []
    layer_roles: Dict[int, str] = {}

    def add_stage(src_indices: List[int], damp: float, role: str, bias_tensor_bank: Optional[torch.Tensor] = None, scale: float = 0.0):
        for s_idx in src_indices:
            safe_s = min(s_idx, n_base - 1)
            new_idx = len(new_layers)
            b_vec = None
            if bias_tensor_bank is not None and scale != 0.0:
                dir_row = bias_tensor_bank[min(safe_s + 1, bias_tensor_bank.shape[0] - 1)]
                b_vec = scale * dir_row
                edits.append(BiasEdit(layer=new_idx, vec=dir_row.clone(), scale=scale))
            w_layer = _shallow_wrap_damped_layer(base_layers[safe_s], new_idx, damping=damp, bias_vec=b_vec)
            new_layers.append(w_layer)
            layer_roles[new_idx] = role

    # 10 Etapas alineadas exactamente con build_k66_macro_network_1_5b (66 Macro-Neuronas)
    add_stage(list(range(0, 7)), 1.0, "early_syntax")                        # 0..6
    add_stage(list(range(7, 13)), damping, "math_cluster", math_dirs, 0.12)   # 7..12 (Algebra)
    add_stage(list(range(9, 15)), damping, "math_cluster", math_dirs, 0.12)   # 13..18 (Euclides)
    add_stage(list(range(7, 14)), 1.0, "general_epistemic")                   # 19..25 (Núcleo Semántico 1)
    add_stage(list(range(11, 17)), damping, "math_cluster", math_dirs, 0.12)  # 26..31 (Modular Exp)
    add_stage(list(range(13, 19)), damping, "ai_causal_cluster", v_ai.unsqueeze(0).repeat(n_base + 1, 1), 0.18)  # 32..37
    add_stage(list(range(14, 21)), 1.0, "general_epistemic")                   # 38..44 (Núcleo Semántico 2)
    add_stage(list(range(15, 22)), damping, "code_cluster", code_dirs, bias_scale)  # 45..51 (Dicts & Estructuras)
    add_stage(list(range(17, 24)), damping, "code_cluster", code_dirs, bias_scale)  # 52..58 (Recursión & Algoritmia)
    add_stage(list(range(21, 28)), 1.0, "synthesis_head")                      # 59..65 (Síntesis & Salida)

    base_model.model.layers = nn.ModuleList(new_layers)
    base_model.config.num_hidden_layers = len(new_layers)
    base_model.config.layer_types = ["full_attention"] * len(new_layers)
    for p in base_model.parameters():
        p.requires_grad_(False)

    field = None
    if attach_field:
        field = ConstelacionV4Field(hidden=hidden_sz, rank=16, pairs=FIELD_PAIRS_66).to(dtype)
        ckpt_primary = os.path.join(ARTIFACTS_V4_DIR, "field_v4_k66_1_5b_best.pt")
        ckpt_general = os.path.join(ARTIFACTS_V4_DIR, "field_v4_k66_general_1_5b_best.pt")
        ckpt = ckpt_primary if os.path.isfile(ckpt_primary) else ckpt_general
        if hidden_sz == 1536 and os.path.isfile(ckpt):
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            field.load_state_dict(state.get("field_state", state))
        for p in field.parameters():
            p.requires_grad_(False)
        field.attach(base_model)

    delta = EditDelta(
        name="k66_shared_1_5b",
        bias_edits=edits,
        meta={
            "n_macro": len(new_layers),
            "base_model": model_hf_id,
            "hidden_size": hidden_sz,
            "layer_roles": layer_roles,
            "damping": damping,
            "bias_scale": bias_scale,
        },
    )
    return base_model, tok, delta, field


def build_weight_shared_k32_0_5b(
    model_hf_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    damping: float = 0.50,
    dtype: torch.dtype = torch.bfloat16,
    attach_field: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, EditDelta, Optional[ConstelacionV4Field]]:
    """Construye el Segundo Sistema Observador Metacognitivo (32 Macro-Neuronas sobre Qwen2.5-0.5B)."""
    tok = AutoTokenizer.from_pretrained(model_hf_id)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_hf_id, dtype=dtype).eval()
    base_layers = list(base_model.model.layers)
    n_base = len(base_layers)
    hidden_sz = base_model.config.hidden_size

    v_meta = LayerWeightSpectrometer()._get_domain_vectors(hidden_sz)["meta_observer"]

    new_layers: List[nn.Module] = []
    edits: List[BiasEdit] = []
    layer_roles: Dict[int, str] = {}

    def add_block(src_indices: List[int], damp: float, role: str, apply_meta_bias: bool = False):
        for s_idx in src_indices:
            safe_s = min(s_idx, n_base - 1)
            new_idx = len(new_layers)
            b_vec = None
            if apply_meta_bias:
                b_vec = 0.15 * v_meta
                edits.append(BiasEdit(layer=new_idx, vec=v_meta.clone(), scale=0.15))
            w_layer = _shallow_wrap_damped_layer(base_layers[safe_s], new_idx, damping=damp, bias_vec=b_vec)
            new_layers.append(w_layer)
            layer_roles[new_idx] = role

    add_block(list(range(0, 12)), 1.0, "early_syntax", False)              # 0..11 Base temprana
    add_block(list(range(10, 14)), damping, "meta_observer_core", True)    # 12..15 Cluster Introspección Espectral
    add_block(list(range(12, 18)), 1.0, "meta_observer_core", False)       # 16..21 Razonamiento Metacognitivo
    add_block(list(range(14, 18)), damping, "meta_observer_core", True)    # 22..25 Cluster Calibración Causal
    add_block(list(range(18, 24)), 1.0, "synthesis_head", False)           # 26..31 Síntesis Narrativa Final

    base_model.model.layers = nn.ModuleList(new_layers)
    base_model.config.num_hidden_layers = len(new_layers)
    base_model.config.layer_types = ["full_attention"] * len(new_layers)
    for p in base_model.parameters():
        p.requires_grad_(False)

    field = None
    if attach_field:
        field = ConstelacionV4Field(hidden=hidden_sz, rank=16, pairs=FIELD_PAIRS_32).to(dtype)
        for p in field.parameters():
            p.requires_grad_(False)
        field.attach(base_model)

    delta = EditDelta(
        name="k32_metacognitive_observer",
        bias_edits=edits,
        meta={
            "n_macro": len(new_layers),
            "base_model": model_hf_id,
            "hidden_size": hidden_sz,
            "layer_roles": layer_roles,
            "damping": damping,
        },
    )
    return base_model, tok, delta, field


class RealTimeHookCollector:
    """Captura activaciones por capa en tiempo real durante un forward pass de inferencia."""

    def __init__(self, system_id: str, max_events: int = 256):
        self.system_id = system_id
        self.stats: Dict[int, Dict[str, float]] = {}
        self.events: deque = deque(maxlen=max_events)
        self._handles: List[Any] = []
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []

    def add_callback(self, fn: Callable[[Dict[str, Any]], None]) -> None:
        self._callbacks.append(fn)

    def clear(self) -> None:
        self.stats.clear()

    def attach(self, model: nn.Module, layer_indices: Optional[List[int]] = None) -> None:
        self.remove()
        layers = model.model.layers
        target_indices = layer_indices if layer_indices is not None else list(range(len(layers)))

        def make_hook(idx: int):
            def hook(_mod, inp, out):
                with torch.no_grad():
                    x_in = inp[0] if isinstance(inp, tuple) and len(inp) > 0 else inp
                    h_out = out[0] if isinstance(out, tuple) else out
                    if not isinstance(x_in, torch.Tensor) or not isinstance(h_out, torch.Tensor):
                        return
                    in_last = x_in[:, -1, :].detach().float()
                    out_last = h_out[:, -1, :].detach().float()
                    act_norm = float(out_last.norm(dim=-1).mean().item())
                    delta_norm = float((out_last - in_last).norm(dim=-1).mean().item())
                    cos_io = float(F.cosine_similarity(in_last, out_last, dim=-1).mean().item())
                    rec = {
                        "system_id": self.system_id,
                        "layer_idx": idx,
                        "act_norm": round(act_norm, 4),
                        "delta_norm": round(delta_norm, 4),
                        "cos_io": round(cos_io, 4),
                        "ts": time.time(),
                    }
                    self.stats[idx] = {"act_norm": act_norm, "delta_norm": delta_norm, "cos_io": cos_io}
                    self.events.append(rec)
                    for cb in self._callbacks:
                        cb(rec)

            return hook

        for idx in target_indices:
            if 0 <= idx < len(layers):
                self._handles.append(layers[idx].register_forward_hook(make_hook(idx)))

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _parse_generated_description(text: str) -> Dict[str, Any]:
    """Extrae atributos estructurados de una descripción generada por el Transformer."""
    out: Dict[str, Any] = {"raw": text.strip()}
    m_role = re.search(r"Rol:\s*([a-zA-Z0-9_]+)", text)
    if m_role:
        out["role"] = m_role.group(1)
    m_state = re.search(r"Estado:\s*([a-zA-Z0-9_]+)", text)
    if m_state:
        out["state"] = m_state.group(1)
    m_damp = re.search(r"Damping:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if m_damp:
        out["damping"] = float(m_damp.group(1))
    m_norm = re.search(r"NormaF:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if m_norm:
        out["frob_down"] = float(m_norm.group(1))
    m_sr = re.search(r"RangoEspectral:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if m_sr:
        out["stable_rank"] = float(m_sr.group(1))
    m_salud = re.search(r"Salud:\s*([a-zA-Z0-9_]+)", text)
    if m_salud:
        out["health"] = m_salud.group(1)
    return out


@dataclass
class AgentObservationResult:
    observer_system: str
    target_system: str
    target_layer_idx: int
    mode: str
    ground_truth: Dict[str, Any]
    predicted_metrics: Dict[str, Any]
    generated_description: str
    parsed_description: Dict[str, Any]
    koopman_innovation_norm: float
    anomaly_score: float
    latency_ms: float


class BaseMetacognitiveAgent:
    """Base común para agentes con capacidad de espectroscopia, predicción latente y descripción LM."""

    def __init__(
        self,
        agent_id: str,
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        layer_roles: Dict[int, str],
        field: Optional[ConstelacionV4Field] = None,
        k_prefix: int = DEFAULT_K_PREFIX,
    ):
        self.agent_id = agent_id
        self.model = model
        self.tok = tok
        self.layer_roles = dict(layer_roles)
        self.field = field
        self.spectrometer = LayerWeightSpectrometer()
        self.hooks = RealTimeHookCollector(system_id=agent_id)

        bank_ids, anc, embed_std = build_metacognitive_subvocab(model, tok)
        self.introspector = WeightIntrospectionModule(
            hidden_lm=model.config.hidden_size,
            k_prefix=k_prefix,
            n_anchors=anc.shape[0],
            init_anchors=anc,
            embed_std=embed_std,
            bank_ids=bank_ids,
        ).float()
        self._latent_memory: Dict[str, torch.Tensor] = {}

    def extract_layer_snapshot(
        self,
        target_agent: "BaseMetacognitiveAgent",
        layer_idx: int,
        is_perturbed: bool = False,
        include_probe_weights: bool = False,
        role_override: Optional[str] = None,
    ) -> LayerWeightSnapshot:
        layers = target_agent.model.model.layers
        layer = layers[layer_idx]
        role = role_override or target_agent.layer_roles.get(layer_idx, "early_syntax")
        if is_perturbed:
            role = "perturbed_anomaly"
        rstats = target_agent.hooks.stats.get(layer_idx)
        probe_mod = target_agent.introspector if include_probe_weights else None
        return self.spectrometer.inspect_decoder_layer(
            layer=layer,
            layer_idx=layer_idx,
            n_total_layers=len(layers),
            system_id=target_agent.agent_id,
            role_label=role,
            is_perturbed=is_perturbed,
            runtime_stats=rstats,
            probe_module=probe_mod,
        )

    def extract_bridge_snapshot(
        self,
        target_agent: "BaseMetacognitiveAgent",
        bridge_idx: int = 0,
        is_perturbed: bool = False,
    ) -> LayerWeightSnapshot:
        if target_agent.field is None or len(target_agent.field.bridges) == 0:
            raise ValueError(f"El agente objetivo {target_agent.agent_id} no posee puentes CONSTELACIÓN v4.")
        keys = sorted(target_agent.field.bridges.keys())
        bkey = keys[bridge_idx % len(keys)]
        bridge = target_agent.field.bridges[bkey]
        return self.spectrometer.inspect_bridge(
            bridge=bridge,
            bridge_idx=bridge_idx % len(keys),
            n_bridges=len(keys),
            system_id=target_agent.agent_id,
            role_label="meta_bridge_koopman",
            is_perturbed=is_perturbed,
        )

    @staticmethod
    def format_predicted_description(fast: Dict[str, Any]) -> str:
        estado = (
            "editada_perturbada"
            if fast["is_perturbed_pred"]
            else ("editada" if fast["is_edited_pred"] else "base_nativa")
        )
        salud = (
            "deriva_espectral"
            if fast["health_pred"] < 0.65
            else ("alerta_moderada" if fast["health_pred"] < 0.85 else "estable")
        )
        return (
            f"Rol: {fast['role_pred']}; Estado: {estado}; "
            f"Damping: {fast['damping_pred']:.2f}; NormaF: {fast['frob_down_pred']:.1f}; "
            f"RangoEspectral: {fast['stable_rank_pred']:.1f}; Bias: {fast['bias_norm_pred']:.2f}; "
            f"Salud: {salud}."
        )

    @torch.no_grad()
    def observe_snapshot_fast(
        self,
        snapshot: LayerWeightSnapshot,
        update_memory: bool = True,
    ) -> Dict[str, Any]:
        """Inferencia metacognitiva neuronal rápida (<2 ms) sobre los pesos de la capa observada."""
        self.introspector.eval()
        key = f"{snapshot.system_id}:{snapshot.module_type}:{snapshot.layer_idx}"
        prev_z = self._latent_memory.get(key)
        pred = self.introspector.encode_and_predict(snapshot.spectral_vector, prev_latent=prev_z)
        if update_memory:
            self._latent_memory[key] = pred["z_tracked"][0].detach().clone()

        role_id = int(torch.argmax(pred["role_logits"][0]).item())
        role_probs = F.softmax(pred["role_logits"][0], dim=-1)
        return {
            "role_pred": ROLE_LABELS[role_id],
            "role_confidence": round(float(role_probs[role_id].item()), 4),
            "damping_pred": round(float(pred["damping"][0].item()), 4),
            "frob_down_pred": round(float(torch.expm1(pred["log_frob_down"][0]).item()), 3),
            "stable_rank_pred": round(float(torch.expm1(pred["log_stable_rank"][0]).item()), 2),
            "bias_norm_pred": round(float(pred["bias_norm"][0].item()), 4),
            "health_pred": round(float(pred["health"][0].item()), 4),
            "is_edited_pred": bool(torch.sigmoid(pred["edited_logit"][0]).item() >= 0.5),
            "is_perturbed_pred": bool(torch.sigmoid(pred["perturbed_logit"][0]).item() >= 0.5),
            "innov_norm": round(float(pred["innov_norm"][0].item()), 4),
            "anomaly_score": round(float(pred["anomaly_prob"][0].item()), 4),
            "prefix_embeds": pred["prefix_embeds"],
            "contrastive_emb": pred["contrastive"][0],
        }

    @torch.no_grad()
    def generate_weight_description(
        self,
        snapshot: LayerWeightSnapshot,
        invariant_prompt: str,
        max_new_tokens: int = 68,
    ) -> Tuple[str, Dict[str, Any]]:
        """Genera con el Transformer completo una descripción de pesos condicionada únicamente en soft-tokens."""
        t0 = time.perf_counter()
        self.model.eval()
        self.introspector.eval()

        fast = self.observe_snapshot_fast(snapshot, update_memory=True)
        prefix = fast["prefix_embeds"].to(dtype=self.model.dtype)
        pre_emb, post_emb = build_sensory_prompt_parts(self.model, self.tok, invariant_prompt)
        full_emb = torch.cat([pre_emb, prefix, post_emb], dim=1)
        attn = torch.ones(1, full_emb.shape[1], dtype=torch.long)

        out_ids = self.model.generate(
            inputs_embeds=full_emb,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            pad_token_id=self.tok.eos_token_id,
        )
        gen_text = self.tok.decode(out_ids[0], skip_special_tokens=True).strip()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        fast["lm_latency_ms"] = round(elapsed_ms, 2)
        return gen_text, fast

    def inspect_target(
        self,
        target_agent: "BaseMetacognitiveAgent",
        layer_idx: int,
        invariant_prompt: str,
        generate_text: bool = True,
        mode: str = "realtime",
        is_perturbed: bool = False,
        include_probe_weights: bool = False,
        target_bridge: bool = False,
        max_new_tokens: int = 68,
    ) -> AgentObservationResult:
        t0 = time.perf_counter()
        if target_bridge:
            snap = self.extract_bridge_snapshot(
                target_agent=target_agent,
                bridge_idx=layer_idx,
                is_perturbed=is_perturbed,
            )
        else:
            snap = self.extract_layer_snapshot(
                target_agent=target_agent,
                layer_idx=layer_idx,
                is_perturbed=is_perturbed,
                include_probe_weights=include_probe_weights,
            )
        if generate_text:
            gen_text, fast = self.generate_weight_description(
                snap, invariant_prompt=invariant_prompt, max_new_tokens=max_new_tokens
            )
        else:
            fast = self.observe_snapshot_fast(snap)
            gen_text = self.format_predicted_description(fast)
        total_ms = (time.perf_counter() - t0) * 1000.0

        pred_clean = {k: v for k, v in fast.items() if not isinstance(v, torch.Tensor)}
        return AgentObservationResult(
            observer_system=self.agent_id,
            target_system=target_agent.agent_id,
            target_layer_idx=layer_idx,
            mode=mode,
            ground_truth=snap.to_report_dict(),
            predicted_metrics=pred_clean,
            generated_description=gen_text,
            parsed_description=_parse_generated_description(gen_text),
            koopman_innovation_norm=float(fast["innov_norm"]),
            anomaly_score=float(fast["anomaly_score"]),
            latency_ms=round(total_ms, 2),
        )


class ReflexivePrimaryAgent(BaseMetacognitiveAgent):
    """Agente 1: Sistema Principal (K66 Macro-Neuronas + Campo v4) con capacidad recíproca."""

    @torch.no_grad()
    def run_task_inference(self, prompt: str, max_new_tokens: int = 24) -> Dict[str, Any]:
        """Ejecuta una tarea real en el Sistema Principal disparando hooks en tiempo real."""
        self.hooks.clear()
        t0 = time.perf_counter()
        formatted = chat_text(self.tok, prompt, system="Be concise and mathematically or algorithmically exact.")
        enc = self.tok(formatted, return_tensors="pt")
        out_ids = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tok.eos_token_id,
        )
        plen = enc["input_ids"].shape[1]
        reply = self.tok.decode(out_ids[0][plen:], skip_special_tokens=True).strip()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "prompt": prompt,
            "reply": reply,
            "latency_ms": round(elapsed_ms, 2),
            "tokens_generated": int(out_ids.shape[1] - plen),
            "captured_layers": len(self.hooks.stats),
        }


class MetacognitiveObserverAgent(BaseMetacognitiveAgent):
    """Agente 2: Segundo Sistema con Metacognición y Monitorización Temporal Koopman."""
    pass


class BidirectionalMetacognitiveSystem:
    """Arquitectura Multi-Agente Metacognitiva Bidireccional en Tiempo Real y por Lotes de Episodios."""

    def __init__(
        self,
        primary: ReflexivePrimaryAgent,
        observer: MetacognitiveObserverAgent,
    ):
        self.s1 = primary
        self.s2 = observer
        self.episode_history: List[Dict[str, Any]] = []

    @classmethod
    def from_pretrained(
        cls,
        primary_hf_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        observer_hf_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
        compact_mode: bool = False,
        checkpoint_path: Optional[str] = None,
    ) -> "BidirectionalMetacognitiveSystem":
        if compact_mode:
            primary_hf_id = "Qwen/Qwen2.5-0.5B-Instruct"
        m1, tok1, d1, f1 = build_weight_shared_k66_1_5b(model_hf_id=primary_hf_id)
        m2, tok2, d2, f2 = build_weight_shared_k32_0_5b(model_hf_id=observer_hf_id)

        s1 = ReflexivePrimaryAgent(
            agent_id="S1_principal",
            model=m1,
            tok=tok1,
            layer_roles=d1.meta["layer_roles"],
            field=f1,
        )
        s2 = MetacognitiveObserverAgent(
            agent_id="S2_observer",
            model=m2,
            tok=tok2,
            layer_roles=d2.meta["layer_roles"],
            field=f2,
        )
        sys_obj = cls(primary=s1, observer=s2)
        if checkpoint_path and os.path.isfile(checkpoint_path):
            sys_obj.load_checkpoint(checkpoint_path)
        return sys_obj

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        for attr, key, k_key, b_key in (
            ("s2", "s2_introspector", "s2_k_prefix", "s2_bank_ids"),
            ("s1", "s1_introspector", "s1_k_prefix", "s1_bank_ids"),
        ):
            if key not in ckpt:
                continue
            agent = getattr(self, attr)
            sd = ckpt[key]
            bank_sd = sd.get("soft_prompt.anchor_bank")
            gate_sd = sd.get("soft_prompt.anchor_gate.2.weight")
            if bank_sd is not None and gate_sd is not None:
                n_anc = int(bank_sd.shape[0])
                k_pref = int(ckpt.get(k_key, gate_sd.shape[0] // max(1, n_anc)))
                if (
                    agent.introspector.soft_prompt.n_anchors != n_anc
                    or agent.introspector.soft_prompt.k_prefix != k_pref
                ):
                    agent.introspector = WeightIntrospectionModule(
                        hidden_lm=agent.model.config.hidden_size,
                        k_prefix=k_pref,
                        n_anchors=n_anc,
                        init_anchors=bank_sd.clone(),
                        embed_std=agent.introspector.soft_prompt.embed_std,
                        bank_ids=ckpt.get(b_key, agent.introspector.bank_ids),
                    ).float()
            agent.introspector.load_state_dict(sd)

    def save_checkpoint(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "s2_introspector": self.s2.introspector.state_dict(),
            "s1_introspector": self.s1.introspector.state_dict(),
            "s2_k_prefix": self.s2.introspector.k_prefix,
            "s1_k_prefix": self.s1.introspector.k_prefix,
            "s2_bank_ids": self.s2.introspector.bank_ids,
            "s1_bank_ids": self.s1.introspector.bank_ids,
            "s1_agent_id": self.s1.agent_id,
            "s2_agent_id": self.s2.agent_id,
            **(extra or {}),
        }
        torch.save(payload, path)

    def run_realtime_episode(
        self,
        prompt: str,
        s1_layer_idx: int = 48,
        s2_layer_idx: int = 14,
        generate_descriptions: bool = True,
        execute_primary_task: bool = True,
        task_max_tokens: int = 16,
    ) -> Dict[str, Any]:
        """Ejecuta un episodio completo con observación bidireccional en tiempo real:
        1) Hooks de S1 capturan dinámicas de activación mientras procesa el prompt.
        2) S2 observa la capa `s1_layer_idx` de S1 (pesos + activaciones) y describe sus pesos.
        3) S1 observa recíprocamente la capa `s2_layer_idx` de S2 (pesos de Transformer + sonda S2)
           y describe los pesos del Observador que lo vigila.
        """
        t_ep = time.perf_counter()
        self.s1.hooks.attach(self.s1.model, layer_indices=[s1_layer_idx, max(0, s1_layer_idx - 1)])
        self.s2.hooks.attach(self.s2.model, layer_indices=[s2_layer_idx])

        try:
            task_out = (
                self.s1.run_task_inference(prompt, max_new_tokens=task_max_tokens)
                if execute_primary_task
                else {"prompt": prompt, "reply": "", "latency_ms": 0.0, "tokens_generated": 0, "captured_layers": 0}
            )

            # Dirección S2 -> S1
            obs_s2_to_s1 = self.s2.inspect_target(
                target_agent=self.s1,
                layer_idx=s1_layer_idx,
                invariant_prompt=S2_TO_S1_PROMPT,
                generate_text=generate_descriptions,
                mode="realtime",
                include_probe_weights=False,
            )

            # Dirección Recíproca S1 -> S2 (incluye pesos reales del módulo observador de S2)
            obs_s1_to_s2 = self.s1.inspect_target(
                target_agent=self.s2,
                layer_idx=s2_layer_idx,
                invariant_prompt=S1_TO_S2_PROMPT,
                generate_text=generate_descriptions,
                mode="realtime_reciprocal",
                include_probe_weights=True,
            )
        finally:
            self.s1.hooks.remove()
            self.s2.hooks.remove()

        ep = {
            "prompt": prompt,
            "total_episode_ms": round((time.perf_counter() - t_ep) * 1000.0, 2),
            "primary_execution": task_out,
            "s2_observes_s1": {
                "target_layer": obs_s2_to_s1.target_layer_idx,
                "ground_truth": obs_s2_to_s1.ground_truth,
                "predicted": obs_s2_to_s1.predicted_metrics,
                "generated_description": obs_s2_to_s1.generated_description,
                "parsed": obs_s2_to_s1.parsed_description,
                "koopman_innov": obs_s2_to_s1.koopman_innovation_norm,
                "latency_ms": obs_s2_to_s1.latency_ms,
            },
            "s1_observes_s2": {
                "target_layer": obs_s1_to_s2.target_layer_idx,
                "ground_truth": obs_s1_to_s2.ground_truth,
                "predicted": obs_s1_to_s2.predicted_metrics,
                "generated_description": obs_s1_to_s2.generated_description,
                "parsed": obs_s1_to_s2.parsed_description,
                "koopman_innov": obs_s1_to_s2.koopman_innovation_norm,
                "latency_ms": obs_s1_to_s2.latency_ms,
            },
        }
        self.episode_history.append(ep)
        return ep

    def run_episodic_batch(
        self,
        episodes: List[Dict[str, Any]],
        generate_descriptions_for_last: bool = True,
        execute_primary_tasks: bool = False,
    ) -> Dict[str, Any]:
        """Ejecuta un lote de episodios (`episodic batch`) y consolida telemetría agregada."""
        t0 = time.perf_counter()
        batch_records: List[Dict[str, Any]] = []
        n = len(episodes)

        for idx, ep_spec in enumerate(episodes):
            do_lm = generate_descriptions_for_last and (idx == n - 1 or bool(ep_spec.get("generate_text", False)))
            rec = self.run_realtime_episode(
                prompt=ep_spec.get("prompt", "Diagnostic episode"),
                s1_layer_idx=int(ep_spec.get("s1_layer_idx", (idx * 7) % len(self.s1.model.model.layers))),
                s2_layer_idx=int(ep_spec.get("s2_layer_idx", (idx * 3) % len(self.s2.model.model.layers))),
                generate_descriptions=do_lm,
                execute_primary_task=execute_primary_tasks,
            )
            batch_records.append(rec)

        s2_role_ok = sum(
            1 for r in batch_records
            if r["s2_observes_s1"]["predicted"]["role_pred"] == r["s2_observes_s1"]["ground_truth"]["role_label"]
        )
        s1_role_ok = sum(
            1 for r in batch_records
            if r["s1_observes_s2"]["predicted"]["role_pred"] == r["s1_observes_s2"]["ground_truth"]["role_label"]
        )
        return {
            "batch_size": n,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "s2_to_s1_role_acc": round(s2_role_ok / max(1, n), 4),
            "s1_to_s2_role_acc": round(s1_role_ok / max(1, n), 4),
            "mean_s2_innov": round(sum(r["s2_observes_s1"]["koopman_innov"] for r in batch_records) / max(1, n), 4),
            "mean_s1_innov": round(sum(r["s1_observes_s2"]["koopman_innov"] for r in batch_records) / max(1, n), 4),
            "episodes": batch_records,
        }
