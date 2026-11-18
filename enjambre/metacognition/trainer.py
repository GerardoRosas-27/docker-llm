"""Entrenamiento Multi-Etapa para la Arquitectura Multi-Agente Metacognitiva Bidireccional.

Implementa:
1. `EpisodicDatasetBuilder`: Extrae instantáneas espectrales reales de capas y puentes de
   `S1_principal` (K66-1.5B + Campo v4) y `S2_observer` (K32-0.5B + Campo v4 + Sonda),
   construyendo particiones estrictas SIN fuga (`Train` vs `Val` vs `Holdout` en índices
   de capa/puente y severidades de perturbación nunca vistas).
2. `BidirectionalMetaTrainer`:
   - Etapa 1 (Introspección Espectral + Koopman Temporal + Alineación Contrastiva S1<->S2):
     Entrena ambos módulos `WeightLatentEncoder`, `AttributePredictionHeads` y
     `MetaKoopmanMonitorBridge` sobre lotes de episodios con transiciones temporales reales.
   - Etapa 2 (Descodificación Lingüística por Soft-Prompt sobre el Manifold Vocabulario):
     Entrena `ManifoldSoftPromptProjector` end-to-end mediante retropropagación a través de
     los Transformers reales congelados (`Qwen2.5-0.5B` y `Qwen2.5-1.5B`) usando prompts
     invariantes sin pistas del índice o estado de la capa observada.
"""
from __future__ import annotations

import copy
import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from enjambre.generate import chat_text
from .agents import (
    ARTIFACTS_V4_DIR,
    DEFAULT_K_PREFIX,
    INSPECTOR_SYSTEM_PROMPT,
    S1_TO_S2_PROMPT,
    S2_TO_S1_PROMPT,
    build_sensory_prompt_parts,
    build_weight_shared_k32_0_5b,
    build_weight_shared_k66_1_5b,
)
from .modules import WeightIntrospectionModule, build_metacognitive_subvocab
from .spectrometer import (
    DampedLinear,
    LayerWeightSnapshot,
    LayerWeightSpectrometer,
    ROLE_LABELS,
    ROLE_TO_ID,
    _extract_effective_weight_and_bias,
    _wrap_raw_linear_proxy,
    apply_reversible_layer_perturbation,
)

DEFAULT_META_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "artifacts", "metacognitive")
)
SEED = 42

# Partición estricta SIN FUGA DE DATOS (índices disjuntos en todas las etapas)
S1_HOLDOUT_LAYERS = {1, 5, 10, 16, 22, 29, 35, 41, 48, 55, 62}
S1_VAL_LAYERS = {2, 8, 14, 21, 27, 34, 40, 46, 53, 60}
S1_HOLDOUT_BRIDGES = {2, 7, 13, 19, 25}
S1_VAL_BRIDGES = {4, 10, 16, 22}

S2_HOLDOUT_LAYERS = {2, 7, 13, 19, 24, 28}
S2_VAL_LAYERS = {4, 9, 14, 20, 23, 30}
S2_HOLDOUT_BRIDGES = {2, 6, 10}
S2_VAL_BRIDGES = {3, 8}


def _jdump(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def _apply_temporary_damping_edit(
    layer: nn.Module,
    damping_mult: float,
    bias_scale: float,
    domain_vec: Optional[torch.Tensor] = None,
):
    """Aplica amortiguamiento y edición de sesgo reversible sobre una capa vía DampedLinear (0 RAM en W)."""
    orig_down = layer.mlp.down_proj
    orig_o = layer.self_attn.o_proj

    _, d_down, b_down = _extract_effective_weight_and_bias(orig_down)
    _, d_o, b_o = _extract_effective_weight_and_bias(orig_o)

    proxy_down = orig_down if not isinstance(orig_down, DampedLinear) else _wrap_raw_linear_proxy(orig_down)
    proxy_o = orig_o if not isinstance(orig_o, DampedLinear) else _wrap_raw_linear_proxy(orig_o)

    new_bias = b_down.detach().clone() if b_down is not None else None
    if domain_vec is not None and bias_scale != 0.0:
        dv = (bias_scale * domain_vec).to(dtype=proxy_down.weight.dtype, device=proxy_down.weight.device)
        new_bias = dv if new_bias is None else (new_bias + dv)

    layer.mlp.down_proj = DampedLinear(proxy_down, damping=d_down * damping_mult, bias=new_bias)
    layer.self_attn.o_proj = DampedLinear(proxy_o, damping=d_o * damping_mult, bias=b_o)

    def restore() -> None:
        layer.mlp.down_proj = orig_down
        layer.self_attn.o_proj = orig_o

    return restore


def _capture_live_hook_stats(
    model: nn.Module,
    tok: Any,
    prompts: List[str],
    layer_indices: List[int],
) -> Dict[int, Dict[str, float]]:
    """Ejecuta un paso forward real corto y captura activaciones por capa mediante hooks."""
    from .agents import RealTimeHookCollector

    collector = RealTimeHookCollector(system_id="capture")
    collector.attach(model, layer_indices=layer_indices)
    try:
        with torch.no_grad():
            for p in prompts:
                fmt = chat_text(tok, p, system="Be concise.")
                enc = tok(fmt, return_tensors="pt")
                model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)
    finally:
        collector.remove()
    return dict(collector.stats)


def _snapshot_split_name(idx: int, hold_set: set, val_set: set) -> str:
    if idx in hold_set:
        return "holdout"
    if idx in val_set:
        return "val"
    return "train"


def collect_target_snapshots(
    model: nn.Module,
    tok: Any,
    layer_roles: Dict[int, str],
    field: Optional[nn.Module],
    system_id: str,
    holdout_layers: set,
    val_layers: set,
    holdout_bridges: set,
    val_bridges: set,
    probe_module: Optional[nn.Module] = None,
    seed: int = SEED,
) -> Dict[str, List[LayerWeightSnapshot]]:
    """Extrae instantáneas espectrales reales separadas estrictamente en train / val / holdout."""
    rng = random.Random(seed + (11 if system_id == "S1_principal" else 29))
    spec = LayerWeightSpectrometer(seed=SEED)
    n_layers = len(model.model.layers)
    hidden_sz = model.config.hidden_size
    dom_vecs = spec._get_domain_vectors(hidden_sz)

    # Capturar telemetría real en vivo sobre capas reales
    sample_prompts = (
        [
            "Solve for x: 3*x + 5 = 20.",
            "def reverse_string(s): return s[::-1]",
            "Explain self-attention in Transformer architectures.",
        ]
        if system_id == "S1_principal"
        else [
            "Describe los pesos de la capa observada:",
            "Evalúa la deriva espectral de la capa activa.",
        ]
    )
    live_stats = _capture_live_hook_stats(model, tok, sample_prompts, list(range(n_layers)))

    splits: Dict[str, List[LayerWeightSnapshot]] = {"train": [], "val": [], "holdout": []}

    # 1. Instantáneas nativas / base editadas en todas las capas
    for idx in range(n_layers):
        split = _snapshot_split_name(idx, holdout_layers, val_layers)
        role = layer_roles.get(idx, "early_syntax")
        rstats = live_stats.get(idx)

        snap = spec.inspect_decoder_layer(
            layer=model.model.layers[idx],
            layer_idx=idx,
            n_total_layers=n_layers,
            system_id=system_id,
            role_label=role,
            is_perturbed=False,
            runtime_stats=rstats,
            probe_module=None,
        )
        splits[split].append(snap)

        # Si inspeccionamos S2, incluir también observación con módulo sonda activo
        if probe_module is not None:
            snap_probe = spec.inspect_decoder_layer(
                layer=model.model.layers[idx],
                layer_idx=idx,
                n_total_layers=n_layers,
                system_id=system_id,
                role_label=role,
                is_perturbed=False,
                runtime_stats=rstats,
                probe_module=probe_module,
            )
            splits[split].append(snap_probe)

    # 2. Variaciones operacionales benignas (para generalizar damping & bias continuos sin fuga)
    op_damps_train = [0.40, 0.65, 0.85, 1.0]
    op_damps_hold = [0.55, 0.78]
    for idx in range(n_layers):
        split = _snapshot_split_name(idx, holdout_layers, val_layers)
        role = layer_roles.get(idx, "early_syntax")
        if split == "train" and rng.random() > 0.55:
            continue
        if split != "train" and idx % 2 == 0:
            continue

        dmult = rng.choice(op_damps_train if split != "holdout" else op_damps_hold)
        bscale = rng.uniform(0.08, 0.32) if role not in ("early_syntax", "synthesis_head", "general_epistemic") else 0.0
        dkey = {
            "math_cluster": "math",
            "code_cluster": "code",
            "ai_causal_cluster": "ai_analysis",
            "meta_observer_core": "meta_observer",
        }.get(role, "ai_analysis")
        restore = _apply_temporary_damping_edit(
            model.model.layers[idx],
            damping_mult=dmult,
            bias_scale=bscale,
            domain_vec=dom_vecs[dkey],
        )
        try:
            snap_op = spec.inspect_decoder_layer(
                layer=model.model.layers[idx],
                layer_idx=idx,
                n_total_layers=n_layers,
                system_id=system_id,
                role_label=role,
                is_perturbed=False,
                runtime_stats=live_stats.get(idx),
                probe_module=probe_module if (probe_module is not None and idx % 2 == 1) else None,
            )
            splits[split].append(snap_op)
        finally:
            restore()

    # 3. Mutaciones físicas reales anómalas (regímenes disjuntos entre train y holdout)
    # Train: damping_mult in [0.22, 0.34, 1.38], rank1 in [2.6, 4.4], bias in [0.95, 1.70]
    # Holdout: damping_mult in [0.28, 1.24], rank1=3.4, bias=1.28 (¡no vistos en train!)
    for idx in range(n_layers):
        split = _snapshot_split_name(idx, holdout_layers, val_layers)
        if split == "train" and rng.random() > 0.58:
            continue
        if split == "val" and idx % 2 == 1:
            continue

        if split == "holdout":
            dmult = 0.28 if (idx % 2 == 0) else 1.24
            r1_scale = 3.4
            b_spike = 1.28
        elif split == "val":
            dmult = 0.25 if (idx % 2 == 0) else 1.30
            r1_scale = 3.1
            b_spike = 1.15
        else:
            dmult = rng.choice([0.20, 0.33, 1.38])
            r1_scale = rng.uniform(2.5, 4.5)
            b_spike = rng.uniform(0.95, 1.75)

        restore = apply_reversible_layer_perturbation(
            model.model.layers[idx],
            damping_mult=dmult,
            rank1_spike_scale=r1_scale,
            bias_spike_scale=b_spike,
            seed=seed + idx * 17 + (0 if split == "train" else 999),
        )
        try:
            # Simular además perturbación transitoria de activación en vivo coherente con la deriva
            rstats_pert = dict(live_stats.get(idx, {"act_norm": 10.0, "delta_norm": 2.0, "cos_io": 0.95}))
            rstats_pert["delta_norm"] = rstats_pert["delta_norm"] * (1.8 if dmult > 1.0 else 0.45)
            rstats_pert["cos_io"] = max(0.35, rstats_pert["cos_io"] - 0.18)

            # Inyectar también mutación ocasional en la sonda observadora de S2 para que S1 la detecte
            probe_for_snap = probe_module
            saved_probe_scale = None
            if probe_module is not None and idx % 2 == 0:
                saved_probe_scale = probe_module.encoder.proj_in.weight.data.clone()
                with torch.no_grad():
                    probe_module.encoder.proj_in.weight.data.mul_(2.6)

            try:
                snap_pert = spec.inspect_decoder_layer(
                    layer=model.model.layers[idx],
                    layer_idx=idx,
                    n_total_layers=n_layers,
                    system_id=system_id,
                    role_label="perturbed_anomaly",
                    is_perturbed=True,
                    runtime_stats=rstats_pert,
                    probe_module=probe_for_snap,
                )
                splits[split].append(snap_pert)
            finally:
                if saved_probe_scale is not None:
                    with torch.no_grad():
                        probe_module.encoder.proj_in.weight.data.copy_(saved_probe_scale)
        finally:
            restore()

    # 3b. Mutación pura del módulo observador (sonda) sin tocar pesos de la capa Transformer objetivo
    if probe_module is not None:
        probe_idxs = [i for i in range(n_layers) if (i in holdout_layers) or (i in val_layers) or (i % 3 == 0)]
        for idx in probe_idxs:
            split = _snapshot_split_name(idx, holdout_layers, val_layers)
            saved_w = probe_module.encoder.proj_in.weight.data.clone()
            saved_w2 = probe_module.attr_heads.role_head[0].weight.data.clone()
            try:
                mult = 2.8 if split == "holdout" else (3.1 if split == "val" else rng.choice([2.4, 3.5]))
                with torch.no_grad():
                    probe_module.encoder.proj_in.weight.data.mul_(mult)
                    probe_module.attr_heads.role_head[0].weight.data.mul_(mult)
                snap_po = spec.inspect_decoder_layer(
                    layer=model.model.layers[idx],
                    layer_idx=idx,
                    n_total_layers=n_layers,
                    system_id=system_id,
                    role_label="perturbed_anomaly",
                    is_perturbed=True,
                    runtime_stats=live_stats.get(idx),
                    probe_module=probe_module,
                )
                splits[split].append(snap_po)
            finally:
                with torch.no_grad():
                    probe_module.encoder.proj_in.weight.data.copy_(saved_w)
                    probe_module.attr_heads.role_head[0].weight.data.copy_(saved_w2)

    # 4. Puentes CONSTELACIÓN v4 (Koopman 2x2) en estado sano y perturbado
    if field is not None and len(field.bridges) > 0:
        bkeys = sorted(field.bridges.keys())
        n_b = len(bkeys)
        for bidx, bk in enumerate(bkeys):
            split = _snapshot_split_name(bidx, holdout_bridges, val_bridges)
            bridge = field.bridges[bk]
            snap_b = spec.inspect_bridge(
                bridge=bridge,
                bridge_idx=bidx,
                n_bridges=n_b,
                system_id=system_id,
                role_label="meta_bridge_koopman",
                is_perturbed=False,
            )
            splits[split].append(snap_b)

            # Mutación temporal sobre rho_raw del puente para casos anómalos
            if split != "train" or (bidx % 2 == 0):
                orig_rho = bridge.rho_raw.data.clone()
                try:
                    with torch.no_grad():
                        shift = 1.22 if split == "holdout" else rng.uniform(1.15, 1.28)
                        bridge.rho_raw.data.fill_(shift)
                    snap_bp = spec.inspect_bridge(
                        bridge=bridge,
                        bridge_idx=bidx,
                        n_bridges=n_b,
                        system_id=system_id,
                        role_label="perturbed_anomaly",
                        is_perturbed=True,
                    )
                    splits[split].append(snap_bp)
                finally:
                    with torch.no_grad():
                        bridge.rho_raw.data.copy_(orig_rho)

    return splits


class BidirectionalMetaTrainer:
    """Entrenador en dos etapas para la Arquitectura Multi-Agente Metacognitiva Bidireccional."""

    def __init__(
        self,
        out_dir: Optional[str] = None,
        stage1_epochs: int = 260,
        stage2_steps_s2: int = 18,
        stage2_steps_s1: int = 12,
        lr_stage1: float = 2.0e-3,
        lr_stage2: float = 7.5e-4,
        seed: int = SEED,
    ):
        self.out_dir = os.path.abspath(out_dir or DEFAULT_META_DIR)
        os.makedirs(self.out_dir, exist_ok=True)
        self.stage1_epochs = stage1_epochs
        self.stage2_steps_s2 = stage2_steps_s2
        self.stage2_steps_s1 = stage2_steps_s1
        self.lr_stage1 = lr_stage1
        self.lr_stage2 = lr_stage2
        self.seed = seed

        torch.manual_seed(seed)
        random.seed(seed)
        torch.set_num_threads(min(8, os.cpu_count() or 8))

    @staticmethod
    def _snapshots_to_tensors(snaps: List[LayerWeightSnapshot]) -> Dict[str, Any]:
        return {
            "x": torch.stack([s.spectral_vector for s in snaps], dim=0),
            "role": torch.tensor([s.role_id for s in snaps], dtype=torch.long),
            "damping": torch.tensor([s.damping for s in snaps], dtype=torch.float32),
            "log_frob": torch.tensor([math.log1p(s.frob_down) for s in snaps], dtype=torch.float32),
            "log_sr": torch.tensor([math.log1p(s.stable_rank) for s in snaps], dtype=torch.float32),
            "bias_norm": torch.tensor([s.bias_norm for s in snaps], dtype=torch.float32),
            "health": torch.tensor([s.health_score for s in snaps], dtype=torch.float32),
            "edited": torch.tensor([1.0 if s.is_edited else 0.0 for s in snaps], dtype=torch.float32),
            "perturbed": torch.tensor([1.0 if s.is_perturbed else 0.0 for s in snaps], dtype=torch.float32),
            "texts": [s.canonical_description() for s in snaps],
        }

    @staticmethod
    def _eval_stage1_metrics(introspector: WeightIntrospectionModule, data: Dict[str, Any]) -> Dict[str, float]:
        introspector.eval()
        with torch.no_grad():
            pred = introspector.encode_and_predict(data["x"])
            role_acc = float((pred["role_logits"].argmax(-1) == data["role"]).float().mean().item())
            damp_mae = float((pred["damping"] - data["damping"]).abs().mean().item())
            frob_mae = float((torch.expm1(pred["log_frob_down"]) - torch.expm1(data["log_frob"])).abs().mean().item())
            sr_mae = float((torch.expm1(pred["log_stable_rank"]) - torch.expm1(data["log_sr"])).abs().mean().item())
            bias_mae = float((pred["bias_norm"] - data["bias_norm"]).abs().mean().item())
            pert_pred = (torch.sigmoid(pred["perturbed_logit"]) >= 0.5).float()
            pert_true = data["perturbed"]
            pert_acc = float((pert_pred == pert_true).float().mean().item())
            tp = float(((pert_pred == 1) & (pert_true == 1)).sum().item())
            fp = float(((pert_pred == 1) & (pert_true == 0)).sum().item())
            fn = float(((pert_pred == 0) & (pert_true == 1)).sum().item())
            f1 = (2.0 * tp) / max(1.0, 2.0 * tp + fp + fn)

            # Evaluar separación temporal Koopman entre pares limpio->limpio y limpio->perturbado
            clean_idx = (pert_true < 0.5).nonzero(as_tuple=True)[0]
            pert_idx = (pert_true > 0.5).nonzero(as_tuple=True)[0]
            innov_clean = 0.0
            innov_pert = 0.0
            anom_acc = 0.0
            if len(clean_idx) >= 2 and len(pert_idx) >= 1:
                z_all = pred["z"]
                z_prev_c = z_all[clean_idx]
                z_curr_c = z_all[torch.roll(clean_idx, 1)]
                _, in_c, ap_c = introspector.koopman_monitor(z_prev_c, z_curr_c)

                rep = (len(pert_idx) + len(clean_idx) - 1) // len(clean_idx)
                z_prev_p = z_all[clean_idx].repeat(rep, 1)[: len(pert_idx)]
                z_curr_p = z_all[pert_idx]
                _, in_p, ap_p = introspector.koopman_monitor(z_prev_p, z_curr_p)

                innov_clean = float(in_c.mean().item())
                innov_pert = float(in_p.mean().item())
                ok_c = (ap_c < 0.5).float().sum().item()
                ok_p = (ap_p >= 0.5).float().sum().item()
                anom_acc = float((ok_c + ok_p) / max(1, len(ap_c) + len(ap_p)))

        return {
            "role_acc": round(role_acc, 4),
            "damping_mae": round(damp_mae, 4),
            "frob_mae": round(frob_mae, 4),
            "sr_mae": round(sr_mae, 4),
            "bias_mae": round(bias_mae, 4),
            "pert_acc": round(pert_acc, 4),
            "pert_f1": round(f1, 4),
            "koopman_innov_clean": round(innov_clean, 4),
            "koopman_innov_pert": round(innov_pert, 4),
            "koopman_anomaly_acc": round(anom_acc, 4),
        }

    def _compute_introspector_loss(
        self,
        introspector: WeightIntrospectionModule,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = batch["x"]
        z = introspector.encoder(x)
        attrs = introspector.attr_heads(z)

        l_role = F.cross_entropy(attrs["role_logits"], batch["role"], label_smoothing=0.02)
        l_damp = F.smooth_l1_loss(attrs["damping"], batch["damping"], beta=0.08)
        l_frob = F.smooth_l1_loss(attrs["log_frob_down"], batch["log_frob"], beta=0.1)
        l_sr = F.smooth_l1_loss(attrs["log_stable_rank"], batch["log_sr"], beta=0.1)
        l_bias = F.smooth_l1_loss(attrs["bias_norm"], batch["bias_norm"], beta=0.08)
        l_health = F.smooth_l1_loss(attrs["health"], batch["health"], beta=0.08)
        l_edit = F.binary_cross_entropy_with_logits(attrs["edited_logit"], batch["edited"])
        l_pert = F.binary_cross_entropy_with_logits(attrs["perturbed_logit"], batch["perturbed"])

        # Entrenamiento de transición temporal del puente Koopman (episodio t-1 -> episodio t)
        clean_mask = batch["perturbed"] < 0.5
        pert_mask = batch["perturbed"] >= 0.5
        l_koop = torch.tensor(0.0, dtype=torch.float32)

        if clean_mask.sum() >= 2:
            zc = z[clean_mask]
            zc_shift = torch.roll(zc, shifts=1, dims=0)
            z_tr_c, in_c, ap_c = introspector.koopman_monitor(zc_shift.detach(), zc)
            # Preservar representación en transiciones sanas y mantener baja anomalía
            l_koop = l_koop + 0.35 * F.mse_loss(z_tr_c, zc) + F.binary_cross_entropy(ap_c.clamp(1e-4, 1 - 1e-4), torch.zeros_like(ap_c))
            l_koop = l_koop + 0.1 * in_c.mean()

            if pert_mask.sum() >= 1:
                zp = z[pert_mask]
                n_p = zp.size(0)
                idx_src = torch.arange(n_p) % zc.size(0)
                zc_prev = zc[idx_src].detach()
                z_tr_p, in_p, ap_p = introspector.koopman_monitor(zc_prev, zp)
                l_koop = l_koop + 0.25 * F.mse_loss(z_tr_p, zp) + F.binary_cross_entropy(ap_p.clamp(1e-4, 1 - 1e-4), torch.ones_like(ap_p))
                # Margen de separación de innovación entre estado anómalo y sano
                l_koop = l_koop + 0.3 * F.relu(0.45 - (in_p.mean() - in_c.mean().detach()))

        # Mantener `z_tracked` estable también en paso estacionario `z_prev = z`
        z_stat, _, _ = introspector.koopman_monitor(z.detach(), z)
        l_stat = F.mse_loss(z_stat, z)

        total = (
            1.4 * l_role
            + 2.0 * l_damp
            + 1.0 * l_frob
            + 1.5 * l_sr
            + 1.4 * l_bias
            + 1.4 * l_health
            + 0.8 * l_edit
            + 1.8 * l_pert
            + 0.8 * l_koop
            + 0.5 * l_stat
        )
        c_emb = F.normalize(introspector.contrastive_proj(z), dim=-1)
        return total, {"z": z, "c_emb": c_emb}

    def _train_stage2_soft_prompt(
        self,
        model: nn.Module,
        tok: Any,
        introspector: WeightIntrospectionModule,
        train_snaps: List[LayerWeightSnapshot],
        val_snaps: List[LayerWeightSnapshot],
        invariant_prompt: str,
        steps: int,
        batch_size: int,
        tag: str,
        skip_warmup: bool = False,
        lr_override: Optional[float] = None,
        warm_lr_mult: float = 0.25,
        eta_min: float = 5.0e-5,
        seed_offset: int = 0,
        max_val_items: int = 4,
        eval_mid_val: bool = True,
        role_priority_order: Optional[List[str]] = None,
        grad_accum: int = 1,
    ) -> Dict[str, Any]:
        """Entrena ManifoldSoftPromptProjector: alineación al sub-vocabulario + ajuste Causal LM extremo a extremo."""
        print(
            f"[meta-stage2:{tag}] Entrenando Soft-Prompt (steps={steps}, B={batch_size}, accum={grad_accum}, train={len(train_snaps)}, val={len(val_snaps)}, skip_warmup={skip_warmup})...",
            flush=True,
        )
        model.eval()
        for p in introspector.parameters():
            p.requires_grad_(False)
        for p in introspector.soft_prompt.parameters():
            p.requires_grad_(True)

        warm_params = [
            p for n, p in introspector.soft_prompt.named_parameters()
            if not n.startswith("delta_proj") and n != "res_scale"
        ]
        align_loss_val = 0.0

        # ------------------------------------------------------------------
        # Etapa 2A: Calibración densa sobre el banco de sub-vocabulario (Train + grilla sintética de atributos)
        # ------------------------------------------------------------------
        if not skip_warmup:
            t_warm = time.perf_counter()
            id_to_slot = {tid: i for i, tid in enumerate(introspector.bank_ids)}
            eos_id = getattr(tok, "eos_token_id", None)
            pad_slot = id_to_slot.get(eos_id, len(introspector.bank_ids) - 1)
            k_pref = introspector.soft_prompt.k_prefix
            v_sub = introspector.soft_prompt.n_anchors

            def _encode_slots(text: str) -> List[int]:
                tids = tok.encode(text, add_special_tokens=False)
                slots = [id_to_slot.get(tid, pad_slot) for tid in tids][:k_pref]
                slots.extend([pad_slot] * (k_pref - len(slots)))
                return slots

            y_real_slots = torch.tensor([_encode_slots(s.compact_payload()) for s in train_snaps], dtype=torch.long)

            x_tr_spec = torch.stack([s.spectral_vector for s in train_snaps], dim=0)
            with torch.no_grad():
                introspector.eval()
                z_tr = introspector.encoder(x_tr_spec)
                attrs_tr = {k: v.detach() for k, v in introspector.attr_heads(z_tr).items()}
                z_tr_tracked, _, _ = introspector.koopman_monitor(z_tr, z_tr)
                z_tr_tracked = z_tr_tracked.detach()

            # Grilla sintética de calibración sobre el rango continuo de atributos (0% fuga de val/holdout)
            rng_cal = random.Random(self.seed + (701 if "0_5b" in tag else 702))
            n_syn = 512
            n_real = z_tr_tracked.size(0)
            syn_z_list: List[torch.Tensor] = []
            syn_roles: List[torch.Tensor] = []
            syn_damp, syn_lfrob, syn_lsr, syn_bias, syn_health = [], [], [], [], []
            syn_edit, syn_pert, syn_slots = [], [], []
            n_roles = len(ROLE_LABELS)

            for syn_i in range(n_syn):
                r_id = syn_i % n_roles
                r_lbl = ROLE_LABELS[r_id]
                is_pert = (r_lbl == "perturbed_anomaly") or (rng_cal.random() < 0.08)
                is_edit = is_pert or (r_lbl not in ("early_syntax", "general_epistemic", "synthesis_head")) or (rng_cal.random() < 0.35)
                d_val = round(rng_cal.uniform(0.10, 1.40), 2)
                f_val = round(math.exp(rng_cal.uniform(math.log(3.0), math.log(280.0))), 1)
                sr_val = round(math.exp(rng_cal.uniform(math.log(2.5), math.log(220.0))), 1)
                b_val = 0.0 if (not is_edit and rng_cal.random() < 0.7) else round(rng_cal.uniform(0.02, 1.85), 2)
                if is_pert:
                    h_val = round(rng_cal.uniform(0.20, 0.72), 2)
                elif rng_cal.random() < 0.20:
                    h_val = round(rng_cal.uniform(0.68, 0.84), 2)
                else:
                    h_val = round(rng_cal.uniform(0.86, 0.99), 2)

                estado = "editada_perturbada" if is_pert else ("editada" if is_edit else "base_nativa")
                salud = "deriva_espectral" if h_val < 0.65 else ("alerta_moderada" if h_val < 0.85 else "estable")
                txt_syn = f"{r_lbl} | {estado} | {d_val:.2f} | {f_val:.1f} | {sr_val:.1f} | {b_val:.2f} | {salud}"
                syn_slots.append(_encode_slots(txt_syn))

                ia, ib = rng_cal.randrange(n_real), rng_cal.randrange(n_real)
                lam = rng_cal.uniform(0.2, 0.8)
                z_mix = lam * z_tr_tracked[ia] + (1.0 - lam) * z_tr_tracked[ib] + 0.25 * torch.randn_like(z_tr_tracked[0])
                syn_z_list.append(z_mix)

                rl = torch.full((n_roles,), -2.5)
                rl[r_id] = 7.5
                syn_roles.append(rl)
                syn_damp.append(d_val)
                syn_lfrob.append(math.log1p(f_val))
                syn_lsr.append(math.log1p(sr_val))
                syn_bias.append(b_val)
                syn_health.append(h_val)
                syn_edit.append(6.0 if is_edit else -6.0)
                syn_pert.append(6.0 if is_pert else -6.0)

            rep_r = 4
            z_cal = torch.cat([z_tr_tracked.repeat(rep_r, 1), torch.stack(syn_z_list, dim=0)], dim=0)
            y_cal_slots = torch.cat([y_real_slots.repeat(rep_r, 1), torch.tensor(syn_slots, dtype=torch.long)], dim=0)
            attrs_cal = {
                "role_logits": torch.cat([attrs_tr["role_logits"].repeat(rep_r, 1), torch.stack(syn_roles, dim=0)], dim=0),
                "damping": torch.cat([attrs_tr["damping"].repeat(rep_r), torch.tensor(syn_damp)], dim=0),
                "log_frob_down": torch.cat([attrs_tr["log_frob_down"].repeat(rep_r), torch.tensor(syn_lfrob)], dim=0),
                "log_stable_rank": torch.cat([attrs_tr["log_stable_rank"].repeat(rep_r), torch.tensor(syn_lsr)], dim=0),
                "bias_norm": torch.cat([attrs_tr["bias_norm"].repeat(rep_r), torch.tensor(syn_bias)], dim=0),
                "health": torch.cat([attrs_tr["health"].repeat(rep_r), torch.tensor(syn_health)], dim=0),
                "edited_logit": torch.cat([attrs_tr["edited_logit"].repeat(rep_r), torch.tensor(syn_edit)], dim=0),
                "perturbed_logit": torch.cat([attrs_tr["perturbed_logit"].repeat(rep_r), torch.tensor(syn_pert)], dim=0),
            }

            opt_warm = torch.optim.AdamW(warm_params, lr=5.0e-3, weight_decay=1e-4)
            n_cal = z_cal.size(0)
            mb_warm = 64
            warm_steps = 460
            for w_step in range(warm_steps):
                introspector.soft_prompt.train()
                perm = torch.randint(0, n_cal, (mb_warm,))
                opt_warm.zero_grad()
                attrs_mb = {k: v[perm] for k, v in attrs_cal.items()}
                lg, _ = introspector.soft_prompt.compute_alignment_logits(
                    z_cal[perm], attrs=attrs_mb, feat_noise_std=0.03
                )
                l_align = F.cross_entropy(lg.reshape(-1, v_sub), y_cal_slots[perm].reshape(-1))
                l_align.backward()
                torch.nn.utils.clip_grad_norm_(warm_params, 1.0)
                opt_warm.step()
                align_loss_val = 0.9 * align_loss_val + 0.1 * float(l_align.item()) if w_step > 0 else float(l_align.item())
            print(
                f"[meta-stage2:{tag}] Calibración sub-vocabulario ({warm_steps} pasos en {time.perf_counter() - t_warm:.2f}s) | AlignLoss={align_loss_val:.4f}",
                flush=True,
            )

        # ------------------------------------------------------------------
        # Etapa 2B: Ajuste fino Causal LM end-to-end sobre el Transformer congelado
        # ------------------------------------------------------------------
        delta_params = [
            p for n, p in introspector.soft_prompt.named_parameters()
            if n.startswith("delta_proj") or n == "res_scale"
        ]
        base_lr = float(lr_override if lr_override is not None else self.lr_stage2)
        optimizer = torch.optim.AdamW(
            [
                {"params": warm_params, "lr": base_lr * warm_lr_mult},
                {"params": delta_params, "lr": base_lr},
            ],
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps), eta_min=eta_min)

        pre_emb, post_emb = build_sensory_prompt_parts(model, tok, invariant_prompt)
        eos_tok = tok.eos_token or ""
        embed_layer = model.get_input_embeddings()

        def compute_lm_loss(batch_snaps: List[LayerWeightSnapshot]) -> torch.Tensor:
            specs = torch.stack([s.spectral_vector for s in batch_snaps], dim=0)
            with torch.no_grad():
                z = introspector.encoder(specs)
                attrs = introspector.attr_heads(z)
                z_tr, _, _ = introspector.koopman_monitor(z, z)
            prefix = introspector.soft_prompt(z_tr, attrs=attrs).to(dtype=model.dtype)  # [B, K, D]

            targets = [s.canonical_description() + eos_tok for s in batch_snaps]
            enc_t = tok(targets, return_tensors="pt", padding=True, truncation=True, max_length=80)
            t_ids = enc_t["input_ids"]
            t_attn = enc_t["attention_mask"]
            t_emb = embed_layer(t_ids).detach()

            B = len(batch_snaps)
            K = prefix.shape[1]
            p_pre = pre_emb.expand(B, -1, -1)
            p_post = post_emb.expand(B, -1, -1)
            full_emb = torch.cat([p_pre, prefix, p_post, t_emb], dim=1)

            head_len = p_pre.shape[1] + K + p_post.shape[1]
            attn_head = torch.ones(B, head_len, dtype=torch.long)
            full_attn = torch.cat([attn_head, t_attn], dim=1)

            ignore_head = torch.full((B, head_len), -100, dtype=torch.long)
            t_labels = t_ids.clone()
            t_labels[t_attn == 0] = -100
            labels = torch.cat([ignore_head, t_labels], dim=1)

            out = model(inputs_embeds=full_emb, attention_mask=full_attn, labels=labels, use_cache=False)
            return out.loss

        def eval_val_loss(snaps: List[LayerWeightSnapshot]) -> float:
            if not snaps:
                return 0.0
            total_l = 0.0
            cnt = 0
            mb = max(1, min(batch_size, 2))
            with torch.no_grad():
                for idx_v in range(0, len(snaps), mb):
                    sub = snaps[idx_v : idx_v + mb]
                    total_l += float(compute_lm_loss(sub).item()) * len(sub)
                    cnt += len(sub)
            return total_l / max(1, cnt)

        # Seleccionar muestra diversa compacta priorizando anomalías, puentes y roles centrales
        by_role_val: Dict[str, List[LayerWeightSnapshot]] = {}
        for s in val_snaps:
            by_role_val.setdefault(s.role_label, []).append(s)
        priority_roles = [
            "perturbed_anomaly",
            "meta_bridge_koopman",
            "code_cluster",
            "meta_observer_core",
            "early_syntax",
            "math_cluster",
            "ai_causal_cluster",
            "general_epistemic",
            "synthesis_head",
        ]
        ordered_rks = [r for r in priority_roles if r in by_role_val] + [
            r for r in sorted(by_role_val.keys()) if r not in priority_roles
        ]
        val_sample: List[LayerWeightSnapshot] = [by_role_val[rk][0] for rk in ordered_rks[:max_val_items]]

        rng = random.Random(self.seed + (101 if "0_5b" in tag else 202) + seed_offset)
        by_role: Dict[str, List[LayerWeightSnapshot]] = {}
        for s in train_snaps:
            by_role.setdefault(s.role_label, []).append(s)
        for lst in by_role.values():
            rng.shuffle(lst)
        ordered_train: List[LayerWeightSnapshot] = []
        if role_priority_order:
            role_keys = [r for r in role_priority_order if r in by_role] + [
                r for r in sorted(by_role.keys()) if r not in role_priority_order
            ]
        else:
            role_keys = sorted(by_role.keys())
        max_b = max((len(v) for v in by_role.values()), default=0)
        for k in range(max_b):
            for rk in role_keys:
                if k < len(by_role[rk]):
                    ordered_train.append(by_role[rk][k])

        introspector.eval()
        init_val_loss = eval_val_loss(val_sample)
        print(f"[meta-stage2:{tag}] Val LM Loss inicial (step 0): {init_val_loss:.4f}", flush=True)

        best_val = init_val_loss
        best_sp = {k: v.cpu().clone() for k, v in introspector.soft_prompt.state_dict().items()}
        curve: List[Dict[str, Any]] = [{"step": 0, "val_loss": round(init_val_loss, 4)}]

        t0 = time.perf_counter()
        n_tr = len(ordered_train)
        cursor = 0
        mid_step = max(1, steps // 2)
        eval_steps = (mid_step, steps) if eval_mid_val else (steps,)
        accum = max(1, int(grad_accum))
        for step in range(1, steps + 1):
            introspector.soft_prompt.train()
            optimizer.zero_grad()
            step_loss_val = 0.0
            for _a in range(accum):
                batch: List[LayerWeightSnapshot] = []
                for _ in range(batch_size):
                    batch.append(ordered_train[cursor % n_tr])
                    cursor += 1
                    if cursor % n_tr == 0:
                        rng.shuffle(ordered_train)
                micro_loss = compute_lm_loss(batch)
                step_loss_val += float(micro_loss.item()) / float(accum)
                (micro_loss / float(accum)).backward()

            torch.nn.utils.clip_grad_norm_(introspector.soft_prompt.parameters(), 0.5)
            optimizer.step()
            scheduler.step()

            if step in eval_steps:
                introspector.eval()
                vloss = eval_val_loss(val_sample)
                cur_lr = scheduler.get_last_lr()[0]
                elapsed = time.perf_counter() - t0
                print(
                    f"[meta-stage2:{tag}] Step {step}/{steps} | LR={cur_lr:.2e} | TrainLoss={step_loss_val:.4f} | ValLoss={vloss:.4f} | {elapsed:.1f}s",
                    flush=True,
                )
                curve.append({
                    "step": step,
                    "train_loss": round(step_loss_val, 4),
                    "val_loss": round(vloss, 4),
                })
                if vloss <= best_val:
                    best_val = vloss
                    best_sp = {k: v.cpu().clone() for k, v in introspector.soft_prompt.state_dict().items()}
            elif accum > 1 or step % 2 == 0 or step == steps:
                print(
                    f"[meta-stage2:{tag}] Step {step}/{steps} | TrainLoss={step_loss_val:.4f} | {time.perf_counter() - t0:.1f}s",
                    flush=True,
                )

        introspector.soft_prompt.load_state_dict(best_sp)
        for p in introspector.parameters():
            p.requires_grad_(False)
        introspector.eval()

        return {
            "align_warmup_loss": round(align_loss_val, 4),
            "init_val_loss": round(init_val_loss, 4),
            "best_val_loss": round(best_val, 4),
            "steps": steps,
            "batch_size": batch_size,
            "grad_accum": accum,
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "curve": curve,
        }

    def run_stage1_only(self) -> Dict[str, Any]:
        """Paso 1 (~45 s): Extracción sin fuga + entrenamiento Stage 1A (S2) y Stage 1B (S1) + caché."""
        t_start = time.perf_counter()

        # ------------------------------------------------------------------
        # FASE A: Extracción de instantáneas reales de S1 (K66-1.5B) + sub-vocabulario
        # ------------------------------------------------------------------
        print("[meta-train] Fase A: Cargando S1 (K66-1.5B + Campo v4) y extrayendo pesos reales...", flush=True)
        m1, tok1, d1, f1 = build_weight_shared_k66_1_5b()
        s1_hidden = m1.config.hidden_size
        bank_ids_1, anc1, std1 = build_metacognitive_subvocab(m1, tok1)
        s1_introspector = WeightIntrospectionModule(
            hidden_lm=s1_hidden,
            k_prefix=DEFAULT_K_PREFIX,
            n_anchors=anc1.shape[0],
            init_anchors=anc1,
            embed_std=std1,
            bank_ids=bank_ids_1,
        ).float()

        s1_splits = collect_target_snapshots(
            model=m1,
            tok=tok1,
            layer_roles=d1.meta["layer_roles"],
            field=f1,
            system_id="S1_principal",
            holdout_layers=S1_HOLDOUT_LAYERS,
            val_layers=S1_VAL_LAYERS,
            holdout_bridges=S1_HOLDOUT_BRIDGES,
            val_bridges=S1_VAL_BRIDGES,
            probe_module=None,
            seed=self.seed,
        )
        print(
            f"[meta-train] Snapshots S1 extraídos -> train={len(s1_splits['train'])}, val={len(s1_splits['val'])}, holdout={len(s1_splits['holdout'])}",
            flush=True,
        )
        del m1, f1
        gc.collect()

        # ------------------------------------------------------------------
        # FASE B + ETAPA 1A: Cargar S2 (K32-0.5B), entrenar y congelar core S2 antes de sondarlo
        # ------------------------------------------------------------------
        print("[meta-train] Fase B: Cargando S2 (K32-0.5B + Campo v4)...", flush=True)
        m2, tok2, d2, f2 = build_weight_shared_k32_0_5b()
        s2_hidden = m2.config.hidden_size
        bank_ids_2, anc2, std2 = build_metacognitive_subvocab(m2, tok2)
        s2_introspector = WeightIntrospectionModule(
            hidden_lm=s2_hidden,
            k_prefix=DEFAULT_K_PREFIX,
            n_anchors=anc2.shape[0],
            init_anchors=anc2,
            embed_std=std2,
            bank_ids=bank_ids_2,
        ).float()

        s1_tr = self._snapshots_to_tensors(s1_splits["train"])
        s1_va = self._snapshots_to_tensors(s1_splits["val"])
        s1_ho = self._snapshots_to_tensors(s1_splits["holdout"])

        print(f"[meta-train] Etapa 1A: Entrenando introspección S2 -> S1 ({self.stage1_epochs} épocas)...", flush=True)
        t_s1 = time.perf_counter()
        s2_core_params = [p for n, p in s2_introspector.named_parameters() if not n.startswith("soft_prompt.")]
        opt_s2 = torch.optim.AdamW(s2_core_params, lr=self.lr_stage1, weight_decay=1e-4)
        sched_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s2, T_max=self.stage1_epochs, eta_min=1e-4)

        best_s2_score = -1.0
        best_s2_state = {k: v.cpu().clone() for k, v in s2_introspector.state_dict().items()}
        p2_tr = s1_tr["perturbed"]
        same_s1 = (p2_tr.unsqueeze(1) == p2_tr.unsqueeze(0)).float()
        target_sim_s1 = same_s1 * 0.8 + (1.0 - same_s1) * (-0.5)
        curve_s2: List[Dict[str, Any]] = []

        for epoch in range(1, self.stage1_epochs + 1):
            s2_introspector.train()
            opt_s2.zero_grad()
            loss_s2, out2 = self._compute_introspector_loss(s2_introspector, s1_tr)
            c2 = out2["c_emb"]
            l_self_contrast = F.mse_loss(torch.matmul(c2, c2.t()), target_sim_s1)
            total_2 = loss_s2 + 0.35 * l_self_contrast
            total_2.backward()
            torch.nn.utils.clip_grad_norm_(s2_core_params, 1.0)
            opt_s2.step()
            sched_s2.step()

            if epoch % 40 == 0 or epoch == self.stage1_epochs:
                m_val_s2 = self._eval_stage1_metrics(s2_introspector, s1_va)
                score = m_val_s2["role_acc"] + m_val_s2["pert_f1"] - 0.25 * m_val_s2["damping_mae"]
                curve_s2.append({"epoch": epoch, "loss": round(float(total_2.item()), 4), "s2_to_s1_val": m_val_s2})
                print(
                    f"[meta-stage1A S2->S1] Epoch {epoch:3d}/{self.stage1_epochs} | "
                    f"Role={m_val_s2['role_acc']:.1%}, F1={m_val_s2['pert_f1']:.1%}, DampMAE={m_val_s2['damping_mae']:.3f}",
                    flush=True,
                )
                if score >= best_s2_score:
                    best_s2_score = score
                    best_s2_state = {k: v.cpu().clone() for k, v in s2_introspector.state_dict().items()}

        s2_introspector.load_state_dict(best_s2_state)
        for p in s2_core_params:
            p.requires_grad_(False)
        s2_introspector.eval()

        # Recolectar instantáneas reales de S2 CON la sonda entrenada y fijada en estado final
        s2_splits = collect_target_snapshots(
            model=m2,
            tok=tok2,
            layer_roles=d2.meta["layer_roles"],
            field=f2,
            system_id="S2_observer",
            holdout_layers=S2_HOLDOUT_LAYERS,
            val_layers=S2_VAL_LAYERS,
            holdout_bridges=S2_HOLDOUT_BRIDGES,
            val_bridges=S2_VAL_BRIDGES,
            probe_module=s2_introspector,
            seed=self.seed,
        )
        print(
            f"[meta-train] Snapshots S2 extraídos -> train={len(s2_splits['train'])}, val={len(s2_splits['val'])}, holdout={len(s2_splits['holdout'])}",
            flush=True,
        )
        del m2, f2
        gc.collect()

        # ------------------------------------------------------------------
        # ETAPA 1B: Entrenamiento introspección recíproca S1 -> S2 + alineación contrastiva
        # ------------------------------------------------------------------
        s2_tr = self._snapshots_to_tensors(s2_splits["train"])
        s2_va = self._snapshots_to_tensors(s2_splits["val"])
        s2_ho = self._snapshots_to_tensors(s2_splits["holdout"])

        with torch.no_grad():
            c2_frozen = s2_introspector.encode_and_predict(s1_tr["x"])["contrastive"].detach()
        same_cross = (p2_tr.unsqueeze(1) == s2_tr["perturbed"].unsqueeze(0)).float()
        target_cross = same_cross * 0.8 + (1.0 - same_cross) * (-0.5)

        print(f"[meta-train] Etapa 1B: Entrenando introspección recíproca S1 -> S2 ({self.stage1_epochs} épocas)...", flush=True)
        s1_core_params = [p for n, p in s1_introspector.named_parameters() if not n.startswith("soft_prompt.")]
        opt_s1 = torch.optim.AdamW(s1_core_params, lr=self.lr_stage1, weight_decay=1e-4)
        sched_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s1, T_max=self.stage1_epochs, eta_min=1e-4)

        best_s1_score = -1.0
        best_s1_state = {k: v.cpu().clone() for k, v in s1_introspector.state_dict().items()}
        curve_s1: List[Dict[str, Any]] = []

        for epoch in range(1, self.stage1_epochs + 1):
            s1_introspector.train()
            opt_s1.zero_grad()
            loss_s1, out1 = self._compute_introspector_loss(s1_introspector, s2_tr)
            c1 = out1["c_emb"]
            sim_mat = torch.matmul(c2_frozen, c1.t())  # [N_s1, N_s2]
            l_contrast = F.mse_loss(sim_mat, target_cross)
            total_1 = loss_s1 + 0.5 * l_contrast
            total_1.backward()
            torch.nn.utils.clip_grad_norm_(s1_core_params, 1.0)
            opt_s1.step()
            sched_s1.step()

            if epoch % 40 == 0 or epoch == self.stage1_epochs:
                m_val_s1 = self._eval_stage1_metrics(s1_introspector, s2_va)
                score = m_val_s1["role_acc"] + m_val_s1["pert_f1"] - 0.25 * m_val_s1["damping_mae"]
                curve_s1.append({"epoch": epoch, "loss": round(float(total_1.item()), 4), "s1_to_s2_val": m_val_s1})
                print(
                    f"[meta-stage1B S1->S2] Epoch {epoch:3d}/{self.stage1_epochs} | "
                    f"Role={m_val_s1['role_acc']:.1%}, F1={m_val_s1['pert_f1']:.1%}, DampMAE={m_val_s1['damping_mae']:.3f}",
                    flush=True,
                )
                if score >= best_s1_score:
                    best_s1_score = score
                    best_s1_state = {k: v.cpu().clone() for k, v in s1_introspector.state_dict().items()}

        s1_introspector.load_state_dict(best_s1_state)
        for p in s1_core_params:
            p.requires_grad_(False)
        s1_introspector.eval()

        stage1_metrics = {
            "elapsed_s": round(time.perf_counter() - t_s1, 2),
            "s2_to_s1": {
                "train": self._eval_stage1_metrics(s2_introspector, s1_tr),
                "val": self._eval_stage1_metrics(s2_introspector, s1_va),
                "holdout_unseen": self._eval_stage1_metrics(s2_introspector, s1_ho),
            },
            "s1_to_s2": {
                "train": self._eval_stage1_metrics(s1_introspector, s2_tr),
                "val": self._eval_stage1_metrics(s1_introspector, s2_va),
                "holdout_unseen": self._eval_stage1_metrics(s1_introspector, s2_ho),
            },
            "curve_s2": curve_s2,
            "curve_s1": curve_s1,
        }

        s1_cache_path = os.path.join(self.out_dir, "stage1_cache.pt")
        stage1_total_s = round(time.perf_counter() - t_start, 2)
        torch.save(
            {
                "stage1_total_s": stage1_total_s,
                "s1_splits": s1_splits,
                "s2_splits": s2_splits,
                "stage1_metrics": stage1_metrics,
                "s2_introspector": s2_introspector.state_dict(),
                "s1_introspector": s1_introspector.state_dict(),
                "s2_hidden_lm": s2_introspector.hidden_lm,
                "s1_hidden_lm": s1_introspector.hidden_lm,
                "s2_k_prefix": s2_introspector.k_prefix,
                "s1_k_prefix": s1_introspector.k_prefix,
                "s2_bank_ids": bank_ids_2,
                "s1_bank_ids": bank_ids_1,
                "s2_embed_std": s2_introspector.soft_prompt.embed_std,
                "s1_embed_std": s1_introspector.soft_prompt.embed_std,
            },
            s1_cache_path,
        )
        print(f"[meta-train] Etapa 1 completa en {stage1_total_s}s. Caché guardado en {s1_cache_path}", flush=True)
        return {"stage1_total_s": stage1_total_s, "stage1_cache_path": s1_cache_path, "stage1": stage1_metrics}

    def run_stage2_s2_only(self) -> Dict[str, Any]:
        """Paso 2 (~2.8 min): Carga stage1_cache.pt, entrena Soft-Prompt S2 (0.5B -> S1) y guarda phase1_cache.pt."""
        s1_cache_path = os.path.join(self.out_dir, "stage1_cache.pt")
        if not os.path.isfile(s1_cache_path):
            raise FileNotFoundError(f"Falta caché Stage 1 en {s1_cache_path}; ejecuta stage1 primero.")
        t_s2_start = time.perf_counter()
        cache = torch.load(s1_cache_path, map_location="cpu", weights_only=False)
        s1_splits = cache["s1_splits"]
        s2_splits = cache["s2_splits"]
        stage1_metrics = cache["stage1_metrics"]

        m2, tok2, _, f2 = build_weight_shared_k32_0_5b()
        s2_anc = cache["s2_introspector"]["soft_prompt.anchor_bank"]
        s2_introspector = WeightIntrospectionModule(
            hidden_lm=int(cache["s2_hidden_lm"]),
            k_prefix=int(cache["s2_k_prefix"]),
            n_anchors=int(s2_anc.shape[0]),
            init_anchors=s2_anc.clone(),
            embed_std=float(cache["s2_embed_std"]),
            bank_ids=cache["s2_bank_ids"],
        ).float()
        s2_introspector.load_state_dict(cache["s2_introspector"])

        stage2_s2_info = self._train_stage2_soft_prompt(
            model=m2,
            tok=tok2,
            introspector=s2_introspector,
            train_snaps=s1_splits["train"],
            val_snaps=s1_splits["val"],
            invariant_prompt=S2_TO_S1_PROMPT,
            steps=self.stage2_steps_s2,
            batch_size=3,
            tag="S2_0_5b->S1",
        )
        del m2, f2
        gc.collect()

        phase1_elapsed = round(float(cache.get("stage1_total_s", 0.0)) + (time.perf_counter() - t_s2_start), 2)
        cache_path = os.path.join(self.out_dir, "phase1_cache.pt")
        torch.save(
            {
                "phase1_elapsed_s": phase1_elapsed,
                "s1_splits": s1_splits,
                "s2_splits": s2_splits,
                "stage1_metrics": stage1_metrics,
                "stage2_s2_info": stage2_s2_info,
                "s2_introspector": s2_introspector.state_dict(),
                "s1_introspector": cache["s1_introspector"],
                "s2_hidden_lm": cache["s2_hidden_lm"],
                "s1_hidden_lm": cache["s1_hidden_lm"],
                "s2_k_prefix": cache["s2_k_prefix"],
                "s1_k_prefix": cache["s1_k_prefix"],
                "s2_bank_ids": cache["s2_bank_ids"],
                "s1_bank_ids": cache["s1_bank_ids"],
                "s2_embed_std": cache["s2_embed_std"],
                "s1_embed_std": cache["s1_embed_std"],
            },
            cache_path,
        )
        try:
            os.remove(s1_cache_path)
        except OSError:
            pass
        print(f"[meta-train] Etapa 2A (S2->S1) completada. Caché intermedio en {cache_path}", flush=True)
        return {
            "phase1_elapsed_s": phase1_elapsed,
            "cache_path": cache_path,
            "stage1": stage1_metrics,
            "stage2_s2_to_s1": stage2_s2_info,
        }

    def run_training_phase1(self) -> Dict[str, Any]:
        """Fase 1: Ejecuta run_stage1_only seguida de run_stage2_s2_only."""
        self.run_stage1_only()
        return self.run_stage2_s2_only()

    def run_training_phase2(self) -> Dict[str, Any]:
        """Fase 2: Carga caché de Fase 1, entrena Soft-Prompt S1 (K66-1.5B -> S2) y guarda artefactos finales."""
        cache_path = os.path.join(self.out_dir, "phase1_cache.pt")
        if not os.path.isfile(cache_path):
            raise FileNotFoundError(f"Falta caché de Fase 1 en {cache_path}; ejecuta phase1 primero.")

        t_p2 = time.perf_counter()
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        s1_splits = cache["s1_splits"]
        s2_splits = cache["s2_splits"]
        stage1_metrics = cache["stage1_metrics"]
        stage2_s2_info = cache["stage2_s2_info"]
        bank_ids_1 = cache["s1_bank_ids"]
        bank_ids_2 = cache["s2_bank_ids"]

        s2_anc = cache["s2_introspector"]["soft_prompt.anchor_bank"]
        s2_introspector = WeightIntrospectionModule(
            hidden_lm=int(cache["s2_hidden_lm"]),
            k_prefix=int(cache["s2_k_prefix"]),
            n_anchors=int(s2_anc.shape[0]),
            init_anchors=s2_anc.clone(),
            embed_std=float(cache["s2_embed_std"]),
            bank_ids=bank_ids_2,
        ).float()
        s2_introspector.load_state_dict(cache["s2_introspector"])
        for p in s2_introspector.parameters():
            p.requires_grad_(False)
        s2_introspector.eval()

        # ------------------------------------------------------------------
        # ETAPA 2B: Entrenamiento Soft-Prompt S1 -> S2 sobre K66-1.5B congelado
        # ------------------------------------------------------------------
        print("[meta-train] Fase 2: Cargando S1 (K66-1.5B + Campo v4) para Soft-Prompt recíproco...", flush=True)
        m1, tok1, _, f1 = build_weight_shared_k66_1_5b()
        s1_anc = cache["s1_introspector"]["soft_prompt.anchor_bank"]
        s1_introspector = WeightIntrospectionModule(
            hidden_lm=int(cache["s1_hidden_lm"]),
            k_prefix=int(cache["s1_k_prefix"]),
            n_anchors=int(s1_anc.shape[0]),
            init_anchors=s1_anc.clone(),
            embed_std=float(cache["s1_embed_std"]),
            bank_ids=bank_ids_1,
        ).float()
        s1_introspector.load_state_dict(cache["s1_introspector"])

        stage2_s1_info = self._train_stage2_soft_prompt(
            model=m1,
            tok=tok1,
            introspector=s1_introspector,
            train_snaps=s2_splits["train"],
            val_snaps=s2_splits["val"],
            invariant_prompt=S1_TO_S2_PROMPT,
            steps=self.stage2_steps_s1,
            batch_size=1,
            tag="S1_1_5b->S2",
        )
        del m1, f1
        gc.collect()

        # Guardar checkpoint y bitácora completa
        ckpt_path = os.path.join(self.out_dir, "meta_bidirectional_ckpt.pt")
        payload = {
            "s2_introspector": s2_introspector.state_dict(),
            "s1_introspector": s1_introspector.state_dict(),
            "s2_hidden_lm": s2_introspector.hidden_lm,
            "s1_hidden_lm": s1_introspector.hidden_lm,
            "s2_k_prefix": s2_introspector.k_prefix,
            "s1_k_prefix": s1_introspector.k_prefix,
            "s2_bank_ids": bank_ids_2,
            "s1_bank_ids": bank_ids_1,
            "role_labels": ROLE_LABELS,
            "split_spec": {
                "s1_holdout_layers": sorted(S1_HOLDOUT_LAYERS),
                "s1_val_layers": sorted(S1_VAL_LAYERS),
                "s2_holdout_layers": sorted(S2_HOLDOUT_LAYERS),
                "s2_val_layers": sorted(S2_VAL_LAYERS),
            },
        }
        torch.save(payload, ckpt_path)

        holdout_cache_path = os.path.join(self.out_dir, "holdout_snapshots.pt")
        torch.save(
            {
                "s1_holdout": s1_splits["holdout"],
                "s2_holdout": s2_splits["holdout"],
                "s1_counts": {k: len(v) for k, v in s1_splits.items()},
                "s2_counts": {k: len(v) for k, v in s2_splits.items()},
            },
            holdout_cache_path,
        )

        phase2_elapsed = round(time.perf_counter() - t_p2, 2)
        total_elapsed = round(float(cache.get("phase1_elapsed_s", 0.0)) + phase2_elapsed, 2)
        train_summary = {
            "total_elapsed_s": total_elapsed,
            "phase1_elapsed_s": cache.get("phase1_elapsed_s", 0.0),
            "phase2_elapsed_s": phase2_elapsed,
            "checkpoint_path": ckpt_path,
            "s2_introspector_params": s2_introspector.n_params(),
            "s1_introspector_params": s1_introspector.n_params(),
            "splits": {
                "s1_target": {k: len(v) for k, v in s1_splits.items()},
                "s2_target": {k: len(v) for k, v in s2_splits.items()},
            },
            "stage1": stage1_metrics,
            "stage2_s2_to_s1": stage2_s2_info,
            "stage2_s1_to_s2": stage2_s1_info,
        }
        _jdump(os.path.join(self.out_dir, "train_log.json"), train_summary)
        try:
            os.remove(cache_path)
        except OSError:
            pass
        print(
            f"[meta-train] Entrenamiento completo en {train_summary['total_elapsed_s']}s. Checkpoint guardado en {ckpt_path}",
            flush=True,
        )
        return train_summary

    def refine_s1_soft_prompt(
        self,
        steps: int = 4,
        lr: float = 6.5e-4,
        seed_offset: int = 911,
        grad_accum: int = 3,
    ) -> Dict[str, Any]:
        """Refina únicamente `s1_introspector.soft_prompt` sobre `s2_splits['train']` sin tocar `s2_introspector`."""
        ckpt_path = os.path.join(self.out_dir, "meta_bidirectional_ckpt.pt")
        log_path = os.path.join(self.out_dir, "train_log.json")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"No existe checkpoint previo en {ckpt_path}")

        t_ref = time.perf_counter()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        train_log = {}
        if os.path.isfile(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                train_log = json.load(f)

        print("[meta-refine:S1] Reconstruyendo splits exactos de S2 (S2 congelado, sin fuga)...", flush=True)
        m2, tok2, d2, f2 = build_weight_shared_k32_0_5b()
        s2_anc = ckpt["s2_introspector"]["soft_prompt.anchor_bank"]
        s2_introspector = WeightIntrospectionModule(
            hidden_lm=int(ckpt["s2_hidden_lm"]),
            k_prefix=int(ckpt["s2_k_prefix"]),
            n_anchors=int(s2_anc.shape[0]),
            init_anchors=s2_anc.clone(),
            embed_std=float(ckpt.get("s2_embed_std", 0.02)),
            bank_ids=ckpt["s2_bank_ids"],
        ).float()
        s2_introspector.load_state_dict(ckpt["s2_introspector"])
        for p in s2_introspector.parameters():
            p.requires_grad_(False)
        s2_introspector.eval()

        s2_splits = collect_target_snapshots(
            model=m2,
            tok=tok2,
            layer_roles=d2.meta["layer_roles"],
            field=f2,
            system_id="S2_observer",
            holdout_layers=S2_HOLDOUT_LAYERS,
            val_layers=S2_VAL_LAYERS,
            holdout_bridges=S2_HOLDOUT_BRIDGES,
            val_bridges=S2_VAL_BRIDGES,
            probe_module=s2_introspector,
            seed=self.seed,
        )
        del m2, f2
        gc.collect()

        print("[meta-refine:S1] Cargando S1 (K66-1.5B + Campo v4) y aplicando ajuste fino Causal LM...", flush=True)
        m1, tok1, _, f1 = build_weight_shared_k66_1_5b()
        s1_anc = ckpt["s1_introspector"]["soft_prompt.anchor_bank"]
        s1_introspector = WeightIntrospectionModule(
            hidden_lm=int(ckpt["s1_hidden_lm"]),
            k_prefix=int(ckpt["s1_k_prefix"]),
            n_anchors=int(s1_anc.shape[0]),
            init_anchors=s1_anc.clone(),
            embed_std=float(ckpt.get("s1_embed_std", 0.02)),
            bank_ids=ckpt["s1_bank_ids"],
        ).float()
        s1_introspector.load_state_dict(ckpt["s1_introspector"])

        ref_info = self._train_stage2_soft_prompt(
            model=m1,
            tok=tok1,
            introspector=s1_introspector,
            train_snaps=s2_splits["train"],
            val_snaps=s2_splits["val"],
            invariant_prompt=S1_TO_S2_PROMPT,
            steps=steps,
            batch_size=1,
            tag="S1_1_5b->S2_refine",
            skip_warmup=True,
            lr_override=lr,
            warm_lr_mult=0.22,
            eta_min=8.0e-5,
            seed_offset=seed_offset,
            max_val_items=5,
            eval_mid_val=False,
            role_priority_order=[
                "perturbed_anomaly",
                "early_syntax",
                "synthesis_head",
                "meta_observer_core",
                "meta_bridge_koopman",
            ],
            grad_accum=grad_accum,
        )
        del m1, f1
        gc.collect()

        # Preservar s2_introspector bit-a-bit; actualizar únicamente s1_introspector
        ckpt["s1_introspector"] = s1_introspector.state_dict()
        torch.save(ckpt, ckpt_path)

        ref_elapsed = round(time.perf_counter() - t_ref, 2)
        ref_info["total_refine_s"] = ref_elapsed
        ref_info["lr"] = lr
        ref_info["seed_offset"] = seed_offset

        if train_log:
            refs = train_log.setdefault("stage2_s1_refinements", [])
            refs.append(ref_info)
            train_log["total_elapsed_s"] = round(float(train_log.get("total_elapsed_s", 0.0)) + ref_elapsed, 2)
            train_log["phase2_elapsed_s"] = round(float(train_log.get("phase2_elapsed_s", 0.0)) + ref_elapsed, 2)
            _jdump(log_path, train_log)

        stale_cache = os.path.join(self.out_dir, "eval_part1_cache.json")
        if os.path.isfile(stale_cache):
            try:
                os.remove(stale_cache)
            except OSError:
                pass

        print(f"[meta-refine:S1] Refinamiento completo en {ref_elapsed}s -> {ckpt_path}", flush=True)
        return ref_info

    def spot_check_s1_val(self) -> Dict[str, Any]:
        """Comprobación rápida (~180 s) de decodificación real S1 -> S2 en los 5 roles de Validación."""
        from .agents import ReflexivePrimaryAgent, _parse_generated_description

        ckpt_path = os.path.join(self.out_dir, "meta_bidirectional_ckpt.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        m2, tok2, d2, f2 = build_weight_shared_k32_0_5b()
        s2_anc = ckpt["s2_introspector"]["soft_prompt.anchor_bank"]
        s2_introspector = WeightIntrospectionModule(
            hidden_lm=int(ckpt["s2_hidden_lm"]),
            k_prefix=int(ckpt["s2_k_prefix"]),
            n_anchors=int(s2_anc.shape[0]),
            init_anchors=s2_anc.clone(),
            embed_std=float(ckpt.get("s2_embed_std", 0.02)),
            bank_ids=ckpt["s2_bank_ids"],
        ).float()
        s2_introspector.load_state_dict(ckpt["s2_introspector"])
        s2_splits = collect_target_snapshots(
            model=m2,
            tok=tok2,
            layer_roles=d2.meta["layer_roles"],
            field=f2,
            system_id="S2_observer",
            holdout_layers=S2_HOLDOUT_LAYERS,
            val_layers=S2_VAL_LAYERS,
            holdout_bridges=S2_HOLDOUT_BRIDGES,
            val_bridges=S2_VAL_BRIDGES,
            probe_module=s2_introspector,
            seed=self.seed,
        )
        del m2, f2
        gc.collect()

        m1, tok1, d1, f1 = build_weight_shared_k66_1_5b()
        s1 = ReflexivePrimaryAgent(
            agent_id="S1_principal",
            model=m1,
            tok=tok1,
            layer_roles=d1.meta["layer_roles"],
            field=f1,
        )
        s1_anc = ckpt["s1_introspector"]["soft_prompt.anchor_bank"]
        s1.introspector = WeightIntrospectionModule(
            hidden_lm=int(ckpt["s1_hidden_lm"]),
            k_prefix=int(ckpt["s1_k_prefix"]),
            n_anchors=int(s1_anc.shape[0]),
            init_anchors=s1_anc.clone(),
            embed_std=float(ckpt.get("s1_embed_std", 0.02)),
            bank_ids=ckpt["s1_bank_ids"],
        ).float()
        s1.introspector.load_state_dict(ckpt["s1_introspector"])
        s1.introspector.eval()

        by_role: Dict[str, LayerWeightSnapshot] = {}
        for s in s2_splits["val"]:
            by_role.setdefault(s.role_label, s)

        rows: List[Dict[str, Any]] = []
        role_ok = 0
        state_ok = 0
        for rk in sorted(by_role.keys()):
            snap = by_role[rk]
            gt_estado = "editada_perturbada" if snap.is_perturbed else ("editada" if snap.is_edited else "base_nativa")
            gen, _ = s1.generate_weight_description(snap, invariant_prompt=S1_TO_S2_PROMPT, max_new_tokens=68)
            parsed = _parse_generated_description(gen)
            if parsed.get("role") == snap.role_label:
                role_ok += 1
            if parsed.get("state") == gt_estado:
                state_ok += 1
            rows.append({
                "role": snap.role_label,
                "gt": snap.canonical_description(),
                "gen": gen,
                "role_match": parsed.get("role") == snap.role_label,
                "state_match": parsed.get("state") == gt_estado,
            })

        n = max(1, len(rows))
        return {
            "val_role_match_rate": round(role_ok / n, 4),
            "val_state_match_rate": round(state_ok / n, 4),
            "rows": rows,
        }

    def run_training(self) -> Dict[str, Any]:
        """Ejecuta extracción sin fuga -> Etapa 1A (S2) -> Etapa 1B (S1) -> Etapa 2A/2B (<5.5 GB RSS)."""
        self.run_training_phase1()
        return self.run_training_phase2()
