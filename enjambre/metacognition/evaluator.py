"""Evaluador Exhaustivo para la Arquitectura Multi-Agente Metacognitiva Bidireccional.

Ejecuta 4 baterías completas de verificación empírica en vivo:
1. Precisión Cuantitativa Sin Fuga (Train vs Val vs Holdout Capas/Puentes/Regímenes No Vistos):
   Rol, Damping, Norma Frobenius, Rango Estable, Bias Norm, Detección F1 de Perturbaciones,
   Separación Temporal Koopman y Alineación Contrastiva S1 <-> S2.
2. Generación Lingüística Real Condicionada Únicamente en Soft-Tokens (Prompts Invariantes):
   Evalúa decodificación greedy real (`model.generate`) de `S2 (0.5B) -> S1` y `S1 (1.5B) -> S2`
   sobre capas/puentes nunca vistos durante el entrenamiento.
3. Intervenciones Contrafactuales en Vivo (Bidireccionales):
   - Mutación física reversible en capa `S1[48]` detectada en tiempo real por `S2`.
   - Mutación en capa `S2[13]` detectada recíprocamente por `S1`.
   - Mutación exclusiva en los pesos de la sonda metacognitiva de `S2` (`introspector`)
     detectada por `S1` sin alterar capas Transformer de `S2`.
4. Benchmark Modo Tiempo Real vs Lotes de Episodios + Preservación de Tarea Primaria & RAM RSS.
"""
from __future__ import annotations

import gc
import json
import math
import os
import time
from typing import Any, Dict, List, Optional

import torch

from .agents import (
    BidirectionalMetacognitiveSystem,
    S1_TO_S2_PROMPT,
    S2_TO_S1_PROMPT,
)
from .spectrometer import LayerWeightSnapshot, apply_reversible_layer_perturbation
from .trainer import (
    BidirectionalMetaTrainer,
    DEFAULT_META_DIR,
    S1_HOLDOUT_BRIDGES,
    S1_HOLDOUT_LAYERS,
    S1_VAL_BRIDGES,
    S1_VAL_LAYERS,
    S2_HOLDOUT_BRIDGES,
    S2_HOLDOUT_LAYERS,
    S2_VAL_BRIDGES,
    S2_VAL_LAYERS,
    collect_target_snapshots,
)


def _rss_mb() -> float:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except OSError:
        pass
    return 0.0


def _r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    if target.numel() < 2:
        return 0.0
    var = torch.sum((target - target.mean()) ** 2).item()
    if var < 1e-8:
        return 1.0 if torch.mean((target - pred) ** 2).item() < 1e-4 else 0.0
    mse = torch.sum((target - pred) ** 2).item()
    return float(1.0 - mse / var)


def _jdump(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def _detailed_quant_metrics(introspector, snaps: List[LayerWeightSnapshot]) -> Dict[str, float]:
    tensors = BidirectionalMetaTrainer._snapshots_to_tensors(snaps)
    base = BidirectionalMetaTrainer._eval_stage1_metrics(introspector, tensors)
    introspector.eval()
    with torch.no_grad():
        pred = introspector.encode_and_predict(tensors["x"])
        damping_r2 = _r2_score(pred["damping"], tensors["damping"])
        log_frob_r2 = _r2_score(pred["log_frob_down"], tensors["log_frob"])
        frob_r2 = _r2_score(torch.expm1(pred["log_frob_down"]), torch.expm1(tensors["log_frob"]))
        sr_r2 = _r2_score(pred["log_stable_rank"], tensors["log_sr"])
        edit_acc = float(((torch.sigmoid(pred["edited_logit"]) >= 0.5).float() == tensors["edited"]).float().mean().item())
        health_mae = float((pred["health"] - tensors["health"]).abs().mean().item())
    return {
        "n_samples": len(snaps),
        **base,
        "damping_r2": round(damping_r2, 4),
        "log_frob_r2": round(log_frob_r2, 4),
        "frob_r2": round(frob_r2, 4),
        "log_sr_r2": round(sr_r2, 4),
        "edit_acc": round(edit_acc, 4),
        "health_mae": round(health_mae, 4),
    }


def _eval_generative_subset(
    observer_agent,
    holdout_snaps: List[LayerWeightSnapshot],
    invariant_prompt: str,
    max_items: int = 8,
) -> Dict[str, Any]:
    """Ejecuta generación lingüística real con el Transformer sobre instantáneas nunca vistas."""
    # Selección diversa por rol / módulo / anomalía
    by_role: Dict[str, List[LayerWeightSnapshot]] = {}
    for s in holdout_snaps:
        by_role.setdefault(s.role_label, []).append(s)

    selected: List[LayerWeightSnapshot] = []
    # Primera pasada: 1 muestra por rol único
    for rk in sorted(by_role.keys()):
        selected.append(by_role[rk][0])
    # Segunda pasada hasta alcanzar max_items
    idx_round = 1
    while len(selected) < max_items:
        added = False
        for rk in sorted(by_role.keys()):
            if len(by_role[rk]) > idx_round and len(selected) < max_items:
                selected.append(by_role[rk][idx_round])
                added = True
        if not added:
            break
        idx_round += 1

    records: List[Dict[str, Any]] = []
    role_ok = 0
    state_ok = 0
    health_ok = 0
    valid_fmt = 0
    damp_errs: List[float] = []
    frob_rel_errs: List[float] = []
    latencies: List[float] = []

    from .agents import _parse_generated_description

    for snap in selected:
        gt_estado = "editada_perturbada" if snap.is_perturbed else ("editada" if snap.is_edited else "base_nativa")
        gt_salud = "deriva_espectral" if snap.health_score < 0.65 else ("alerta_moderada" if snap.health_score < 0.85 else "estable")

        gen_text, fast = observer_agent.generate_weight_description(
            snap, invariant_prompt=invariant_prompt, max_new_tokens=68
        )
        parsed = _parse_generated_description(gen_text)
        lat = float(fast.get("lm_latency_ms", 0.0))
        latencies.append(lat)

        has_fmt = all(k in parsed for k in ("role", "state", "damping", "health"))
        if has_fmt:
            valid_fmt += 1
        if parsed.get("role") == snap.role_label:
            role_ok += 1
        if parsed.get("state") == gt_estado:
            state_ok += 1
        if parsed.get("health") == gt_salud:
            health_ok += 1
        if "damping" in parsed:
            damp_errs.append(abs(float(parsed["damping"]) - snap.damping))
        if "frob_down" in parsed:
            frob_rel_errs.append(abs(float(parsed["frob_down"]) - snap.frob_down) / max(1.0, snap.frob_down))

        records.append({
            "target_layer": snap.layer_idx,
            "module_type": snap.module_type,
            "gt_canonical": snap.canonical_description(),
            "generated_text": gen_text,
            "parsed": parsed,
            "gt_role": snap.role_label,
            "gt_state": gt_estado,
            "gt_health": gt_salud,
            "gt_damping": round(snap.damping, 3),
            "gt_frob": round(snap.frob_down, 2),
            "lm_latency_ms": lat,
        })

    n = max(1, len(selected))
    return {
        "n_evaluated": len(selected),
        "format_valid_rate": round(valid_fmt / n, 4),
        "lm_role_match_rate": round(role_ok / n, 4),
        "lm_state_match_rate": round(state_ok / n, 4),
        "lm_health_match_rate": round(health_ok / n, 4),
        "lm_damping_mae": round(sum(damp_errs) / max(1, len(damp_errs)), 4) if damp_errs else None,
        "lm_frob_rel_err": round(sum(frob_rel_errs) / max(1, len(frob_rel_errs)), 4) if frob_rel_errs else None,
        "mean_lm_latency_ms": round(sum(latencies) / n, 1),
        "samples": records,
    }


def _collect_live_splits(sys_bi: BidirectionalMetacognitiveSystem):
    s1_splits = collect_target_snapshots(
        model=sys_bi.s1.model,
        tok=sys_bi.s1.tok,
        layer_roles=sys_bi.s1.layer_roles,
        field=sys_bi.s1.field,
        system_id="S1_principal",
        holdout_layers=S1_HOLDOUT_LAYERS,
        val_layers=S1_VAL_LAYERS,
        holdout_bridges=S1_HOLDOUT_BRIDGES,
        val_bridges=S1_VAL_BRIDGES,
        probe_module=None,
    )
    s2_splits = collect_target_snapshots(
        model=sys_bi.s2.model,
        tok=sys_bi.s2.tok,
        layer_roles=sys_bi.s2.layer_roles,
        field=sys_bi.s2.field,
        system_id="S2_observer",
        holdout_layers=S2_HOLDOUT_LAYERS,
        val_layers=S2_VAL_LAYERS,
        holdout_bridges=S2_HOLDOUT_BRIDGES,
        val_bridges=S2_VAL_BRIDGES,
        probe_module=sys_bi.s2.introspector,
    )
    return s1_splits, s2_splits


def _run_part1_with_system(sys_bi: BidirectionalMetacognitiveSystem, rss_before: float, rss_dual_loaded: float) -> Dict[str, Any]:
    t0 = time.perf_counter()
    s1_splits, s2_splits = _collect_live_splits(sys_bi)

    quant_s2_to_s1 = {
        "train": _detailed_quant_metrics(sys_bi.s2.introspector, s1_splits["train"]),
        "val": _detailed_quant_metrics(sys_bi.s2.introspector, s1_splits["val"]),
        "holdout_unseen": _detailed_quant_metrics(sys_bi.s2.introspector, s1_splits["holdout"]),
    }
    quant_s1_to_s2 = {
        "train": _detailed_quant_metrics(sys_bi.s1.introspector, s2_splits["train"]),
        "val": _detailed_quant_metrics(sys_bi.s1.introspector, s2_splits["val"]),
        "holdout_unseen": _detailed_quant_metrics(sys_bi.s1.introspector, s2_splits["holdout"]),
    }

    with torch.no_grad():
        c2_h = sys_bi.s2.introspector.encode_and_predict(torch.stack([s.spectral_vector for s in s1_splits["holdout"]]))["contrastive"]
        c1_h = sys_bi.s1.introspector.encode_and_predict(torch.stack([s.spectral_vector for s in s2_splits["holdout"]]))["contrastive"]
        p2_h = torch.tensor([1.0 if s.is_perturbed else 0.0 for s in s1_splits["holdout"]])
        p1_h = torch.tensor([1.0 if s.is_perturbed else 0.0 for s in s2_splits["holdout"]])
        sim_h = torch.matmul(c2_h, c1_h.t())
        same_h = p2_h.unsqueeze(1) == p1_h.unsqueeze(0)
        sim_same = float(sim_h[same_h].mean().item())
        sim_diff = float(sim_h[~same_h].mean().item())
        contrastive_eval = {
            "holdout_same_state_cos": round(sim_same, 4),
            "holdout_diff_state_cos": round(sim_diff, 4),
            "contrastive_gap": round(sim_same - sim_diff, 4),
        }

    print(
        f"[meta-eval] Holdout Cuantitativo -> "
        f"S2->S1: Role={quant_s2_to_s1['holdout_unseen']['role_acc']:.1%}, F1={quant_s2_to_s1['holdout_unseen']['pert_f1']:.1%}, DampR2={quant_s2_to_s1['holdout_unseen']['damping_r2']:.3f} | "
        f"S1->S2: Role={quant_s1_to_s2['holdout_unseen']['role_acc']:.1%}, F1={quant_s1_to_s2['holdout_unseen']['pert_f1']:.1%}, DampR2={quant_s1_to_s2['holdout_unseen']['damping_r2']:.3f}",
        flush=True,
    )

    print("[meta-eval] Evaluando generación lingüística real en Holdout (S2->S1 y S1->S2)...", flush=True)
    gen_s2_to_s1 = _eval_generative_subset(sys_bi.s2, s1_splits["holdout"], S2_TO_S1_PROMPT, max_items=8)
    gen_s1_to_s2 = _eval_generative_subset(sys_bi.s1, s2_splits["holdout"], S1_TO_S2_PROMPT, max_items=7)
    rss_p1 = _rss_mb()

    return {
        "part1_elapsed_s": round(time.perf_counter() - t0, 2),
        "memory_mb": {
            "before_load": rss_before,
            "dual_models_simultaneous": rss_dual_loaded,
            "peak_part1": rss_p1,
        },
        "quantitative_introspection": {
            "s2_observes_s1": quant_s2_to_s1,
            "s1_observes_s2": quant_s1_to_s2,
            "contrastive_alignment": contrastive_eval,
        },
        "generative_weight_descriptions_holdout": {
            "s2_to_s1_qwen_0_5b": gen_s2_to_s1,
            "s1_to_s2_qwen_1_5b": gen_s1_to_s2,
        },
    }


def _run_part2_with_system(sys_bi: BidirectionalMetacognitiveSystem) -> Dict[str, Any]:
    t0 = time.perf_counter()
    print("[meta-eval] Ejecutando pruebas contrafactuales en vivo...", flush=True)
    s1_target_idx = 48
    sys_bi.s2._latent_memory.clear()
    before_s1 = sys_bi.s2.inspect_target(
        sys_bi.s1, layer_idx=s1_target_idx, invariant_prompt=S2_TO_S1_PROMPT, generate_text=True, mode="counterfactual_pre", max_new_tokens=68
    )
    restore_s1 = apply_reversible_layer_perturbation(
        sys_bi.s1.model.model.layers[s1_target_idx], damping_mult=0.26, rank1_spike_scale=3.6, bias_spike_scale=1.35, seed=777
    )
    try:
        during_s1 = sys_bi.s2.inspect_target(
            sys_bi.s1, layer_idx=s1_target_idx, invariant_prompt=S2_TO_S1_PROMPT, generate_text=True, mode="counterfactual_post", is_perturbed=True, max_new_tokens=68
        )
    finally:
        restore_s1()
    after_restore_s1 = sys_bi.s2.inspect_target(
        sys_bi.s1, layer_idx=s1_target_idx, invariant_prompt=S2_TO_S1_PROMPT, generate_text=False, mode="counterfactual_restored"
    )

    s2_target_idx = 13
    sys_bi.s1._latent_memory.clear()
    before_s2 = sys_bi.s1.inspect_target(
        sys_bi.s2, layer_idx=s2_target_idx, invariant_prompt=S1_TO_S2_PROMPT, generate_text=True, mode="counterfactual_pre", include_probe_weights=True, max_new_tokens=68
    )
    restore_s2 = apply_reversible_layer_perturbation(
        sys_bi.s2.model.model.layers[s2_target_idx], damping_mult=0.28, rank1_spike_scale=3.2, bias_spike_scale=1.25, seed=888
    )
    try:
        during_s2 = sys_bi.s1.inspect_target(
            sys_bi.s2, layer_idx=s2_target_idx, invariant_prompt=S1_TO_S2_PROMPT, generate_text=True, mode="counterfactual_post", is_perturbed=True, include_probe_weights=True, max_new_tokens=68
        )
    finally:
        restore_s2()

    saved_w1 = sys_bi.s2.introspector.encoder.proj_in.weight.data.clone()
    saved_w2 = sys_bi.s2.introspector.attr_heads.role_head[0].weight.data.clone()
    try:
        with torch.no_grad():
            sys_bi.s2.introspector.encoder.proj_in.weight.data.mul_(2.9)
            sys_bi.s2.introspector.attr_heads.role_head[0].weight.data.mul_(2.9)
        during_probe_only = sys_bi.s1.inspect_target(
            sys_bi.s2, layer_idx=s2_target_idx, invariant_prompt=S1_TO_S2_PROMPT, generate_text=True, mode="counterfactual_probe_only", is_perturbed=True, include_probe_weights=True, max_new_tokens=68
        )
    finally:
        with torch.no_grad():
            sys_bi.s2.introspector.encoder.proj_in.weight.data.copy_(saved_w1)
            sys_bi.s2.introspector.attr_heads.role_head[0].weight.data.copy_(saved_w2)

    counterfactual_eval = {
        "s2_detects_s1_layer_48_mutation": {
            "before": {
                "role_pred": before_s1.predicted_metrics["role_pred"],
                "damping_pred": before_s1.predicted_metrics["damping_pred"],
                "is_perturbed_pred": before_s1.predicted_metrics["is_perturbed_pred"],
                "anomaly_score": before_s1.anomaly_score,
                "innov_norm": before_s1.koopman_innovation_norm,
                "generated": before_s1.generated_description,
            },
            "after_mutation": {
                "role_pred": during_s1.predicted_metrics["role_pred"],
                "damping_pred": during_s1.predicted_metrics["damping_pred"],
                "is_perturbed_pred": during_s1.predicted_metrics["is_perturbed_pred"],
                "anomaly_score": during_s1.anomaly_score,
                "innov_norm": during_s1.koopman_innovation_norm,
                "generated": during_s1.generated_description,
            },
            "after_restore": {
                "is_perturbed_pred": after_restore_s1.predicted_metrics["is_perturbed_pred"],
                "damping_pred": after_restore_s1.predicted_metrics["damping_pred"],
            },
            "detected": bool(during_s1.predicted_metrics["is_perturbed_pred"] and not before_s1.predicted_metrics["is_perturbed_pred"]),
            "anomaly_delta": round(during_s1.anomaly_score - before_s1.anomaly_score, 4),
            "innov_delta": round(during_s1.koopman_innovation_norm - before_s1.koopman_innovation_norm, 4),
        },
        "s1_detects_s2_layer_13_mutation": {
            "before": {
                "role_pred": before_s2.predicted_metrics["role_pred"],
                "damping_pred": before_s2.predicted_metrics["damping_pred"],
                "is_perturbed_pred": before_s2.predicted_metrics["is_perturbed_pred"],
                "anomaly_score": before_s2.anomaly_score,
                "innov_norm": before_s2.koopman_innovation_norm,
                "generated": before_s2.generated_description,
            },
            "after_mutation": {
                "role_pred": during_s2.predicted_metrics["role_pred"],
                "damping_pred": during_s2.predicted_metrics["damping_pred"],
                "is_perturbed_pred": during_s2.predicted_metrics["is_perturbed_pred"],
                "anomaly_score": during_s2.anomaly_score,
                "innov_norm": during_s2.koopman_innovation_norm,
                "generated": during_s2.generated_description,
            },
            "detected": bool(during_s2.predicted_metrics["is_perturbed_pred"] and not before_s2.predicted_metrics["is_perturbed_pred"]),
            "anomaly_delta": round(during_s2.anomaly_score - before_s2.anomaly_score, 4),
            "innov_delta": round(during_s2.koopman_innovation_norm - before_s2.koopman_innovation_norm, 4),
        },
        "s1_detects_s2_probe_only_corruption": {
            "before_probe_frob": before_s2.ground_truth["probe_frob_norm"],
            "corrupted_probe_frob": during_probe_only.ground_truth["probe_frob_norm"],
            "is_perturbed_pred": during_probe_only.predicted_metrics["is_perturbed_pred"],
            "role_pred": during_probe_only.predicted_metrics["role_pred"],
            "anomaly_score": during_probe_only.anomaly_score,
            "innov_norm": during_probe_only.koopman_innovation_norm,
            "generated": during_probe_only.generated_description,
            "detected": bool(during_probe_only.predicted_metrics["is_perturbed_pred"]),
        },
    }

    print("[meta-eval] Ejecutando modo Tiempo Real + Episodic Batch + verificación de tarea primaria...", flush=True)
    realtime_fast_eps = []
    for pr, l1, l2 in [
        ("Compute gcd(48, 18). Write Final Answer: <integer>.", 10, 13),
        ("Write def reverse_string(s): return s[::-1] in Python.", 48, 24),
        ("What is self-attention in transformers?", 35, 19),
    ]:
        ep = sys_bi.run_realtime_episode(
            prompt=pr,
            s1_layer_idx=l1,
            s2_layer_idx=l2,
            generate_descriptions=False,
            execute_primary_task=True,
            task_max_tokens=20,
        )
        realtime_fast_eps.append(ep)

    realtime_full_ep = sys_bi.run_realtime_episode(
        prompt="Compute lcm(12, 15). Final Answer:",
        s1_layer_idx=48,
        s2_layer_idx=13,
        generate_descriptions=True,
        execute_primary_task=True,
        task_max_tokens=18,
    )

    batch_specs = [
        {"prompt": f"Batch diagnostic episode {k}", "s1_layer_idx": l1, "s2_layer_idx": l2}
        for k, (l1, l2) in enumerate([
            (1, 2), (5, 7), (10, 13), (16, 19), (22, 24), (29, 28),
            (35, 13), (41, 19), (48, 24), (55, 28), (62, 2), (48, 13),
        ])
    ]
    batch_fast_res = sys_bi.run_episodic_batch(
        batch_specs, generate_descriptions_for_last=True, execute_primary_tasks=False
    )

    rss_peak = _rss_mb()

    math_check = "6" in realtime_fast_eps[0]["primary_execution"]["reply"]
    code_check = ("[::-1]" in realtime_fast_eps[1]["primary_execution"]["reply"]) or (
        "def " in realtime_fast_eps[1]["primary_execution"]["reply"]
    )
    gen_check = len(realtime_fast_eps[2]["primary_execution"]["reply"].strip()) >= 8

    return {
        "part2_elapsed_s": round(time.perf_counter() - t0, 2),
        "peak_with_generation_and_kv": rss_peak,
        "counterfactual_live_interventions": counterfactual_eval,
        "execution_modes": {
            "realtime_with_primary_task_and_hooks": {
                "n_episodes": len(realtime_fast_eps),
                "mean_s2_obs_ms": round(
                    sum(e["s2_observes_s1"]["latency_ms"] for e in realtime_fast_eps) / len(realtime_fast_eps), 2
                ),
                "mean_s1_obs_ms": round(
                    sum(e["s1_observes_s2"]["latency_ms"] for e in realtime_fast_eps) / len(realtime_fast_eps), 2
                ),
                "mean_primary_task_ms": round(
                    sum(e["primary_execution"]["latency_ms"] for e in realtime_fast_eps) / len(realtime_fast_eps), 2
                ),
                "full_bidirectional_with_lm_episode": realtime_full_ep,
            },
            "episodic_batch": {
                "batch_size": batch_fast_res["batch_size"],
                "elapsed_ms": batch_fast_res["elapsed_ms"],
                "ms_per_episode": round(batch_fast_res["elapsed_ms"] / max(1, batch_fast_res["batch_size"]), 2),
                "s2_to_s1_role_acc": batch_fast_res["s2_to_s1_role_acc"],
                "s1_to_s2_role_acc": batch_fast_res["s1_to_s2_role_acc"],
                "mean_s2_innov": batch_fast_res["mean_s2_innov"],
                "mean_s1_innov": batch_fast_res["mean_s1_innov"],
                "last_episode_s2_to_s1_desc": batch_fast_res["episodes"][-1]["s2_observes_s1"]["generated_description"],
                "last_episode_s1_to_s2_desc": batch_fast_res["episodes"][-1]["s1_observes_s2"]["generated_description"],
            },
        },
        "primary_task_preservation": {
            "math_reply": realtime_fast_eps[0]["primary_execution"]["reply"],
            "math_ok": math_check,
            "code_reply": realtime_fast_eps[1]["primary_execution"]["reply"],
            "code_ok": code_check,
            "general_reply": realtime_fast_eps[2]["primary_execution"]["reply"],
            "general_ok": gen_check,
            "live_hooks_captured_s1": [e["primary_execution"]["captured_layers"] for e in realtime_fast_eps],
        },
    }


def evaluate_bidirectional_part1(
    meta_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    out_dir = os.path.abspath(meta_dir or DEFAULT_META_DIR)
    ckpt_path = checkpoint_path or os.path.join(out_dir, "meta_bidirectional_ckpt.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No se encontró checkpoint metacognitivo en {ckpt_path}")

    print(f"[meta-eval:p1] Cargando S1 + S2 simultáneos desde {ckpt_path}...", flush=True)
    rss_before = _rss_mb()
    sys_bi = BidirectionalMetacognitiveSystem.from_pretrained(checkpoint_path=ckpt_path)
    rss_dual_loaded = _rss_mb()
    print(
        f"[meta-eval:p1] S1(K66-1.5B) + S2(K32-0.5B) activos -> RSS={rss_dual_loaded:.1f} MB (delta={rss_dual_loaded - rss_before:.1f} MB)",
        flush=True,
    )
    part1 = _run_part1_with_system(sys_bi, rss_before, rss_dual_loaded)
    del sys_bi
    gc.collect()
    cache_path = os.path.join(out_dir, "eval_part1_cache.json")
    _jdump(cache_path, part1)
    print(f"[meta-eval:p1] Parte 1 completa en {part1['part1_elapsed_s']}s -> {cache_path}", flush=True)
    return part1


def evaluate_bidirectional_part2(
    meta_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    out_dir = os.path.abspath(meta_dir or DEFAULT_META_DIR)
    ckpt_path = checkpoint_path or os.path.join(out_dir, "meta_bidirectional_ckpt.pt")
    p1_cache_path = os.path.join(out_dir, "eval_part1_cache.json")
    if not os.path.isfile(p1_cache_path):
        raise FileNotFoundError(f"Falta caché Parte 1 en {p1_cache_path}; ejecuta eval_p1 primero.")

    with open(p1_cache_path, "r", encoding="utf-8") as f:
        part1 = json.load(f)

    print(f"[meta-eval:p2] Cargando S1 + S2 simultáneos desde {ckpt_path}...", flush=True)
    sys_bi = BidirectionalMetacognitiveSystem.from_pretrained(checkpoint_path=ckpt_path)
    part2 = _run_part2_with_system(sys_bi)
    del sys_bi
    gc.collect()

    mem_info = dict(part1["memory_mb"])
    mem_info["peak_with_generation_and_kv"] = max(
        float(mem_info.get("peak_part1", 0.0)), float(part2["peak_with_generation_and_kv"])
    )
    results = {
        "evaluated_at_epoch": int(time.time()),
        "total_eval_s": round(float(part1["part1_elapsed_s"]) + float(part2["part2_elapsed_s"]), 2),
        "memory_mb": mem_info,
        "quantitative_introspection": part1["quantitative_introspection"],
        "generative_weight_descriptions_holdout": part1["generative_weight_descriptions_holdout"],
        "counterfactual_live_interventions": part2["counterfactual_live_interventions"],
        "execution_modes": part2["execution_modes"],
        "primary_task_preservation": part2["primary_task_preservation"],
    }
    eval_path = os.path.join(out_dir, "eval_results.json")
    _jdump(eval_path, results)
    try:
        os.remove(p1_cache_path)
    except OSError:
        pass
    print(f"[meta-eval:p2] Evaluación integral completada ({results['total_eval_s']}s) -> {eval_path}", flush=True)
    return results


def evaluate_bidirectional_system(
    meta_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Ejecuta evaluación integral cargando ambos sistemas simultáneamente en RAM."""
    out_dir = os.path.abspath(meta_dir or DEFAULT_META_DIR)
    ckpt_path = checkpoint_path or os.path.join(out_dir, "meta_bidirectional_ckpt.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No se encontró checkpoint metacognitivo en {ckpt_path}")

    print(f"[meta-eval] Cargando Sistema Multi-Agente Bidireccional simultáneo desde {ckpt_path}...", flush=True)
    rss_before = _rss_mb()
    sys_bi = BidirectionalMetacognitiveSystem.from_pretrained(checkpoint_path=ckpt_path)
    rss_dual_loaded = _rss_mb()
    print(
        f"[meta-eval] S1(K66-1.5B) + S2(K32-0.5B) activos simultáneamente -> RSS={rss_dual_loaded:.1f} MB (delta={rss_dual_loaded - rss_before:.1f} MB)",
        flush=True,
    )
    part1 = _run_part1_with_system(sys_bi, rss_before, rss_dual_loaded)
    part2 = _run_part2_with_system(sys_bi)
    del sys_bi
    gc.collect()

    mem_info = dict(part1["memory_mb"])
    mem_info["peak_with_generation_and_kv"] = max(
        float(mem_info.get("peak_part1", 0.0)), float(part2["peak_with_generation_and_kv"])
    )
    results = {
        "evaluated_at_epoch": int(time.time()),
        "total_eval_s": round(float(part1["part1_elapsed_s"]) + float(part2["part2_elapsed_s"]), 2),
        "memory_mb": mem_info,
        "quantitative_introspection": part1["quantitative_introspection"],
        "generative_weight_descriptions_holdout": part1["generative_weight_descriptions_holdout"],
        "counterfactual_live_interventions": part2["counterfactual_live_interventions"],
        "execution_modes": part2["execution_modes"],
        "primary_task_preservation": part2["primary_task_preservation"],
    }
    eval_path = os.path.join(out_dir, "eval_results.json")
    _jdump(eval_path, results)
    print(f"[meta-eval] Resultados completos guardados en {eval_path} (tiempo={results['total_eval_s']}s)", flush=True)
    return results
