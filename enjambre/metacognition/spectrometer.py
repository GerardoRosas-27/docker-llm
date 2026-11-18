"""Extractor Espectral y Estructural de Pesos en Vivo (LayerWeightSpectrometer).

Permite:
1. Envolver proyecciones lineales con amortiguamiento por compartición virtual de memoria
   (`DampedLinear`), preservando equivalencia algebraica exacta con `mul_(damping)` sin
   duplicar gigabytes de RAM al operar dos modelos Transformer simultáneamente.
2. Inspeccionar en vivo los tensores reales de pesos de cualquier capa Transformer (`Qwen2DecoderLayer`),
   macro-neurona (`K32`/`K66`), módulo metacognitivo o puente del campo (`ConstelacionV4Bridge`),
   calculando invariantes espectrales, normas de Frobenius, rango estable, entropía espectral,
   esparsidad, alineación direccional con vectores de edición (`BiasEdit`) y un bosquejo
   ortogonal determinista Johnson-Lindenstrauss (`D_spec = 96`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SPECTRAL_INVARIANT_DIM = 32
JL_SKETCH_DIM = 64
SPECTRAL_VECTOR_DIM = SPECTRAL_INVARIANT_DIM + JL_SKETCH_DIM  # 96

ROLE_LABELS: List[str] = [
    "early_syntax",          # 0: capas base iniciales (0..6)
    "math_cluster",          # 1: clusters algebra / euclides / modular / combinatoria
    "code_cluster",          # 2: clusters dicts / recursion / algoritmos
    "general_epistemic",     # 3: clusters fisica / biologia / historia / filosofia
    "ai_causal_cluster",     # 4: clusters analisis de IA & deduccion causal
    "synthesis_head",        # 5: nucleo semantico & sintesis superior
    "meta_observer_core",    # 6: capas del segundo sistema observador metacognitivo
    "meta_bridge_koopman",   # 7: puentes ConstelacionV4 / MetaKoopman
    "perturbed_anomaly",     # 8: capa con intervencion anomala / deriva espectral
]
ROLE_TO_ID: Dict[str, int] = {r: i for i, r in enumerate(ROLE_LABELS)}


class DampedLinear(nn.Module):
    """Capa lineal con compartición virtual de pesos y factor de amortiguamiento escalar.

    Algebraicamente idéntica a `W_eff = damping * base_weight` con sesgo independiente `bias`,
    pero sin clonar el tensor `[out_features, in_features]` en memoria principal.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        damping: float = 1.0,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.base_weight = base_linear.weight  # referencia compartida sin copia RAM
        self.damping = float(damping)
        if bias is not None:
            self.bias = nn.Parameter(
                bias.detach().to(dtype=base_linear.weight.dtype, device=base_linear.weight.device).clone(),
                requires_grad=False,
            )
        elif base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

    @property
    def weight(self) -> torch.Tensor:
        if self.damping == 1.0:
            return self.base_weight
        return self.base_weight * self.damping

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.base_weight, bias=None)
        if self.damping != 1.0:
            out = out * self.damping
        if self.bias is not None:
            out = out + self.bias
        return out


@dataclass
class LayerWeightSnapshot:
    """Telemetría cuantitativa y vector espectral extraído de una capa viva."""

    system_id: str                 # "S1_principal" | "S2_observer"
    layer_idx: int
    n_total_layers: int
    module_type: str               # "decoder_layer" | "meta_bridge" | "observer_with_probe"
    role_label: str
    role_id: int
    damping: float
    frob_down: float
    frob_o: float
    frob_up: float
    frob_qkv: float
    bias_norm: float
    spectral_norm: float
    stable_rank: float
    spectral_entropy: float
    sparsity: float
    kurtosis: float
    domain_cosines: Dict[str, float]
    runtime_act_norm: float
    runtime_delta_norm: float
    runtime_cos_io: float
    probe_frob_norm: float
    is_edited: bool
    is_perturbed: bool
    health_score: float
    spectral_vector: torch.Tensor  # [96] float32
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_report_dict(self) -> Dict[str, Any]:
        return {
            "system_id": self.system_id,
            "layer_idx": self.layer_idx,
            "module_type": self.module_type,
            "role_label": self.role_label,
            "damping": round(self.damping, 4),
            "frob_down": round(self.frob_down, 3),
            "frob_o": round(self.frob_o, 3),
            "frob_up": round(self.frob_up, 3),
            "bias_norm": round(self.bias_norm, 4),
            "spectral_norm": round(self.spectral_norm, 4),
            "stable_rank": round(self.stable_rank, 2),
            "spectral_entropy": round(self.spectral_entropy, 4),
            "sparsity": round(self.sparsity, 4),
            "probe_frob_norm": round(self.probe_frob_norm, 3),
            "runtime_act_norm": round(self.runtime_act_norm, 3),
            "runtime_delta_norm": round(self.runtime_delta_norm, 3),
            "is_edited": self.is_edited,
            "is_perturbed": self.is_perturbed,
            "health_score": round(self.health_score, 4),
            "domain_cosines": {k: round(v, 4) for k, v in self.domain_cosines.items()},
        }

    def canonical_description(self) -> str:
        """Descripción textual canónica determinista anclada en las métricas reales del tensor."""
        estado = "editada_perturbada" if self.is_perturbed else ("editada" if self.is_edited else "base_nativa")
        salud = "deriva_espectral" if self.health_score < 0.65 else ("alerta_moderada" if self.health_score < 0.85 else "estable")
        return (
            f"Rol: {self.role_label}; Estado: {estado}; "
            f"Damping: {self.damping:.2f}; NormaF: {self.frob_down:.1f}; "
            f"RangoEspectral: {self.stable_rank:.1f}; Bias: {self.bias_norm:.2f}; "
            f"Salud: {salud}."
        )

    def compact_payload(self) -> str:
        """Representación semántica compacta (~34 tokens) para proyección de soft-tokens."""
        estado = "editada_perturbada" if self.is_perturbed else ("editada" if self.is_edited else "base_nativa")
        salud = "deriva_espectral" if self.health_score < 0.65 else ("alerta_moderada" if self.health_score < 0.85 else "estable")
        return (
            f"{self.role_label} | {estado} | {self.damping:.2f} | "
            f"{self.frob_down:.1f} | {self.stable_rank:.1f} | {self.bias_norm:.2f} | {salud}"
        )


def _extract_effective_weight_and_bias(mod: nn.Module) -> Tuple[torch.Tensor, float, Optional[torch.Tensor]]:
    """Retorna (base_weight, damping_scale, bias) sin duplicar tensores grandes en RAM."""
    if isinstance(mod, DampedLinear):
        return mod.base_weight, float(mod.damping), mod.bias
    return mod.weight, 1.0, getattr(mod, "bias", None)


@torch.no_grad()
def _power_top_singular_values(
    W: torch.Tensor,
    damping: float = 1.0,
    k: int = 4,
    n_iter: int = 8,
    seed: int = 7,
) -> Tuple[float, torch.Tensor, float, float, float, float]:
    """Calcula rápidamente valores singulares dominantes vía iteración por bloques sin SVD completo.

    Retorna: (frob_norm, top_k_sv [k], stable_rank, spectral_entropy, sparsity, kurtosis).
    Opera por bloques para mantener RAM mínima aun en matrices grandes (p. ej. [1536, 8960]).
    """
    Wf = W.detach().float()
    rows, cols = Wf.shape
    frob_base = float(torch.linalg.norm(Wf).item())
    frob = frob_base * abs(damping)

    g = torch.Generator(device=Wf.device).manual_seed(seed)
    kk = min(k + 2, min(rows, cols))
    # Sketch aleatorio para iteración de subespacio (Halko-Martinsson-Tropp rápido)
    if rows <= cols:
        Q = torch.randn(rows, kk, generator=g, device=Wf.device, dtype=torch.float32)
        Q, _ = torch.linalg.qr(Q, mode="reduced")
        for _ in range(n_iter):
            Z = Wf.t() @ Q          # [cols, kk]
            Q_new = Wf @ Z          # [rows, kk]
            Q, _ = torch.linalg.qr(Q_new, mode="reduced")
        B = Q.t() @ Wf              # [kk, cols]
    else:
        Q = torch.randn(cols, kk, generator=g, device=Wf.device, dtype=torch.float32)
        Q, _ = torch.linalg.qr(Q, mode="reduced")
        for _ in range(n_iter):
            Z = Wf @ Q              # [rows, kk]
            Q_new = Wf.t() @ Z      # [cols, kk]
            Q, _ = torch.linalg.qr(Q_new, mode="reduced")
        B = Wf @ Q                  # [rows, kk]

    svs = torch.linalg.svdvals(B) * abs(damping)
    svs_top = svs[:k]
    if svs_top.numel() < k:
        svs_top = F.pad(svs_top, (0, k - svs_top.numel()))

    sigma1 = float(svs_top[0].item()) if svs_top.numel() > 0 else 0.0
    stable_rank = (frob * frob) / max(sigma1 * sigma1, 1e-8)

    p = svs_top.clamp_min(1e-8)
    p = p / p.sum()
    entropy = float(-(p * torch.log(p)).sum().item() / math.log(max(2, len(p))))

    # Muestreo determinista para esparsidad & curtosis sin recorrer gigabytes por completo
    flat = Wf.reshape(-1)
    step = max(1, flat.numel() // 16384)
    samp = flat[::step] * damping
    std_s = float(samp.std().item()) + 1e-8
    sparsity = float((samp.abs() < (0.05 * std_s)).float().mean().item())
    mean_s = float(samp.mean().item())
    kurt = float((((samp - mean_s) / std_s) ** 4).mean().item())
    return frob, svs_top.cpu(), float(stable_rank), float(entropy), float(sparsity), float(kurt)


class LayerWeightSpectrometer:
    """Computa descriptores espectrales de tamaño fijo `[96]` sobre pesos reales de cualquier capa."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._jl_left: Dict[Tuple[int, int], torch.Tensor] = {}
        self._jl_right: Dict[Tuple[int, int], torch.Tensor] = {}
        self._domain_basis: Dict[int, Dict[str, torch.Tensor]] = {}

    def _get_jl_pair(self, rows: int, cols: int, out_r: int = 8, out_c: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """Genera proyecciones deterministas Johnson-Lindenstrauss (8x8 = 64 dims) para cualquier tamaño matricial."""
        kl = (rows, out_r)
        if kl not in self._jl_left:
            g = torch.Generator().manual_seed(self.seed + rows * 31 + out_r)
            L = torch.randn(out_r, rows, generator=g, dtype=torch.float32) / math.sqrt(rows)
            self._jl_left[kl] = L
        kr = (cols, out_c)
        if kr not in self._jl_right:
            g = torch.Generator().manual_seed(self.seed + cols * 97 + out_c)
            R = torch.randn(cols, out_c, generator=g, dtype=torch.float32) / math.sqrt(cols)
            self._jl_right[kr] = R
        return self._jl_left[kl], self._jl_right[kr]

    def _get_domain_vectors(self, hidden_size: int) -> Dict[str, torch.Tensor]:
        """Subespacios direccionales canónicos ortogonalizados por Gram-Schmidt para medir sesgos."""
        if hidden_size not in self._domain_basis:
            g = torch.Generator().manual_seed(42)
            # Base compatible con las direcciones usadas en network.py (v_ai, v_causal, etc.)
            names = ["ai_analysis", "causal_logic", "math", "code", "meta_observer"]
            basis: List[torch.Tensor] = []
            mapping: Dict[str, torch.Tensor] = {}
            for name in names:
                v = torch.randn(hidden_size, generator=g, dtype=torch.float32)
                for b in basis:
                    v = v - torch.dot(v, b) * b
                v = v / (v.norm() + 1e-8)
                basis.append(v)
                mapping[name] = v
            self._domain_basis[hidden_size] = mapping
        return self._domain_basis[hidden_size]

    @torch.no_grad()
    def _compute_jl_sketch(self, W: torch.Tensor, damping: float = 1.0) -> torch.Tensor:
        Wf = W.detach().float()
        rows, cols = Wf.shape
        L, R = self._get_jl_pair(rows, cols, out_r=8, out_c=8)
        L = L.to(Wf.device)
        R = R.to(Wf.device)
        sketch = (L @ Wf @ R) * damping  # [8, 8]
        return sketch.reshape(-1).cpu()   # [64]

    @torch.no_grad()
    def inspect_decoder_layer(
        self,
        layer: nn.Module,
        layer_idx: int,
        n_total_layers: int,
        system_id: str,
        role_label: str,
        is_edited: Optional[bool] = None,
        is_perturbed: bool = False,
        runtime_stats: Optional[Dict[str, float]] = None,
        probe_module: Optional[nn.Module] = None,
    ) -> LayerWeightSnapshot:
        """Extrae espectro, normas, alineaciones y sketch real de una capa Qwen2DecoderLayer / Macro-Neurona.

        Nota de diseño sin fuga de etiquetas (`zero label leakage`):
        `spectral_vector` contiene EXCLUSIVAMENTE observables físicos medidos sobre los
        tensores reales (pesos, sesgos, telemetría de hooks y sonda) — nunca etiquetas objetivo
        como `is_perturbed`, `is_edited` o `health`.
        """
        W_down, damp_down, b_down = _extract_effective_weight_and_bias(layer.mlp.down_proj)
        W_o, damp_o, _ = _extract_effective_weight_and_bias(layer.self_attn.o_proj)
        W_up, damp_up, _ = _extract_effective_weight_and_bias(layer.mlp.up_proj)
        W_q, damp_q, _ = _extract_effective_weight_and_bias(layer.self_attn.q_proj)
        W_k, damp_k, _ = _extract_effective_weight_and_bias(layer.self_attn.k_proj)
        W_v, damp_v, _ = _extract_effective_weight_and_bias(layer.self_attn.v_proj)

        damping_calc = damp_down
        effective_damping_gt = float(getattr(layer.mlp.down_proj, "effective_damping", damp_down))
        frob_down, svs_down, sr_down, ent_down, spars_down, kurt_down = _power_top_singular_values(
            W_down, damping=damping_calc, k=4, n_iter=6, seed=self.seed + layer_idx
        )
        frob_o = float(torch.linalg.norm(W_o.detach().float()).item()) * abs(damp_o)
        frob_up = float(torch.linalg.norm(W_up.detach().float()).item()) * abs(damp_up)
        frob_qkv = (
            float(torch.linalg.norm(W_q.detach().float()).item()) * abs(damp_q)
            + float(torch.linalg.norm(W_k.detach().float()).item()) * abs(damp_k)
            + float(torch.linalg.norm(W_v.detach().float()).item()) * abs(damp_v)
        )

        # Estadística de cola / factor de cresta sobre W_down físico
        flat_w = W_down.detach().float().reshape(-1)
        step = max(1, flat_w.numel() // 16384)
        samp = flat_w[::step] * damping_calc
        rms_s = float(torch.sqrt((samp * samp).mean()).item()) + 1e-8
        crest_factor = float(samp.abs().max().item()) / rms_s

        # Normas & alineaciones del sesgo (BiasEdit)
        hidden_size = W_down.shape[0]
        if b_down is not None:
            bf = b_down.detach().float().cpu()
            bias_norm = float(bf.norm().item())
        else:
            bf = torch.zeros(hidden_size, dtype=torch.float32)
            bias_norm = 0.0

        dom_vecs = self._get_domain_vectors(hidden_size)
        domain_cosines: Dict[str, float] = {}
        if bias_norm > 1e-6:
            bu = bf / bias_norm
            for dname, dvec in dom_vecs.items():
                domain_cosines[dname] = float(torch.dot(bu, dvec).item())
        else:
            for dname in dom_vecs:
                domain_cosines[dname] = 0.0

        # Detección de módulos observadores activos (cuando S1 inspecciona una capa de S2 con su sonda)
        probe_frob = 0.0
        probe_spec = 0.0
        if probe_module is not None:
            core_submods = [
                getattr(probe_module, attr)
                for attr in ("encoder", "attr_heads", "koopman_monitor", "contrastive_proj")
                if hasattr(probe_module, attr)
            ]
            if core_submods:
                params_iter = [p for sm in core_submods for p in sm.parameters() if p.numel() > 0]
            else:
                params_iter = [p for p in probe_module.parameters() if p.numel() > 0]
            p_norms = [float(p.detach().float().norm().item()) for p in params_iter]
            probe_frob = math.sqrt(sum(n * n for n in p_norms)) if p_norms else 0.0
            probe_spec = max(p_norms) if p_norms else 0.0

        # Estadísticas en tiempo real de activación si fueron capturadas por forward hooks
        rstats = runtime_stats or {}
        act_norm = float(rstats.get("act_norm", 0.0))
        delta_norm = float(rstats.get("delta_norm", 0.0))
        cos_io = float(rstats.get("cos_io", 1.0))

        # Determinar etiquetas ground-truth únicamente a efectos de evaluación/supervisión
        if is_edited is None:
            is_edited = (abs(effective_damping_gt - 1.0) > 1e-3) or (bias_norm > 1e-3) or (probe_frob > 0.0)

        ratio_o_up = frob_o / max(frob_up, 1e-6)
        ratio_down_up = frob_down / max(frob_up, 1e-6)
        ratio_o_qkv = frob_o / max(frob_qkv, 1e-6)

        sigma1 = float(svs_down[0].item())
        sv_conc = sigma1 / max(float(svs_down[3].item()), 1e-6)
        # Indicador de salud ground-truth derivado de los invariantes físicos y condición de perturbación
        health = 1.0
        if is_perturbed:
            health = max(0.15, 0.55 - 0.12 * abs(effective_damping_gt - 0.5))
        elif effective_damping_gt < 0.15 or effective_damping_gt > 1.05:
            health = 0.45
        elif sr_down < 4.0:
            health = 0.60
        elif kurt_down > 18.0:
            health = 0.72

        # Construir vector espectral de 32 observables físicos (SIN etiquetas objetivo)
        inv = torch.tensor([
            layer_idx / max(1, n_total_layers - 1),
            1.0 if system_id == "S1_principal" else 0.0,
            ratio_o_qkv * 5.0,
            math.log1p(frob_down) / 5.0,
            math.log1p(frob_o) / 5.0,
            math.log1p(frob_up) / 5.0,
            math.log1p(frob_qkv) / 5.0,
            ratio_o_up,
            ratio_down_up,
            math.log1p(bias_norm),
            math.log1p(sigma1) / 3.0,
            float(svs_down[1].item()) / max(sigma1, 1e-6),
            float(svs_down[2].item()) / max(sigma1, 1e-6),
            float(svs_down[3].item()) / max(sigma1, 1e-6),
            math.log1p(sr_down) / 6.0,
            ent_down,
            spars_down,
            min(kurt_down, 50.0) / 25.0,
            domain_cosines.get("ai_analysis", 0.0),
            domain_cosines.get("causal_logic", 0.0),
            domain_cosines.get("math", 0.0),
            domain_cosines.get("code", 0.0),
            domain_cosines.get("meta_observer", 0.0),
            math.log1p(act_norm) / 5.0,
            math.log1p(delta_norm) / 5.0,
            cos_io,
            math.log1p(probe_frob) / 4.0,
            math.log1p(probe_spec) / 3.0,
            min(sv_conc, 30.0) / 15.0,
            min(crest_factor, 40.0) / 20.0,
            math.log1p(bias_norm / max(frob_down, 1e-6) * 100.0),
            float(hidden_size) / 2048.0,
        ], dtype=torch.float32)

        sketch = self._compute_jl_sketch(W_down, damping=damping_calc)
        sketch_scaled = torch.tanh(sketch * 0.5)
        vec = torch.cat([inv, sketch_scaled], dim=0)

        mod_type = "observer_with_probe" if probe_module is not None else "decoder_layer"
        effective_role = "perturbed_anomaly" if is_perturbed else role_label
        return LayerWeightSnapshot(
            system_id=system_id,
            layer_idx=layer_idx,
            n_total_layers=n_total_layers,
            module_type=mod_type,
            role_label=effective_role,
            role_id=ROLE_TO_ID.get(effective_role, 0),
            damping=float(effective_damping_gt),
            frob_down=frob_down,
            frob_o=frob_o,
            frob_up=frob_up,
            frob_qkv=frob_qkv,
            bias_norm=bias_norm,
            spectral_norm=sigma1,
            stable_rank=sr_down,
            spectral_entropy=ent_down,
            sparsity=spars_down,
            kurtosis=kurt_down,
            domain_cosines=domain_cosines,
            runtime_act_norm=act_norm,
            runtime_delta_norm=delta_norm,
            runtime_cos_io=cos_io,
            probe_frob_norm=probe_frob,
            is_edited=bool(is_edited),
            is_perturbed=bool(is_perturbed),
            health_score=float(health),
            spectral_vector=vec,
            meta={"hidden_size": hidden_size, "svs_top": [round(float(v), 4) for v in svs_down], "crest": round(crest_factor, 2)},
        )

    @torch.no_grad()
    def inspect_bridge(
        self,
        bridge: nn.Module,
        bridge_idx: int,
        n_bridges: int,
        system_id: str = "S1_principal",
        role_label: str = "meta_bridge_koopman",
        is_perturbed: bool = False,
    ) -> LayerWeightSnapshot:
        """Extrae descriptor espectral real de un puente ConstelacionV4Bridge (Koopman 2x2 + Π̂)."""
        W_up = bridge.up.weight.detach().float()
        W_down = bridge.down.weight.detach().float()
        W_nd = bridge.neuro_down.weight.detach().float() if hasattr(bridge, "neuro_down") else torch.zeros_like(W_down)

        frob_down, svs_down, sr_down, ent_down, spars_down, kurt_down = _power_top_singular_values(
            W_up, damping=1.0, k=4, n_iter=5, seed=self.seed + 500 + bridge_idx
        )
        frob_up = float(torch.linalg.norm(W_up).item())
        frob_o = float(torch.linalg.norm(W_down).item())
        frob_nd = float(torch.linalg.norm(W_nd).item())

        rho_mean = float(bridge.rho_raw.detach().float().mean().item()) if hasattr(bridge, "rho_raw") else 0.95
        rho_max = float(bridge.rho_raw.detach().float().max().item()) if hasattr(bridge, "rho_raw") else 0.95
        omega_std = float(bridge.omega.detach().float().std().item()) if hasattr(bridge, "omega") else 0.1
        gate_b = float(bridge.gate.bias.detach().float().mean().item()) if hasattr(bridge, "gate") else -4.0
        kappa = float(bridge.kappa.detach().float().item()) if hasattr(bridge, "kappa") else 0.25

        health = 0.45 if (is_perturbed or rho_mean > 1.04) else 0.96
        sigma1 = float(svs_down[0].item())
        sv_conc = sigma1 / max(float(svs_down[3].item()), 1e-6)

        inv = torch.tensor([
            bridge_idx / max(1, n_bridges - 1),
            1.0 if system_id == "S1_principal" else 0.0,
            rho_mean,
            math.log1p(frob_down + frob_nd) / 5.0,
            math.log1p(frob_o) / 5.0,
            math.log1p(frob_up) / 5.0,
            math.log1p(frob_nd) / 5.0,
            frob_o / max(frob_up, 1e-6),
            (frob_down + frob_nd) / max(frob_up, 1e-6),
            abs(gate_b) / 5.0,
            math.log1p(sigma1) / 3.0,
            float(svs_down[1].item()) / max(sigma1, 1e-6),
            float(svs_down[2].item()) / max(sigma1, 1e-6),
            float(svs_down[3].item()) / max(sigma1, 1e-6),
            math.log1p(sr_down) / 6.0,
            ent_down,
            spars_down,
            min(kurt_down, 50.0) / 25.0,
            omega_std,
            kappa,
            0.0,
            0.0,
            1.0 if system_id == "S2_observer" else 0.0,
            0.0,
            0.0,
            1.0,
            math.log1p(frob_o + frob_nd) / 4.0,
            rho_max,
            min(sv_conc, 30.0) / 15.0,
            abs(rho_mean - 0.95) * 4.0,
            abs(kappa) * 2.0,
            float(W_up.shape[1]) / 2048.0,
        ], dtype=torch.float32)

        sketch = torch.tanh(self._compute_jl_sketch(W_up, damping=1.0) * 0.5)
        vec = torch.cat([inv, sketch], dim=0)

        effective_role = "perturbed_anomaly" if is_perturbed else role_label
        return LayerWeightSnapshot(
            system_id=system_id,
            layer_idx=bridge_idx,
            n_total_layers=n_bridges,
            module_type="meta_bridge",
            role_label=effective_role,
            role_id=ROLE_TO_ID[effective_role],
            damping=rho_mean,
            frob_down=frob_up + frob_o + frob_nd,
            frob_o=frob_o,
            frob_up=frob_up,
            frob_qkv=0.0,
            bias_norm=abs(gate_b),
            spectral_norm=sigma1,
            stable_rank=sr_down,
            spectral_entropy=ent_down,
            sparsity=spars_down,
            kurtosis=kurt_down,
            domain_cosines={"omega_std": omega_std, "kappa": kappa},
            runtime_act_norm=0.0,
            runtime_delta_norm=0.0,
            runtime_cos_io=1.0,
            probe_frob_norm=frob_o + frob_nd,
            is_edited=True,
            is_perturbed=is_perturbed,
            health_score=health,
            spectral_vector=vec,
            meta={"rho_mean": rho_mean, "src_tgt": f"{getattr(bridge, 'src_layer', -1)}_{getattr(bridge, 'tgt_layer', -1)}"},
        )


def apply_reversible_layer_perturbation(
    layer: nn.Module,
    damping_mult: float = 0.25,
    rank1_spike_scale: float = 2.0,
    bias_spike_scale: float = 0.8,
    seed: int = 123,
) -> Callable[[], None]:
    """Aplica una mutación física real y reversible sobre `mlp.down_proj` y `self_attn.o_proj`
    de una única capa sin contaminar tensores compartidos por capas hermanas.

    Retorna una función `restore()` para revertir el estado exacto y liberar RAM.
    """
    orig_down = layer.mlp.down_proj
    orig_o = layer.self_attn.o_proj

    W_d_base, damp_d, b_d = _extract_effective_weight_and_bias(orig_down)
    W_o_base, damp_o, b_o = _extract_effective_weight_and_bias(orig_o)

    g = torch.Generator().manual_seed(seed)
    rows_d, cols_d = W_d_base.shape
    new_damp_d = float(damp_d * damping_mult)
    new_damp_o = float(damp_o * damping_mult)

    # Construir peso denso temporal mutado con inyección de pico espectral rango-1
    u = torch.randn(rows_d, 1, generator=g, dtype=torch.float32)
    u = u / (u.norm() + 1e-8)
    v = torch.randn(1, cols_d, generator=g, dtype=torch.float32)
    v = v / (v.norm() + 1e-8)
    W_mut = (W_d_base.detach().float() * new_damp_d + rank1_spike_scale * (u @ v)).to(
        dtype=W_d_base.dtype, device=W_d_base.device
    )

    bias_mut = torch.zeros(rows_d, dtype=W_d_base.dtype, device=W_d_base.device)
    if b_d is not None:
        bias_mut = bias_mut + b_d.detach()
    if bias_spike_scale != 0.0:
        b_dir = torch.randn(rows_d, generator=g, dtype=torch.float32)
        b_dir = b_dir / (b_dir.norm() + 1e-8)
        bias_mut = bias_mut + (bias_spike_scale * b_dir).to(dtype=W_d_base.dtype, device=W_d_base.device)

    mut_down = nn.Linear(cols_d, rows_d, bias=True, dtype=W_d_base.dtype)
    mut_down.weight = nn.Parameter(W_mut, requires_grad=False)
    mut_down.bias = nn.Parameter(bias_mut, requires_grad=False)
    mut_down.effective_damping = new_damp_d  # metadato para evaluación ground-truth (no entra al vector espectral)

    # Para o_proj usamos DampedLinear con el nuevo factor multiplicado (0 RAM adicional)
    mut_o = DampedLinear(
        orig_o if not isinstance(orig_o, DampedLinear) else _wrap_raw_linear_proxy(orig_o),
        damping=new_damp_o,
        bias=b_o,
    )

    layer.mlp.down_proj = mut_down
    layer.self_attn.o_proj = mut_o

    def restore() -> None:
        layer.mlp.down_proj = orig_down
        layer.self_attn.o_proj = orig_o

    return restore


def _wrap_raw_linear_proxy(damped: DampedLinear) -> nn.Linear:
    proxy = nn.Linear(
        damped.in_features,
        damped.out_features,
        bias=damped.bias is not None,
        dtype=damped.base_weight.dtype,
    )
    proxy.weight = damped.base_weight
    if damped.bias is not None:
        proxy.bias = damped.bias
    return proxy

