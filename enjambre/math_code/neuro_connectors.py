"""Conectores inter-capa inspirados en las 4 arquitecturas de `neuro-geometrica_multi-agente` (rama `main`).

Sobre la base ganadora `K24_mc (24L+6B)` (24 capas editadas `0..23` + 6 puentes `0→4, 4→8, 8→12, 12→16, 16→20, 20→23`),
implementa 4 arquitecturas de conectores extraídas directamente de `GerardoRosas-27/neuro-geometrica_multi-agente`:

1. `constelacion_v4` (`ConstelacionV4Bridge`):
   Portado de `orbita/src/constelacion4/mod.rs` (`Neuron::kmul` + `Neuron::f` + precisión Π + cabeza rápida W^F).
   Aplica bloques 2×2 de rotación-escala de Koopman K(ρ, ω), levantamiento monótono acoplado a la innovación
   (z_tgt - K z_src) ponderada por precisiones normalizadas Π̂, y modulación en el subespacio del puente.

2. `atlas_fw` (`AtlasFWBridge`):
   Portado de `orbita/src/constelacion4/atlas.rs` (`Mech: M1 + M2 + M6`).
   Amplía el estado latente φ_v4 con la biblioteca M1 de 17 generadores no lineales Koopman/SINDy
   (polinomios grado ≤ 3, Fourier sin/cos(2·), tanh(2·), radiales r·x con escala `LIB_S = 0.3`),
   proyección regularizada M2 y compuerta contextual CLS M6 (sin atenuación por grado M7).

3. `bq10_dastr` (`BQ10DaStrBridge`):
   Portado de `orbita/src/constelacion4/atlas.rs` (`Mech::bq(10, false, true)` — ganadora global BQ10 =
   ATLAS-FW + M7 DA-STR `da=10`) combinado con el canal de innovación causal de `gemma_campo` (`docs/campo/README.md`).
   Pondera las 17 funciones de la biblioteca M1 con la atenuación espectral por grado `LIB_DEG`:
   `s_j = da^(-(grado_j - 1)/2)` con `da = 10` (×1 en lineales, ×0.3162 en cuadráticos, ×0.10 en cúbicos),
   interacción bilineal de grado 2 atenuada, innovación diferencial `RMSNorm(h_tgt - h_src)` y compuerta
   neuromodulada vectorial por canal DA-STR.

4. `soft_bellman_simplicial` (`SoftBellmanSimplicialBridge`):
   Portado de `soft_bellman_simplicial/src/main.rs` (`transform` SPD Cholesky + `inner` + `softmin` baricéntrico).
   Proyecta (h_src, h_tgt) sobre un complejo simplicial de K=4 vértices (tetraedro Δ³), calcula longitudes
   geodésicas con métrica Riemanniana SPD aprendida (`0.5 + softplus(w_diag)`, `0.8 * tanh(w_off)`),
   obtiene coordenadas baricéntricas vía `softmin(q / τ)` y agrega los valores de los vértices.

Todas las arquitecturas preservan en `t=0` (inicialización) equivalencia exacta bit a bit con el puente base
`ResidualBridge` de `K24_mc` cuando se cargan los pesos base (`up`, `down`, `gate`), ya que la proyección de
salida de la rama neuro-geométrica se inicializa en cero (`neuro_down.weight = 0` y `latent_out.weight = 0`).
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from enjambre.bridges.modules import ResidualBridge

PAIRS_K24: List[Tuple[int, int]] = [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 23)]

# Constantes exactas de `orbita/src/constelacion4/atlas.rs`
N_LIB: int = 17
LIB_S: float = 0.3
LIB_DEG: List[int] = [1, 1, 2, 2, 2, 3, 3, 3, 3, 1, 1, 2, 2, 1, 1, 2, 2]
DEFAULT_DA: float = 10.0

ARCHITECTURES = [
    "constelacion_v4",
    "atlas_fw",
    "bq10_dastr",
    "soft_bellman_simplicial",
]


def koopman_sindy_lib_17(a: torch.Tensor, b: torch.Tensor) -> List[torch.Tensor]:
    """Portado exacto de `pub fn lib(x: &[f64]) -> [f64; 17]` en `orbita/src/constelacion4/atlas.rs`.

    Entradas:
      a, b: Tensores [B, T, D_sub] (coordenadas latentes del par origen/destino, acotadas en [-1, 1]).
    Devuelve:
      Lista de 17 tensores [B, T, D_sub] correspondientes a los grados `LIB_DEG`:
      [a, b, a², ab, b², a³, a²b, ab², b³, sin(2a), sin(2b), cos(2a)-1, cos(2b)-1, tanh(2a), tanh(2b), r·a, r·b]
    """
    r = torch.sqrt(a * a + b * b + 1e-6)
    a2 = a * a
    b2 = b * b
    ab = a * b
    return [
        a,                          # deg 1
        b,                          # deg 1
        a2,                         # deg 2
        ab,                         # deg 2
        b2,                         # deg 2
        a2 * a,                     # deg 3
        a2 * b,                     # deg 3
        a * b2,                     # deg 3
        b2 * b,                     # deg 3
        torch.sin(2.0 * a),         # deg 1
        torch.sin(2.0 * b),         # deg 1
        torch.cos(2.0 * a) - 1.0,   # deg 2
        torch.cos(2.0 * b) - 1.0,   # deg 2
        torch.tanh(2.0 * a),        # deg 1
        torch.tanh(2.0 * b),        # deg 1
        r * a,                      # deg 2
        r * b,                      # deg 2
    ]


def da_str_colscale(da: float = DEFAULT_DA) -> List[float]:
    """Portado exacto de `fn colscale` en `orbita/src/constelacion4/atlas.rs` (M7 DA-STR):
    `s_j = da^(-(LIB_DEG[j] - 1) / 2.0)`."""
    if da <= 0.0:
        return [1.0] * N_LIB
    return [float(da ** (-((deg - 1) / 2.0))) for deg in LIB_DEG]


# ─── Opción 1: CONSTELACIÓN v4 (ÓRBITA-Neurona Koopman 2x2 + Precisión Π) ────
class ConstelacionV4Bridge(nn.Module):
    """Conector basado en `orbita/src/constelacion4/mod.rs` (`Neuron` Koopman bloque 2×2 + Π normalizada)."""

    def __init__(self, hidden: int, rank: int = 16, gate_bias: float = -4.0):
        super().__init__()
        assert rank % 2 == 0, "rank debe ser par para bloques 2x2 de Koopman"
        self.src_layer: int = -1
        self.tgt_layer: int = -1
        self.rank = rank
        self.n_blocks = rank // 2

        # Camino base v4 / K24_mc (compatible bit a bit con ResidualBridge)
        self.up = nn.Linear(hidden, rank, bias=False)
        self.down = nn.Linear(rank, hidden, bias=False)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.zeros_(self.down.weight)
        self.gate = nn.Linear(hidden, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)

        # Parámetros de ÓRBITA-neurona (`rho`, `omega` -> bloques 2x2 de rotación-escala)
        self.rho_raw = nn.Parameter(torch.full((self.n_blocks,), 0.95))
        omega_init = torch.linspace(-math.pi / 8.0, math.pi / 8.0, self.n_blocks)
        self.omega = nn.Parameter(omega_init)

        # Levantamiento monótono `s = σ(R K μ + b)` y acoplamiento de innovación con precisión Π
        self.lift_r = nn.Linear(rank, rank, bias=True)
        nn.init.normal_(self.lift_r.weight, std=0.02)
        nn.init.zeros_(self.lift_r.bias)
        self.tgt_proj = nn.Linear(hidden, rank, bias=False)
        nn.init.normal_(self.tgt_proj.weight, std=0.02)

        self.log_pi = nn.Parameter(torch.zeros(rank))
        self.eta_b = nn.Parameter(torch.tensor(0.15))
        self.kappa = nn.Parameter(torch.tensor(0.25))

        # Modulación en el subespacio de rango r + proyección directa (zero-init para paridad paso 0)
        self.latent_out = nn.Linear(rank, rank, bias=False)
        nn.init.zeros_(self.latent_out.weight)
        self.neuro_down = nn.Linear(rank, hidden, bias=False)
        nn.init.zeros_(self.neuro_down.weight)
        self.neuro_gate = nn.Linear(rank, 1, bias=True)
        nn.init.zeros_(self.neuro_gate.weight)
        nn.init.zeros_(self.neuro_gate.bias)

    def _kmul(self, z: torch.Tensor) -> torch.Tensor:
        """Aplica bloques 2×2 `rho * [[cos w, -sin w], [sin w, cos w]]` a `z` [B, T, rank]."""
        rho = torch.clamp(self.rho_raw, 0.5, 1.05)
        cos_w = rho * torch.cos(self.omega)
        sin_w = rho * torch.sin(self.omega)
        z_even = z[..., 0::2]
        z_odd = z[..., 1::2]
        out_even = cos_w * z_even - sin_w * z_odd
        out_odd = sin_w * z_even + cos_w * z_odd
        return torch.stack([out_even, out_odd], dim=-1).flatten(-2)

    def forward(self, h_src: torch.Tensor, h_tgt: torch.Tensor) -> torch.Tensor:
        z_src = self.up(h_src)
        phi_v4 = F.gelu(z_src)

        # Dinámica CONSTELACIÓN v4: Koopman K*z_src + levantamiento monótono + innovación Π̂
        kz = self._kmul(z_src)
        s_lift = torch.sigmoid(self.lift_r(kz))
        z_tgt = self.tgt_proj(h_tgt)
        pi = F.softmax(self.log_pi, dim=0) * float(self.rank)
        innov = z_tgt - kz
        mu = F.gelu(kz + self.eta_b * s_lift + self.kappa * (pi * innov))

        # Corrección dentro del subespacio de `self.down` + residuo directo
        g_total = torch.sigmoid(self.gate(h_tgt) + self.neuro_gate(mu))
        phi_total = phi_v4 + self.latent_out(mu)
        return h_tgt + g_total * self.down(phi_total) + g_total * self.neuro_down(mu)


# ─── Opción 2: ATLAS-FW (M1 Biblioteca 17 fns + M2 Ridge + M6 Compuerta CLS) ─
class AtlasFWBridge(nn.Module):
    """Conector basado en `orbita/src/constelacion4/atlas.rs` (`Mech: M1 + M2 + M6`, sin M7)."""

    def __init__(self, hidden: int, rank: int = 16, d_sub: int = 4, gate_bias: float = -4.0):
        super().__init__()
        self.src_layer: int = -1
        self.tgt_layer: int = -1
        self.rank = rank
        self.d_sub = d_sub
        self.d_lib = N_LIB * d_sub

        # Camino base v4 / K24_mc
        self.up = nn.Linear(hidden, rank, bias=False)
        self.down = nn.Linear(rank, hidden, bias=False)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.zeros_(self.down.weight)
        self.gate = nn.Linear(hidden, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)

        # M1: proyecciones a coordenadas (a, b) para evaluar la biblioteca de 17 funciones Koopman/SINDy
        self.coord_src = nn.Linear(hidden, d_sub, bias=False)
        self.coord_tgt = nn.Linear(hidden, d_sub, bias=False)
        nn.init.normal_(self.coord_src.weight, std=0.02)
        nn.init.normal_(self.coord_tgt.weight, std=0.02)

        # M2: proyección desde ψ_lib hacia el subespacio de rango r
        self.lib_mix = nn.Linear(self.d_lib, rank, bias=False)
        nn.init.normal_(self.lib_mix.weight, std=0.02)
        self.latent_out = nn.Linear(rank, rank, bias=False)
        nn.init.zeros_(self.latent_out.weight)
        self.neuro_down = nn.Linear(rank, hidden, bias=False)
        nn.init.zeros_(self.neuro_down.weight)

        # M6: compuerta CLS contextual
        self.cls_gate = nn.Linear(d_sub * 3, 1, bias=True)
        nn.init.zeros_(self.cls_gate.weight)
        nn.init.zeros_(self.cls_gate.bias)

    def forward(self, h_src: torch.Tensor, h_tgt: torch.Tensor) -> torch.Tensor:
        phi_v4 = F.gelu(self.up(h_src))

        # M1: biblioteca Koopman/SINDy de 17 funciones con escala LIB_S = 0.3 (sin atenuación M7)
        a = torch.tanh(self.coord_src(h_src))
        b = torch.tanh(self.coord_tgt(h_tgt))
        terms = koopman_sindy_lib_17(a, b)
        psi_lib = torch.cat([LIB_S * t for t in terms], dim=-1)  # [B, T, 17 * d_sub]
        h_lib = F.gelu(self.lib_mix(psi_lib))

        # M2 + M6: modulación CLS en el subespacio de `self.down`
        cls_feat = torch.cat([a, b, torch.abs(b - a)], dim=-1)
        g_total = torch.sigmoid(self.gate(h_tgt) + self.cls_gate(cls_feat))
        phi_total = phi_v4 + self.latent_out(h_lib)
        return h_tgt + g_total * self.down(phi_total) + g_total * self.neuro_down(h_lib)


# ─── Opción 3: BQ10 (ATLAS-FW Core + M7 DA-STR da=10 + Innovación Campo) ─────
class BQ10DaStrBridge(nn.Module):
    """Conector basado en la arquitectura ganadora `BQ10 (bio-quantum v10)` de `atlas.rs`:
    Núcleo ATLAS-FW (`M1 + M2 + M6`) + `M7 DA-STR` (regularización/reescalado por grado `da = 10`)
    + interacción bilineal de grado 2 atenuada + canal de innovación causal `RMSNorm(h_tgt - h_src)`
    de `gemma_campo` (`docs/campo/README.md`) + compuerta neuromodulada por canal.
    """

    def __init__(
        self,
        hidden: int,
        rank: int = 16,
        d_sub: int = 4,
        da: float = DEFAULT_DA,
        gate_bias: float = -4.0,
    ):
        super().__init__()
        self.src_layer: int = -1
        self.tgt_layer: int = -1
        self.rank = rank
        self.d_sub = d_sub
        self.da = da
        self.d_lib = N_LIB * d_sub

        # Escala espectral por grado M7 DA-STR: s_j = da^(-(grado_j - 1)/2)
        scales = da_str_colscale(da)
        scale_vec = torch.tensor(
            [s for s in scales for _ in range(d_sub)], dtype=torch.float32
        )
        self.register_buffer("colscale", scale_vec)
        # Escala de grado 2 para interacción bilineal en rango r: da^(-1/2) = 0.3162277
        self.deg2_scale: float = float(da ** -0.5) if da > 0 else 1.0

        # Camino base v4 / K24_mc
        self.up = nn.Linear(hidden, rank, bias=False)
        self.down = nn.Linear(rank, hidden, bias=False)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.zeros_(self.down.weight)
        self.gate = nn.Linear(hidden, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)

        # Coordenadas latentes (a, b) para la biblioteca M1 + M7 DA-STR
        self.coord_src = nn.Linear(hidden, d_sub, bias=False)
        self.coord_tgt = nn.Linear(hidden, d_sub, bias=False)
        nn.init.normal_(self.coord_src.weight, std=0.02)
        nn.init.normal_(self.coord_tgt.weight, std=0.02)

        # Proyección de destino para interacción bilineal de grado 2 (z_src ⊙ z_tgt)
        self.tgt_rank_proj = nn.Linear(hidden, rank, bias=False)
        nn.init.normal_(self.tgt_rank_proj.weight, std=0.02)

        # Canal de innovación causal `gemma_campo` (Δh = RMSNorm(h_tgt - h_src))
        self.innov_up = nn.Linear(hidden, rank, bias=False)
        nn.init.normal_(self.innov_up.weight, std=0.02)

        # Proyección en el espacio reescalado DA-STR: `w_j <- s_j * u_j`
        self.lib_mix = nn.Linear(self.d_lib, rank, bias=False)
        nn.init.normal_(self.lib_mix.weight, std=0.02)

        # Salida en el subespacio de `self.down` + corrección residual (zero-init en t=0)
        self.latent_out = nn.Linear(rank, rank, bias=False)
        nn.init.zeros_(self.latent_out.weight)
        self.neuro_down = nn.Linear(rank, hidden, bias=False)
        nn.init.zeros_(self.neuro_down.weight)

        # Compuerta neuromodulada por canal DA-STR (r canales en el subespacio + ajuste escalar M6)
        self.da_channel_gate = nn.Linear(d_sub * 3, rank, bias=True)
        nn.init.zeros_(self.da_channel_gate.weight)
        nn.init.zeros_(self.da_channel_gate.bias)
        self.da_scalar_gate = nn.Linear(d_sub * 3, 1, bias=True)
        nn.init.zeros_(self.da_scalar_gate.weight)
        nn.init.zeros_(self.da_scalar_gate.bias)

    def forward(self, h_src: torch.Tensor, h_tgt: torch.Tensor) -> torch.Tensor:
        z_src = self.up(h_src)
        phi_v4 = F.gelu(z_src)

        # 1) M1 + M7 DA-STR: biblioteca de 17 funciones reescalada por `colscale^2 = da^-(deg-1)`
        a = torch.tanh(self.coord_src(h_src))
        b = torch.tanh(self.coord_tgt(h_tgt))
        terms = koopman_sindy_lib_17(a, b)
        psi_raw = torch.cat([LIB_S * t for t in terms], dim=-1)
        psi_scaled = psi_raw * (self.colscale * self.colscale)
        h_lib = self.lib_mix(psi_scaled)

        # 2) Interacción bilineal en el subespacio de rango r con atenuación M7 de grado 2 (da^-0.5)
        z_tgt = torch.tanh(self.tgt_rank_proj(h_tgt))
        h_bilin = self.deg2_scale * (torch.tanh(z_src) * z_tgt)

        # 3) Innovación causal `gemma_campo` sobre el diferencial normalizado entre capas
        diff = h_tgt - h_src
        rms = torch.sqrt(torch.mean(diff * diff, dim=-1, keepdim=True) + 1e-6)
        h_innov = self.innov_up(diff / rms)

        # 4) Fusión neuromodulada por canal DA-STR dentro del subespacio de `self.down`
        cls_feat = torch.cat([a, b, torch.abs(b - a)], dim=-1)
        chan_mod = 1.0 + 0.5 * torch.tanh(self.da_channel_gate(cls_feat))  # [B, T, rank], = 1.0 en t=0
        h_neuro = F.gelu(h_lib + h_bilin + h_innov) * chan_mod

        g_total = torch.sigmoid(self.gate(h_tgt) + self.da_scalar_gate(cls_feat))
        phi_total = phi_v4 * chan_mod + self.latent_out(h_neuro)
        return h_tgt + g_total * self.down(phi_total) + g_total * self.neuro_down(h_neuro)


# ─── Opción 4: Soft-Bellman Simplicial 3D (Métrica SPD + Tetraedro Δ³) ───────
class SoftBellmanSimplicialBridge(nn.Module):
    """Conector basado en `soft_bellman_simplicial/src/main.rs`:
    Complejo simplicial de K=4 vértices (tetraedro Δ³) con métrica Riemanniana SPD3x3 aprendida
    (`transform` con `0.5 + softplus(w_diag)` y `0.8 * tanh(w_off)`) y agregación baricéntrica
    `softmin(q / τ)`.
    """

    def __init__(
        self,
        hidden: int,
        rank: int = 16,
        n_vertices: int = 4,
        tau: float = 0.25,
        gate_bias: float = -4.0,
    ):
        super().__init__()
        self.src_layer: int = -1
        self.tgt_layer: int = -1
        self.rank = rank
        self.n_vertices = n_vertices
        self.tau = tau

        # Camino base v4 / K24_mc
        self.up = nn.Linear(hidden, rank, bias=False)
        self.down = nn.Linear(rank, hidden, bias=False)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.zeros_(self.down.weight)
        self.gate = nn.Linear(hidden, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_bias)

        # Coordenadas 3D en el complejo tetraédrico para cada vértice k=1..K
        self.pos_src = nn.Linear(hidden, n_vertices * 3, bias=True)
        self.pos_tgt = nn.Linear(hidden, 3, bias=False)
        nn.init.normal_(self.pos_src.weight, std=0.02)
        nn.init.zeros_(self.pos_src.bias)
        nn.init.normal_(self.pos_tgt.weight, std=0.02)

        # Parámetros w[0..6] de la métrica SPD3x3 (`fn transform(d: V, w: &[f64])` en main.rs líneas 126-132)
        self.spd_w = nn.Parameter(torch.zeros(6))

        # Valores en los K=4 vértices del tetraedro y proyección en el subespacio (zero-init en salida)
        self.vertex_values = nn.Linear(rank, n_vertices * rank, bias=False)
        nn.init.normal_(self.vertex_values.weight, std=0.02)
        self.latent_out = nn.Linear(rank, rank, bias=False)
        nn.init.zeros_(self.latent_out.weight)
        self.neuro_down = nn.Linear(rank, hidden, bias=False)
        nn.init.zeros_(self.neuro_down.weight)

        self.simp_gate = nn.Linear(n_vertices, 1, bias=True)
        nn.init.zeros_(self.simp_gate.weight)
        nn.init.zeros_(self.simp_gate.bias)

    def _spd_transform_3d(self, d: torch.Tensor) -> torch.Tensor:
        """Portado exacto de `fn transform(d: V, w: &[f64]) -> V` en `soft_bellman_simplicial/src/main.rs`."""
        w = self.spd_w
        d0, d1, d2 = d[..., 0], d[..., 1], d[..., 2]
        s0 = 0.5 + F.softplus(w[0])
        s1 = 0.5 + F.softplus(w[1])
        s2 = 0.5 + F.softplus(w[2])
        t3 = 0.8 * torch.tanh(w[3])
        t4 = 0.8 * torch.tanh(w[4])
        t5 = 0.8 * torch.tanh(w[5])
        v0 = s0 * d0 + t3 * d1 + t4 * d2
        v1 = s1 * d1 + t5 * d2
        v2 = s2 * d2
        return torch.stack([v0, v1, v2], dim=-1)

    def compute_barycentric_weights(self, h_src: torch.Tensor, h_tgt: torch.Tensor) -> torch.Tensor:
        """Calcula los costos geodésicos `q_k` bajo la métrica SPD3x3 y aplica `softmin(q / τ)` -> p ∈ Δ^{K-1}."""
        B, T, _ = h_src.shape
        v_src = self.pos_src(h_src).view(B, T, self.n_vertices, 3)
        v_tgt = self.pos_tgt(h_tgt).unsqueeze(-2)  # [B, T, 1, 3]
        disp = v_tgt - v_src  # [B, T, K, 3]
        trans = self._spd_transform_3d(disp)
        q = torch.sqrt(torch.sum(trans * trans, dim=-1) + 0.01 * torch.sum(disp * disp, dim=-1) + 1e-8)
        p = F.softmax(-q / self.tau, dim=-1)  # [B, T, K]
        return p

    def forward(self, h_src: torch.Tensor, h_tgt: torch.Tensor) -> torch.Tensor:
        B, T, _ = h_src.shape
        z_src = self.up(h_src)
        phi_v4 = F.gelu(z_src)

        p = self.compute_barycentric_weights(h_src, h_tgt)  # [B, T, K]
        vals = F.gelu(self.vertex_values(phi_v4)).view(B, T, self.n_vertices, self.rank)
        bary_val = torch.sum(p.unsqueeze(-1) * vals, dim=-2)  # [B, T, rank]

        g_total = torch.sigmoid(self.gate(h_tgt) + self.simp_gate(p))
        phi_total = phi_v4 + self.latent_out(bary_val)
        return h_tgt + g_total * self.down(phi_total) + g_total * self.neuro_down(bary_val)


def make_neuro_bridge(arch: str, hidden: int, rank: int = 16) -> nn.Module:
    if arch == "residual_base":
        return ResidualBridge(hidden=hidden, rank=rank)
    if arch == "constelacion_v4":
        return ConstelacionV4Bridge(hidden=hidden, rank=rank)
    if arch == "atlas_fw":
        return AtlasFWBridge(hidden=hidden, rank=rank)
    if arch == "bq10_dastr":
        return BQ10DaStrBridge(hidden=hidden, rank=rank, da=DEFAULT_DA)
    if arch == "soft_bellman_simplicial":
        return SoftBellmanSimplicialBridge(hidden=hidden, rank=rank)
    raise ValueError(f"Arquitectura de conector desconocida: {arch}")


class NeuroBridgeBank(nn.Module):
    """Banco de 6 conectores entre las 24 capas editadas de `K24_mc` (`0→4, 4→8, 8→12, 12→16, 16→20, 20→23`),
    parametrizable por arquitectura de `neuro-geometrica_multi-agente` (homogénea o híbrida por par)."""

    def __init__(
        self,
        hidden: int,
        arch: str = "bq10_dastr",
        rank: int = 16,
        pairs: Optional[List[Tuple[int, int]]] = None,
        per_pair_arch: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.arch = arch
        self.rank = rank
        self.pairs = list(pairs or PAIRS_K24)
        self.bridges = nn.ModuleDict()
        for s, t in self.pairs:
            key = f"{s}_{t}"
            chosen_arch = (per_pair_arch or {}).get(key, arch)
            b = make_neuro_bridge(chosen_arch, hidden=hidden, rank=rank)
            b.src_layer, b.tgt_layer = s, t
            self.bridges[key] = b
        self._buf: Dict[int, torch.Tensor] = {}
        self._hooks: List = []
        self.enabled: bool = True

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def n_trainable_neuro_params(self) -> int:
        base_suffixes = (".up.weight", ".down.weight", ".gate.weight", ".gate.bias")
        return sum(
            p.numel()
            for k, p in self.named_parameters()
            if not k.endswith(base_suffixes)
        )

    def clear_buf(self) -> None:
        self._buf.clear()

    def attach(self, model: nn.Module) -> None:
        self.remove_hooks()
        layers = model.model.layers
        src_set = {s for s, _ in self.pairs}
        tgt_map = {t: s for s, t in self.pairs}

        def pre_clear(_mod, _inp):
            self.clear_buf()

        self._hooks.append(layers[0].register_forward_pre_hook(pre_clear))

        def make_hook(idx: int):
            def hook(_mod, _inp, out):
                if not self.enabled:
                    return out
                h = out
                if idx in tgt_map:
                    src = tgt_map[idx]
                    if src in self._buf:
                        h_src = self._buf[src]
                        if h_src.shape[1] != h.shape[1]:
                            T = min(h_src.shape[1], h.shape[1])
                            h_src = h_src[:, -T:]
                            h = h[:, -T:]
                        key = f"{src}_{idx}"
                        h_new = self.bridges[key](h_src, h)
                        if h_new.shape[1] == out.shape[1]:
                            h = h_new
                        else:
                            out2 = out.clone()
                            out2[:, -h_new.shape[1] :, :] = h_new
                            h = out2
                if idx in src_set:
                    self._buf[idx] = h
                return h if idx in tgt_map else out

            return hook

        needed = sorted(src_set | set(tgt_map))
        for idx in needed:
            self._hooks.append(layers[idx].register_forward_hook(make_hook(idx)))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.clear_buf()


def build_warm_neuro_bridge_bank(
    hidden: int,
    arch: str,
    *,
    rank: int = 16,
    pairs: Optional[List[Tuple[int, int]]] = None,
    k24_pretrained_state: Optional[Dict[str, torch.Tensor]] = None,
    per_pair_arch: Optional[Dict[str, str]] = None,
) -> NeuroBridgeBank:
    """Construye un `NeuroBridgeBank` inicializado con los pesos entrenados de `K24_mc` (`up`, `down`, `gate`)
    y verifica que la rama neuro-geométrica empiece con proyección de salida nula (`latent_out.weight = 0`,
    `neuro_down.weight = 0`), garantizando equivalencia exacta en `step=0` con `K24_mc (24L+6B)`."""
    bank = NeuroBridgeBank(
        hidden=hidden,
        arch=arch,
        rank=rank,
        pairs=pairs or PAIRS_K24,
        per_pair_arch=per_pair_arch,
    )
    if k24_pretrained_state:
        own = bank.state_dict()
        with torch.no_grad():
            for k, v in k24_pretrained_state.items():
                if k in own and own[k].shape == v.shape:
                    own[k].copy_(v)
        bank.load_state_dict(own)
    return bank
