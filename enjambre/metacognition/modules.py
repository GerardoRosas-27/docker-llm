"""Módulos Neuronales Entrenables de Introspección y Metacognición Bidireccional.

Componentes principales:
1. WeightLatentEncoder: Proyecta el vector espectral de pesos [96] de una capa observada
   al espacio latente metacognitivo común [256].
2. AttributePredictionHeads: Cabezales multi-tarea que predicen en <1 ms el rol funcional,
   factor de amortiguamiento (damping), norma de Frobenius, rango estable, magnitud de sesgo,
   estado de edición/perturbación y salud espectral únicamente desde tensores de peso.
3. ManifoldSoftPromptProjector: Traduce el latente metacognitivo [256] a `K_prefix` tokens
   virtuales continuos anclados en el manifold exacto de embeddings del Transformer observador,
   permitiendo que el LLM genere lenguaje natural fundamentado directamente sobre los pesos.
4. MetaKoopmanMonitorBridge: Envoltura sobre `ConstelacionV4Bridge` (bloques 2x2 de Koopman
   K(ρ, ω) + precisión Π̂) para rastrear la evolución temporal episodio a episodio o en
   tiempo real y emitir alertas de innovación causal ante derivas o mutaciones de pesos.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from enjambre.math_code.neuro_connectors import ConstelacionV4Bridge
from .spectrometer import ROLE_LABELS, SPECTRAL_VECTOR_DIM


class WeightLatentEncoder(nn.Module):
    """Codificador residual desde descriptores espectrales reales [96] al espacio común [256]."""

    def __init__(self, in_dim: int = SPECTRAL_VECTOR_DIM, latent_dim: int = 256, dropout: float = 0.05):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.in_norm = nn.LayerNorm(in_dim)
        self.proj_in = nn.Linear(in_dim, latent_dim)
        self.block1 = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.out_norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.proj_in(self.in_norm(x)))
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.out_norm(h)


class AttributePredictionHeads(nn.Module):
    """Cabezales cuantitativos y de diagnóstico sobre el estado espectral de la capa observada."""

    def __init__(self, latent_dim: int = 256, n_roles: int = len(ROLE_LABELS)):
        super().__init__()
        self.role_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, n_roles),
        )
        self.scalar_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, 5),  # [damping, log_frob_down, log_stable_rank, bias_norm, health]
        )
        self.binary_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),   # [is_edited_logit, is_perturbed_logit]
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        role_logits = self.role_head(z)
        raw_scalars = self.scalar_head(z)
        damping = 1.5 * torch.sigmoid(raw_scalars[..., 0])
        log_frob_down = F.softplus(raw_scalars[..., 1])
        log_stable_rank = F.softplus(raw_scalars[..., 2])
        bias_norm = F.softplus(raw_scalars[..., 3])
        health = torch.sigmoid(raw_scalars[..., 4])
        bin_logits = self.binary_head(z)
        return {
            "role_logits": role_logits,
            "damping": damping,
            "log_frob_down": log_frob_down,
            "log_stable_rank": log_stable_rank,
            "bias_norm": bias_norm,
            "health": health,
            "edited_logit": bin_logits[..., 0],
            "perturbed_logit": bin_logits[..., 1],
        }


class ManifoldSoftPromptProjector(nn.Module):
    """Proyecta el estado latente + atributos neuronales predichos a `k_prefix` tokens continuos.

    Combina una distribución de atención sobre `anchor_bank` (embeddings reales del sub-vocabulario
    del Transformer congelado, registrados como buffer en el manifold exacto) con una perturbación
    residual acotada (`delta_proj`) optimizada extremo a extremo contra la pérdida Causal LM.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_lm: int,
        k_prefix: int = 38,
        n_anchors: int = 24,
        init_anchors: Optional[torch.Tensor] = None,
        embed_std: float = 0.02,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_lm = hidden_lm
        self.k_prefix = k_prefix
        self.embed_std = float(embed_std)
        self.cond_dim = int(cond_dim if cond_dim is not None else (latent_dim + len(ROLE_LABELS) + 7))

        if init_anchors is not None:
            anc = init_anchors.detach().float().cpu()
            self.n_anchors = int(anc.shape[0])
            self.register_buffer("anchor_bank", anc.clone())
        else:
            self.n_anchors = n_anchors
            self.register_buffer("anchor_bank", torch.randn(n_anchors, hidden_lm) * embed_std)

        self.cond_proj = nn.Sequential(
            nn.Linear(self.cond_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(384, 384),
            nn.GELU(),
        )
        self.anchor_gate = nn.Sequential(
            nn.Linear(384, 384),
            nn.GELU(),
            nn.Linear(384, k_prefix * self.n_anchors),
        )
        self.delta_proj = nn.Sequential(
            nn.Linear(384, 384),
            nn.GELU(),
            nn.Linear(384, k_prefix * hidden_lm),
        )
        nn.init.normal_(self.delta_proj[-1].weight, std=1e-3)
        nn.init.zeros_(self.delta_proj[-1].bias)
        self.res_scale = nn.Parameter(torch.tensor(1.0))

    def _build_cond(self, z: torch.Tensor, attrs: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        if attrs is None:
            extra_dim = self.cond_dim - z.shape[-1]
            if extra_dim > 0:
                pad = torch.zeros(z.size(0), extra_dim, dtype=z.dtype, device=z.device)
                return torch.cat([z, pad], dim=-1)
            return z[..., : self.cond_dim]

        role_p = F.softmax(attrs["role_logits"], dim=-1)
        sc = torch.stack(
            [
                attrs["damping"],
                attrs["log_frob_down"] / 5.0,
                attrs["log_stable_rank"] / 5.0,
                attrs["bias_norm"],
                attrs["health"],
                torch.sigmoid(attrs["edited_logit"]),
                torch.sigmoid(attrs["perturbed_logit"]),
            ],
            dim=-1,
        )
        return torch.cat([z, role_p * 3.0, sc * 3.0], dim=-1)

    def compute_alignment_logits(
        self,
        z: torch.Tensor,
        attrs: Optional[Dict[str, torch.Tensor]] = None,
        feat_noise_std: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cond = self._build_cond(z, attrs)
        if feat_noise_std > 0.0 and self.training:
            cond = cond + feat_noise_std * torch.randn_like(cond)
        h = self.cond_proj(cond)
        logits = self.anchor_gate(h).view(z.size(0), self.k_prefix, self.n_anchors)
        return logits, h

    def forward(
        self,
        z: torch.Tensor,
        attrs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B = z.size(0)
        logits, h = self.compute_alignment_logits(z, attrs=attrs, feat_noise_std=0.0)
        weights = F.softmax(logits, dim=-1)                       # [B, K, A]
        base = torch.matmul(weights, self.anchor_bank)            # [B, K, D]
        delta = torch.tanh(self.delta_proj(h)).view(B, self.k_prefix, self.hidden_lm)
        return base + self.res_scale * self.embed_std * delta


class MetaKoopmanMonitorBridge(nn.Module):
    """Monitor dinámico temporal de episodios basado en `ConstelacionV4Bridge`.

    Mantiene estabilidad espectral (bloques 2x2 de Koopman con precisión normalizada Π̂) entre
    el estado latente anterior de metacognición y el estado actual observado de la capa/episodio.
    """

    def __init__(self, latent_dim: int = 256, rank: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.bridge = ConstelacionV4Bridge(hidden=latent_dim, rank=rank, gate_bias=-1.5)
        self.anomaly_head = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, z_prev: torch.Tensor, z_curr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retorna: (z_tracked [B, D], innov_norm [B], anomaly_prob [B])."""
        hp = z_prev.unsqueeze(1) if z_prev.dim() == 2 else z_prev
        hc = z_curr.unsqueeze(1) if z_curr.dim() == 2 else z_curr
        out = self.bridge(hp, hc).squeeze(1)
        innov = out - (z_curr if z_curr.dim() == 2 else z_curr.squeeze(1))
        innov_norm = torch.norm(innov, dim=-1)
        pair_feat = torch.cat([z_curr if z_curr.dim() == 2 else z_curr.squeeze(1), innov], dim=-1)
        anomaly_prob = torch.sigmoid(self.anomaly_head(pair_feat)).squeeze(-1)
        return out, innov_norm, anomaly_prob


class WeightIntrospectionModule(nn.Module):
    """Módulo neuronal completo de introspección de pesos para un agente (S1 o S2)."""

    def __init__(
        self,
        hidden_lm: int,
        in_dim: int = SPECTRAL_VECTOR_DIM,
        latent_dim: int = 256,
        k_prefix: int = 38,
        n_anchors: int = 24,
        init_anchors: Optional[torch.Tensor] = None,
        embed_std: float = 0.02,
        bank_ids: Optional[List[int]] = None,
    ):
        super().__init__()
        self.hidden_lm = hidden_lm
        self.latent_dim = latent_dim
        self.k_prefix = k_prefix
        self.bank_ids: List[int] = list(bank_ids) if bank_ids is not None else []

        self.encoder = WeightLatentEncoder(in_dim=in_dim, latent_dim=latent_dim)
        self.attr_heads = AttributePredictionHeads(latent_dim=latent_dim)
        self.koopman_monitor = MetaKoopmanMonitorBridge(latent_dim=latent_dim, rank=16)
        self.soft_prompt = ManifoldSoftPromptProjector(
            latent_dim=latent_dim,
            hidden_lm=hidden_lm,
            k_prefix=k_prefix,
            n_anchors=n_anchors,
            init_anchors=init_anchors,
            embed_std=embed_std,
        )
        # Proyección contrastiva para alineación recíproca simétrica S1 <-> S2
        self.contrastive_proj = nn.Linear(latent_dim, 128)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode_and_predict(
        self,
        spec_vec: torch.Tensor,
        prev_latent: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if spec_vec.dim() == 1:
            spec_vec = spec_vec.unsqueeze(0)
        z = self.encoder(spec_vec)
        attrs = self.attr_heads(z)
        z_prev = z.detach() if prev_latent is None else prev_latent
        if z_prev.dim() == 1:
            z_prev = z_prev.unsqueeze(0).expand_as(z)
        z_tracked, innov_norm, anomaly_prob = self.koopman_monitor(z_prev, z)
        prefix_embeds = self.soft_prompt(z_tracked, attrs=attrs)
        contrastive_emb = F.normalize(self.contrastive_proj(z), dim=-1)
        return {
            "z": z,
            "z_tracked": z_tracked,
            "innov_norm": innov_norm,
            "anomaly_prob": anomaly_prob,
            "prefix_embeds": prefix_embeds,
            "contrastive": contrastive_emb,
            **attrs,
        }


def build_metacognitive_subvocab(model: nn.Module, tok: Any) -> Tuple[List[int], torch.Tensor, float]:
    """Construye el sub-vocabulario determinista de tokens reales del Transformer observador."""
    emb = model.get_input_embeddings().weight.detach().float()
    embed_std = float(emb.std().item())

    words = (
        ROLE_LABELS
        + [f" {r}" for r in ROLE_LABELS]
        + ["base_nativa", "editada", "editada_perturbada", " base_nativa", " editada", " editada_perturbada"]
        + ["estable", "alerta_moderada", "deriva_espectral", " estable", " alerta_moderada", " deriva_espectral"]
        + [" | ", " |", "| ", "|", ".", ",", "-", ":", ";", " "]
        + [str(d) for d in range(10)]
        + [f" {d}" for d in range(10)]
        + [f"{d}." for d in range(10)]
        + [f"0.{d}" for d in range(10)]
        + [f"1.{d}" for d in range(6)]
    )
    bank_ids: List[int] = []
    seen = set()
    for w in words:
        for tid in tok.encode(w, add_special_tokens=False):
            if tid not in seen:
                seen.add(tid)
                bank_ids.append(tid)
    # Garantizar cobertura de todo dígito o separador presente en payloads compactos
    for tid in tok.encode("0123456789 . | base_nativa editada_perturbada deriva_espectral", add_special_tokens=False):
        if tid not in seen:
            seen.add(tid)
            bank_ids.append(tid)
    eos_id = getattr(tok, "eos_token_id", None)
    if eos_id is not None and eos_id not in seen:
        seen.add(eos_id)
        bank_ids.append(int(eos_id))

    anc = emb[bank_ids].cpu().clone()
    return bank_ids, anc, embed_std


def build_vocab_anchors(model: nn.Module, tok: Any, n_anchors: int = 24) -> Tuple[torch.Tensor, float]:
    """Extrae embeddings iniciales reales del vocabulario metacognitivo del Transformer."""
    _, anc, embed_std = build_metacognitive_subvocab(model, tok)
    if n_anchors is not None and anc.shape[0] > n_anchors and n_anchors <= 24:
        return anc[:n_anchors].clone(), embed_std
    return anc, embed_std

