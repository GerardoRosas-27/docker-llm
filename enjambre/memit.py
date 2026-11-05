"""Memoria por edición de pesos: ROME (rank-1 con covarianza) y MEMIT (batch multi-capa).

Ambos editan mlp.down_proj (W: d_model x d_ff). Claves k = entrada de down_proj en el
último token del sujeto. Covarianza C = E[k k^T] estimada con una muestra pequeña de texto.

ROME (un hecho, capa l):   ΔW = (v* - W k) (C^-1 k)^T / (k^T C^-1 k)
MEMIT (n hechos, capas R): para cada l en R (en orden):
    resid_i = (z_i - h_i^{L})/(|R| - j)   ;  ΔW_l = Resid K^T (λ C + K K^T)^-1
z_i se obtiene optimizando un δ sobre la salida del bloque L=max(R) (Adam, solo δ).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .generate import chat_text


@dataclass
class LowRankEdit:
    layer: int
    U: torch.Tensor   # d_out x r
    V: torch.Tensor   # r x d_in      ΔW = U @ V
    kind: str = "lowrank"


@dataclass
class FactSpec:
    subject: str
    obj: str
    train_prompts: List[str]     # usados para optimizar z y promediar k
    para_prompts: List[str]      # paráfrasis held-out (no vistas)
    target_tpl: str = "The capital of {s} is {o}."


# ---------------- helpers ----------------

def _layers(model):
    return model.model.layers


def _subject_last_pos(tok, ids: List[int], subject: str) -> int:
    for variant in (" " + subject, subject):
        sub = tok(variant, add_special_tokens=False).input_ids
        n = len(sub)
        for i in range(len(ids) - n, -1, -1):
            if ids[i:i + n] == sub:
                return i + n - 1
    # fallback: buscar último token que decodifica dentro del sujeto
    for i in range(len(ids) - 1, -1, -1):
        piece = tok.decode([ids[i]]).strip()
        if piece and piece in subject:
            return i
    return len(ids) - 1


class _Cap:
    def __init__(self):
        self.v = None


@torch.no_grad()
def compute_cov(model, tok, texts: Sequence[str], layer: int, max_len: int = 128) -> Tuple[torch.Tensor, int]:
    lin = _layers(model)[layer].mlp.down_proj
    d = lin.in_features
    C = torch.zeros(d, d, dtype=torch.float64)
    n = 0
    cap = _Cap()
    h = lin.register_forward_hook(lambda m, i, o: setattr(cap, "v", i[0].detach()))
    try:
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids
            model(ids)
            k = cap.v[0].double()
            C += k.T @ k
            n += k.shape[0]
    finally:
        h.remove()
    return (C / max(1, n)).float(), n


@torch.no_grad()
def keys_at_subject(model, tok, prompts: List[str], subject: str, layer: int) -> torch.Tensor:
    lin = _layers(model)[layer].mlp.down_proj
    cap = _Cap()
    h = lin.register_forward_hook(lambda m, i, o: setattr(cap, "v", i[0].detach()))
    ks = []
    try:
        for p in prompts:
            ids = tok(p, return_tensors="pt").input_ids
            pos = _subject_last_pos(tok, ids[0].tolist(), subject)
            model(ids)
            ks.append(cap.v[0, pos].float())
    finally:
        h.remove()
    return torch.stack(ks).mean(0)


@torch.no_grad()
def hidden_at_subject(model, tok, prompts: List[str], subject: str, layer: int) -> torch.Tensor:
    """Salida del bloque `layer` (= hidden_states[layer+1]) en el último token del sujeto."""
    hs = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids
        pos = _subject_last_pos(tok, ids[0].tolist(), subject)
        out = model(ids, output_hidden_states=True)
        hs.append(out.hidden_states[layer + 1][0, pos].float())
    return torch.stack(hs).mean(0)


def _full_prompts(tok, fact: FactSpec, prompts: List[str]) -> List[str]:
    return [chat_text(tok, p.format(s=fact.subject)) for p in prompts]


def optimize_delta(
    model, tok, fact: FactSpec, layer: int, where: str, steps: int = 25, lr: float = 0.5,
    clamp_factor: float = 4.0, wd: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimiza δ sumado en el último token del sujeto a la salida de
    where='mlp' (down_proj de `layer`, ROME) o where='block' (bloque `layer`, MEMIT).
    Devuelve (δ*, valor_base en esa posición) promediados sobre prompts de entrenamiento."""
    prompts = _full_prompts(tok, fact, fact.train_prompts)
    target = fact.target_tpl.format(s=fact.subject, o=fact.obj)
    batch = []
    for p in prompts:
        q = tok(p, return_tensors="pt").input_ids[0]
        full = tok(p + target, return_tensors="pt").input_ids[0]
        pos = _subject_last_pos(tok, q.tolist(), fact.subject)
        batch.append((full, len(q), pos))
    for p_ in model.parameters():
        p_.requires_grad_(False)
    mod = _layers(model)[layer].mlp.down_proj if where == "mlp" else _layers(model)[layer]
    # norma de referencia
    base_vals = []
    cur = {"pos": 0, "delta": None}

    def hook(_m, _i, out):
        is_t = isinstance(out, tuple)
        h = out[0] if is_t else out
        if cur["delta"] is None:
            base_vals.append(h[0, cur["pos"]].detach().float().clone())
            return out
        h = h.clone()
        h[0, cur["pos"]] = h[0, cur["pos"]] + cur["delta"]
        return (h,) + tuple(out[1:]) if is_t else h

    handle = mod.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for full, ql, pos in batch:
                cur["pos"] = pos
                model(full[None])
        base = torch.stack(base_vals).mean(0)
        ref = base.norm().item() + 1e-6
        delta = nn.Parameter(torch.zeros_like(base))
        opt = torch.optim.Adam([delta], lr=lr * ref / 50.0)
        cur["delta"] = delta
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            for full, ql, pos in batch:
                cur["pos"] = pos
                logits = model(full[None]).logits[0]
                loss = loss + nn.functional.cross_entropy(logits[ql - 1:-1], full[ql:])
            loss = loss / len(batch) + wd * (delta.norm() / ref) ** 2
            loss.backward()
            opt.step()
            with torch.no_grad():
                mx = clamp_factor * ref
                if delta.norm() > mx:
                    delta.mul_(mx / delta.norm())
    finally:
        handle.remove()
    return delta.detach().clone(), base


def _apply_lowrank(model, e: LowRankEdit, sign: float = 1.0):
    W = _layers(model)[e.layer].mlp.down_proj.weight
    with torch.no_grad():
        W.add_(sign * (e.U @ e.V).to(W.dtype))


class WeightMemory:
    """Guarda ediciones low-rank; aplicar/revertir exacto (resta del mismo ΔW)."""

    def __init__(self, model, tok, cov_texts: Sequence[str], rome_layer: int, memit_layers: List[int],
                 memit_lambda: float = 2.0, cov_eps: float = 1e-2):
        self.model, self.tok = model, tok
        self.cov_texts = list(cov_texts)
        self.rome_layer = rome_layer
        self.memit_layers = memit_layers
        self.lam = memit_lambda
        self.cov_eps = cov_eps
        self._cov: Dict[int, torch.Tensor] = {}
        self.cov_tokens = 0
        self.applied: List[LowRankEdit] = []
        self.facts: List[FactSpec] = []
        self.method: Optional[str] = None

    def cov(self, layer: int) -> torch.Tensor:
        if layer not in self._cov:
            C, n = compute_cov(self.model, self.tok, self.cov_texts, layer)
            self.cov_tokens = n
            self._cov[layer] = C
        return self._cov[layer]

    def clear(self):
        for e in reversed(self.applied):
            _apply_lowrank(self.model, e, -1.0)
        self.applied.clear()
        self.facts.clear()
        self.method = None

    # ---------- ROME ----------
    def write_rome(self, fact: FactSpec, layer: Optional[int] = None) -> LowRankEdit:
        L = self.rome_layer if layer is None else layer
        prompts = _full_prompts(self.tok, fact, fact.train_prompts)
        k = keys_at_subject(self.model, self.tok, prompts, fact.subject, L)
        delta, _ = optimize_delta(self.model, self.tok, fact, L, "mlp")
        C = self.cov(L)
        Cr = C + self.cov_eps * C.diagonal().mean() * torch.eye(C.shape[0])
        u = torch.linalg.solve(Cr, k)
        u = u / (k @ u)
        e = LowRankEdit(layer=L, U=delta[:, None], V=u[None, :])
        _apply_lowrank(self.model, e)
        self.applied.append(e)
        self.facts.append(fact)
        self.method = "rome"
        return e

    # ---------- MEMIT ----------
    def write_memit(self, facts: List[FactSpec]) -> List[LowRankEdit]:
        R = self.memit_layers
        Lz = R[-1]
        zs, subj_prompts = [], []
        for f in facts:
            d, base = optimize_delta(self.model, self.tok, f, Lz, "block")
            zs.append(base + d)
            subj_prompts.append(_full_prompts(self.tok, f, f.train_prompts))
        Z = torch.stack(zs)                                    # n x d_model
        edits = []
        for j, L in enumerate(R):
            K = torch.stack([keys_at_subject(self.model, self.tok, ps, f.subject, L)
                             for f, ps in zip(facts, subj_prompts)])        # n x d_ff
            H = torch.stack([hidden_at_subject(self.model, self.tok, ps, f.subject, Lz)
                             for f, ps in zip(facts, subj_prompts)])        # n x d_model
            resid = (Z - H) / (len(R) - j)                     # n x d_model
            C = self.cov(L)
            # escala λ relativa: tr(C)/d vs ||k||^2 medio, para que λ sea comparable entre modelos
            kk = (K.norm(dim=1) ** 2).mean()
            # λ relativo: diag(λC) ~ λ·||k||². Se normaliza con la MEDIANA de diag(C) (no la media):
            # en 360M/0.5B unos pocos tokens con activaciones masivas inflan la media y dejaban
            # el sistema casi sin regularizar (modelo destrozado). Ver docs.
            lam = self.lam * kk / C.diagonal().median()
            A = lam * C + K.T @ K + 1e-6 * kk * torch.eye(C.shape[0])
            V = torch.linalg.solve(A, K.T).T                   # n x d_ff  = (A^-1 K^T)^T
            e = LowRankEdit(layer=L, U=resid.T.contiguous(), V=V.contiguous())
            _apply_lowrank(self.model, e)
            self.applied.append(e)
            edits.append(e)
        self.facts.extend(facts)
        self.method = "memit"
        return edits
