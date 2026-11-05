"""Ediciones de pesos (sin LoRA / sin fine-tuning de matrices W).

- Sesgo en mlp.down_proj = steering de activación horneado (edicion_pesos).
- Memoria ROME/MEMIT: ΔW=(v*-W k̂)⊗k̂. Init cerrado (logit-lens) + Adam corto
  **solo sobre el vector v** (documentado). Alternativas: teacher / logit_lens puro.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class BiasEdit:
    layer: int
    vec: torch.Tensor
    scale: float = 1.0
    kind: str = "bias"


@dataclass
class Rank1Edit:
    layer: int
    k: torch.Tensor
    delta_v: torch.Tensor
    fact_id: str = ""
    kind: str = "rank1"


@dataclass
class EditDelta:
    name: str
    bias_edits: List[BiasEdit] = field(default_factory=list)
    rank1_edits: List[Rank1Edit] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def nbytes(self) -> int:
        n = sum(b.vec.numel() * b.vec.element_size() for b in self.bias_edits)
        for r in self.rank1_edits:
            n += r.k.numel() * r.k.element_size() + r.delta_v.numel() * r.delta_v.element_size()
        return n


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-8)


class EditableModel:
    def __init__(self, model: nn.Module):
        self.model = model
        self._bias_backup: Dict[Tuple[int, str], Optional[torch.Tensor]] = {}
        self._weight_backup: Dict[int, torch.Tensor] = {}
        self._active: List[str] = []
        self._applied_rank1: List[Rank1Edit] = []

    def layers(self):
        return self.model.model.layers

    def n_layers(self) -> int:
        return len(self.layers())

    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def _down(self, layer: int) -> nn.Linear:
        return self.layers()[layer].mlp.down_proj

    def apply_bias(self, edit: BiasEdit) -> None:
        lin = self._down(edit.layer)
        key = (edit.layer, "bias")
        if key not in self._bias_backup:
            self._bias_backup[key] = None if lin.bias is None else lin.bias.detach().clone()
        vec = (edit.scale * edit.vec).to(dtype=lin.weight.dtype, device=lin.weight.device)
        if lin.bias is None:
            lin.bias = nn.Parameter(vec.clone())
        else:
            with torch.no_grad():
                base = torch.zeros_like(vec) if self._bias_backup[key] is None else self._bias_backup[key]
                lin.bias.copy_(base + vec)

    def revert_biases(self) -> None:
        for (layer, _), bak in list(self._bias_backup.items()):
            lin = self._down(layer)
            if bak is None:
                lin.bias = None
            else:
                with torch.no_grad():
                    lin.bias.copy_(bak)
        self._bias_backup.clear()

    def apply_rank1(self, edit: Rank1Edit) -> None:
        lin = self._down(edit.layer)
        if edit.layer not in self._weight_backup:
            self._weight_backup[edit.layer] = lin.weight.detach().clone()
        kn = edit.k.to(dtype=lin.weight.dtype, device=lin.weight.device)
        kn = kn / (kn.norm() + 1e-8)
        dv = edit.delta_v.to(dtype=lin.weight.dtype, device=lin.weight.device)
        with torch.no_grad():
            lin.weight.add_(torch.outer(dv, kn))
        self._applied_rank1.append(edit)

    def revert_rank1(self) -> None:
        for layer, bak in self._weight_backup.items():
            with torch.no_grad():
                self._down(layer).weight.copy_(bak)
        self._weight_backup.clear()
        self._applied_rank1.clear()

    def apply_delta(self, delta: EditDelta, *, keep_memory: bool = True) -> None:
        self.revert_biases()
        if not keep_memory:
            self.revert_rank1()
        for b in delta.bias_edits:
            self.apply_bias(b)
        for r in delta.rank1_edits:
            self.apply_rank1(r)
        self._active = [delta.name]

    def revert_all(self) -> None:
        self.revert_biases()
        self.revert_rank1()
        self._active = []

    def apply_only_biases(self, delta: EditDelta) -> None:
        self.revert_biases()
        for b in delta.bias_edits:
            self.apply_bias(b)
        self._active = [delta.name]


@torch.no_grad()
def last_token_hidden(model, tok, texts: List[str], batch_size: int = 8) -> torch.Tensor:
    device = next(model.parameters()).device
    chunks = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        last_idx = enc["attention_mask"].sum(1) - 1
        hs = torch.stack(out.hidden_states)
        B = hs.shape[1]
        chunks.append(hs[:, torch.arange(B), last_idx, :].cpu())
    return torch.cat(chunks, dim=1)


@torch.no_grad()
def mean_activation_diff(model, tok, texts_pos, texts_neg, batch_size: int = 8) -> torch.Tensor:
    return last_token_hidden(model, tok, texts_pos, batch_size).mean(1) - \
        last_token_hidden(model, tok, texts_neg, batch_size).mean(1)


def bake_steering_bias(direction_at_layer: torch.Tensor, layer: int, scale: float) -> BiasEdit:
    return BiasEdit(layer=layer, vec=direction_at_layer.detach().float().clone(), scale=scale)


def _find_subseq(hay: List[int], needle: List[int]) -> Optional[int]:
    if not needle:
        return None
    n = len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i:i + n] == needle:
            return i
    return None


@torch.no_grad()
def mlp_key_at_subject(model, tok, prompt: str, subject: str, layer: int) -> torch.Tensor:
    device = next(model.parameters()).device
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    sub_ids = tok(subject, add_special_tokens=False).input_ids
    pos = _find_subseq(ids[0].tolist(), sub_ids)
    subj_pos = (pos + len(sub_ids) - 1) if pos is not None else ids.shape[1] - 1
    captured = {}

    def hook(_mod, inp, _out):
        captured["k"] = inp[0][0, subj_pos, :].detach().cpu()

    h = model.model.layers[layer].mlp.down_proj.register_forward_hook(hook)
    try:
        model(ids)
    finally:
        h.remove()
    return captured["k"]


def distinctive_object_token(tok, object_text: str):
    words = [w for w in object_text.replace(",", " ").split() if w.strip()]
    best = max(words, key=len) if words else object_text
    ids = tok(best, add_special_tokens=False).input_ids or tok(" " + best, add_special_tokens=False).input_ids
    return (ids[0] if ids else 0), best


@torch.no_grad()
def closed_form_value_via_pinv(model, tok, prompt_with_fact, layer, key) -> torch.Tensor:
    device = next(model.parameters()).device
    ids = tok(prompt_with_fact, return_tensors="pt").input_ids.to(device)
    captured = {}

    def hook(_mod, _inp, out):
        captured["v"] = out[0, -1, :].detach().cpu()

    h = model.model.layers[layer].mlp.down_proj.register_forward_hook(hook)
    try:
        model(ids)
    finally:
        h.remove()
    W = model.model.layers[layer].mlp.down_proj.weight.detach().cpu().float()
    kn = key.float() / (key.float().norm() + 1e-8)
    return captured["v"].float() - (W @ kn)


@torch.no_grad()
def closed_form_value_logit_lens(model, tok, object_text, key, layer, amplify: float = 8.0) -> torch.Tensor:
    obj_id, _ = distinctive_object_token(tok, object_text)
    u = unit(model.lm_head.weight.detach().cpu().float()[obj_id])
    W = model.model.layers[layer].mlp.down_proj.weight.detach().cpu().float()
    kn = key.float() / (key.float().norm() + 1e-8)
    Wk = W @ kn
    return amplify * u * (Wk.norm() + 1.0) - Wk


def optimize_value_vector(
    model, tok, query_prompt: str, object_text: str, layer: int, key: torch.Tensor,
    steps: int = 40, lr: float = 0.2,
) -> torch.Tensor:
    """Adam SOLO sobre v∈R^d. Congela todos los pesos del modelo."""
    device = next(model.parameters()).device
    was_grad = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    v0 = closed_form_value_logit_lens(model, tok, object_text, key, layer, amplify=6.0)
    v = nn.Parameter(v0.to(device).float().clone())
    opt = torch.optim.Adam([v], lr=lr)

    full = query_prompt + object_text
    ids = tok(full, return_tensors="pt").input_ids.to(device)
    q_len = tok(query_prompt, return_tensors="pt").input_ids.shape[1]

    try:
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)

            def hook(_mod, _inp, out):
                out = out.clone()
                out[0, q_len - 1, :] = v
                return out

            handle = model.model.layers[layer].mlp.down_proj.register_forward_hook(hook)
            try:
                logits = model(ids).logits
            finally:
                handle.remove()
            pred = logits[0, q_len - 1:-1, :]
            tgt = ids[0, q_len:]
            if tgt.numel() == 0:
                break
            loss = nn.functional.cross_entropy(pred, tgt)
            loss.backward()
            opt.step()
    finally:
        for p, g in zip(model.parameters(), was_grad):
            p.requires_grad_(g)

    with torch.no_grad():
        W = model.model.layers[layer].mlp.down_proj.weight.detach().float()
        kn = key.to(device).float() / (key.float().norm() + 1e-8)
        delta = v.detach().float().cpu() - (W @ kn.to(W.device)).cpu()
        # limitar norma para reducir daño colateral (bleed a prompts no relacionados)
        max_n = 35.0
        n = delta.norm().clamp_min(1e-8)
        if n > max_n:
            delta = delta * (max_n / n)
    return delta


def make_memory_edit(
    model, tok, subject: str, obj: str, layer: int, template: str = None,
    fact_id: str = "", amplify: float = 8.0, method: str = "rome_v",
    query_prompt: str = None,
) -> Rank1Edit:
    """query_prompt debe ser el mismo string que se usará al leer (p.ej. chat template)."""
    q = query_prompt or template or f"What is {subject}? Answer briefly:"
    key = mlp_key_at_subject(model, tok, q, subject, layer)
    if method == "teacher":
        delta_v = closed_form_value_via_pinv(model, tok, f"{subject} is {obj}.", layer, key)
    elif method == "logit_lens":
        delta_v = closed_form_value_logit_lens(model, tok, obj, key, layer, amplify=amplify)
    else:
        delta_v = optimize_value_vector(model, tok, q, obj, layer, key)
    return Rank1Edit(layer=layer, k=key, delta_v=delta_v, fact_id=fact_id or f"{subject}->{obj}")
