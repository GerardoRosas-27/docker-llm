"""Memoria compartida: ROME rank-1 (Adam solo sobre v; W congelada).

Un hecho por capa (DEFAULT_FACT_LAYERS). Delta clampado para limitar bleed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .edits import EditableModel, Rank1Edit, make_memory_edit
from .generate import chat_text, generate_one

DEFAULT_FACT_LAYERS = [14, 16, 18, 20, 22, 24]


@dataclass
class Fact:
    subject: str
    obj: str
    fact_id: str
    layer: int
    edit: Rank1Edit


@dataclass
class SharedMemory:
    em: EditableModel
    tok: object
    layer: int = 18
    facts: Dict[str, Fact] = field(default_factory=dict)
    _layer_cursor: int = 0
    fact_layers: List[int] = field(default_factory=lambda: list(DEFAULT_FACT_LAYERS))

    def _alloc_layer(self) -> int:
        used = {f.layer for f in self.facts.values()}
        for L in self.fact_layers:
            if L not in used:
                return L
        L = self.fact_layers[self._layer_cursor % len(self.fact_layers)]
        self._layer_cursor += 1
        return L

    def _query_prompt(self, subject: str) -> str:
        return chat_text(self.tok, build_recall_prompt(subject))

    def write(self, subject: str, obj: str, fact_id: Optional[str] = None, layer: Optional[int] = None) -> Fact:
        fid = fact_id or f"{subject}::{obj}"
        layer = self._alloc_layer() if layer is None else layer
        self.em.revert_rank1()
        q = self._query_prompt(subject)
        edit = make_memory_edit(
            self.em.model, self.tok, subject, obj, layer,
            fact_id=fid, method="rome_v", query_prompt=q,
        )
        fact = Fact(subject=subject, obj=obj, fact_id=fid, layer=layer, edit=edit)
        self.facts[fid] = fact
        for f in self.facts.values():
            self.em.apply_rank1(f.edit)
        return fact

    def forget(self, fact_id: str) -> bool:
        if fact_id not in self.facts:
            return False
        del self.facts[fact_id]
        self.em.revert_rank1()
        for f in self.facts.values():
            self.em.apply_rank1(f.edit)
        return True

    def clear(self) -> None:
        self.facts.clear()
        self.em.revert_rank1()

    def list_facts(self) -> List[Dict]:
        return [
            {"fact_id": f.fact_id, "subject": f.subject, "object": f.obj, "layer": f.layer}
            for f in self.facts.values()
        ]

    def first_token(self, subject: str) -> str:
        """Primer token greedy tras el prompt de recall (métrica limpia de ROME)."""
        q = self._query_prompt(subject)
        ids = self.tok(q, return_tensors="pt").input_ids.to(next(self.em.model.parameters()).device)
        with torch.no_grad():
            logits = self.em.model(ids).logits[0, -1]
            tid = int(logits.argmax())
        return self.tok.decode([tid])


MEMORY_WRITE_FACTS = [
    ("BlergCorp", "robotics", "blerg"),
    ("Zorpium", "purplemetal", "zorp"),
    ("QuibCity", "FogHarbor", "quib"),
    ("Nexlify", "beekeeping", "nex"),
]

MEMORY_UNRELATED = [
    ("The capital of France is", "Paris"),
    ("Water freezes at", "0"),
    ("The chemical symbol for gold is", "Au"),
    ("Shakespeare wrote", "Hamlet"),
]


def build_recall_prompt(subject: str) -> str:
    return f"What is {subject}? Answer briefly:"


def recall_check(output: str, obj: str) -> bool:
    low = output.lower().replace(" ", "")
    target = obj.lower().replace(" ", "")
    if target in low:
        return True
    # prefijo del objeto (≥4) al inicio de la respuesta
    head = output.lower().lstrip()[: max(12, len(target) + 2)]
    return target[: max(4, len(target) // 2)] in head.replace(" ", "")


def first_token_ok(token: str, obj: str) -> bool:
    t = token.lower().strip()
    o = obj.lower().strip()
    return o.startswith(t[: max(3, len(t))]) or t.startswith(o[: max(3, min(5, len(o)))]) or o[:4] in t or t[:4] in o
