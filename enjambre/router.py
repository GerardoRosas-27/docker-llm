"""Router keyword + embeddings bag-of-words (sin dependencias extra)."""
from __future__ import annotations
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .experts import ExpertSpec, all_expert_specs


def _tok(s: str) -> List[str]:
    return re.findall(r"[a-záéíóúüñ0-9_]+", s.lower())


class Router:
    """Enruta a uno o más expertos.

    Estrategias:
      - keyword: puntuación por solapamiento de keywords / id / name
      - embedding: TF bag-of-words coseno contra un prototipo por experto
      - hybrid: 0.6*keyword + 0.4*embedding
    """

    def __init__(self, specs: Optional[List[ExpertSpec]] = None):
        self.specs = specs or all_expert_specs()
        self._proto: Dict[str, Counter] = {}
        for e in self.specs:
            bag = Counter()
            for text in [e.id, e.name, e.description, e.system_prompt, " ".join(e.keywords)]:
                bag.update(_tok(text))
            for it in e.tune_items[:3]:
                bag.update(_tok(it.prompt))
            self._proto[e.id] = bag

    def route_keyword(self, text: str, top_k: int = 1) -> List[Tuple[str, float]]:
        low = text.lower()
        scores = []
        for e in self.specs:
            sc = 0.0
            for kw in e.keywords:
                if kw.lower() in low:
                    sc += 1.0 + 0.1 * len(kw.split())
            if e.id.replace("_", " ") in low:
                sc += 2.0
            scores.append((e.id, sc))
        scores.sort(key=lambda x: -x[1])
        if scores[0][1] <= 0:
            # fallback: embedding
            return self.route_embedding(text, top_k=top_k)
        return scores[:top_k]

    def route_embedding(self, text: str, top_k: int = 1) -> List[Tuple[str, float]]:
        q = Counter(_tok(text))
        scored = [(eid, _cos(q, proto)) for eid, proto in self._proto.items()]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def route(self, text: str, strategy: str = "hybrid", top_k: int = 1) -> List[Tuple[str, float]]:
        if strategy == "keyword":
            return self.route_keyword(text, top_k)
        if strategy == "embedding":
            return self.route_embedding(text, top_k)
        # hybrid
        kw = {e: s for e, s in self.route_keyword(text, top_k=len(self.specs))}
        emb = {e: s for e, s in self.route_embedding(text, top_k=len(self.specs))}
        # normalizar
        def norm(d):
            m = max(d.values()) or 1.0
            return {k: v / m for k, v in d.items()}
        kw, emb = norm(kw), norm(emb)
        scored = [(eid, 0.6 * kw.get(eid, 0) + 0.4 * emb.get(eid, 0)) for eid in kw]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def route_one(self, text: str, strategy: str = "hybrid") -> str:
        return self.route(text, strategy=strategy, top_k=1)[0][0]


def _cos(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
