"""Router heurístico por keywords / task_id explícito — NO es razonamiento LLM.

Si el prompt contiene 'task:<id>' o el id literal, despacha a esa tarea.
Si no, puntúa por solapamiento de keywords. Empate → None (fallback base).
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple

from .catalog import TaskSpec, all_tasks


def route(prompt: str, tasks: Optional[List[TaskSpec]] = None) -> Tuple[Optional[str], Dict]:
    tasks = tasks or all_tasks()
    p = prompt.strip()
    # explicit dispatch
    m = re.search(r"task:([a-z0-9_]+)", p, re.I)
    if m:
        tid = m.group(1)
        if any(t.id == tid for t in tasks):
            return tid, {"method": "explicit_task_id", "score": 1.0, "note": "dispatch by id, not router accuracy"}
    low = p.lower()
    if any(t.id.lower() == low for t in tasks):
        return low, {"method": "exact_id", "score": 1.0}
    best_id, best_score = None, 0.0
    details = []
    for t in tasks:
        score = 0.0
        hits = []
        for kw in t.keywords:
            k = kw.lower()
            if len(k) <= 2:
                if re.search(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", low):
                    score += 1.0
                    hits.append(kw)
            elif k in low:
                score += 1.0
                hits.append(kw)
        # mild boost if task id token appears as whole id
        if re.search(rf"(?<![a-z0-9]){re.escape(t.id)}(?![a-z0-9])", low):
            score += 2.0
            hits.append(t.id)
        details.append({"id": t.id, "score": score, "hits": hits})
        if score > best_score:
            best_score, best_id = score, t.id
    if best_score <= 0:
        return None, {"method": "keyword", "score": 0.0, "fallback": "base", "details_top": sorted(details, key=lambda d: -d["score"])[:5]}
    return best_id, {"method": "keyword", "score": best_score, "details_top": sorted(details, key=lambda d: -d["score"])[:5]}
