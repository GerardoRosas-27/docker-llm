"""Ciclo de auto-mejora acotado (N episodios), no rutina de app.

Usa experiencias verificadas (MemoryNet) + tune/val del catálogo para proponer
nuevas ediciones (capa/escala/dirección). Compara en VAL (nunca heldout).
Promueve atómicamente solo si supera al incumbente con control de PPL; si no,
rechaza y conserva el previo (rollback implícito al no promover).
"""
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from enjambre.edits import EditDelta, bake_steering_bias, last_token_hidden, mean_activation_diff, unit
from enjambre.generate import chat_text, generate, perplexity

from .bank import ACTIVE, CANDIDATE, EXPERIMENTAL, QUARANTINE, ExpertBank
from .catalog import TaskSpec, Item
from .memory_net import MemoryNet
from .snapshot import SnapshotActivator

PPL_GUARD = 1.5
DEFAULT_LAYER_FRACS = [0.27, 0.33, 0.40]
DEFAULT_REL_SCALES = [0.02, 0.08, 0.25]


def _acc(model, tok, items: Sequence[Item], max_new: int = 32) -> Tuple[float, List[str], List[bool]]:
    outs = generate(model, tok, [it.prompt for it in items], max_new_tokens=max_new, batch_size=4)
    flags = [bool(it.check(o)) for it, o in zip(items, outs)]
    return sum(flags) / max(1, len(flags)), outs, flags


def compute_task_direction(model, tok, task: TaskSpec, good_prompts: Optional[List[str]] = None) -> torch.Tensor:
    """Contraste system+train vs train sin system; opcional boost con experiencias buenas."""
    sys_pos = task.system_prompt
    pos = [chat_text(tok, it.prompt, sys_pos) for it in task.train_items]
    neg = [chat_text(tok, it.prompt, None) for it in task.train_items]
    if good_prompts:
        # añade hasta 4 experiencias verificadas como positivos extra
        for gp in good_prompts[:4]:
            pos.append(chat_text(tok, gp, sys_pos))
            neg.append(chat_text(tok, gp, None))
    direction = mean_activation_diff(model, tok, pos, neg, batch_size=4)
    out = []
    for i in range(direction.shape[0]):
        v = direction[i]
        out.append(unit(v) if v.norm() > 1e-6 else v)
    return torch.stack(out)


def residual_norms(model, tok, probes: List[str]) -> torch.Tensor:
    h = last_token_hidden(model, tok, probes, batch_size=4)
    return h.norm(dim=-1).median(1).values  # [n_layers+1]


def grid_search_delta(
    model, tok, act: SnapshotActivator, task: TaskSpec, direction: torch.Tensor,
    *, layer_idxs: List[int], rel_scales: List[float], norms: torch.Tensor,
    max_new: int = 32, base_ppl: Optional[float] = None,
) -> Tuple[EditDelta, Dict[str, Any]]:
    """Selecciona (layer, rel_scale) maximizando TUNE (no val, no heldout)."""
    if base_ppl is None:
        base_ppl = perplexity(model, tok)
    base_tune, _, _ = _acc(model, tok, task.tune_items, max_new)
    best = None
    rows = []
    for L in layer_idxs:
        for r in rel_scales:
            scale = float(r * float(norms[L + 1]))
            delta = EditDelta(
                name=task.id,
                bias_edits=[bake_steering_bias(direction[L + 1], L, 1.0)],
                meta={"kind": "single", "layer": L, "rel": r, "abs_scale": scale},
            )
            # bake_steering_bias uses scale on unit vec; we want abs scale = rel*norm
            delta.bias_edits[0].scale = scale
            with act.session(delta):
                tune_acc, _, _ = _acc(model, tok, task.tune_items, max_new)
                ppl = perplexity(model, tok)
            ok_ppl = ppl <= PPL_GUARD * base_ppl
            rows.append({"layer": L, "rel": r, "tune_acc": round(tune_acc, 4), "ppl": round(ppl, 3), "ok_ppl": ok_ppl})
            key = (int(ok_ppl), tune_acc if ok_ppl else -1.0, -abs(ppl - base_ppl))
            if best is None or key > best[0]:
                best = (key, delta, tune_acc, ppl)
    assert best is not None
    delta = best[1]
    info = {
        "base_tune": round(base_tune, 4),
        "best_tune": round(best[2], 4),
        "best_ppl": round(best[3], 3),
        "base_ppl": round(base_ppl, 3),
        "grid": rows,
        "chosen": delta.meta,
        "tune_no_gain": best[2] <= base_tune + 1e-9,
    }
    return delta, info


class AutoImprover:
    """Presupuesto por N episodios de mejora por tarea (no daemon continuo)."""

    def __init__(
        self,
        model, tok, act: SnapshotActivator, bank: ExpertBank, memory: MemoryNet,
        *, layer_fracs: Optional[List[float]] = None, rel_scales: Optional[List[float]] = None,
        max_new: int = 32, min_val_margin: float = 0.05,
    ):
        self.model, self.tok, self.act = model, tok, act
        self.bank, self.memory = bank, memory
        self.layer_fracs = layer_fracs or list(DEFAULT_LAYER_FRACS)
        self.rel_scales = rel_scales or list(DEFAULT_REL_SCALES)
        self.max_new = max_new
        self.min_val_margin = min_val_margin
        nL = len(model.model.layers)
        self.layer_idxs = sorted({min(nL - 1, max(0, round(f * nL))) for f in self.layer_fracs})

    def _norms(self, task: TaskSpec) -> torch.Tensor:
        probes = [chat_text(self.tok, it.prompt) for it in task.train_items[:2]]
        return residual_norms(self.model, self.tok, probes)

    def evaluate_version(self, task: TaskSpec, version_id: Optional[str], split: str = "val") -> Dict[str, Any]:
        items = {"tune": task.tune_items, "val": task.val_items, "heldout": task.heldout_items}[split]
        delta = self.bank.load_delta(version_id) if version_id else None
        with self.act.session(delta):
            acc, outs, flags = _acc(self.model, self.tok, items, self.max_new)
            ppl = perplexity(self.model, self.tok)
        return {
            "split": split, "n": len(items), "acc": round(acc, 4), "ppl": round(ppl, 3),
            "n_ok": int(sum(flags)), "version_id": version_id,
            "samples": [{"prompt": it.prompt[:80], "out": o[:120], "ok": bool(f)} for it, o, f in zip(items[:3], outs[:3], flags[:3])],
        }

    def build_initial(self, task: TaskSpec) -> Dict[str, Any]:
        """Construye primer delta; activo solo si gana a base en VAL (no heldout)."""
        t0 = time.time()
        good = [r["prompt"] for r in self.memory.retrieve_good(task.id, limit=4)]
        direction = compute_task_direction(self.model, self.tok, task, good)
        norms = self._norms(task)
        base_ppl = perplexity(self.model, self.tok)
        delta, info = grid_search_delta(
            self.model, self.tok, self.act, task, direction,
            layer_idxs=self.layer_idxs, rel_scales=self.rel_scales, norms=norms,
            max_new=self.max_new, base_ppl=base_ppl,
        )
        # eval base vs candidate on VAL
        base_val = self.evaluate_version(task, None, "val")
        # register as candidate first
        ev = self.bank.register(
            task.id, delta, status=CANDIDATE, metrics={"tune": info, "base_val": base_val},
            provenance={"kind": "initial_build", "good_prompts_used": len(good)},
            make_active=False,
        )
        cand_val = self.evaluate_version(task, ev.version_id, "val")
        accept = (
            cand_val["acc"] + 1e-9 >= base_val["acc"] + self.min_val_margin
            and cand_val["ppl"] <= PPL_GUARD * base_ppl
            and not info["tune_no_gain"]
        )
        # also accept weak equality with tune gain and ppl ok if margin 0 path — keep strict margin
        if accept:
            self.bank.promote_atomic(task.id, ev.version_id)
            status = ACTIVE
        else:
            # keep experimental (fallback base honesto)
            self.bank.set_status(ev.version_id, EXPERIMENTAL)
            status = EXPERIMENTAL
        detail = {
            "accepted": accept, "status": status, "version_id": ev.version_id,
            "base_val": base_val, "cand_val": cand_val, "tune": info, "seconds": round(time.time() - t0, 1),
        }
        self.memory.log_improve(task.id, "initial_build", accept, detail)
        return detail

    def propose_replacement(self, task: TaskSpec) -> Dict[str, Any]:
        """Propone candidato nuevo usando experiencias buenas; NO promueve aún."""
        t0 = time.time()
        good_rows = self.memory.retrieve_good(task.id, limit=8)
        good = [r["prompt"] for r in good_rows]
        direction = compute_task_direction(self.model, self.tok, task, good)
        norms = self._norms(task)
        base_ppl = perplexity(self.model, self.tok)
        # perturb grid: prefer layers not used by incumbent if possible
        inc = self.bank.get_active(task.id)
        avoid_layer = None
        if inc:
            d0 = self.bank.load_delta(inc.version_id)
            if d0.bias_edits:
                avoid_layer = d0.bias_edits[0].layer
        layers = [L for L in self.layer_idxs if L != avoid_layer] or list(self.layer_idxs)
        delta, info = grid_search_delta(
            self.model, self.tok, self.act, task, direction,
            layer_idxs=layers, rel_scales=self.rel_scales, norms=norms,
            max_new=self.max_new, base_ppl=base_ppl,
        )
        parent = inc.version_id if inc else None
        ev = self.bank.register(
            task.id, delta, status=CANDIDATE, metrics={"tune": info},
            provenance={"kind": "replacement_proposal", "good_prompts_used": len(good), "avoid_layer": avoid_layer},
            parent_version_id=parent, make_active=False,
        )
        detail = {"version_id": ev.version_id, "parent": parent, "tune": info, "seconds": round(time.time() - t0, 1)}
        self.memory.log_improve(task.id, "propose_replacement", None, detail)
        return detail

    def evaluate_and_maybe_promote(self, task: TaskSpec, candidate_version_id: str) -> Dict[str, Any]:
        """Compara candidato vs incumbente (o base) en VAL; promueve atómicamente solo si gana."""
        inc = self.bank.get_active(task.id)
        base_ppl = perplexity(self.model, self.tok)
        if inc:
            inc_val = self.evaluate_version(task, inc.version_id, "val")
            incumbent_id = inc.version_id
            incumbent_acc = inc_val["acc"]
        else:
            inc_val = self.evaluate_version(task, None, "val")
            incumbent_id = None
            incumbent_acc = inc_val["acc"]
        cand_val = self.evaluate_version(task, candidate_version_id, "val")
        better = cand_val["acc"] + 1e-9 >= incumbent_acc + self.min_val_margin
        ppl_ok = cand_val["ppl"] <= PPL_GUARD * base_ppl
        accepted = bool(better and ppl_ok)
        if accepted:
            promo = self.bank.promote_atomic(task.id, candidate_version_id)
            action = "promote"
        else:
            self.bank.set_status(candidate_version_id, QUARANTINE if cand_val["acc"] < incumbent_acc else EXPERIMENTAL)
            promo = {"task_id": task.id, "active": incumbent_id, "rejected": candidate_version_id}
            action = "reject"
        detail = {
            "action": action, "accepted": accepted, "incumbent": incumbent_id,
            "candidate": candidate_version_id, "inc_val": inc_val, "cand_val": cand_val,
            "min_val_margin": self.min_val_margin, "ppl_ok": ppl_ok, "promo": promo,
        }
        self.memory.log_improve(task.id, action, accepted, detail)
        return detail

    def auto_improve_task(self, task: TaskSpec, *, episodes: int = 1) -> Dict[str, Any]:
        """Hasta `episodes` ciclos propose→evaluate por tarea. Heldout no se toca."""
        history = []
        for i in range(max(1, episodes)):
            prop = self.propose_replacement(task)
            decision = self.evaluate_and_maybe_promote(task, prop["version_id"])
            history.append({"episode": i, "proposal": prop, "decision": decision})
            # si aceptó, un episodio más puede seguir; si rechazó todos igual registramos
        return {"task_id": task.id, "episodes": episodes, "history": history,
                "active": (self.bank.get_active(task.id).version_id if self.bank.get_active(task.id) else None)}

    def auto_improve(self, tasks: List[TaskSpec], *, episodes: int = 1, max_tasks: Optional[int] = None) -> Dict[str, Any]:
        selected = tasks[: max_tasks or len(tasks)]
        results = []
        for t in selected:
            results.append(self.auto_improve_task(t, episodes=episodes))
        n_acc = sum(1 for r in results for h in r["history"] if h["decision"]["accepted"])
        n_rej = sum(1 for r in results for h in r["history"] if not h["decision"]["accepted"])
        return {"n_tasks": len(selected), "episodes": episodes, "accepted": n_acc, "rejected": n_rej, "results": results}
