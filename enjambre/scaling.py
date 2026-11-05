"""Pipeline model-agnóstico: expertos (steering horneado, 1 o 3 capas) + swarm + memoria.

Uso:  python -m enjambre.scaling --model smollm2-360m [--max-eval 20] [--max-new 64]
Escribe artifacts/<model-key>/{delta_*.pt, scaling_report.json}.
"""
from __future__ import annotations
import argparse
import json
import os
import resource
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from enjambre.build_experts import compute_direction
from enjambre.config import (EXPERT_LAYER_FRACS, MEMIT_LAYER_FRACS, ROME_LAYER_FRAC, STEER_MULTI_REL_SCALES,
                             STEER_REL_SCALES, artifacts_dir, resolve, scale_layers)
from enjambre.edits import BiasEdit, EditDelta, EditableModel, last_token_hidden
from enjambre.experts import all_expert_specs
from enjambre.generate import chat_text, generate, perplexity
from enjambre.router import Router


PPL_GUARD = float(os.environ.get("ENJAMBRE_PPL_GUARD", "1.5"))


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _acc(model, tok, items, max_new):
    outs = generate(model, tok, [it.prompt for it in items], max_new_tokens=max_new)
    ok = [bool(it.check(o)) for it, o in zip(items, outs)]
    return sum(ok) / max(1, len(ok)), outs, ok


def make_delta(eid, direction, layers, abs_scales, meta):
    return EditDelta(name=eid, bias_edits=[BiasEdit(layer=L, vec=direction[L + 1].float().clone(), scale=c)
                                           for L, c in zip(layers, abs_scales)], meta=meta)


def save_delta(path, d: EditDelta):
    torch.save({"name": d.name, "edits": [{"layer": b.layer, "scale": b.scale, "vec": b.vec} for b in d.bias_edits],
                "layer": d.bias_edits[0].layer, "scale": d.bias_edits[0].scale, "vec": d.bias_edits[0].vec,
                "meta": {k: v for k, v in d.meta.items() if k != "grid"}}, path)


def run(model_key: str, max_eval: int, max_new: int, skip_memory: bool = False) -> Dict[str, Any]:
    cfg = resolve(model_key)
    out_dir = artifacts_dir(cfg.key)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(min(8, os.cpu_count() or 8))
    T0 = time.time()
    tok = AutoTokenizer.from_pretrained(cfg.hf_id)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.hf_id, dtype=torch.float32).eval()
    t_load = time.time() - T0
    em = EditableModel(model)
    nL = em.n_layers()
    layers = scale_layers(EXPERT_LAYER_FRACS, nL)
    specs = all_expert_specs()
    for s in specs:
        s.eval_items = s.eval_items[:max_eval]
    base_ppl = perplexity(model, tok)
    # norma mediana del residual por capa (para escalar el steering)
    probe = [chat_text(tok, it.prompt) for s in specs for it in s.tune_items[:2]]
    norms = last_token_hidden(model, tok, probe).norm(dim=-1).median(1).values
    router = Router()
    rep: Dict[str, Any] = {"model": cfg.hf_id, "key": cfg.key, "n_layers": nL,
                           "hidden": model.config.hidden_size, "expert_layers": layers,
                           "rel_scales": STEER_REL_SCALES, "multi_rel_scales": STEER_MULTI_REL_SCALES,
                           "base_ppl": round(base_ppl, 3), "max_new_tokens": max_new,
                           "load_seconds": round(t_load, 1), "tasks": {}}
    deltas: Dict[str, EditDelta] = {}
    t_exp = time.time()
    for spec in specs:
        t1 = time.time()
        em.revert_all()
        direction = compute_direction(model, tok, spec)
        base_tune, _, _ = _acc(model, tok, spec.tune_items, max_new)
        cands = []
        for L in layers:
            for r in STEER_REL_SCALES:
                cands.append(("single", [L], [r * float(norms[L + 1])], r))
        for r in STEER_MULTI_REL_SCALES:
            cands.append(("multi", layers, [r * float(norms[L + 1]) for L in layers], r))
        grid, best = [], None
        for kind, Ls, cs, r in cands:
            d = make_delta(spec.id, direction, Ls, cs, {"kind": kind, "layers": Ls, "rel": r})
            em.apply_only_biases(d)
            acc, _, _ = _acc(model, tok, spec.tune_items, max_new)
            ppl = perplexity(model, tok)
            em.revert_biases()
            grid.append({"kind": kind, "layers": Ls, "rel": r, "tune_acc": round(acc, 4), "ppl": round(ppl, 3)})
            # guardia de PPL: configs que rompen el modelo (PPL > 1.5x base) quedan descartadas
            ok_ppl = ppl <= PPL_GUARD * base_ppl
            key = (int(ok_ppl), acc if ok_ppl else 0.0, -abs(ppl - base_ppl))
            if best is None or key > best[0]:
                best = (key, d)
        d = best[1]
        # si ninguna config mejora el tune sobre la base -> experto = identidad (no-op honesto)
        noop = best[0][1] <= base_tune
        deltas[spec.id] = d
        save_delta(os.path.join(out_dir, f"delta_{spec.id}.pt"), d)
        b_acc, _, _ = _acc(model, tok, spec.eval_items, max_new)
        em.apply_only_biases(d)
        e_acc, e_outs, _ = _acc(model, tok, spec.eval_items, max_new)
        e_ppl = perplexity(model, tok)
        em.revert_biases()
        rep["tasks"][spec.id] = {
            "n_tune": len(spec.tune_items), "n_eval": len(spec.eval_items),
            "tune_base": round(base_tune, 4), "tune_best": round(best[0][1], 4), "ppl_guard": PPL_GUARD, "tune_no_gain": noop,
            "chosen": d.meta, "base_acc": round(b_acc, 4), "expert_acc": round(e_acc, 4),
            "ppl_expert": round(e_ppl, 3), "grid": grid, "seconds": round(time.time() - t1, 1),
            "samples_expert": [{"p": it.prompt[:80], "out": o[:160]} for it, o in zip(spec.eval_items[:3], e_outs)],
        }
        print(f"[{cfg.key}] {spec.id:12s} base={b_acc:.2f} exp={e_acc:.2f} tune {base_tune:.2f}->{best[0][1]:.2f} "
              f"{d.meta} ppl={e_ppl:.2f} ({time.time()-t1:.0f}s) rss={rss_mb():.0f}MB", flush=True)
    rep["experts_seconds"] = round(time.time() - t_exp, 1)
    # swarm: router -> experto enrutado (batch por experto)
    t_sw = time.time()
    rc = rt = 0
    for spec in specs:
        groups: Dict[str, List] = {}
        for it in spec.eval_items:
            eid = router.route_one(it.prompt)
            rt += 1; rc += int(eid == spec.id)
            groups.setdefault(eid, []).append(it)
        ok = 0
        for eid, its in groups.items():
            em.revert_biases()
            if eid in deltas:
                em.apply_only_biases(deltas[eid])
            a, _, _ = _acc(model, tok, its, max_new)
            ok += round(a * len(its))
        em.revert_biases()
        rep["tasks"][spec.id]["swarm_acc"] = round(ok / len(spec.eval_items), 4)
    rep["router_accuracy"] = round(rc / rt, 4)
    rep["swarm_seconds"] = round(time.time() - t_sw, 1)
    print(f"[{cfg.key}] router={rep['router_accuracy']} swarm {rep['swarm_seconds']}s", flush=True)
    em.revert_all()
    if not skip_memory:
        from enjambre.membench import run_memory_bench
        rl = scale_layers([ROME_LAYER_FRAC], nL)[0]
        ml = scale_layers(MEMIT_LAYER_FRACS, nL)
        rep["memory"] = run_memory_bench(model, tok, rl, ml, n_facts=(4, 8))
    rep["total_seconds"] = round(time.time() - T0, 1)
    rep["peak_rss_mb"] = round(rss_mb(), 0)
    with open(os.path.join(out_dir, "scaling_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"[{cfg.key}] DONE {rep['total_seconds']}s peak_rss={rep['peak_rss_mb']}MB", flush=True)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--max-eval", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--skip-memory", action="store_true")
    a = ap.parse_args()
    run(a.model, a.max_eval, a.max_new, a.skip_memory)
