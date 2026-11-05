"""Benchmarks: base vs experto vs swarm; router; memoria; PPL."""
from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, List, Optional

import torch

from .experts import all_expert_specs
from .generate import generate, perplexity
from .memory import MEMORY_UNRELATED, MEMORY_WRITE_FACTS, build_recall_prompt, recall_check
from .orchestrator import Swarm


def eval_items(model, tok, items, max_new_tokens=80):
    outs = generate(model, tok, [it.prompt for it in items], max_new_tokens=max_new_tokens)
    ok = [bool(it.check(o)) for it, o in zip(items, outs)]
    return sum(ok) / max(1, len(ok)), outs, ok


def run_task_table(swarm: Swarm) -> Dict[str, Any]:
    """(a) base en todas las tareas (b) cada experto en la suya (c) swarm+router e2e."""
    rows = []
    router_correct = 0
    router_total = 0
    t0 = time.time()

    for spec in all_expert_specs():
        # base
        swarm.em.revert_all()
        # reaplicar memoria si hubiera — aquí bench de expertos sin memoria
        base_acc, base_outs, _ = eval_items(swarm.em.model, swarm.tok, spec.eval_items)

        # experto
        if spec.id in swarm.deltas:
            swarm.apply_expert(spec.id, keep_memory=False)
            exp_acc, exp_outs, _ = eval_items(swarm.em.model, swarm.tok, spec.eval_items)
            ppl_e = perplexity(swarm.em.model, swarm.tok)
            layer = swarm.deltas[spec.id].meta.get("layer")
            scale = swarm.deltas[spec.id].meta.get("scale")
        else:
            exp_acc, exp_outs, ppl_e, layer, scale = None, [], None, None, None

        # swarm router e2e (cada prompt se enruta)
        swarm_ok = 0
        routed = []
        for it in spec.eval_items:
            r = swarm.route_and_run(it.prompt, strategy="hybrid")
            good = bool(it.check(r["output"]))
            swarm_ok += int(good)
            routed.append(r["expert"])
            router_total += 1
            router_correct += int(r["expert"] == spec.id)

        swarm_acc = swarm_ok / max(1, len(spec.eval_items))
        rows.append({
            "task": spec.id,
            "n_eval": len(spec.eval_items),
            "base_acc": round(base_acc, 4),
            "expert_acc": None if exp_acc is None else round(exp_acc, 4),
            "swarm_acc": round(swarm_acc, 4),
            "layer": layer,
            "scale": scale,
            "ppl_expert": ppl_e if ppl_e is None else round(ppl_e, 3),
            "router_match_rate": round(sum(1 for e in routed if e == spec.id) / max(1, len(routed)), 4),
        })
        print(
            f"[bench] {spec.id:12s} base={base_acc:.3f} expert={exp_acc} swarm={swarm_acc:.3f}",
            flush=True,
        )

    swarm.em.revert_all()
    base_ppl = perplexity(swarm.em.model, swarm.tok)
    return {
        "rows": rows,
        "router_accuracy": round(router_correct / max(1, router_total), 4),
        "base_ppl": round(base_ppl, 3),
        "seconds": round(time.time() - t0, 1),
    }


def run_memory_bench(swarm: Swarm, layer: Optional[int] = None) -> Dict[str, Any]:
    """Memoria: (1) recall tras escribir UN hecho (limpia entre hechos);
    (2) recall con los 4 hechos a la vez (interferencia); unrelated + PPL."""
    from .memory import first_token_ok
    t0 = time.time()
    swarm.memory.clear()
    swarm.em.revert_all()

    # --- baseline ---
    base_ft = base_txt = 0
    base_outs = []
    for subject, obj, _fid in MEMORY_WRITE_FACTS:
        out = swarm.generate(build_recall_prompt(subject), expert_id=None)
        ft = swarm.memory.first_token(subject)
        ok_ft = first_token_ok(ft, obj)
        ok_txt = recall_check(out, obj)
        base_ft += int(ok_ft); base_txt += int(ok_txt)
        base_outs.append({"subject": subject, "out": out, "first_token": ft, "ok_ft": ok_ft, "ok_txt": ok_txt})

    # --- one-at-a-time (primary) ---
    single_ft = single_txt = 0
    single_outs = []
    single_unrelated = []
    for subject, obj, fid in MEMORY_WRITE_FACTS:
        swarm.memory.clear()
        info = swarm.memory_write(subject, obj, fact_id=fid, layer=None)
        r = swarm.memory_read(subject)
        ft = swarm.memory.first_token(subject)
        ok_ft = first_token_ok(ft, obj)
        ok_txt = recall_check(r["output"], obj)
        single_ft += int(ok_ft); single_txt += int(ok_txt)
        # unrelated spot-check with this single edit
        u_prompt, u_needle = MEMORY_UNRELATED[0]
        u_out = swarm.generate(u_prompt, expert_id=None)
        u_ok = u_needle.lower() in u_out.lower()
        single_unrelated.append(u_ok)
        single_outs.append({
            "subject": subject, "layer": info["layer"], "out": r["output"],
            "first_token": ft, "ok_ft": ok_ft, "ok_txt": ok_txt, "target": obj,
            "unrelated_paris_ok": u_ok,
        })

    # --- multi-fact batch (interference) ---
    swarm.memory.clear()
    layers_used = []
    for subject, obj, fid in MEMORY_WRITE_FACTS:
        info = swarm.memory_write(subject, obj, fact_id=fid, layer=None)
        layers_used.append(info["layer"])
    multi_ft = multi_txt = 0
    multi_outs = []
    for subject, obj, fid in MEMORY_WRITE_FACTS:
        r = swarm.memory_read(subject)
        ft = swarm.memory.first_token(subject)
        ok_ft = first_token_ok(ft, obj)
        ok_txt = recall_check(r["output"], obj)
        multi_ft += int(ok_ft); multi_txt += int(ok_txt)
        multi_outs.append({"subject": subject, "out": r["output"], "first_token": ft,
                           "ok_ft": ok_ft, "ok_txt": ok_txt, "target": obj})

    unrelated_ok = 0
    unrelated_outs = []
    for prompt, needle in MEMORY_UNRELATED:
        out = swarm.generate(prompt, expert_id=None)
        ok = needle.lower() in out.lower()
        unrelated_ok += int(ok)
        unrelated_outs.append({"prompt": prompt, "out": out, "needle": needle, "ok": ok})

    ppl_mem = perplexity(swarm.em.model, swarm.tok)
    facts = list(swarm.memory.facts.values())
    swarm.memory.clear()
    ppl_base = perplexity(swarm.em.model, swarm.tok)
    for f in facts:
        swarm.memory.facts[f.fact_id] = f
        swarm.em.apply_rank1(f.edit)

    n = len(MEMORY_WRITE_FACTS); nu = len(MEMORY_UNRELATED)
    return {
        "n_facts": n,
        "layers": layers_used,
        "recall_base_first_token": round(base_ft / n, 4),
        "recall_base_text": round(base_txt / n, 4),
        "recall_single_first_token": round(single_ft / n, 4),
        "recall_single_text": round(single_txt / n, 4),
        "recall_multi_first_token": round(multi_ft / n, 4),
        "recall_multi_text": round(multi_txt / n, 4),
        "unrelated_ok_with_single_avg": round(sum(single_unrelated) / max(1, len(single_unrelated)), 4),
        "unrelated_ok_with_multi": round(unrelated_ok / nu, 4),
        "ppl_base": round(ppl_base, 3),
        "ppl_memory_multi": round(ppl_mem, 3),
        "samples_base": base_outs,
        "samples_single": single_outs,
        "samples_multi": multi_outs,
        "samples_unrelated_multi": unrelated_outs,
        "seconds": round(time.time() - t0, 1),
        "method": (
            "ROME: chat-aligned k*; v* = logit-lens init + Adam≤40 on v only (W frozen); "
            "ΔW=(v*-Wk̂)⊗k̂, ||Δv||≤35; one fact per layer. "
            "Primary metric = write/recall one fact at a time; multi-fact shows interference."
        ),
    }


def run_full_bench(swarm: Swarm, out_path: str) -> Dict[str, Any]:
    print("[bench] tareas + router…", flush=True)
    tasks = run_task_table(swarm)
    print("[bench] memoria…", flush=True)
    mem = run_memory_bench(swarm)
    report = {
        "tasks": tasks,
        "memory": mem,
        "status": swarm.status(),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[bench] escrito {out_path}", flush=True)
    return report
