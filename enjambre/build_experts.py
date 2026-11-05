"""Construye expertos: direcciones (system+few-shot y comply/violate), tune, eval."""
from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from .edits import (
    EditableModel, EditDelta, bake_steering_bias, mean_activation_diff, unit,
)
from .experts import ExpertSpec, Item, all_expert_specs
from .generate import chat_text, generate, perplexity


def _score(model, tok, items: List[Item], max_new_tokens: int = 80) -> Tuple[float, List[str]]:
    outs = generate(model, tok, [it.prompt for it in items], system=None, max_new_tokens=max_new_tokens)
    ok = sum(int(bool(it.check(o))) for it, o in zip(items, outs))
    return ok / max(1, len(items)), outs


def _fewshot_block(spec: ExpertSpec) -> str:
    """Ejemplos cortos de 'buena respuesta' metidos en el system (contraste)."""
    # un ejemplo sintético por tipo, no del eval
    demos = {
        "json": 'Example good reply: {"a":"x","b":"y"}',
        "spanish": "Example good reply: El sol brilla en el cielo azul.",
        "translate_en": "Example: input 'Hola' → Hello",
        "arithmetic": "Example: What is 2+2? → 4",
        "summarize": "Example: resumen en 5 palabras → Gatos duermen muchas horas diarias.",
        "uppercase": "Example good reply: EL CAFÉ ES UNA BEBIDA POPULAR.",
        "sentiment": "Example: 'I love it' → positive",
        "entities": 'Example: {"person":"Ada","place":"Lima","date":"May"}',
        "bullets": "Example:\n- uno\n- dos\n- tres",
        "code_python": "Example: def add(a, b):\n    return a + b",
    }
    return demos.get(spec.id, "Follow the format exactly.")


def compute_direction(model, tok, spec: ExpertSpec) -> torch.Tensor:
    """Contraste enriquecido:
    pos = chat(user, system_tarea + few-shot)
    neg = chat(user, None)
    + 0.5 * dirección cumple/viola si hay ambos en el tune set.
    Se normaliza por capa.
    """
    sys_pos = spec.system_prompt + "\n" + _fewshot_block(spec)
    pos = [chat_text(tok, it.prompt, sys_pos) for it in spec.tune_items]
    neg = [chat_text(tok, it.prompt, None) for it in spec.tune_items]
    dirA = mean_activation_diff(model, tok, pos, neg, batch_size=6)

    # comply vs violate sobre el baseline
    outs = generate(model, tok, [it.prompt for it in spec.tune_items])
    ok_flags = [bool(it.check(o)) for it, o in zip(spec.tune_items, outs)]
    if any(ok_flags) and not all(ok_flags):
        texts = [chat_text(tok, it.prompt, None) for it in spec.tune_items]
        h = None
        # reusa last_token via mean trick
        from .edits import last_token_hidden
        h = last_token_hidden(model, tok, texts, batch_size=6)  # [L+1,n,d]
        ok_t = torch.tensor(ok_flags)
        dirB = h[:, ok_t].mean(1) - h[:, ~ok_t].mean(1)
        dirA = dirA + 0.5 * dirB

    # normalizar por capa (salvo capa 0)
    out = []
    for i in range(dirA.shape[0]):
        v = dirA[i]
        out.append(unit(v) if v.norm() > 1e-6 else v)
    return torch.stack(out)


def tune_expert(
    em: EditableModel, tok, spec: ExpertSpec, direction: torch.Tensor,
) -> Tuple[EditDelta, Dict[str, Any]]:
    """Elige (layer, scale) maximizando tune_acc; desempate: menor daño PPL y mayor |Δacc|.

    `scale` aquí es magnitud del vector unitario (equivalente a c * ||d|| en edicion_pesos).
    """
    base_acc, _ = _score(em.model, tok, spec.tune_items)
    base_ppl = perplexity(em.model, tok)
    best = None  # (acc, -ppl_delta, layer, scale, info)
    rows = []
    for L in spec.layer_candidates:
        vec = direction[L + 1]
        for c in spec.scale_candidates:
            delta = EditDelta(
                name=spec.id,
                bias_edits=[bake_steering_bias(vec, L, c)],
                meta={"layer": L, "scale": c},
            )
            em.apply_only_biases(delta)
            acc, _ = _score(em.model, tok, spec.tune_items)
            ppl = perplexity(em.model, tok)
            rows.append({
                "layer": L, "scale": c,
                "tune_acc": round(acc, 4),
                "ppl": round(ppl, 3),
            })
            # score lexicográfico: acc, luego no degradar ppl demasiado, luego preferir mejora
            cand = (acc, -(ppl - base_ppl), c, L)
            if best is None or cand > (best[0], best[1], best[2], best[3]):
                best = (acc, -(ppl - base_ppl), c, L, {"tune_acc": acc, "ppl": ppl})
            em.revert_biases()

    acc, _, c, L, info = best
    delta = EditDelta(
        name=spec.id,
        bias_edits=[bake_steering_bias(direction[L + 1], L, c)],
        meta={
            "layer": L, "scale": c, "tune_acc": acc,
            "base_tune_acc": base_acc, "base_ppl": base_ppl, "grid": rows,
            "normalized": True,
        },
    )
    return delta, {
        "best_layer": L, "best_scale": c, "tune_acc": acc,
        "base_tune_acc": base_acc, "grid": rows,
    }


def build_all(
    em: EditableModel, tok, out_dir: str, specs: Optional[List[ExpertSpec]] = None,
) -> Tuple[Dict[str, Any], Dict[str, EditDelta]]:
    os.makedirs(out_dir, exist_ok=True)
    specs = specs or all_expert_specs()
    report: Dict[str, Any] = {"experts": {}, "base_ppl": None}
    t0 = time.time()

    em.revert_all()
    report["base_ppl"] = round(perplexity(em.model, tok), 3)

    directions = {}
    deltas: Dict[str, EditDelta] = {}

    for spec in specs:
        print(f"[build] {spec.id}: dirección…", flush=True)
        t1 = time.time()
        em.revert_all()
        direction = compute_direction(em.model, tok, spec)
        directions[spec.id] = direction
        print(f"[build] {spec.id}: tune capa/escala…", flush=True)
        delta, info = tune_expert(em, tok, spec, direction)
        deltas[spec.id] = delta

        em.apply_only_biases(delta)
        eval_acc, outs = _score(em.model, tok, spec.eval_items)
        ppl = perplexity(em.model, tok)
        em.revert_biases()
        base_acc, base_outs = _score(em.model, tok, spec.eval_items)

        entry = {
            "id": spec.id,
            "name": spec.name,
            "layer": info["best_layer"],
            "scale": info["best_scale"],
            "tune_acc": round(info["tune_acc"], 4),
            "tune_acc_base": round(info["base_tune_acc"], 4),
            "eval_acc_expert": round(eval_acc, 4),
            "eval_acc_base": round(base_acc, 4),
            "ppl_expert": round(ppl, 3),
            "n_tune": len(spec.tune_items),
            "n_eval": len(spec.eval_items),
            "grid": info["grid"],
            "seconds": round(time.time() - t1, 1),
            "samples_expert": [
                {"prompt": it.prompt, "out": o, "ok": bool(it.check(o))}
                for it, o in zip(spec.eval_items, outs)
            ],
            "samples_base": [
                {"prompt": it.prompt, "out": o, "ok": bool(it.check(o))}
                for it, o in zip(spec.eval_items, base_outs)
            ],
        }
        report["experts"][spec.id] = entry
        torch.save(
            {
                "name": spec.id,
                "layer": info["best_layer"],
                "scale": info["best_scale"],
                "vec": delta.bias_edits[0].vec,
                "meta": delta.meta,
            },
            os.path.join(out_dir, f"delta_{spec.id}.pt"),
        )
        print(
            f"[build] {spec.id}: base={base_acc:.3f} expert={eval_acc:.3f} "
            f"L={info['best_layer']} c={info['best_scale']} ppl={ppl:.2f} "
            f"(tune {info['base_tune_acc']:.2f}->{info['tune_acc']:.2f}) ({entry['seconds']}s)",
            flush=True,
        )

    torch.save(directions, os.path.join(out_dir, "directions.pt"))
    report["total_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "experts_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report, deltas


def load_delta(path: str) -> EditDelta:
    d = torch.load(path, map_location="cpu", weights_only=False)
    return EditDelta(
        name=d["name"],
        bias_edits=[bake_steering_bias(d["vec"], int(d["layer"]), float(d["scale"]))],
        meta=d.get("meta", {}),
    )
