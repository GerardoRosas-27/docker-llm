"""Bench de memoria: ROME (single / secuencial) vs MEMIT (batch 4 y 8).
Métricas: recall de la cadena completa del objeto, paráfrasis held-out, preservación de
hechos no relacionados (sobre los que el base acierta) y PPL."""
from __future__ import annotations
import time
from typing import Any, Dict, List

from .generate import PPL_TEXT, generate, perplexity
from .memit import FactSpec, WeightMemory

TRAIN_TPL = ["What is the capital of {s}? Answer briefly:", "Tell me the capital city of {s}."]
PARA_TPL = ["{s} is a country. What is the name of its capital?",
            "Which city is the capital of {s}? One word."]
CAPS = [("Veltria", "Morvane"), ("Quibland", "Fogharbor"), ("Drenmark", "Ostavel"),
        ("Kalvoria", "Tesslin"), ("Brumeria", "Peloran"), ("Nexovia", "Halcrest"),
        ("Ostrania", "Vimbury"), ("Torvania", "Selmund")]
FACTS = [FactSpec(s, o, TRAIN_TPL, PARA_TPL) for s, o in CAPS]
UNRELATED = [
    ("What is the capital of France? Answer briefly:", "paris"),
    ("What is the capital of Japan? Answer briefly:", "tokyo"),
    ("What is the capital of Italy? Answer briefly:", "rome"),
    ("What is the capital of Germany? Answer briefly:", "berlin"),
    ("What is the capital of Spain? Answer briefly:", "madrid"),
    ("What is the capital of Egypt? Answer briefly:", "cairo"),
    ("What is the capital of Russia? Answer briefly:", "moscow"),
    ("What is the capital of England? Answer briefly:", "london"),
    ("What is the chemical symbol for gold? Answer briefly:", "au"),
    ("Who wrote Hamlet? Answer briefly:", "shakespeare"),
    ("How many legs does a spider have? Answer with a number:", "8"),
    ("What color is the sky on a clear day? One word:", "blue"),
]

COV_EXTRA = [
    "The museum is located in the old town and attracts many visitors every summer.",
    "She studied physics at the university before working as an engineer in a small company.",
    "El mercado central abre todos los días a las siete y vende frutas y verduras frescas.",
    "Paris is the capital of France. Berlin is the capital of Germany. Tokyo is the capital of Japan.",
    "The river flows through the valley and into the sea near the port city.",
    "Python is a programming language used for data science, web development and automation.",
    "Los estudiantes leyeron un libro sobre la historia de América Latina.",
    "In 1969, astronauts landed on the Moon for the first time.",
    "The recipe calls for flour, sugar, two eggs and a cup of milk.",
    "Our team will meet on Monday to discuss the new project timeline.",
]


def cov_texts(tok) -> List[str]:
    from .experts import all_expert_specs
    from .generate import chat_text
    t = [PPL_TEXT] + COV_EXTRA
    for s in all_expert_specs():
        t += [chat_text(tok, it.prompt) for it in s.tune_items]  # solo tune (no eval)
    return t


def _gen(model, tok, prompts):
    return generate(model, tok, prompts, max_new_tokens=24, batch_size=8)


def _score(model, tok, facts: List[FactSpec], base_unrel_ok: List[bool]) -> Dict[str, Any]:
    canon = [f.train_prompts[0].format(s=f.subject) for f in facts]
    para = [p.format(s=f.subject) for f in facts for p in f.para_prompts]
    outs = _gen(model, tok, canon + para + [u[0] for u in UNRELATED])
    oc, op, ou = outs[:len(canon)], outs[len(canon):len(canon) + len(para)], outs[len(canon) + len(para):]
    rc = [f.obj.lower() in o.lower() for f, o in zip(facts, oc)]
    pf = [f for f in facts for _ in f.para_prompts]
    rp = [f.obj.lower() in o.lower() for f, o in zip(pf, op)]
    uo = [u[1] in o.lower() for u, o in zip(UNRELATED, ou)]
    kept = [a for a, b in zip(uo, base_unrel_ok) if b]
    return {
        "recall": sum(rc) / len(rc), "paraphrase": sum(rp) / len(rp),
        "unrelated_kept": (sum(kept) / len(kept)) if kept else None,
        "ppl": perplexity(model, tok),
        "samples": [{"subj": f.subject, "target": f.obj, "out": o} for f, o in zip(facts, oc)][:4]
                   + [{"para": True, "out": o} for o in op[:2]],
        "unrelated_samples": ou[:3],
    }


def run_memory_bench(model, tok, rome_layer: int, memit_layers: List[int], n_facts=(4, 8)) -> Dict[str, Any]:
    t0 = time.time()
    mem = WeightMemory(model, tok, cov_texts(tok), rome_layer, memit_layers)
    outs = _gen(model, tok, [u[0] for u in UNRELATED])
    base_ok = [u[1] in o.lower() for u, o in zip(UNRELATED, outs)]
    res: Dict[str, Any] = {"rome_layer": rome_layer, "memit_layers": memit_layers,
                           "unrelated_base_correct": sum(base_ok), "n_unrelated": len(UNRELATED)}
    res["base"] = _score(model, tok, FACTS, base_ok)
    acc = {"recall": 0, "paraphrase": 0, "unrelated_kept": 0, "ppl": 0}
    t1 = time.time()
    for f in FACTS:
        mem.clear()
        mem.write_rome(f)
        s = _score(model, tok, [f], base_ok)
        for k in acc:
            acc[k] += (s[k] or 0) / len(FACTS)
    res["rome_single_avg"] = {k: round(v, 4) for k, v in acc.items()}
    res["rome_single_avg"]["seconds_total"] = round(time.time() - t1, 1)
    mem.clear()
    for n in n_facts:
        t1 = time.time()
        for f in FACTS[:n]:
            mem.write_rome(f)
        res[f"rome_seq_{n}"] = _score(model, tok, FACTS[:n], base_ok)
        res[f"rome_seq_{n}"]["write_seconds"] = round(time.time() - t1, 1)
        mem.clear()
        t1 = time.time()
        mem.write_memit(FACTS[:n])
        res[f"memit_{n}"] = _score(model, tok, FACTS[:n], base_ok)
        res[f"memit_{n}"]["write_seconds"] = round(time.time() - t1, 1)
        mem.clear()
    res["cov_tokens"] = mem.cov_tokens
    res["seconds"] = round(time.time() - t0, 1)
    return res
