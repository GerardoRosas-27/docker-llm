"""Evalúa exactitud en precios y telas sobre el holdout (generación greedy)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluar(model, tok, holdout: list[dict], max_new_tokens: int = 40) -> dict:
    model.eval()
    aciertos = {"precio": [0, 0], "tela": [0, 0]}
    ejemplos = []
    t0 = time.time()
    for r in holdout:
        msgs = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["pregunta"]}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        ok = r["esperado"].lower() in txt.lower().replace(",", "")
        aciertos[r["campo"]][0] += ok
        aciertos[r["campo"]][1] += 1
        if len(ejemplos) < 6:
            ejemplos.append({"pregunta": r["pregunta"], "esperado": r["esperado"], "respuesta": txt, "ok": ok})
    res = {c: round(a / n, 4) for c, (a, n) in aciertos.items()}
    res["total"] = round(sum(a for a, _ in aciertos.values()) / sum(n for _, n in aciertos.values()), 4)
    res["n"] = sum(n for _, n in aciertos.values())
    res["segundos"] = round(time.time() - t0, 1)
    res["ejemplos"] = ejemplos
    return res


def cargar_holdout(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--holdout", default=str(Path(__file__).parent / "data" / "holdout.jsonl"))
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    print(json.dumps(evaluar(m, tok, cargar_holdout(a.holdout)), ensure_ascii=False, indent=2))
