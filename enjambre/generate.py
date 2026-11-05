"""Generación greedy compartida (CPU, semillas fijas)."""
from __future__ import annotations
import math
from typing import List, Optional

import torch


def chat_text(tok, prompt: str, system: Optional[str] = None) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(
    model, tok, prompts: List[str], *, system: Optional[str] = None,
    max_new_tokens: int = 80, batch_size: int = 8,
) -> List[str]:
    device = next(model.parameters()).device
    outs: List[str] = []
    for i in range(0, len(prompts), batch_size):
        batch = [chat_text(tok, p, system) for p in prompts[i:i + batch_size]]
        enc = tok(batch, return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            repetition_penalty=1.0,
        )
        plen = enc["input_ids"].shape[1]
        for row in gen:
            outs.append(tok.decode(row[plen:], skip_special_tokens=True).strip())
    return outs


@torch.no_grad()
def generate_one(model, tok, prompt: str, **kw) -> str:
    return generate(model, tok, [prompt], **kw)[0]


PPL_TEXT = (
    "The city library opened its doors early on Saturday morning. Families arrived with "
    "children who wanted to borrow picture books, while students looked for quiet tables.\n"
    "Water boils at one hundred degrees Celsius at sea level.\n"
    "La biblioteca de la ciudad abrió temprano el sábado. Las familias llegaron con niños.\n"
    "The Earth orbits the Sun once every year, and the Moon orbits the Earth roughly once every month."
)


@torch.no_grad()
def perplexity(model, tok, text: str = PPL_TEXT) -> float:
    device = next(model.parameters()).device
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    loss = model(ids, labels=ids).loss.item()
    return float(math.exp(loss))
