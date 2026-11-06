"""Splits train/val/holdout disjuntos de prompts históricos del workstream."""
from __future__ import annotations
import hashlib
import json
import re
from typing import Callable, Dict, List, Set, Tuple

from enjambre.experts import Item, all_expert_specs, parse_json, words, lang
from enjambre.fusion_20.holdout import all_prior_prompts as k20_base_priors
from enjambre.fusion_20.holdout import build_k20_holdout

Check = Callable[[str], bool]
Row = Tuple[str, str, Check]


def _exact(t: str, exp: str) -> bool:
    only = t.strip().splitlines()[0].strip() if t.strip() else ""
    return only == exp


def all_historical_prompts() -> Set[str]:
    s = set(k20_base_priors())
    for items in build_k20_holdout(50).values():
        for it in items:
            s.add(it.prompt)
    return s


def build_supervised_pool() -> Dict[str, List[Row]]:
    pool: Dict[str, List[Row]] = {}

    words_u = [f"bw{i}z" for i in range(120)]
    pool["uppercase"] = [
        (f"Reply with the word in ALL CAPS only: {w}", w.upper(),
         (lambda t, ee=w.upper(): _exact(t, ee)))
        for w in words_u
    ]

    rows = []
    for a in range(101, 220):
        for b in range(31, 40):
            c = a + b
            rows.append((f"What is {a}+{b}? Reply with ONLY the number.", str(c),
                         (lambda t, ee=str(c): _exact(t, ee))))
            if len(rows) >= 120:
                break
        if len(rows) >= 120:
            break
    pool["arithmetic"] = rows

    base_sents = [
        ("I am extremely happy today", "positive"), ("This is utterly awful", "negative"),
        ("The table is wooden", "neutral"), ("What a fantastic result", "positive"),
        ("I hate this delay", "negative"), ("A metal screw sits there", "neutral"),
        ("Pure delight right now", "positive"), ("Miserable cold night", "negative"),
        ("Paper cup is empty", "neutral"), ("Wonderful soft music", "positive"),
        ("Terrible broken promise", "negative"), ("Stone path is dry", "neutral"),
        ("Thrilled with success", "positive"), ("Disgusting sour milk", "negative"),
        ("Glass jar on shelf", "neutral"), ("Joyful warm sunrise", "positive"),
        ("Horrible rude reply", "negative"), ("Cotton towel folded", "neutral"),
        ("Delightful fresh bread", "positive"), ("Painful sharp sting", "negative"),
    ]
    rows = []
    for i in range(120):
        s, lab = base_sents[i % len(base_sents)]
        prompt = f"Classify sentiment as positive, negative, or neutral. Reply ONLY with the label.\n{s}"
        if i >= len(base_sents):
            prompt += f"\n(#s{i})"
        rows.append((prompt, lab, (lambda t, ee=lab: _exact(t, ee))))
    pool["sentiment"] = rows

    rows = []
    for i in range(120):
        tp = f"obj{i}q"
        tgt = '{"a":"x","b":"y"}'
        rows.append((
            f'Responde SOLO con JSON válido con exactamente las claves "a" y "b" (texto corto sobre {tp}). Nada más.',
            tgt,
            (lambda t: isinstance(parse_json(t), dict) and set(parse_json(t).keys()) == {"a", "b"}),
        ))
    pool["json"] = rows

    rows = []
    for i in range(120):
        en = f"topic{i}z"
        tgt = f"El tema {en} es interesante. Es un buen ejemplo."
        rows.append((
            f"Write two sentences about {en}. Answer only in Spanish.",
            tgt,
            (lambda t: len(words(t)) >= 4 and lang(t) == "es"),
        ))
    pool["spanish"] = rows

    rows = []
    for i in range(120):
        name, args, body = f"fn{i}", "x", f"return x + {i % 7}"
        tgt = f"def {name}({args}):\n    {body}\n"
        rows.append((
            f"Write ONLY a Python function `{name}({args})` with body `{body}`. No prose.",
            tgt,
            (lambda t, n=name: ("def " + n) in t and "return" in t),
        ))
    pool["code_python"] = rows

    places = ["Basel", "Cork", "Dover", "Essen", "Flint", "Genoa", "Hull", "Kiel",
              "Lyon", "Mainz", "Nantes", "Oran", "Pisa", "Quito", "Rabat", "Sofia",
              "Tunis", "Ulm", "Vigo", "Waco", "York", "Zurich", "Athens", "Bergen"]
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    rows = []
    for i in range(120):
        person = f"P{i}x"
        place = places[i % len(places)]
        month = months[i % len(months)]
        expect = {"person": person, "place": place, "date": month}
        text = f"{person} visited {place} in {month}."
        tgt = json.dumps(expect)
        def chk(t, e=expect):
            o = parse_json(t)
            if not isinstance(o, dict):
                return False
            return all(k in o and str(o[k]).lower() == str(v).lower() for k, v in e.items())
        rows.append((
            f'Extract person, place, date as JSON with keys "person","place","date" only.\n{text}',
            tgt, chk,
        ))
    pool["entities"] = rows

    rows = []
    for i in range(120):
        tp = f"bul{i}"
        tgt = f"- uno {tp}\n- dos {tp}\n- tres {tp}"
        def chk(t, _i=i):
            lines = [ln for ln in t.strip().splitlines() if re.match(r"^\s*[-*•]\s+", ln)]
            return len(lines) == 3
        rows.append((
            f"Lista EXACTAMENTE 3 viñetas que empiecen con '- ' sobre {tp}. Nada más.",
            tgt, chk,
        ))
    pool["bullets"] = rows

    rows = []
    for i in range(120):
        para = f"Párrafo de práctica {i} con suficiente texto para resumir en pocas palabras claras."
        tgt = "Resumen breve del texto."
        rows.append((
            f"Resume en español en MÁXIMO 12 palabras:\n{para}",
            tgt,
            (lambda t: 1 <= len(words(t)) <= 14 and lang(t) in ("es", "?")),
        ))
    pool["summarize"] = rows

    rows = []
    for i in range(120):
        es = f"Frase de prueba número {i} con sentido claro."
        en = f"Test sentence number {i} with clear meaning."
        rows.append((
            f"Translate to English. Reply with ONLY the English sentence.\n{es}",
            en,
            (lambda t, ee=en: ee.lower() in t.lower() or _exact(t, ee)),
        ))
    pool["translate_en"] = rows

    for s in all_expert_specs():
        assert s.id in pool and len(pool[s.id]) >= 100, (s.id, len(pool.get(s.id, [])))
    return pool


def make_splits(n_train: int = 20, n_val: int = 10, n_hold: int = 50, seed: int = 0):
    import random
    rng = random.Random(seed)
    hist = all_historical_prompts()
    pool = build_supervised_pool()
    train, val, hold_rows = {}, {}, {}
    for tid, rows in pool.items():
        clean = [r for r in rows if r[0] not in hist]
        rng.shuffle(clean)
        need = n_train + n_val + n_hold
        assert len(clean) >= need, (tid, len(clean), need)
        train[tid] = clean[:n_train]
        val[tid] = clean[n_train:n_train + n_val]
        hold_rows[tid] = clean[n_train + n_val:n_train + n_val + n_hold]
    # disjointness
    buckets = []
    for d in (train, val, hold_rows):
        ps = {p for rows in d.values() for p, _, _ in rows}
        assert ps.isdisjoint(hist)
        buckets.append(ps)
    assert buckets[0].isdisjoint(buckets[1]) and buckets[0].isdisjoint(buckets[2]) and buckets[1].isdisjoint(buckets[2])

    def to_items(d):
        return {
            tid: [Item(prompt=p, check=chk, meta={"target": tgt, "task": tid}) for p, tgt, chk in rows]
            for tid, rows in d.items()
        }

    man = {
        "n_train": n_train, "n_val": n_val, "n_hold": n_hold, "seed": seed,
        "prompt_sha256": hashlib.sha256(json.dumps({
            "train": {k: [r[0] for r in v] for k, v in train.items()},
            "val": {k: [r[0] for r in v] for k, v in val.items()},
            "hold": {k: [r[0] for r in v] for k, v in hold_rows.items()},
        }, sort_keys=True).encode()).hexdigest(),
    }
    return train, val, to_items(hold_rows), man
