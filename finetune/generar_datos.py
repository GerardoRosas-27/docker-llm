"""Genera datos sintéticos de una tienda de mochilas (semilla fija).

Salida (en finetune/data/, no se sube a git):
  catalogo.json  - productos: modelo, precio, tela, capacidad, color
  train.jsonl    - conversaciones de ventas en formato chat (messages)
  holdout.jsonl  - preguntas con redacción que NO aparece en train, para evaluar
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

LINEAS = ["Andes", "Sierra", "Cumbre", "Ruta", "Nómada", "Volcán", "Selva", "Bahía",
          "Desierto", "Glaciar", "Pampa", "Cañón", "Laguna", "Mesa", "Tundra"]
TIPOS = ["Urbana", "Trek", "Escolar", "Laptop", "Viaje"]
TELAS = ["nylon balístico", "poliéster reciclado", "lona de algodón", "cordura 500D",
         "ripstop", "piel sintética", "mezclilla"]
COLORES = ["negro", "azul marino", "verde olivo", "gris", "rojo", "arena", "mostaza"]

TPL_TRAIN = {
    "precio": ["¿Cuánto cuesta la mochila {m}?", "Precio de la {m}", "¿Qué precio tiene la {m}?"],
    "tela": ["¿De qué tela es la mochila {m}?", "¿Qué material usa la {m}?", "Tela de la {m}"],
    "capacidad": ["¿Qué capacidad tiene la {m}?", "¿Cuántos litros le caben a la {m}?"],
    "color": ["¿En qué color viene la {m}?", "Color de la {m}"],
    "ficha": ["Dame la ficha de la mochila {m}", "Háblame de la {m}"],
}
TPL_HOLDOUT = {
    "precio": "Oye, ¿en cuánto me sale la mochila {m}?",
    "tela": "¿Con qué tela está hecha la {m}?",
}
SYSTEM = "Eres el asistente de ventas de Mochilas Rosas. Responde con datos exactos del catálogo."


def catalogo(rng: random.Random, n: int) -> list[dict]:
    nombres = [f"{l} {t}" for l in LINEAS for t in TIPOS]
    rng.shuffle(nombres)
    out = []
    for i, nombre in enumerate(nombres[:n]):
        out.append({
            "sku": f"MR-{i + 1:03d}",
            "modelo": nombre,
            "precio": rng.randrange(499, 2999, 50),
            "tela": rng.choice(TELAS),
            "capacidad_l": rng.choice([12, 15, 20, 25, 30, 35, 40, 45, 55]),
            "color": rng.choice(COLORES),
        })
    return out


def respuesta(p: dict, campo: str) -> str:
    m = p["modelo"]
    if campo == "precio":
        return f"La mochila {m} cuesta ${p['precio']} MXN."
    if campo == "tela":
        return f"La mochila {m} está hecha de {p['tela']}."
    if campo == "capacidad":
        return f"La mochila {m} tiene {p['capacidad_l']} litros de capacidad."
    if campo == "color":
        return f"La mochila {m} viene en color {p['color']}."
    return (f"Mochila {m} ({p['sku']}): ${p['precio']} MXN, tela {p['tela']}, "
            f"{p['capacidad_l']} L, color {p['color']}.")


def chat(q: str, a: str) -> dict:
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": q},
                         {"role": "assistant", "content": a}]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--productos", type=int, default=30)
    ap.add_argument("--out", default=str(Path(__file__).parent / "data"))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    cat = catalogo(rng, a.productos)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "catalogo.json").write_text(json.dumps(cat, ensure_ascii=False, indent=2), encoding="utf-8")

    train = []
    for p in cat:
        for campo, tpls in TPL_TRAIN.items():
            for t in tpls:
                train.append(chat(t.format(m=p["modelo"]), respuesta(p, campo)))
    # preguntas de comparación / rango de precio
    for _ in range(len(cat) * 2):
        x, y = rng.sample(cat, 2)
        barata = x if x["precio"] <= y["precio"] else y
        train.append(chat(f"¿Cuál es más barata, la {x['modelo']} o la {y['modelo']}?",
                          f"La {barata['modelo']} es más barata: ${barata['precio']} MXN."))
    rng.shuffle(train)
    holdout = [{"sku": p["sku"], "campo": c, "pregunta": t.format(m=p["modelo"]),
                "esperado": str(p["precio"]) if c == "precio" else p["tela"], "system": SYSTEM}
               for p in cat for c, t in TPL_HOLDOUT.items()]
    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "holdout.jsonl", "w", encoding="utf-8") as f:
        for r in holdout:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(cat)} productos, {len(train)} ejemplos de entrenamiento, {len(holdout)} holdout -> {out}")


if __name__ == "__main__":
    main()
