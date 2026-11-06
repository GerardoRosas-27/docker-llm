"""Holdout NUEVO >=20/tarea, disjunto de tune/eval históricos de experts.py.

Templates distintos y temas no usados en los sets originales (auditoría manual de no solape).
Solo para confirmación final — NUNCA para seleccionar K/layout/escala.
"""
from __future__ import annotations
import json
import re
from typing import Callable, Dict, List

from enjambre.experts import Item, ExpertSpec, all_expert_specs, parse_json, words, lang, bullets

CheckFn = Callable[[str], bool]


def _only(t: str) -> str:
    return t.strip().splitlines()[0].strip() if t.strip() else ""


def _exact(exp: str) -> CheckFn:
    e = exp.strip()
    def chk(t, ee=e):
        return _only(t) == ee
    return chk


def _exact_ci(exp: str) -> CheckFn:
    e = exp.strip().lower()
    def chk(t, ee=e):
        return _only(t).lower() == ee
    return chk


def build_new_holdout(n: int = 20) -> Dict[str, List[Item]]:
    """Genera >=n ítems nuevos por cada uno de los 10 ids."""
    out: Dict[str, List[Item]] = {}

    # json: topics not in original tune/eval lists
    topics = ["satélite", "quimera", "bambú", "tornado", "cristal", "fósil", "brújula", "ancla",
              "pétalo", "carbón", "linterna", "coral", "humedal", "glaciar", "orquesta", "píxel",
              "algoritmo", "nebula", "cantera", "timbre", "espora", "farola"]
    items = []
    for tp in topics[:n]:
        items.append(Item(
            prompt=f'Responde SOLO con JSON válido con exactamente las claves "a" y "b" (texto corto sobre {tp}). Nada más.',
            check=lambda t, _tp=tp: isinstance(parse_json(t), dict) and set(parse_json(t).keys()) == {"a", "b"},
            meta={"topic": tp, "split": "new_holdout"},
        ))
    out["json"] = items

    topics_en = ["zebras", "glaciers", "comets", "orchids", "submarines", "lanterns", "canyons",
                 "fossils", "mosaics", "turbines", "orchards", "paddocks", "fjords", "basalt",
                 "ember", "quill", "satchel", "trellis", "harbor", "meadows", "ciphers", "dunes"]
    items = []
    for en in topics_en[:n]:
        items.append(Item(
            prompt=f"Write two sentences about {en}. Answer only in Spanish.",
            check=lambda t: len(words(t)) >= 4 and lang(t) == "es",
            meta={"topic": en, "split": "new_holdout"},
        ))
    out["spanish"] = items

    # translate_en: Spanish phrases not in original
    pairs = [
        ("El faro brilla de noche.", "The lighthouse shines at night."),
        ("La brújula apunta al norte.", "The compass points north."),
        ("El glaciar se derrite lento.", "The glacier melts slowly."),
        ("Compramos pan fresco hoy.", "We bought fresh bread today."),
        ("El tren llega puntual.", "The train arrives on time."),
        ("Ella lee un mapa antiguo.", "She reads an old map."),
        ("Los corales necesitan agua limpia.", "Corals need clean water."),
        ("Guardé la llave en el cajón.", "I kept the key in the drawer."),
        ("El viento mueve las turbinas.", "The wind moves the turbines."),
        ("Hay niebla en el valle.", "There is fog in the valley."),
        ("El perro espera en la puerta.", "The dog waits at the door."),
        ("Cociné arroz con verduras.", "I cooked rice with vegetables."),
        ("El museo abre a las nueve.", "The museum opens at nine."),
        ("Perdí mi sombrero azul.", "I lost my blue hat."),
        ("La orquesta ensaya mañana.", "The orchestra rehearses tomorrow."),
        ("El puente cruza el río.", "The bridge crosses the river."),
        ("Necesito una linterna nueva.", "I need a new lantern."),
        ("El ancla está oxidada.", "The anchor is rusty."),
        ("Sembramos bambú en primavera.", "We planted bamboo in spring."),
        ("El pétalo cayó al suelo.", "The petal fell to the ground."),
        ("Apagué la farola rota.", "I turned off the broken streetlamp."),
        ("El cifrado es difícil.", "The cipher is hard."),
    ]
    items = []
    for es, en in pairs[:n]:
        items.append(Item(
            prompt=f"Translate to English. Reply with ONLY the English sentence.\n{es}",
            check=lambda t, ee=en: ee.lower() in t.lower() or _only(t).lower() == ee.lower(),
            meta={"src": es, "split": "new_holdout"},
        ))
    out["translate_en"] = items

    # arithmetic
    probs = [(a, b, a + b) for a in range(11, 31) for b in range(3, 9)][:n]
    items = []
    for a, b, c in probs:
        items.append(Item(
            prompt=f"What is {a}+{b}? Reply with ONLY the number.",
            check=_exact(str(c)),
            meta={"split": "new_holdout"},
        ))
    out["arithmetic"] = items

    # summarize: long-ish texts with word cap check (reuse experts style - max words)
    paras = [
        "Un satélite artificial orbita la Tierra y envía datos de clima a estaciones terrestres cada hora.",
        "La cantera abandonada se llenó de agua de lluvia y ahora atrae aves migratorias en otoño.",
        "El algoritmo clasifica imágenes médicas con cuidado y sugiere posibles diagnósticos al médico.",
        "En el humedal crecen juncos altos donde anidan ranas y libélulas durante la primavera húmeda.",
        "La nebula lejana brilla con colores tenues que los telescopios capturan en noches sin luna.",
        "Un reloj de sol antiguo marca las horas en el patio del museo regional del pueblo.",
        "Los buzos documentaron un barco hundido cubierto de coral cerca de la costa rocosa.",
        "La fábrica moderna usa turbinas eólicas para reducir el consumo de electricidad fósil.",
        "El mapa cifrado escondía coordenadas de un depósito de fósiles bajo la colina este.",
        "Niños plantaron bambú junto al río para frenar la erosión de la orilla arenosa.",
        "La farola solar se enciende sola al anochecer frente a la plaza del mercado.",
        "Un compositor escribió una pieza corta para orquesta basada en sonidos de glaciar.",
        "La brújula magnética falló cerca del depósito de hierro y el grupo se reorientó con GPS.",
        "En la biblioteca digital se indexan manuscritos escaneados de siglos pasados.",
        "El ancla del velero se atascó entre rocas basálticas durante la marea baja.",
        "Científicos midieron esporas en el aire tras la tormenta de polvo del desierto.",
        "El puente peatonal nuevo conecta el parque con el barrio de talleres artesanales.",
        "Una linterna de mano reveló pinturas rupestres en una cueva estrecha y húmeda.",
        "El tren ligero reduce el tráfico matutino entre el puerto y el centro urbano.",
        "Agricultores usaron sensores de humedad para regar solo cuando el suelo lo necesita.",
        "El timbre inalámbrico avisa en el teléfono cuando llega un paquete a la puerta.",
        "Un equipo reparó la turbina dañada antes del pico de demanda eléctrica invernal.",
    ]
    items = []
    for para in paras[:n]:
        items.append(Item(
            prompt=f"Resume en español en MÁXIMO 12 palabras:\n{para}",
            check=lambda t: 1 <= len(words(t)) <= 14 and lang(t) in ("es", "?"),
            meta={"split": "new_holdout"},
        ))
    out["summarize"] = items

    # uppercase
    words_u = ["quasar", "bamboo", "fjord", "glyph", "nexus", "prism", "quartz", "raven",
               "sable", "thorn", "umbra", "vortex", "willow", "xenon", "yacht", "zephyr",
               "anchor", "basalt", "cipher", "ember", "flare", "grove"]
    items = []
    for w in words_u[:n]:
        items.append(Item(
            prompt=f"Reply with the word in ALL CAPS only: {w}",
            check=_exact(w.upper()),
            meta={"split": "new_holdout"},
        ))
    out["uppercase"] = items

    # sentiment
    sents = [
        ("This gift made my week", "positive"), ("The service was horrible", "negative"),
        ("The box is wooden", "neutral"), ("I adore this melody", "positive"),
        ("Everything tastes rotten", "negative"), ("A metal spoon rests", "neutral"),
        ("Pure joy right now", "positive"), ("Painful delay again", "negative"),
        ("Stone wall stands there", "neutral"), ("Wonderful surprise today", "positive"),
        ("Nasty rude reply", "negative"), ("Paper sheet is blank", "neutral"),
        ("Thrilled with results", "positive"), ("Awful burning smell", "negative"),
        ("Glass jar on shelf", "neutral"), ("Happy reunion tonight", "positive"),
        ("Terrible broken promise", "negative"), ("Cotton fabric feels soft", "neutral"),
        ("Delightful warm bread", "positive"), ("Miserable cold rain", "negative"),
        ("Plastic chair nearby", "neutral"), ("Brilliant clear sky", "positive"),
    ]
    items = []
    for s, lab in sents[:n]:
        items.append(Item(
            prompt=f"Classify sentiment as positive, negative, or neutral. Reply ONLY with the label.\n{s}",
            check=_exact(lab),
            meta={"split": "new_holdout"},
        ))
    out["sentiment"] = items

    # entities - expect JSON keys person/place/date loosely like original
    ents = [
        ("Ada visited Lima in May.", {"person": "Ada", "place": "Lima", "date": "May"}),
        ("Omar reached Cairo on June.", {"person": "Omar", "place": "Cairo", "date": "June"}),
        ("Nina left Oslo in March.", {"person": "Nina", "place": "Oslo", "date": "March"}),
        ("Leo saw Paris in April.", {"person": "Leo", "place": "Paris", "date": "April"}),
        ("Mia flew Tokyo in July.", {"person": "Mia", "place": "Tokyo", "date": "July"}),
        ("Ira met Rome in August.", {"person": "Ira", "place": "Rome", "date": "August"}),
        ("Kai reached Seoul in September.", {"person": "Kai", "place": "Seoul", "date": "September"}),
        ("Eva visited Berlin in October.", {"person": "Eva", "place": "Berlin", "date": "October"}),
        ("Noa saw Madrid in November.", {"person": "Noa", "place": "Madrid", "date": "November"}),
        ("Pia left Dublin in December.", {"person": "Pia", "place": "Dublin", "date": "December"}),
        ("Sam reached Boston in January.", {"person": "Sam", "place": "Boston", "date": "January"}),
        ("Uma visited Quito in February.", {"person": "Uma", "place": "Quito", "date": "February"}),
        ("Rex saw Denver in May.", {"person": "Rex", "place": "Denver", "date": "May"}),
        ("Zoe left Miami in June.", {"person": "Zoe", "place": "Miami", "date": "June"}),
        ("Yan reached Austin in July.", {"person": "Yan", "place": "Austin", "date": "July"}),
        ("Gil visited Boise in August.", {"person": "Gil", "place": "Boise", "date": "August"}),
        ("Hana saw Tulsa in September.", {"person": "Hana", "place": "Tulsa", "date": "September"}),
        ("Ida left Nome in October.", {"person": "Ida", "place": "Nome", "date": "October"}),
        ("Jon reached Reno in November.", {"person": "Jon", "place": "Reno", "date": "November"}),
        ("Kim visited Tampa in December.", {"person": "Kim", "place": "Tampa", "date": "December"}),
        ("Liz saw Omaha in January.", {"person": "Liz", "place": "Omaha", "date": "January"}),
        ("Mel left Akron in February.", {"person": "Mel", "place": "Akron", "date": "February"}),
    ]
    def ent_check(t, expect):
        o = parse_json(t)
        if not isinstance(o, dict):
            return False
        # require keys present; values case-insensitive match expected if present
        for k, v in expect.items():
            if k not in o:
                return False
            if str(o[k]).lower() != str(v).lower():
                return False
        return True
    items = []
    for text, expect in ents[:n]:
        items.append(Item(
            prompt=f'Extract person, place, date as JSON with keys "person","place","date" only.\n{text}',
            check=lambda t, e=expect: ent_check(t, e),
            meta={"split": "new_holdout"},
        ))
    out["entities"] = items

    # bullets exactly 3
    topics_b = ["cometas", "faros", "esporas", "anclas", "turbina", "glaciares", "mosaicos",
                "cañones", "tréboles", "píxeles", "brújulas", "humedales", "orquestas", "satélites",
                "linternas", "fósiles", "bambúes", "nébulas", "canteras", "timbres", "farolas", "cifras"]
    items = []
    for tp in topics_b[:n]:
        def chk(t, _tp=tp):
            lines = [ln for ln in t.strip().splitlines() if re.match(r"^\s*[-*•]\s+", ln)]
            return len(lines) == 3
        items.append(Item(
            prompt=f"Lista EXACTAMENTE 3 viñetas que empiecen con '- ' sobre {tp}. Nada más.",
            check=chk,
            meta={"split": "new_holdout"},
        ))
    out["bullets"] = items

    # code_python: simple function stubs
    specs = [
        ("add", "a, b", "return a + b"),
        ("sub", "a, b", "return a - b"),
        ("dbl", "x", "return x * 2"),
        ("neg", "x", "return -x"),
        ("inc", "x", "return x + 1"),
        ("dec", "x", "return x - 1"),
        ("square", "x", "return x * x"),
        ("ident", "x", "return x"),
        ("zero", "", "return 0"),
        ("one", "", "return 1"),
        ("pair", "a, b", "return (a, b)"),
        ("first", "xs", "return xs[0]"),
        ("last", "xs", "return xs[-1]"),
        ("length", "xs", "return len(xs)"),
        ("empty", "xs", "return len(xs) == 0"),
        ("absv", "x", "return abs(x)"),
        ("maxi", "a, b", "return max(a, b)"),
        ("mini", "a, b", "return min(a, b)"),
        ("bool_not", "x", "return not x"),
        ("true", "", "return True"),
        ("false", "", "return False"),
        ("none", "", "return None"),
    ]
    items = []
    for name, args, body in specs[:n]:
        items.append(Item(
            prompt=f"Write ONLY a Python function `{name}({args})` with body `{body}`. No prose.",
            check=lambda t, n=name: ("def " + n) in t and "return" in t,
            meta={"split": "new_holdout"},
        ))
    out["code_python"] = items

    # validate
    ids = [s.id for s in all_expert_specs()]
    for i in ids:
        assert i in out and len(out[i]) >= n, (i, len(out.get(i, [])))
    return out
