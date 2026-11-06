"""Holdout confirmación K20: >=50/tarea, disjunto de TODOS los splits previos."""
from __future__ import annotations
import hashlib
import json
import re
from typing import Callable, Dict, List, Set

from enjambre.experts import Item, all_expert_specs, parse_json, words, lang
from enjambre.fusion_sweep.holdout import build_new_holdout
from enjambre.fusion_confirm12.confirm_holdout import build_confirm_holdout

CheckFn = Callable[[str], bool]


def _only(t: str) -> str:
    return t.strip().splitlines()[0].strip() if t.strip() else ""


def _exact(exp: str) -> CheckFn:
    e = exp.strip()
    def chk(t, ee=e):
        return _only(t) == ee
    return chk


def all_prior_prompts() -> Set[str]:
    prompts: Set[str] = set()
    for s in all_expert_specs():
        for it in list(s.tune_items) + list(s.eval_items):
            prompts.add(it.prompt)
    for items in build_new_holdout(20).values():
        for it in items:
            prompts.add(it.prompt)
    for items in build_confirm_holdout(50).values():
        for it in items:
            prompts.add(it.prompt)
    return prompts


def build_k20_holdout(n: int = 50) -> Dict[str, List[Item]]:
    assert n >= 20
    out: Dict[str, List[Item]] = {}

    topics = [
        "ajuar", "báculo", "crisol", "diáfano", "efebo", "fronda", "gárgola", "hélice",
        "ítem", "jacinto", "káiser-nuevo", "lacre", "mármol", "nódulo", "ópalo", "papiro-x",
        "quirófano", "rótulo", "sílex-x", "tálero-x", "úlcera", "vórtice", "wéstern-x", "yermo",
        "zafir", "ámbar-x", "bóveda", "cúpula", "dintel", "éstasis", "fósforo", "glifo",
        "hércules", "índigo", "júbilo", "kermés", "lámina", "mástil", "náutica", "óbolo",
        "pétalo-x", "químio", "réplica-x", "sátiro", "tímpano", "úngula", "vial", "wadi",
        "yóquey-x", "zócalo-x", "alerce", "brezo",
    ]
    items = []
    for tp in topics[:n]:
        items.append(Item(
            prompt=f'Responde SOLO con JSON válido con exactamente las claves "a" y "b" (texto corto sobre {tp}). Nada más.',
            check=lambda t, _tp=tp: isinstance(parse_json(t), dict) and set(parse_json(t).keys()) == {"a", "b"},
            meta={"topic": tp, "split": "k20_holdout", "family": "json"},
        ))
    out["json"] = items

    topics_en = [
        "agate", "birch", "cedar", "dahlia", "ember-x", "fjord-x", "granite", "hazel",
        "iris-x", "juniper", "krill", "larch", "maple", "nutmeg", "onyx", "pewter",
        "quince", "resin", "spruce", "thyme", "urchin", "violet", "willow-x", "xylem",
        "yucca", "zinc", "anvil-x", "basil", "clove", "drift", "eider", "fern",
        "guava", "heron", "ivory", "jute", "kelp-x", "lotus", "mango", "nori",
        "olive", "peach", "quinoa-x", "reed", "sage", "teak", "umber", "vine",
        "wheat", "yarn", "zincite", "almond-x",
    ]
    items = []
    for en in topics_en[:n]:
        items.append(Item(
            prompt=f"Write two sentences about {en}. Answer only in Spanish.",
            check=lambda t: len(words(t)) >= 4 and lang(t) == "es",
            meta={"topic": en, "split": "k20_holdout", "family": "spanish"},
        ))
    out["spanish"] = items

    pairs = [
        ("El alerce crece en la montaña.", "The larch grows in the mountain."),
        ("La bóveda cubre la iglesia.", "The vault covers the church."),
        ("El dintel sostiene la puerta.", "The lintel supports the door."),
        ("Compré tomillo fresco hoy.", "I bought fresh thyme today."),
        ("El mástil vibra con el viento.", "The mast vibrates with the wind."),
        ("La gárgola escupe agua de lluvia.", "The gargoyle spits rainwater."),
        ("El ópalo brilla con colores.", "The opal shines with colors."),
        ("Hay lacre en el sobre.", "There is sealing wax on the envelope."),
        ("El yermo está seco y vacío.", "The wasteland is dry and empty."),
        ("La cúpula refleja el sol.", "The dome reflects the sun."),
        ("El jacinto florece en primavera.", "The hyacinth blooms in spring."),
        ("Guardé el rótulo en la caja.", "I kept the label in the box."),
        ("El quirófano estaba limpio.", "The operating room was clean."),
        ("La fronda da sombra fresca.", "The frond gives cool shade."),
        ("El báculo es de madera antigua.", "The staff is of old wood."),
        ("Sembramos brezo junto al camino.", "We planted heather by the path."),
        ("El crisol funde el metal.", "The crucible melts the metal."),
        ("La hélice gira muy rápido.", "The propeller spins very fast."),
        ("El nódulo aparece en la radiografía.", "The nodule appears on the X-ray."),
        ("El zafir es de un azul intenso.", "The sapphire is an intense blue."),
        ("Hay glifos en la piedra.", "There are glyphs on the stone."),
        ("El juniper crece cerca del río.", "The juniper grows near the river."),
        ("La lámina es muy delgada.", "The sheet is very thin."),
        ("El pewter brilla opaco.", "The pewter shines dull."),
        ("Comí quince en el postre.", "I ate quince for dessert."),
        ("El spruce es un árbol alto.", "The spruce is a tall tree."),
        ("La urchin vive en el fondo.", "The urchin lives on the bottom."),
        ("El xylem transporta agua.", "The xylem transports water."),
        ("La yucca tiene hojas duras.", "The yucca has hard leaves."),
        ("El zinc protege el acero.", "Zinc protects the steel."),
        ("Hay basil en la cocina.", "There is basil in the kitchen."),
        ("El clove aromatiza el té.", "The clove flavors the tea."),
        ("La fern crece en la sombra.", "The fern grows in the shade."),
        ("El heron pesca en el lago.", "The heron fishes in the lake."),
        ("Compré jute para tejer.", "I bought jute for weaving."),
        ("El lotus flota en el estanque.", "The lotus floats in the pond."),
        ("El mango está maduro.", "The mango is ripe."),
        ("Hay nori en el sushi.", "There is nori in the sushi."),
        ("La olive es verde oscura.", "The olive is dark green."),
        ("El peach es jugoso.", "The peach is juicy."),
        ("El reed crece en el pantano.", "The reed grows in the swamp."),
        ("La sage huele fuerte.", "Sage smells strong."),
        ("El teak es madera dura.", "Teak is hard wood."),
        ("El umber es un pigmento.", "Umber is a pigment."),
        ("La vine trepa el muro.", "The vine climbs the wall."),
        ("El wheat cubre el campo.", "Wheat covers the field."),
        ("Hay yarn de lana suave.", "There is soft wool yarn."),
        ("El wadi está seco en verano.", "The wadi is dry in summer."),
        ("La ungula del caballo es fuerte.", "The horse's hoof is strong."),
        ("El vial contiene medicina.", "The vial contains medicine."),
        ("El kermés tiñe de rojo.", "Kermes dyes red."),
        ("El índigo colorea la tela.", "Indigo colors the fabric."),
    ]
    items = []
    for es, en in pairs[:n]:
        items.append(Item(
            prompt=f"Translate to English. Reply with ONLY the English sentence.\n{es}",
            check=lambda t, ee=en: ee.lower() in t.lower() or _only(t).lower() == ee.lower(),
            meta={"src": es, "split": "k20_holdout", "family": "translate_en"},
        ))
    out["translate_en"] = items

    # arithmetic ranges unused before (confirm used 41-70; sweep 11-30)
    probs = [(a, b, a + b) for a in range(72, 99) for b in range(21, 29)][:n]
    items = []
    for a, b, c in probs:
        items.append(Item(
            prompt=f"What is {a}+{b}? Reply with ONLY the number.",
            check=_exact(str(c)),
            meta={"split": "k20_holdout", "family": "arithmetic"},
        ))
    out["arithmetic"] = items

    paras = [
        "Un carpintero talló un báculo de alerce y lo barnizó con resina natural del bosque cercano.",
        "La bóveda del templo mostraba glifos antiguos que los arqueólogos fotografiaron con cuidado.",
        "Ingenieros probaron una hélice nueva en el túnel de viento antes del primer vuelo de prueba.",
        "En el yermo crecen solo brezos resistentes tras años de sequía extrema en la región.",
        "El quirófano recibió lámparas LED que reducen el calor sobre el campo quirúrgico.",
        "Una joyería exhibió ópalos y zafiros tallados con láminas de oro muy finas.",
        "El crisol de laboratorio alcanzó temperaturas altas para fundir aleaciones de zinc.",
        "Botánicos midieron el crecimiento de juniper y spruce tras una temporada lluviosa.",
        "El dintel de piedra fue reforzado con barras de acero ocultas tras la restauración.",
        "Cocineros usaron tomillo, clavo y salvia para un guiso lento de cordero local.",
        "La cúpula de cristal filtra luz índigo al mediodía en el atrio del museo.",
        "Pescadores observaron un heron acechando peces junto a los reeds del estuario.",
        "El mástil de teak resistió la tormenta gracias a refuerzos de fibra de carbono.",
        "Estudiantes dibujaron gárgolas del claustro como ejercicio de perspectiva lineal.",
        "Un médico señaló un nódulo pequeño en la radiografía y pidió seguimiento en tres meses.",
        "La fábrica de pewter recicló sobras para fabricar vasos utilitarios sin plomo.",
        "Agricultores plantaron quince y mango en terrazas con riego por goteo eficiente.",
        "El wadi solo lleva agua tras las lluvias de primavera en el desierto costero.",
        "Químicos extrajeron pigmento umber de suelos locales para restaurar frescos antiguos.",
        "Una tejedora compró jute y yarn de lana para tapices de motivos náuticos.",
        "El vial sellado contenía una muestra de agua del manantial para análisis bacteriano.",
        "Escaladores usaron lacre improvisado para marcar rutas en la pared de granito.",
        "El jardín botánico añadió yucca y lotus a la sección de plantas resistentes.",
        "Marineros repararon la fronda de una vela rota con parches de tela impermeable.",
        "El laboratorio cultivó helechos (fern) bajo luz controlada para un estudio genético.",
        "Una feria rural vendió guava, peach y olive aceite de primera presión en frío.",
        "El xylem de muestras de maple se examinó al microscopio tras un invierno crudo.",
        "Arquitectos diseñaron una cúpula ligera sobre la biblioteca municipal renovada.",
        "El kermés histórico tiñó paños de prueba con un rojo profundo estable al lavado.",
        "Buzos hallaron urchins cerca de un arrecife dañado por anclas de embarcaciones.",
        "El rótulo del museo explica el origen del báculo ceremonial en tres idiomas.",
        "Ingenieros civiles midieron vibraciones del dintel tras el paso de camiones pesados.",
        "Una startup hizo envases de teak residual certificado para cosmética natural.",
        "El índigo sintético reemplazó parcialmente al natural en la industria textil local.",
        "Pastores guiaron rebaños por el yermo buscando pozos estacionales marcados en mapas.",
        "El basil fresco del huerto se usó en pesto para el comedor escolar del pueblo.",
        "Geólogos catalogaron agate y onyx en un depósito aluvial tras crecidas del río.",
        "La náutica escuela entrenó a novatos a izar velas con mástil de aluminio ligero.",
        "Un herbolario secó thyme y sage en bastidores de madera bajo techo ventilado.",
        "La gárgola restaurada volvió a canalizar agua lejos de los muros de piedra caliza.",
        "Químicos midieron fósforo en suelos agrícolas para ajustar fertilización de primavera.",
        "El júbilo de la multitud llenó la plaza tras el anuncio del festival de música.",
        "Artesanos tallaron glifo a glifo una réplica educativa para talleres escolares.",
        "El olive grove produjo menos aceite tras una helada tardía en abril.",
        "Una bióloga marcó herons anillados para estudiar migraciones costeras anuales.",
        "El pewter antiguo del ajuar familiar se restauró sin alterar marcas de orfebre.",
        "Ingenieros probaron resin epoxi en uniones de spruce para muebles exteriores.",
        "El wheat del valle se cosechó temprano por pronóstico de granizo severo.",
        "Estudiantes midieron el diámetro de un nódulo pulmonar en un simulador médico.",
        "La bóveda celeste del planetario proyectó constelaciones con un nuevo láser.",
        "Un ceramista usó engobe índigo sobre piezas de barro cocido a alta temperatura.",
        "El vine trepó la pérgola hasta cubrir por completo el patio trasero en verano.",
    ]
    items = []
    for para in paras[:n]:
        items.append(Item(
            prompt=f"Resume en español en MÁXIMO 12 palabras:\n{para}",
            check=lambda t: 1 <= len(words(t)) <= 14 and lang(t) in ("es", "?"),
            meta={"split": "k20_holdout", "family": "summarize"},
        ))
    out["summarize"] = items

    words_u = [
        "agate", "birch", "cedar", "dahlia", "fjordx", "granite", "hazel", "irisx",
        "juniper", "krill", "larch", "maple", "nutmeg", "onyx", "pewter", "quince",
        "resin", "spruce", "thyme", "urchin", "violet", "xylem", "yucca", "zinc",
        "basil", "clove", "drift", "eider", "fern", "guava", "heron", "ivory",
        "jute", "lotus", "mango", "nori", "olive", "peach", "reed", "sage",
        "teak", "umber", "vine", "wheat", "yarn", "wadi", "vial", "grommet",
        "lintel", "mast", "opal", "crucible",
    ]
    items = []
    for w in words_u[:n]:
        items.append(Item(
            prompt=f"Reply with the word in ALL CAPS only: {w}",
            check=_exact(w.upper()),
            meta={"split": "k20_holdout", "family": "uppercase", "word": w},
        ))
    out["uppercase"] = items

    sents = [
        ("Absolutely wonderful news", "positive"), ("Completely ruined my day", "negative"),
        ("A granite block rests", "neutral"), ("So grateful for help", "positive"),
        ("Disgusting filthy kitchen", "negative"), ("The lintel is stone", "neutral"),
        ("Perfect calm weekend", "positive"), ("Bitter cruel remark", "negative"),
        ("Pewter cup is empty", "neutral"), ("Thrilled beyond belief", "positive"),
        ("Horrible aching tooth", "negative"), ("Maple leaf on ground", "neutral"),
        ("Joyful bright morning", "positive"), ("Miserable wet shoes", "negative"),
        ("Cedar plank is dry", "neutral"), ("Delightful fresh thyme", "positive"),
        ("Awful sticky floor", "negative"), ("Onyx ring on tray", "neutral"),
        ("Fantastic clever plan", "positive"), ("Dreadful long night", "negative"),
        ("Spruce cone is closed", "neutral"), ("Lovely gentle breeze", "positive"),
        ("Nasty rude gesture", "negative"), ("Teak chair is firm", "neutral"),
        ("Blissful quiet nap", "positive"), ("Rotten sour milk", "negative"),
        ("Jute bag is full", "neutral"), ("Superb clear design", "positive"),
        ("Ghastly burnt smell", "negative"), ("Reed mat lies flat", "neutral"),
        ("Cheerful warm soup", "positive"), ("Ugly harsh words", "negative"),
        ("Zinc sheet is thin", "neutral"), ("Radiant sunny porch", "positive"),
        ("Wretched cold room", "negative"), ("Vial sits on shelf", "neutral"),
        ("Ecstatic final match", "positive"), ("Vile bitter coffee", "negative"),
        ("Yarn spool is neat", "neutral"), ("Glorious soft bread", "positive"),
        ("Appalling late train", "negative"), ("Wadi bed is dusty", "neutral"),
        ("Charming small cafe", "positive"), ("Bleak empty hall", "negative"),
        ("Olive jar is sealed", "neutral"), ("Happy reunion lunch", "positive"),
        ("Terrible broken clasp", "negative"), ("Basil pot is green", "neutral"),
        ("Wonderful soft music", "positive"), ("Painful sharp sting", "negative"),
        ("Clove jar is closed", "neutral"), ("Brilliant quick fix", "positive"),
    ]
    items = []
    for s, lab in sents[:n]:
        items.append(Item(
            prompt=f"Classify sentiment as positive, negative, or neutral. Reply ONLY with the label.\n{s}",
            check=_exact(lab),
            meta={"split": "k20_holdout", "family": "sentiment"},
        ))
    out["sentiment"] = items

    ents = [
        ("Ava visited Bergen in March.", {"person": "Ava", "place": "Bergen", "date": "March"}),
        ("Ben reached Cusco in April.", {"person": "Ben", "place": "Cusco", "date": "April"}),
        ("Cia left Dakar in May.", {"person": "Cia", "place": "Dakar", "date": "May"}),
        ("Dan saw Erbil in June.", {"person": "Dan", "place": "Erbil", "date": "June"}),
        ("Eve flew Fez in July.", {"person": "Eve", "place": "Fez", "date": "July"}),
        ("Fay met Ghent in August.", {"person": "Fay", "place": "Ghent", "date": "August"}),
        ("Gus reached Hilo in September.", {"person": "Gus", "place": "Hilo", "date": "September"}),
        ("Hana visited Ibiza in October.", {"person": "Hana", "place": "Ibiza", "date": "October"}),
        ("Ian saw Juneau in November.", {"person": "Ian", "place": "Juneau", "date": "November"}),
        ("Joy left Kobe in December.", {"person": "Joy", "place": "Kobe", "date": "December"}),
        ("Ken reached Leeds in January.", {"person": "Ken", "place": "Leeds", "date": "January"}),
        ("Lia visited Metz in February.", {"person": "Lia", "place": "Metz", "date": "February"}),
        ("Max saw Nice in March.", {"person": "Max", "place": "Nice", "date": "March"}),
        ("Nia left Omsk in April.", {"person": "Nia", "place": "Omsk", "date": "April"}),
        ("Omar reached Perm in May.", {"person": "Omar", "place": "Perm", "date": "May"}),
        ("Pia visited Riga in June.", {"person": "Pia", "place": "Riga", "date": "June"}),
        ("Quin saw Split in July.", {"person": "Quin", "place": "Split", "date": "July"}),
        ("Rae left Tulsa in August.", {"person": "Rae", "place": "Tulsa", "date": "August"}),
        ("Sam reached Ulm in September.", {"person": "Sam", "place": "Ulm", "date": "September"}),
        ("Tess visited Vigo in October.", {"person": "Tess", "place": "Vigo", "date": "October"}),
        ("Uri saw Waco in November.", {"person": "Uri", "place": "Waco", "date": "November"}),
        ("Val left York in December.", {"person": "Val", "place": "York", "date": "December"}),
        ("Wes reached Zurich in January.", {"person": "Wes", "place": "Zurich", "date": "January"}),
        ("Xia visited Accra in February.", {"person": "Xia", "place": "Accra", "date": "February"}),
        ("Yan saw Basel in March.", {"person": "Yan", "place": "Basel", "date": "March"}),
        ("Zoe left Cork in April.", {"person": "Zoe", "place": "Cork", "date": "April"}),
        ("Abe reached Dover in May.", {"person": "Abe", "place": "Dover", "date": "May"}),
        ("Bea visited Essen in June.", {"person": "Bea", "place": "Essen", "date": "June"}),
        ("Cal saw Flint in July.", {"person": "Cal", "place": "Flint", "date": "July"}),
        ("Dee left Genoa in August.", {"person": "Dee", "place": "Genoa", "date": "August"}),
        ("Eli reached Hull in September.", {"person": "Eli", "place": "Hull", "date": "September"}),
        ("Flo visited Innsbruck in October.", {"person": "Flo", "place": "Innsbruck", "date": "October"}),
        ("Gil saw Jeddah in November.", {"person": "Gil", "place": "Jeddah", "date": "November"}),
        ("Hue left Kiel in December.", {"person": "Hue", "place": "Kiel", "date": "December"}),
        ("Ivy reached Lyon in January.", {"person": "Ivy", "place": "Lyon", "date": "January"}),
        ("Jed visited Mainz in February.", {"person": "Jed", "place": "Mainz", "date": "February"}),
        ("Kim saw Nantes in March.", {"person": "Kim", "place": "Nantes", "date": "March"}),
        ("Lou left Oran in April.", {"person": "Lou", "place": "Oran", "date": "April"}),
        ("Meg reached Pisa in May.", {"person": "Meg", "place": "Pisa", "date": "May"}),
        ("Ned visited Quito in June.", {"person": "Ned", "place": "Quito", "date": "June"}),
        ("Ora saw Recife in July.", {"person": "Ora", "place": "Recife", "date": "July"}),
        ("Pam left Sofia in August.", {"person": "Pam", "place": "Sofia", "date": "August"}),
        ("Raj reached Tunis in September.", {"person": "Raj", "place": "Tunis", "date": "September"}),
        ("Sue visited Uppsala in October.", {"person": "Sue", "place": "Uppsala", "date": "October"}),
        ("Tim saw Valencia in November.", {"person": "Tim", "place": "Valencia", "date": "November"}),
        ("Uma left Wuhan in December.", {"person": "Uma", "place": "Wuhan", "date": "December"}),
        ("Vic reached Xiamen in January.", {"person": "Vic", "place": "Xiamen", "date": "January"}),
        ("Wen visited Yangon in February.", {"person": "Wen", "place": "Yangon", "date": "February"}),
        ("Xan saw Zagreb in March.", {"person": "Xan", "place": "Zagreb", "date": "March"}),
        ("Yul left Aarhus in April.", {"person": "Yul", "place": "Aarhus", "date": "April"}),
        ("Zed reached Bilbao in May.", {"person": "Zed", "place": "Bilbao", "date": "May"}),
        ("Ann visited Cardiff in June.", {"person": "Ann", "place": "Cardiff", "date": "June"}),
    ]
    def ent_check(t, expect):
        o = parse_json(t)
        if not isinstance(o, dict):
            return False
        for k, v in expect.items():
            if k not in o or str(o[k]).lower() != str(v).lower():
                return False
        return True
    items = []
    for text, expect in ents[:n]:
        items.append(Item(
            prompt=f'Extract person, place, date as JSON with keys "person","place","date" only.\n{text}',
            check=lambda t, e=expect: ent_check(t, e),
            meta={"split": "k20_holdout", "family": "entities"},
        ))
    out["entities"] = items

    topics_b = [
        "alerce", "bóvedas", "crisoles", "dinteles", "gárgolas", "hélices", "jacintos",
        "lacres", "mármoles", "nódulos", "ópalos", "quirófanos", "rótulos", "yermos",
        "zafiros", "cúpulas", "frondas", "glifos", "índigo", "júbilos", "láminas",
        "mástiles", "pewter", "quince", "spruce", "thyme", "urchins", "xylem", "yucca",
        "zinc", "basil", "clove", "ferns", "guava", "herons", "jute", "lotus", "mango",
        "nori", "olive", "peach", "reeds", "sage", "teak", "umber", "vines", "wheat",
        "yarn", "wadi", "vials", "lintels", "masts",
    ]
    items = []
    for tp in topics_b[:n]:
        def chk(t, _tp=tp):
            lines = [ln for ln in t.strip().splitlines() if re.match(r"^\s*[-*•]\s+", ln)]
            return len(lines) == 3
        items.append(Item(
            prompt=f"Lista EXACTAMENTE 3 viñetas que empiecen con '- ' sobre {tp}. Nada más.",
            check=chk,
            meta={"split": "k20_holdout", "family": "bullets"},
        ))
    out["bullets"] = items

    specs_fn = [
        ("mul3", "x", "return x * 3"),
        ("div2", "x", "return x / 2"),
        ("rem", "a, b", "return a % b"),
        ("floordiv", "a, b", "return a // b"),
        ("pow4", "x", "return x ** 4"),
        ("neg2", "x", "return -2 * x"),
        ("abs2", "x", "return abs(x) * 2"),
        ("clamp1", "x", "return min(1, max(0, x))"),
        ("is_pos", "x", "return x > 0"),
        ("is_neg", "x", "return x < 0"),
        ("third", "xs", "return xs[2]"),
        ("mid3", "xs", "return xs[1:4]"),
        ("rev2", "xs", "return list(reversed(xs))"),
        ("sum_xs", "xs", "return sum(xs)"),
        ("mean_xs", "xs", "return sum(xs) / len(xs)"),
        ("join_sp", "a, b", "return a + ' ' + b"),
        ("rep3", "s", "return s * 3"),
        ("cap1", "s", "return s.capitalize()"),
        ("title1", "s", "return s.title()"),
        ("isdigit1", "s", "return s.isdigit()"),
        ("list2", "a, b", "return [a, b]"),
        ("tuple2", "a, b", "return (a, b)"),
        ("dict1", "k, v", "return {k: v}"),
        ("get0", "xs", "return xs[0] if xs else None"),
        ("len2", "a, b", "return len(a) + len(b)"),
        ("max3", "a, b, c", "return max(a, b, c)"),
        ("min3", "a, b, c", "return min(a, b, c)"),
        ("avg2", "a, b", "return (a + b) / 2"),
        ("gt0", "x", "return x > 0"),
        ("lte0", "x", "return x <= 0"),
        ("eq0", "x", "return x == 0"),
        ("neq0", "x", "return x != 0"),
        ("and3", "a, b, c", "return a and b and c"),
        ("or3", "a, b, c", "return a or b or c"),
        ("not2", "x", "return not bool(x)"),
        ("str2", "x", "return str(x) * 2"),
        ("int1", "x", "return int(x)"),
        ("float1", "x", "return float(x)"),
        ("bool1", "x", "return bool(x)"),
        ("none2", "", "return None"),
        ("true2", "", "return True"),
        ("false2", "", "return False"),
        ("zero2", "", "return 0"),
        ("one2", "", "return 1"),
        ("two", "", "return 2"),
        ("pair_sum", "a, b", "return (a + b, a * b)"),
        ("safe_div", "a, b", "return a / b if b else 0"),
        ("endswith_x", "s", "return s.endswith('x')"),
        ("startswith_a", "s", "return s.startswith('a')"),
        ("strip2", "s", "return s.strip()"),
        ("lower2", "s", "return s.lower()"),
        ("upper2", "s", "return s.upper()"),
    ]
    items = []
    for name, args, body in specs_fn[:n]:
        items.append(Item(
            prompt=f"Write ONLY a Python function `{name}({args})` with body `{body}`. No prose.",
            check=lambda t, n=name: ("def " + n) in t and "return" in t,
            meta={"split": "k20_holdout", "family": "code_python", "fn": name},
        ))
    out["code_python"] = items

    ids = [s.id for s in all_expert_specs()]
    for i in ids:
        assert i in out and len(out[i]) >= n, (i, len(out.get(i, [])))
    return out


def assert_no_overlap(holdout: Dict[str, List[Item]]) -> None:
    prior = all_prior_prompts()
    for tid, items in holdout.items():
        for it in items:
            assert it.prompt not in prior, f"overlap {tid}: {it.prompt[:70]}"


def holdout_manifest(holdout: Dict[str, List[Item]]) -> Dict:
    audit = {tid: [{"prompt": it.prompt, "meta": it.meta} for it in items] for tid, items in holdout.items()}
    blob = json.dumps(audit, sort_keys=True, ensure_ascii=False).encode()
    return {
        "n_per_task": {k: len(v) for k, v in holdout.items()},
        "split": "k20_holdout",
        "prompt_sha256": hashlib.sha256(blob).hexdigest(),
        "items": audit,
    }
