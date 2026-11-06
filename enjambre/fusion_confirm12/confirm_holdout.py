"""Holdout de confirmación INDEPENDIENTE >=50/tarea.

Disjunto de tune/eval históricos y del holdout n=20 del barrido.
Semilla de generación fija; prompts hasheados en manifesto.
"""
from __future__ import annotations
import hashlib
import json
import re
from typing import Callable, Dict, List, Set

from enjambre.experts import Item, all_expert_specs, parse_json, words, lang
from enjambre.fusion_sweep.holdout import build_new_holdout

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


def build_confirm_holdout(n: int = 50) -> Dict[str, List[Item]]:
    """n>=50 preferido; genera n ítems nuevos por cada tarea."""
    assert n >= 20
    out: Dict[str, List[Item]] = {}

    # --- json: topics nuevos (no en experts ni holdout barrido) ---
    topics = [
        "zirconio", "nácar", "teclado", "caldera", "sílex", "manglar", "cuarcita", "órbita",
        "pistacho", "yunque", "kárate", "láser", "ómnix", "quetzal", "ráfaga", "sílice",
        "túnel", "úvula", "válvula", "wombat", "xilófono", "yeti", "zócalo", "áureo",
        "bélico", "cíclope", "dátil", "ébano", "féretro", "góndola", "híbrido", "ícono",
        "jáquima", "káiser", "líquen", "módulo", "níquel", "óxidos", "páramo", "quínoa",
        "réplica", "sándalo", "tálero", "úrico", "vértice", "wéstern", "yóquey", "zéjel",
        "almendra", "brújula-solar", "cascabel", "dendrita",
    ]
    items = []
    for tp in topics[:n]:
        items.append(Item(
            prompt=f'Responde SOLO con JSON válido con exactamente las claves "a" y "b" (texto corto sobre {tp}). Nada más.',
            check=lambda t, _tp=tp: isinstance(parse_json(t), dict) and set(parse_json(t).keys()) == {"a", "b"},
            meta={"topic": tp, "split": "confirm12_holdout", "family": "json"},
        ))
    out["json"] = items

    topics_en = [
        "kelp", "lynxes", "mantas", "nacre", "otters", "papyrus", "quolls", "radon",
        "saffron", "tundra", "ulna", "vellum", "walrus", "xeric", "yarrow", "zircon",
        "basilisk", "citrine", "dugong", "eucalyptus", "feldspar", "ginkgo", "halite",
        "ibexes", "jasper", "kaolin", "lichen", "myrrh", "nitrate", "obsidian",
        "peridot", "quartzite", "rutile", "spinel", "topaz", "uraninite", "vivianite",
        "wolfram", "xenolith", "yttrium", "zeolite", "ambergris", "bismuth", "cobalt",
        "dolomite", "emerald", "fluorite", "garnet", "hematite", "iodine", "jadeite", "kyanite",
    ]
    items = []
    for en in topics_en[:n]:
        items.append(Item(
            prompt=f"Write two sentences about {en}. Answer only in Spanish.",
            check=lambda t: len(words(t)) >= 4 and lang(t) == "es",
            meta={"topic": en, "split": "confirm12_holdout", "family": "spanish"},
        ))
    out["spanish"] = items

    pairs = [
        ("El sílex corta la cuerda.", "The flint cuts the rope."),
        ("La góndola cruza el canal.", "The gondola crosses the canal."),
        ("El líquen crece en la roca.", "The lichen grows on the rock."),
        ("Compré quínoa en el mercado.", "I bought quinoa at the market."),
        ("El páramo está cubierto de niebla.", "The moor is covered in fog."),
        ("La válvula controla el flujo.", "The valve controls the flow."),
        ("El wombat cava un túnel.", "The wombat digs a tunnel."),
        ("El xilófono suena alegre.", "The xylophone sounds cheerful."),
        ("Hay zirconio en la aleación.", "There is zirconium in the alloy."),
        ("El yeti es una leyenda.", "The yeti is a legend."),
        ("El zócalo necesita pintura.", "The baseboard needs paint."),
        ("El nácar brilla en la concha.", "The nacre shines in the shell."),
        ("La órbita es casi circular.", "The orbit is almost circular."),
        ("El yunque amplifica el golpe.", "The anvil amplifies the blow."),
        ("El manglar filtra el agua.", "The mangrove filters the water."),
        ("La cuarcita es muy dura.", "Quartzite is very hard."),
        ("El pistacho crece en verano.", "The pistachio grows in summer."),
        ("El láser marca el punto.", "The laser marks the point."),
        ("El quetzal vive en la selva.", "The quetzal lives in the jungle."),
        ("La ráfaga dobló el mástil.", "The gust bent the mast."),
        ("El túnel atraviesa la colina.", "The tunnel goes through the hill."),
        ("La úvula vibra al hablar.", "The uvula vibrates when speaking."),
        ("El módulo aterrizó limpio.", "The module landed cleanly."),
        ("El níquel resiste la corrosión.", "Nickel resists corrosion."),
        ("La réplica es casi perfecta.", "The replica is almost perfect."),
        ("El sándalo huele intenso.", "Sandalwood smells intense."),
        ("El vértice apunta al cielo.", "The vertex points to the sky."),
        ("El yóquey ganó la carrera.", "The jockey won the race."),
        ("El áureo color dorado brilla.", "The golden color shines."),
        ("El cíclope tiene un ojo.", "The cyclops has one eye."),
        ("El dátil es dulce y suave.", "The date is sweet and soft."),
        ("El ébano es madera oscura.", "Ebony is dark wood."),
        ("La góndola espera turistas.", "The gondola waits for tourists."),
        ("El híbrido combina dos razas.", "The hybrid combines two breeds."),
        ("El ícono es fácil de ver.", "The icon is easy to see."),
        ("El káiser gobernó un imperio.", "The kaiser ruled an empire."),
        ("El módulo solar carga bien.", "The solar module charges well."),
        ("Los óxidos tiñen la roca.", "Oxides stain the rock."),
        ("La quínoa es rica en proteína.", "Quinoa is rich in protein."),
        ("El tálero era una moneda.", "The thaler was a coin."),
        ("El wéstern cuenta una historia.", "The western tells a story."),
        ("El zéjel es un poema antiguo.", "The zajal is an old poem."),
        ("La almendra se tuesta fácil.", "The almond roasts easily."),
        ("El cascabel advierte peligro.", "The rattle warns of danger."),
        ("La dendrita crece en el cristal.", "The dendrite grows in the crystal."),
        ("El volcán expulsó ceniza fina.", "The volcano expelled fine ash."),
        ("El teclado tiene teclas quietas.", "The keyboard has quiet keys."),
        ("La sílice forma el vidrio.", "Silica forms glass."),
        ("El kárate requiere disciplina.", "Karate requires discipline."),
        ("El féretro era de madera.", "The coffin was made of wood."),
        ("La jáquima sujeta al caballo.", "The hackamore holds the horse."),
        ("El úrico ácido preocupa al médico.", "Uric acid worries the doctor."),
    ]
    items = []
    for es, en in pairs[:n]:
        items.append(Item(
            prompt=f"Translate to English. Reply with ONLY the English sentence.\n{es}",
            check=lambda t, ee=en: ee.lower() in t.lower() or _only(t).lower() == ee.lower(),
            meta={"src": es, "split": "confirm12_holdout", "family": "translate_en"},
        ))
    out["translate_en"] = items

    # arithmetic: distinct ranges from prior (prior used 11-30 + 3-8)
    probs = [(a, b, a + b) for a in range(41, 71) for b in range(11, 19)][:n]
    items = []
    for a, b, c in probs:
        items.append(Item(
            prompt=f"What is {a}+{b}? Reply with ONLY the number.",
            check=_exact(str(c)),
            meta={"split": "confirm12_holdout", "family": "arithmetic", "a": a, "b": b},
        ))
    out["arithmetic"] = items

    paras = [
        "Un geólogo midió cuarcita en la cantera norte y anotó dureza excepcional en su libreta de campo.",
        "La estación polar registró vientos que doblaron antenas y retrasaron el envío de datos climáticos.",
        "Ingenieros calibraron un láser industrial para cortar aleaciones de níquel sin deformar las piezas.",
        "En el manglar costero anidan aves migratorias que llegan cada invierno desde latitudes altas.",
        "Un archivista digitalizó partituras de zéjel andalusí para una biblioteca universitaria abierta.",
        "La sonda espacial ajustó su órbita tras detectar micrometeoritos cerca del cinturón de asteroides.",
        "Agricultores probaron quínoa resistente a sequía en terrazas del páramo andino este año.",
        "El museo exhibe un xilófono ceremonial tallado en ébano con incrustaciones de nácar.",
        "Biólogos rastrearon wombats nocturnos con collares GPS para mapear sus túneles subterráneos.",
        "Una fábrica redujo óxidos contaminantes instalando filtros nuevos en la línea de fundición.",
        "El equipo de rescate usó cascabeles y silbatos para localizar personas en la niebla del valle.",
        "Científicos cultivaron líquenes en laboratorio para estudiar su resistencia a radiación UV.",
        "El puerto instaló válvulas inteligentes que cortan fugas de combustible en segundos.",
        "Una startup fabricó teclados ergonómicos con carcasa de bambú reciclado y switches silenciosos.",
        "El volcán dormido expulsó una fina capa de ceniza que cubrió viñedos a veinte kilómetros.",
        "Médicos midieron ácido úrico en pacientes y recomendaron cambios dietéticos graduales.",
        "Un artesano restauró una góndola veneciana con madera de alerce y barniz marino.",
        "La red de sensores midió sílice en el aire tras obras de demolición en el barrio industrial.",
        "Pilotos de drones mapearon dendritas de hielo en glaciares para modelos climáticos locales.",
        "El laboratorio sintetizó pigmentos a partir de hematita y cobalto para tintas estables.",
        "Guardabosques reintrodujeron quetzales en un corredor biológico conectado al parque nacional.",
        "Una orquesta ensayó una pieza corta inspirada en ráfagas de viento sobre el acantilado.",
        "El reactor experimental usó zirconio en el revestimiento por su baja captura neutrónica.",
        "Estudiantes construyeron un módulo solar portátil que carga teléfonos en tres horas de sol.",
        "El mercado nocturno vendió dátiles y almendras tostadas junto a puestos de té de sándalo.",
        "Geofísicos hallaron un xenolito raro en lava antigua cerca de la antigua caldera.",
        "La clínica usó ultrasonido para evaluar la úvula en pacientes con apnea del sueño leve.",
        "Un documental filmó ibexes en riscos de cuarcita durante la temporada de celo primaveral.",
        "Ingenieros civiles reforzaron el zócalo del puente peatonal tras detectar microfisuras.",
        "La cooperativa molió pistachos locales para producir pasta sin aditivos artificiales.",
        "Astrónomos midieron el brillo de un cometa con filtro de fluorita en el telescopio.",
        "El taller de kárate inauguró tatamis nuevos fabricados con materiales reciclados.",
        "Una mina abandonada se convirtió en hábitat de murciélagos tras sellar pozos peligrosos.",
        "Químicos purificaron wollastonita para cerámica de alta temperatura en hornos industriales.",
        "El faro automático usa energía eólica y un backup de baterías de litio-ferro-fosfato.",
        "Paleontólogos catalogaron fósiles de peces en lutitas negras del período Devónico medio.",
        "La biblioteca pública digitalizó cartas de un yóquey del siglo XIX halladas en un baúl.",
        "Bomberos entrenaron con simuladores de túneles llenos de humo denso y baja visibilidad.",
        "Un vivero cultivó ginkgo y yarrow para un jardín terapéutico en el hospital regional.",
        "La startup midió radon en sótanos urbanos y ofreció kits de ventilación a bajo costo.",
        "Arqueólogos hallaron un tálero de plata bajo el piso de una casa colonial restaurada.",
        "El instituto midió topaz y spinel en un depósito aluvial tras lluvias torrenciales.",
        "Cocineros locales prepararon guisos con kelp deshidratado traído de la costa atlántica.",
        "Un equipo reparó la antena del módulo lunar de prueba dañada por una ráfaga de polvo.",
        "La reserva protege otters de río tras prohibir redes de pesca ilegales en el cauce.",
        "Ingenieros de sonido calibraron micrófonos para grabar el eco en una cueva de basalto.",
        "El observatorio detectó un eclipse parcial con cámaras protegidas por filtros de mylar.",
        "Una cooperativa tejió vellum sintético para restaurar mapas antiguos sin dañar originales.",
        "Médicos veterinarios vacunaron lynxes en cautiverio antes de liberarlos en el corredor norte.",
        "El laboratorio aisló enzimas de líquenes capaces de degradar plásticos a escala piloto.",
        "Pilotos comerciales reportaron turbulencia moderada sobre el páramo a ocho mil metros.",
        "La feria del libro presentó una réplica tipográfica de un zéjel impreso en el siglo XV.",
    ]
    items = []
    for para in paras[:n]:
        items.append(Item(
            prompt=f"Resume en español en MÁXIMO 12 palabras:\n{para}",
            check=lambda t: 1 <= len(words(t)) <= 14 and lang(t) in ("es", "?"),
            meta={"split": "confirm12_holdout", "family": "summarize"},
        ))
    out["summarize"] = items

    words_u = [
        "kelp", "lynx", "nacre", "otter", "papyrus", "quoll", "radon", "saffron",
        "tundra", "ulna", "vellum", "walrus", "xeric", "yarrow", "zircon", "beryl",
        "citrine", "dugong", "feldspar", "ginkgo", "halite", "ibex", "jasper", "kaolin",
        "lichen", "myrrh", "nitrate", "obsidian", "peridot", "pumice", "rutile", "spinel",
        "topaz", "wolfram", "zeolite", "bismuth", "cobalt", "dolomite", "emerald",
        "fluorite", "garnet", "hematite", "iodine", "jadeite", "kyanite", "magnetite",
        "nickel", "olivine", "pyrite", "quartzite", "rutabaga", "sulfur",
    ]
    items = []
    for w in words_u[:n]:
        items.append(Item(
            prompt=f"Reply with the word in ALL CAPS only: {w}",
            check=_exact(w.upper()),
            meta={"split": "confirm12_holdout", "family": "uppercase", "word": w},
        ))
    out["uppercase"] = items

    sents = [
        ("This victory feels incredible", "positive"), ("The delay ruined everything", "negative"),
        ("A ceramic bowl sits there", "neutral"), ("I love this quiet morning", "positive"),
        ("Disgusting mold everywhere", "negative"), ("The shelf holds books", "neutral"),
        ("Best news of the year", "positive"), ("Cruel unfair treatment", "negative"),
        ("Metal hinge is closed", "neutral"), ("Grateful beyond words", "positive"),
        ("Horrible bitter aftertaste", "negative"), ("Wooden crate is empty", "neutral"),
        ("Pure happiness tonight", "positive"), ("Painful sharp headache", "negative"),
        ("Glass pane reflects light", "neutral"), ("Wonderful calm evening", "positive"),
        ("Awful sticky mess", "negative"), ("Cotton towel is folded", "neutral"),
        ("Thrilled by the outcome", "positive"), ("Miserable endless queue", "negative"),
        ("Plastic lid fits well", "neutral"), ("Delightful soft music", "positive"),
        ("Terrible broken latch", "negative"), ("Stone path is dry", "neutral"),
        ("Joyful reunion ahead", "positive"), ("Nasty sour smell", "negative"),
        ("Paper clip on desk", "neutral"), ("Fantastic clear answer", "positive"),
        ("Dreadful cold wind", "negative"), ("Clay pot holds soil", "neutral"),
        ("Cheerful bright smile", "positive"), ("Ugly rude comment", "negative"),
        ("Iron bar lies flat", "neutral"), ("Blissful warm tea", "positive"),
        ("Wretched long wait", "negative"), ("Copper wire is thin", "neutral"),
        ("Radiant sunny day", "positive"), ("Ghastly burnt toast", "negative"),
        ("Brick wall stands firm", "neutral"), ("Lovely gentle rain", "positive"),
        ("Vile harsh insult", "negative"), ("Steel bolt is tight", "neutral"),
        ("Ecstatic final score", "positive"), ("Rotten damp basement", "negative"),
        ("Marble tile is smooth", "neutral"), ("Glorious fresh bread", "positive"),
        ("Appalling rude service", "negative"), ("Linen cloth is clean", "neutral"),
        ("Superb clever idea", "positive"), ("Bleak empty street", "negative"),
        ("Rubber seal is intact", "neutral"), ("Charming quiet cafe", "positive"),
    ]
    items = []
    for s, lab in sents[:n]:
        items.append(Item(
            prompt=f"Classify sentiment as positive, negative, or neutral. Reply ONLY with the label.\n{s}",
            check=_exact(lab),
            meta={"split": "confirm12_holdout", "family": "sentiment"},
        ))
    out["sentiment"] = items

    ents = [
        ("Rita visited Kyoto in April.", {"person": "Rita", "place": "Kyoto", "date": "April"}),
        ("Hugo reached Bergen in May.", {"person": "Hugo", "place": "Bergen", "date": "May"}),
        ("Ivy left Cusco in June.", {"person": "Ivy", "place": "Cusco", "date": "June"}),
        ("Owen saw Lisbon in July.", {"person": "Owen", "place": "Lisbon", "date": "July"}),
        ("Nora flew Helsinki in August.", {"person": "Nora", "place": "Helsinki", "date": "August"}),
        ("Paul met Prague in September.", {"person": "Paul", "place": "Prague", "date": "September"}),
        ("Quinn reached Taipei in October.", {"person": "Quinn", "place": "Taipei", "date": "October"}),
        ("Ruth visited Vienna in November.", {"person": "Ruth", "place": "Vienna", "date": "November"}),
        ("Seth saw Warsaw in December.", {"person": "Seth", "place": "Warsaw", "date": "December"}),
        ("Tina left Athens in January.", {"person": "Tina", "place": "Athens", "date": "January"}),
        ("Umar reached Nairobi in February.", {"person": "Umar", "place": "Nairobi", "date": "February"}),
        ("Vera visited Hanoi in March.", {"person": "Vera", "place": "Hanoi", "date": "March"}),
        ("Wade saw Manila in April.", {"person": "Wade", "place": "Manila", "date": "April"}),
        ("Xena left Quito in May.", {"person": "Xena", "place": "Quito", "date": "May"}),
        ("Yves reached Oslo in June.", {"person": "Yves", "place": "Oslo", "date": "June"}),
        ("Zara visited Seoul in July.", {"person": "Zara", "place": "Seoul", "date": "July"}),
        ("Aya saw Cairo in August.", {"person": "Aya", "place": "Cairo", "date": "August"}),
        ("Ben left Rome in September.", {"person": "Ben", "place": "Rome", "date": "September"}),
        ("Cora reached Delhi in October.", {"person": "Cora", "place": "Delhi", "date": "October"}),
        ("Drew visited Perth in November.", {"person": "Drew", "place": "Perth", "date": "November"}),
        ("Elle saw Dallas in December.", {"person": "Elle", "place": "Dallas", "date": "December"}),
        ("Finn left Calgary in January.", {"person": "Finn", "place": "Calgary", "date": "January"}),
        ("Gia reached Tucson in February.", {"person": "Gia", "place": "Tucson", "date": "February"}),
        ("Hank visited Juneau in March.", {"person": "Hank", "place": "Juneau", "date": "March"}),
        ("Ina saw Boise in April.", {"person": "Ina", "place": "Boise", "date": "April"}),
        ("Joel left Reno in May.", {"person": "Joel", "place": "Reno", "date": "May"}),
        ("Kira reached Tulsa in June.", {"person": "Kira", "place": "Tulsa", "date": "June"}),
        ("Lars visited Omaha in July.", {"person": "Lars", "place": "Omaha", "date": "July"}),
        ("Mira saw Akron in August.", {"person": "Mira", "place": "Akron", "date": "August"}),
        ("Ned left Tampa in September.", {"person": "Ned", "place": "Tampa", "date": "September"}),
        ("Ona reached Denver in October.", {"person": "Ona", "place": "Denver", "date": "October"}),
        ("Pete visited Miami in November.", {"person": "Pete", "place": "Miami", "date": "November"}),
        ("Rae saw Boston in December.", {"person": "Rae", "place": "Boston", "date": "December"}),
        ("Sid left Dublin in January.", {"person": "Sid", "place": "Dublin", "date": "January"}),
        ("Tess reached Berlin in February.", {"person": "Tess", "place": "Berlin", "date": "February"}),
        ("Ula visited Madrid in March.", {"person": "Ula", "place": "Madrid", "date": "March"}),
        ("Vic saw Tokyo in April.", {"person": "Vic", "place": "Tokyo", "date": "April"}),
        ("Willa left Paris in May.", {"person": "Willa", "place": "Paris", "date": "May"}),
        ("Xander reached Lima in June.", {"person": "Xander", "place": "Lima", "date": "June"}),
        ("Yara visited Rome in July.", {"person": "Yara", "place": "Rome", "date": "July"}),
        ("Zeke saw Oslo in August.", {"person": "Zeke", "place": "Oslo", "date": "August"}),
        ("Ann left Seoul in September.", {"person": "Ann", "place": "Seoul", "date": "September"}),
        ("Bea reached Cairo in October.", {"person": "Bea", "place": "Cairo", "date": "October"}),
        ("Carl visited Hanoi in November.", {"person": "Carl", "place": "Hanoi", "date": "November"}),
        ("Deb saw Manila in December.", {"person": "Deb", "place": "Manila", "date": "December"}),
        ("Eli left Quito in January.", {"person": "Eli", "place": "Quito", "date": "January"}),
        ("Fay reached Kyoto in February.", {"person": "Fay", "place": "Kyoto", "date": "February"}),
        ("Gus visited Bergen in March.", {"person": "Gus", "place": "Bergen", "date": "March"}),
        ("Hana saw Lisbon in April.", {"person": "Hana", "place": "Lisbon", "date": "April"}),
        ("Ian left Helsinki in May.", {"person": "Ian", "place": "Helsinki", "date": "May"}),
        ("Joy reached Prague in June.", {"person": "Joy", "place": "Prague", "date": "June"}),
        ("Ken visited Taipei in July.", {"person": "Ken", "place": "Taipei", "date": "July"}),
    ]
    def ent_check(t, expect):
        o = parse_json(t)
        if not isinstance(o, dict):
            return False
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
            meta={"split": "confirm12_holdout", "family": "entities"},
        ))
    out["entities"] = items

    topics_b = [
        "kelp", "zirconio", "manglares", "góndolas", "líquenes", "páramos", "quínoa",
        "sándalo", "wombats", "xilófonos", "válvulas", "órbitas", "cuarcita", "nácar",
        "teclados", "volcanes", "sílex", "yunque", "láseres", "quetzales", "ráfagas",
        "túneles", "módulos", "níquel", "óxidos", "réplicas", "vértices", "yóqueys",
        "zéjeles", "almendras", "cascabeles", "dendritas", "ibexes", "ginkgo", "halita",
        "jaspe", "caolín", "mirra", "obsidiana", "peridoto", "espinela", "topacio",
        "wolframio", "zeolita", "bismuto", "cobalto", "dolomita", "esmeralda", "fluorita",
        "granate", "hematita", "yodo", "jadeíta",
    ]
    items = []
    for tp in topics_b[:n]:
        def chk(t, _tp=tp):
            lines = [ln for ln in t.strip().splitlines() if re.match(r"^\s*[-*•]\s+", ln)]
            return len(lines) == 3
        items.append(Item(
            prompt=f"Lista EXACTAMENTE 3 viñetas que empiecen con '- ' sobre {tp}. Nada más.",
            check=chk,
            meta={"split": "confirm12_holdout", "family": "bullets"},
        ))
    out["bullets"] = items

    specs_fn = [
        ("prod", "a, b", "return a * b"),
        ("quot", "a, b", "return a // b"),
        ("mod", "a, b", "return a % b"),
        ("triple", "x", "return x * 3"),
        ("half", "x", "return x / 2"),
        ("clamp0", "x", "return max(0, x)"),
        ("sgn", "x", "return 1 if x > 0 else (-1 if x < 0 else 0)"),
        ("is_even", "x", "return x % 2 == 0"),
        ("is_odd", "x", "return x % 2 != 0"),
        ("second", "xs", "return xs[1]"),
        ("rev", "xs", "return xs[::-1]"),
        ("head2", "xs", "return xs[:2]"),
        ("tail2", "xs", "return xs[-2:]"),
        ("count", "xs", "return len(xs)"),
        ("sum2", "a, b", "return a + b"),
        ("diff2", "a, b", "return a - b"),
        ("pow2", "x", "return x ** 2"),
        ("pow3", "x", "return x ** 3"),
        ("floor1", "x", "return int(x)"),
        ("as_str", "x", "return str(x)"),
        ("as_bool", "x", "return bool(x)"),
        ("swap", "a, b", "return (b, a)"),
        ("mid", "a, b", "return (a + b) / 2"),
        ("gt", "a, b", "return a > b"),
        ("lt", "a, b", "return a < b"),
        ("eq", "a, b", "return a == b"),
        ("neq", "a, b", "return a != b"),
        ("and2", "a, b", "return a and b"),
        ("or2", "a, b", "return a or b"),
        ("xor2", "a, b", "return bool(a) != bool(b)"),
        ("id_str", "s", "return s"),
        ("upper1", "s", "return s.upper()"),
        ("lower1", "s", "return s.lower()"),
        ("len_s", "s", "return len(s)"),
        ("first_c", "s", "return s[0]"),
        ("last_c", "s", "return s[-1]"),
        ("repeat2", "s", "return s * 2"),
        ("strip1", "s", "return s.strip()"),
        ("join2", "a, b", "return a + b"),
        ("list1", "x", "return [x]"),
        ("tuple1", "x", "return (x,)"),
        ("set1", "x", "return {x}"),
        ("none_fn", "", "return None"),
        ("true_fn", "", "return True"),
        ("false_fn", "", "return False"),
        ("zero_fn", "", "return 0"),
        ("one_fn", "", "return 1"),
        ("neg1", "x", "return -x"),
        ("abs1", "x", "return abs(x)"),
        ("maxi3", "a, b, c", "return max(a, b, c)"),
        ("mini3", "a, b, c", "return min(a, b, c)"),
        ("sum3", "a, b, c", "return a + b + c"),
    ]
    items = []
    for name, args, body in specs_fn[:n]:
        items.append(Item(
            prompt=f"Write ONLY a Python function `{name}({args})` with body `{body}`. No prose.",
            check=lambda t, n=name: ("def " + n) in t and "return" in t,
            meta={"split": "confirm12_holdout", "family": "code_python", "fn": name},
        ))
    out["code_python"] = items

    ids = [s.id for s in all_expert_specs()]
    for i in ids:
        assert i in out and len(out[i]) >= n, (i, len(out.get(i, [])))
    return out


def historical_prompt_set() -> Set[str]:
    prompts: Set[str] = set()
    for s in all_expert_specs():
        for it in list(s.tune_items) + list(s.eval_items):
            prompts.add(it.prompt)
    for items in build_new_holdout(20).values():
        for it in items:
            prompts.add(it.prompt)
    return prompts


def assert_no_overlap(holdout: Dict[str, List[Item]]) -> None:
    hist = historical_prompt_set()
    for tid, items in holdout.items():
        for it in items:
            assert it.prompt not in hist, f"overlap {tid}: {it.prompt[:60]}"


def holdout_manifest(holdout: Dict[str, List[Item]]) -> Dict:
    audit = {
        tid: [{"prompt": it.prompt, "meta": it.meta} for it in items]
        for tid, items in holdout.items()
    }
    blob = json.dumps(audit, sort_keys=True, ensure_ascii=False).encode()
    return {
        "n_per_task": {k: len(v) for k, v in holdout.items()},
        "split": "confirm12_holdout",
        "seed_note": "deterministic lists in confirm_holdout.py",
        "prompt_sha256": hashlib.sha256(blob).hexdigest(),
        "items": audit,
    }
