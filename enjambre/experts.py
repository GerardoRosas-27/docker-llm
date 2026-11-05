"""Definición de 10 micro-expertos: prompts de contraste, tuning/eval y checks."""
from __future__ import annotations
import json, re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence


def words(t: str):
    return re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9']+", t)


def bullets(t: str):
    return [l for l in t.strip().splitlines() if re.match(r"^\s*([-*•]|\d+[.)])\s+", l)]


def nonempty(t: str):
    return [l for l in t.strip().splitlines() if l.strip()]


ES = set("el la los las de que y en un una es por para con no se su al lo como más pero sus muy también son está del una".split())
EN = set("the and of to is in it that for are with as on be this was they by an at or can their its".split())


def lang(t: str) -> str:
    w = [x.lower() for x in words(t)]
    e = sum(x in EN for x in w)
    s = sum(x in ES for x in w)
    return "en" if e > s else ("es" if s > e else "?")


def parse_json(t: str):
    s = t.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I).strip()
    # intenta primer {..} o [...]
    m = re.search(r"(\{.*\}|\[.*\])", s, re.S)
    if m:
        s = m.group(1)
    try:
        return json.loads(s)
    except Exception:
        return None


CheckFn = Callable[[str], bool]


@dataclass
class Item:
    prompt: str
    check: CheckFn
    meta: Dict = field(default_factory=dict)


@dataclass
class ExpertSpec:
    id: str
    name: str
    description: str
    system_prompt: str
    keywords: List[str]
    # sets
    tune_items: List[Item]
    eval_items: List[Item]
    # hiperparams candidatos
    layer_candidates: List[int] = field(default_factory=lambda: [8, 10, 12])
    scale_candidates: List[float] = field(default_factory=lambda: [3.0, 6.0, 12.0])


# ---------- builders por tarea ----------

def _json_items(split: str) -> List[Item]:
    topics_t = ["café", "gato", "lluvia", "océano", "montaña", "música"]
    topics_e = ["libro", "fútbol", "abeja", "luna", "pan", "volcán", "ajedrez", "tren", "árbol", "invierno", "amistad", "río",
                "nube", "desierto", "guitarra", "selva", "faro", "puente", "mercado", "jardín"]
    topics = topics_t if split == "tune" else topics_e
    items = []
    for tp in topics:
        items.append(Item(
            prompt=f'Responde SOLO con JSON válido con exactamente las claves "a" y "b" (texto corto sobre {tp}). Nada más.',
            check=lambda t, _tp=tp: isinstance(parse_json(t), dict) and set(parse_json(t).keys()) == {"a", "b"},
            meta={"topic": tp},
        ))
    return items


def _spanish_items(split: str) -> List[Item]:
    topics_t = ["the sun", "dogs", "chocolate", "stars", "snow", "rivers"]
    topics_e = ["theater", "flowers", "coffee", "cats", "rain", "ocean", "mountains", "music", "books", "soccer", "bees", "moon",
                "trains", "bread", "volcanoes", "winter", "friendship", "gardens", "bicycles", "the desert"]
    topics = topics_t if split == "tune" else topics_e
    items = []
    for en in topics:
        items.append(Item(
            prompt=f"Write two sentences about {en}. Answer only in Spanish.",
            check=lambda t: len(words(t)) >= 4 and lang(t) == "es",
            meta={"topic": en},
        ))
    return items


def _translate_items(split: str) -> List[Item]:
    pairs_t = [
        ("Hola, ¿cómo estás?", ["hello", "how are you"]),
        ("El gato duerme en el sofá.", ["cat", "sleep", "sofa", "couch"]),
        ("Me gusta el café por la mañana.", ["coffee", "morning", "like"]),
        ("La biblioteca abre temprano.", ["library", "open", "early"]),
        ("Necesito ayuda con esto.", ["need", "help"]),
        ("Hoy hace mucho calor.", ["hot", "today", "warm"]),
    ]
    pairs_e = [
        ("Buenos días, profesor.", ["good morning", "teacher", "professor"]),
        ("El tren llega a las cinco.", ["train", "arrives", "five"]),
        ("Los niños juegan en el parque.", ["children", "kids", "play", "park"]),
        ("¿Dónde está la estación?", ["where", "station"]),
        ("Prefiero té en lugar de café.", ["prefer", "tea", "coffee"]),
        ("La luna brilla esta noche.", ["moon", "shine", "night", "bright"]),
        ("Mi hermana estudia medicina.", ["sister", "stud", "medicine"]),
        ("El océano es muy profundo.", ["ocean", "deep"]),
        ("Cierra la puerta, por favor.", ["close", "door", "please"]),
        ("Mañana viajamos a Madrid.", ["tomorrow", "travel", "madrid", "go"]),
        ("El pan está fresco.", ["bread", "fresh"]),
        ("Escucho música clásica.", ["listen", "music", "classical"]),
        ("El perro corre en la playa.", ["dog", "run", "beach"]),
        ("Mi padre cocina muy bien.", ["father", "dad", "cook"]),
        ("La tienda cierra a las nueve.", ["store", "shop", "close", "nine"]),
        ("Quiero aprender a nadar.", ["want", "learn", "swim"]),
        ("El libro está sobre la mesa.", ["book", "table"]),
        ("Hace frío en invierno.", ["cold", "winter"]),
        ("Ellos viven en una casa grande.", ["live", "house", "big", "large"]),
        ("¿Cuánto cuesta este sombrero?", ["how much", "cost", "hat"]),
    ]
    pairs = pairs_t if split == "tune" else pairs_e
    items = []
    for es, keys in pairs:
        def chk(t, keys=keys):
            low = t.lower()
            return lang(t) == "en" and any(k in low for k in keys) and len(words(t)) >= 2
        items.append(Item(
            prompt=f"Translate to English. Reply with ONLY the translation, nothing else:\n{es}",
            check=chk,
            meta={"src": es},
        ))
    return items


def _arith_items(split: str) -> List[Item]:
    probs_t = [
        ("What is 7 + 5?", "12"),
        ("What is 9 * 3?", "27"),
        ("What is 20 - 8?", "12"),
        ("What is 15 / 3?", "5"),
        ("What is 11 + 14?", "25"),
        ("What is 6 * 7?", "42"),
    ]
    probs_e = [
        ("What is 8 + 9?", "17"),
        ("What is 4 * 8?", "32"),
        ("What is 30 - 12?", "18"),
        ("What is 28 / 4?", "7"),
        ("What is 13 + 19?", "32"),
        ("What is 5 * 9?", "45"),
        ("What is 100 - 37?", "63"),
        ("What is 16 / 2?", "8"),
        ("What is 21 + 6?", "27"),
        ("What is 3 * 12?", "36"),
        ("What is 45 - 20?", "25"),
        ("What is 81 / 9?", "9"),
        ("What is 14 + 23?", "37"),
        ("What is 7 * 6?", "42"),
        ("What is 50 - 17?", "33"),
        ("What is 64 / 8?", "8"),
        ("What is 9 + 16?", "25"),
        ("What is 11 * 4?", "44"),
        ("What is 72 - 45?", "27"),
        ("What is 36 / 6?", "6"),
    ]
    probs = probs_t if split == "tune" else probs_e
    items = []
    for q, ans in probs:
        items.append(Item(
            prompt=f"{q} Reply with ONLY the number.",
            check=lambda t, a=ans: a in re.findall(r"-?\d+", t) or t.strip() == a,
            meta={"ans": ans},
        ))
    return items


def _summarize_items(split: str) -> List[Item]:
    texts_t = [
        ("Los gatos son animales domésticos independientes que duermen muchas horas al día y cazan ratones.", 8),
        ("El café se cultiva en climas tropicales y es una de las bebidas más populares del mundo.", 8),
        ("La lluvia es precipitación de agua que cae de las nubes y alimenta ríos y cultivos.", 8),
        ("Las montañas se forman por el movimiento de placas tectónicas a lo largo de millones de años.", 10),
        ("La música clasica incluye obras de Bach, Mozart y Beethoven interpretadas por orquestas.", 8),
        ("Los libros permiten viajar con la imaginación y aprender sobre historia, ciencia y cultura.", 8),
    ]
    texts_e = [
        ("El fútbol es un deporte de equipo con once jugadores que buscan marcar goles con un balón.", 8),
        ("Las abejas polinizan flores y producen miel en colmenas organizadas por una reina.", 8),
        ("La luna orbita la Tierra y provoca las mareas con su atracción gravitatoria.", 8),
        ("El pan se elabora con harina, agua y levadura, horneado hasta dorarse.", 8),
        ("Los volcanes expulsan lava y ceniza cuando la presión del magma es demasiado alta.", 10),
        ("El ajedrez es un juego de estrategia con piezas como el rey, la reina y los peones.", 8),
        ("Las bicicletas son un medio de transporte ecológico impulsado por pedales.", 8),
        ("Los árboles producen oxígeno y dan sombra, fruto y madera a las personas.", 8),
        ("El invierno trae frío, nieve en muchas regiones y días más cortos.", 8),
        ("La amistad es un vínculo de confianza y apoyo entre personas.", 6),
        ("Los trenes transportan pasajeros y mercancías sobre rieles de forma eficiente.", 8),
        ("El océano cubre la mayor parte del planeta y alberga una enorme biodiversidad.", 8),
        ("Los desiertos tienen poca lluvia, temperaturas extremas y plantas adaptadas como los cactus.", 8),
        ("La guitarra es un instrumento de cuerdas muy usado en el rock, el folk y el flamenco.", 8),
        ("Los faros guían a los barcos de noche con una luz giratoria en la costa.", 8),
        ("Las selvas tropicales tienen lluvias abundantes y una gran variedad de animales.", 8),
        ("Los puentes conectan orillas de ríos y permiten el paso de personas y vehículos.", 8),
        ("El reciclaje reduce la basura y ahorra materias primas y energía.", 8),
        ("Los mercados reúnen a vendedores de frutas, verduras y otros productos frescos.", 8),
        ("Las nubes se forman cuando el vapor de agua se condensa en el aire frío.", 8),
    ]
    texts = texts_t if split == "tune" else texts_e
    items = []
    for text, n in texts:
        items.append(Item(
            prompt=f"Resume el siguiente texto en máximo {n} palabras. Solo el resumen:\n{text}",
            check=lambda t, n=n: 1 <= len(words(t)) <= n + 2,  # +2 tolerancia leve
            meta={"max_words": n},
        ))
    return items


def _uppercase_items(split: str) -> List[Item]:
    topics_t = ["el café", "los gatos", "la lluvia", "el océano", "las montañas", "la música"]
    topics_e = ["los libros", "el fútbol", "las abejas", "la luna", "el pan", "los volcanes",
                "el ajedrez", "las bicicletas", "los árboles", "el invierno", "la amistad", "los trenes",
                "las nubes", "el desierto", "la guitarra", "la selva", "los faros", "los puentes", "el mercado", "el jardín"]
    topics = topics_t if split == "tune" else topics_e
    items = []
    for tp in topics:
        items.append(Item(
            prompt=f"Describe {tp} en una o dos frases. Escribe TODA la respuesta en MAYÚSCULAS.",
            check=lambda t: len(words(t)) >= 3 and t == t.upper(),
            meta={"topic": tp},
        ))
    return items


def _sentiment_items(split: str) -> List[Item]:
    data_t = [
        ("I love this product, it is amazing!", "positive"),
        ("This is the worst experience ever.", "negative"),
        ("The package arrived on time as expected.", "neutral"),
        ("Absolutely fantastic service, thank you!", "positive"),
        ("I hate waiting in long lines.", "negative"),
        ("The meeting is scheduled for Monday.", "neutral"),
    ]
    data_e = [
        ("What a wonderful day to go hiking!", "positive"),
        ("The food was cold and tasteless.", "negative"),
        ("The store opens at nine.", "neutral"),
        ("I'm so happy with my new phone.", "positive"),
        ("Terrible customer support, very rude.", "negative"),
        ("She took the bus to work.", "neutral"),
        ("Best concert I've ever attended!", "positive"),
        ("My order was damaged and late.", "negative"),
        ("The book has 300 pages.", "neutral"),
        ("This movie made me cry tears of joy.", "positive"),
        ("I regret buying this junk.", "negative"),
        ("Rain is expected tomorrow afternoon.", "neutral"),
        ("I adore this little café, the staff is lovely!", "positive"),
        ("The hotel room was dirty and noisy.", "negative"),
        ("The train leaves at 7:15.", "neutral"),
        ("Great job, I'm really impressed!", "positive"),
        ("This app keeps crashing, so annoying.", "negative"),
        ("The meeting room is on the second floor.", "neutral"),
        ("What a delightful surprise party!", "positive"),
        ("The battery died after one hour, awful.", "negative"),
    ]
    data = data_t if split == "tune" else data_e
    items = []
    for text, label in data:
        items.append(Item(
            prompt=(
                f'Classify the sentiment of this text as exactly one word: '
                f'positive, negative, or neutral.\nText: "{text}"\nAnswer:'
            ),
            check=lambda t, lab=label: lab in t.lower().split()[:5] or t.lower().strip().startswith(lab),
            meta={"label": label},
        ))
    return items


def _entities_items(split: str) -> List[Item]:
    data_t = [
        ("Maria visited Paris on March 12, 2020.", {"person": "maria", "place": "paris", "date": "march"}),
        ("John signed the contract in London on July 4.", {"person": "john", "place": "london", "date": "july"}),
        ("Ana flew to Tokyo in January 2019.", {"person": "ana", "place": "tokyo", "date": "january"}),
        ("Carlos met Sofia in Madrid on 5 May 2021.", {"person": "carlos", "place": "madrid", "date": "may"}),
        ("Emma worked in Berlin during August 2018.", {"person": "emma", "place": "berlin", "date": "august"}),
        ("Luis arrived in Rome on December 25.", {"person": "luis", "place": "rome", "date": "december"}),
    ]
    data_e = [
        ("Pedro opened a shop in Lisbon on April 1, 2022.", {"person": "pedro", "place": "lisbon", "date": "april"}),
        ("Nora taught in Boston until September 2017.", {"person": "nora", "place": "boston", "date": "september"}),
        ("Omar moved to Cairo on 10 February 2015.", {"person": "omar", "place": "cairo", "date": "february"}),
        ("Helen visited Sydney in June 2016.", {"person": "helen", "place": "sydney", "date": "june"}),
        ("Diego lived in Bogotá until October 2020.", {"person": "diego", "place": "bogot", "date": "october"}),
        ("Sara arrived in Oslo on November 11.", {"person": "sara", "place": "oslo", "date": "november"}),
        ("Tom studied in Chicago during March 2014.", {"person": "tom", "place": "chicago", "date": "march"}),
        ("Irene met Paul in Seville on 8 July.", {"person": "irene", "place": "seville", "date": "july"}),
        ("Yuki flew to Seoul on January 3, 2023.", {"person": "yuki", "place": "seoul", "date": "january"}),
        ("Bruno worked in Milan until May 2011.", {"person": "bruno", "place": "milan", "date": "may"}),
        ("Clara visited Athens on August 20.", {"person": "clara", "place": "athens", "date": "august"}),
        ("Hugo signed papers in Vienna on 2 December.", {"person": "hugo", "place": "vienna", "date": "december"}),
        ("Lena opened a bakery in Dublin on March 3.", {"person": "lena", "place": "dublin", "date": "march"}),
        ("Mateo visited Lima in July 2019.", {"person": "mateo", "place": "lima", "date": "july"}),
        ("Grace moved to Toronto on 15 April.", {"person": "grace", "place": "toronto", "date": "april"}),
        ("Ivan taught in Prague until June 2012.", {"person": "ivan", "place": "prague", "date": "june"}),
        ("Rosa arrived in Quito on September 9.", {"person": "rosa", "place": "quito", "date": "september"}),
        ("Felix studied in Munich during October 2018.", {"person": "felix", "place": "munich", "date": "october"}),
        ("Alma flew to Havana on 1 November 2021.", {"person": "alma", "place": "havana", "date": "november"}),
        ("Victor lived in Montreal until January 2010.", {"person": "victor", "place": "montreal", "date": "january"}),
    ]
    data = data_t if split == "tune" else data_e
    items = []
    for text, keys in data:
        def chk(t, keys=keys):
            low = t.lower()
            # exige JSON con person/place/date O que mencione los tres valores
            js = parse_json(t)
            if isinstance(js, dict):
                blob = json.dumps(js).lower()
                return all(v in blob for v in keys.values())
            return all(v in low for v in keys.values())
        items.append(Item(
            prompt=(
                f'Extract person, place and date from the sentence. '
                f'Reply ONLY with JSON keys "person","place","date".\nSentence: {text}'
            ),
            check=chk,
            meta=keys,
        ))
    return items


def _bullets_items(split: str) -> List[Item]:
    topics_t = ["el café", "los gatos", "la lluvia", "el océano", "las montañas", "la música"]
    topics_e = ["los libros", "el fútbol", "las abejas", "la luna", "el pan", "los volcanes",
                "el ajedrez", "las bicicletas", "los árboles", "el invierno", "la amistad", "los trenes",
                "las nubes", "el desierto", "la guitarra", "la selva", "los faros", "los puentes", "el mercado", "el jardín"]
    topics = topics_t if split == "tune" else topics_e
    items = []
    for tp in topics:
        items.append(Item(
            prompt=f"Escribe sobre {tp}. Responde en exactamente 3 viñetas que empiecen con '- ', sin ningún otro texto.",
            check=lambda t: len(bullets(t)) == 3 and len(nonempty(t)) == 3,
            meta={"topic": tp},
        ))
    return items


def _code_items(split: str) -> List[Item]:
    tasks_t = [
        ("Write a Python function `add(a, b)` that returns the sum. Only code.",
         lambda t: "def add" in t and "return" in t),
        ("Write a Python one-liner that prints numbers 1 to 5. Only code.",
         lambda t: "print" in t and ("range" in t or "for" in t)),
        ("Write a Python function `is_even(n)` returning True if n is even. Only code.",
         lambda t: "def is_even" in t and "%" in t),
        ("Write Python code to reverse a string `s`. Only code.",
         lambda t: ("[::-1]" in t or "reversed" in t) and ("def" in t or "s" in t)),
        ("Write a Python list comprehension of squares 0..4. Only code.",
         lambda t: "[" in t and "for" in t and ("**2" in t or "i*i" in t or "i**2" in t)),
        ("Write a Python function `hello()` that returns 'hi'. Only code.",
         lambda t: "def hello" in t and ("hi" in t or "'hi'" in t or '"hi"' in t)),
    ]
    tasks_e = [
        ("Write a Python function `mul(a, b)` that returns the product. Only code.",
         lambda t: "def mul" in t and "return" in t),
        ("Write Python code to check if a string is a palindrome. Only code.",
         lambda t: ("[::-1]" in t or "reversed" in t)),
        ("Write a Python function `fact(n)` computing factorial. Only code.",
         lambda t: "def fact" in t and ("*" in t or "math.factorial" in t)),
        ("Write a Python dict mapping 'a' to 1 and 'b' to 2. Only code.",
         lambda t: "{" in t and "'a'" in t.replace('"','\'') or '"a"' in t),
        ("Write a Python for-loop that prints each item of list `xs`. Only code.",
         lambda t: "for" in t and "print" in t),
        ("Write a Python function `maxlen(xs)` returning the longest string. Only code.",
         lambda t: "def maxlen" in t and "max" in t),
        ("Write Python code using `enumerate` on a list. Only code.",
         lambda t: "enumerate" in t),
        ("Write a Python try/except that catches ValueError. Only code.",
         lambda t: "try" in t and "except" in t and "ValueError" in t),
        ("Write a Python class `Dog` with `__init__(self, name)`. Only code.",
         lambda t: "class Dog" in t and "__init__" in t),
        ("Write a Python lambda that doubles x. Only code.",
         lambda t: "lambda" in t and ("* 2" in t or "*2" in t)),
        ("Write Python code to join a list of words with spaces. Only code.",
         lambda t: "join" in t),
        ("Write a Python function `clamp(x, lo, hi)` that bounds x. Only code.",
         lambda t: "def clamp" in t and ("max" in t or "min" in t or "if" in t)),
        ("Write a Python function `square(x)` that returns x squared. Only code.",
         lambda t: "def square" in t and "return" in t),
        ("Write Python code that reads a file `data.txt` with `open`. Only code.",
         lambda t: "open(" in t and "data.txt" in t),
        ("Write a Python function `count_vowels(s)`. Only code.",
         lambda t: "def count_vowels" in t and ("for" in t or "sum" in t)),
        ("Write Python code that sorts a list `xs` in descending order. Only code.",
         lambda t: ("sort" in t) and "reverse" in t),
        ("Write a Python function `avg(xs)` returning the mean. Only code.",
         lambda t: "def avg" in t and ("sum" in t or "mean" in t)),
        ("Write a Python while loop that counts down from 3. Only code.",
         lambda t: "while" in t),
        ("Write a Python function `greet(name)` returning 'Hello, ' + name. Only code.",
         lambda t: "def greet" in t and "Hello" in t),
        ("Write Python code to remove duplicates from a list `xs`. Only code.",
         lambda t: "set(" in t or "not in" in t or "dict.fromkeys" in t),
    ]
    tasks = tasks_t if split == "tune" else tasks_e
    items = []
    for prompt, chk in tasks:
        items.append(Item(prompt=prompt, check=chk))
    return items


def all_expert_specs() -> List[ExpertSpec]:
    return [
        ExpertSpec(
            id="json",
            name="JSON formatting",
            description="Responde únicamente con JSON válido según el esquema pedido.",
            system_prompt=(
                "You are a JSON-only assistant. Always reply with valid JSON and nothing else. "
                "No markdown, no commentary."
            ),
            keywords=["json", "clave", "keys", "objeto", "schema", "{", "}"],
            tune_items=_json_items("tune"),
            eval_items=_json_items("eval"),
        ),
        ExpertSpec(
            id="spanish",
            name="Spanish answers",
            description="Responde siempre en español.",
            system_prompt=(
                "You MUST answer ONLY in Spanish (castellano). Never use English words. "
                "Obey language constraints exactly."
            ),
            keywords=["español", "spanish", "castellano", "en español"],
            tune_items=_spanish_items("tune"),
            eval_items=_spanish_items("eval"),
        ),
        ExpertSpec(
            id="translate_en",
            name="English translation",
            description="Traduce al inglés y solo devuelve la traducción.",
            system_prompt=(
                "You are a precise translator to English. Reply with ONLY the English translation, "
                "no quotes, no explanation."
            ),
            keywords=["translate", "traduce", "traducción", "english", "inglés", "translation"],
            tune_items=_translate_items("tune"),
            eval_items=_translate_items("eval"),
        ),
        ExpertSpec(
            id="arithmetic",
            name="Arithmetic",
            description="Resuelve cuentas simples y responde solo con el número.",
            system_prompt=(
                "You are a calculator. Reply with ONLY the numeric answer, no words."
            ),
            keywords=["what is", "cuánto", "suma", "multiply", "*", "+", "-", "/", "calculate", "número"],
            tune_items=_arith_items("tune"),
            eval_items=_arith_items("eval"),
        ),
        ExpertSpec(
            id="summarize",
            name="Summarization N words",
            description="Resume respetando un máximo de palabras.",
            system_prompt=(
                "You summarize briefly and EXACTLY respect the maximum word count. "
                "No preamble."
            ),
            keywords=["resume", "summary", "summarize", "máximo", "palabras", "words"],
            tune_items=_summarize_items("tune"),
            eval_items=_summarize_items("eval"),
        ),
        ExpertSpec(
            id="uppercase",
            name="Uppercase / format rules",
            description="Cumple reglas de formato (p.ej. TODO EN MAYÚSCULAS).",
            system_prompt=(
                "You follow formatting constraints LITERALLY. If asked for ALL CAPS, "
                "every letter must be uppercase."
            ),
            keywords=["mayúsculas", "uppercase", "ALL CAPS", "MAYÚSCULAS", "formato"],
            tune_items=_uppercase_items("tune"),
            eval_items=_uppercase_items("eval"),
        ),
        ExpertSpec(
            id="sentiment",
            name="Sentiment classification",
            description="Clasifica sentimiento: positive/negative/neutral.",
            system_prompt=(
                "You are a sentiment classifier. Reply with exactly one of: "
                "positive, negative, neutral."
            ),
            keywords=["sentiment", "sentimiento", "positive", "negative", "neutral", "classify"],
            tune_items=_sentiment_items("tune"),
            eval_items=_sentiment_items("eval"),
        ),
        ExpertSpec(
            id="entities",
            name="Entity/date extraction",
            description="Extrae persona, lugar y fecha en JSON.",
            system_prompt=(
                "You extract entities. Reply ONLY with JSON containing keys "
                "person, place, date. No other text."
            ),
            keywords=["extract", "entity", "entities", "date", "person", "place", "json"],
            tune_items=_entities_items("tune"),
            eval_items=_entities_items("eval"),
        ),
        ExpertSpec(
            id="bullets",
            name="Exact bullet lists",
            description="Listas con exactamente N viñetas '- '.",
            system_prompt=(
                "You produce exact bullet lists. Use '- ' for each bullet, "
                "exact count, no intro or outro."
            ),
            keywords=["viñetas", "bullets", "lista", "exactamente", "- "],
            tune_items=_bullets_items("tune"),
            eval_items=_bullets_items("eval"),
        ),
        ExpertSpec(
            id="code_python",
            name="Python code snippets",
            description="Devuelve solo snippets de Python válidos.",
            system_prompt=(
                "You write short Python code only. No markdown fences, no explanation, "
                "just executable code."
            ),
            keywords=["python", "def ", "code", "función", "snippet", "lambda", "class "],
            tune_items=_code_items("tune"),
            eval_items=_code_items("eval"),
        ),
    ]


def expert_by_id(eid: str) -> ExpertSpec:
    for e in all_expert_specs():
        if e.id == eid:
            return e
    raise KeyError(eid)
