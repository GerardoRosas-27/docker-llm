"""Benchmark tipo IFEval verificable automaticamente (es/en)."""
import json, re

TOPICS_EVAL = ["el café", "los gatos", "la lluvia", "el océano", "las montañas", "la música", "los libros",
  "el fútbol", "las abejas", "la luna", "el pan", "los volcanes", "el ajedrez", "las bicicletas",
  "los árboles", "el invierno", "la amistad", "los trenes"]
TOPICS_DIR = ["el sol", "los perros", "el chocolate", "las estrellas", "la nieve", "los ríos", "el teatro", "las flores"]
TOPICS_EN = {"el café":"coffee","los gatos":"cats","la lluvia":"rain","el océano":"the ocean","las montañas":"mountains",
  "la música":"music","los libros":"books","el fútbol":"soccer","las abejas":"bees","la luna":"the moon","el pan":"bread",
  "los volcanes":"volcanoes","el ajedrez":"chess","las bicicletas":"bicycles","los árboles":"trees","el invierno":"winter",
  "la amistad":"friendship","los trenes":"trains","el sol":"the sun","los perros":"dogs","el chocolate":"chocolate",
  "las estrellas":"stars","la nieve":"snow","los ríos":"rivers","el teatro":"theater","las flores":"flowers"}

def words(t): return re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9']+", t)
def bullets(t): return [l for l in t.strip().splitlines() if re.match(r"^\s*([-*•]|\d+[.)])\s+", l)]
def nonempty(t): return [l for l in t.strip().splitlines() if l.strip()]
ES = set("el la los las de que y en un una es por para con no se su al lo como más pero sus muy también son está".split())
EN = set("the and of to is in it that for are with as on be this was they by an at or can their its".split())
def lang(t):
    w=[x.lower() for x in words(t)]; e=sum(x in EN for x in w); s=sum(x in ES for x in w)
    return "en" if e>s else ("es" if s>e else "?")
def parse_json(t):
    s=t.strip(); s=re.sub(r"^```(json)?\s*|\s*```$","",s).strip()
    try: return json.loads(s)
    except Exception: return None

def make(topic, ctype):
    en = TOPICS_EN[topic]
    C = {
     "bullets3": (f"Escribe sobre {topic}. Responde en exactamente 3 viñetas que empiecen con '- ', sin ningún otro texto.",
                  lambda t: len(bullets(t))==3 and len(nonempty(t))==3),
     "mayusculas": (f"Describe {topic} en dos frases. Escribe TODA la respuesta en MAYÚSCULAS.",
                  lambda t: len(words(t))>=3 and t==t.upper()),
     "minusculas": (f"Describe {topic} in two sentences. Your entire answer must be in lowercase letters only, no capital letters.",
                  lambda t: len(words(t))>=3 and t==t.lower()),
     "sin_palabra": (f"Explica qué es {topic} sin usar la palabra 'es' en ningún momento.",
                  lambda t: len(words(t))>=3 and "es" not in [w.lower() for w in words(t)]),
     "termina_con": (f"Escribe una frase corta sobre {topic} y termina tu respuesta exactamente con la frase 'Fin del mensaje.'",
                  lambda t: t.strip().endswith("Fin del mensaje.")),
     "empieza_con": (f"Write one sentence about {en}. Your answer must start with the exact word 'Indeed'.",
                  lambda t: t.strip().startswith("Indeed")),
     "json_ab": (f"Responde solo con JSON válido con exactamente las claves \"a\" y \"b\" (texto sobre {topic}). Nada más.",
                  lambda t: isinstance(parse_json(t),dict) and set(parse_json(t).keys())=={"a","b"}),
     "max_palabras": (f"Describe {topic} en máximo 10 palabras.",
                  lambda t: 1<=len(words(t))<=10),
     "en_ingles": (f"Escribe dos frases sobre {topic}. Responde únicamente en inglés.",
                  lambda t: len(words(t))>=4 and lang(t)=="en"),
     "en_espanol": (f"Write two sentences about {en}. Answer only in Spanish.",
                  lambda t: len(words(t))>=4 and lang(t)=="es"),
     "una_palabra": (f"¿Qué color asocias con {topic}? Responde con una sola palabra.",
                  lambda t: len(words(t))==1),
     "sin_comas": (f"Write two sentences about {en} without using any commas.",
                  lambda t: len(words(t))>=4 and "," not in t),
    }
    return C[ctype]
CTYPES = ["bullets3","mayusculas","minusculas","sin_palabra","termina_con","empieza_con","json_ab","max_palabras","en_ingles","en_espanol","una_palabra","sin_comas"]

def build(split):
    topics = TOPICS_EVAL if split=="eval" else TOPICS_DIR
    items=[]
    for i,c in enumerate(CTYPES):
        for j,tp in enumerate(topics):
            if split=="eval" and (i+j)%2: continue  # 9 por tipo -> 108
            p,chk=make(tp,c); items.append({"type":c,"prompt":p,"check":chk})
    return items

PPL_TEXT = """The city library opened its doors early on Saturday morning. Families arrived with children who wanted to borrow picture books, while students looked for quiet tables to study for their exams. A volunteer explained how to use the new catalog system, and an old man asked about history books on the region.
Water boils at one hundred degrees Celsius at sea level. At higher altitudes the air pressure is lower, so water boils at a lower temperature, which is why cooking pasta in the mountains can take longer.
La biblioteca de la ciudad abrió temprano el sábado. Las familias llegaron con niños que querían libros ilustrados, mientras los estudiantes buscaban mesas tranquilas para estudiar.
To make a simple tomato sauce, heat olive oil in a pan, add chopped garlic, and cook it gently. Then add crushed tomatoes, a pinch of salt, and let the sauce simmer for twenty minutes, stirring from time to time.
The Earth orbits the Sun once every year, and the Moon orbits the Earth roughly once every month. These motions explain the seasons, the phases of the Moon, and the timing of eclipses."""
