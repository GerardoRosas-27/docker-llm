"""Catálogo auditable de 50 tareas mono-especialista.

Splits: train (dirección), tune (selección hiperparams), val (gating reemplazo),
heldout (bench final — NUNCA usar para seleccionar/promover).
Checks deterministas (parse/regex/exact), no substring laxo.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

CheckFn = Callable[[str], bool]


@dataclass
class Item:
    prompt: str
    check: CheckFn
    meta: Dict = field(default_factory=dict)


@dataclass
class TaskSpec:
    id: str
    name: str
    family: str
    description: str
    keywords: List[str]
    system_prompt: str
    train_items: List[Item]
    tune_items: List[Item]
    val_items: List[Item]
    heldout_items: List[Item]

    def all_non_heldout(self) -> List[Item]:
        return self.train_items + self.tune_items + self.val_items


def _words(t: str):
    return re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9']+", t)


def _parse_json(t: str):
    s = t.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I).strip()
    m = re.search(r"(\{.*\}|\[.*\])", s, re.S)
    if m:
        s = m.group(1)
    try:
        return json.loads(s)
    except Exception:
        return None


def _only_token(t: str) -> str:
    return t.strip().splitlines()[0].strip() if t.strip() else ""


def _exact_ci(expected: str) -> CheckFn:
    exp = expected.strip().lower()
    def chk(t: str, e=exp) -> bool:
        return _only_token(t).lower() == e
    return chk


def _exact(expected: str) -> CheckFn:
    exp = expected.strip()
    def chk(t: str, e=exp) -> bool:
        return _only_token(t) == e
    return chk


def _regex(pat: str, flags=0) -> CheckFn:
    rx = re.compile(pat, flags)
    def chk(t: str, r=rx) -> bool:
        return bool(r.fullmatch(_only_token(t))) or bool(r.search(t.strip()))
    return chk


def _mk_splits(rows: List[tuple], make_item, n_train=4, n_tune=4, n_val=4, n_held=5):
    assert len(rows) >= n_train + n_tune + n_val + n_held, f"need >= {n_train+n_tune+n_val+n_held} got {len(rows)}"
    i = 0
    tr = [make_item(r) for r in rows[i:i+n_train]]; i += n_train
    tu = [make_item(r) for r in rows[i:i+n_tune]]; i += n_tune
    va = [make_item(r) for r in rows[i:i+n_val]]; i += n_val
    he = [make_item(r) for r in rows[i:i+n_held]]
    return tr, tu, va, he


def _task(tid, name, family, desc, keywords, system, rows, make_item) -> TaskSpec:
    tr, tu, va, he = _mk_splits(rows, make_item)
    return TaskSpec(tid, name, family, desc, keywords, system, tr, tu, va, he)


def _build_catalog() -> List[TaskSpec]:
    tasks: List[TaskSpec] = []

    # ---- family: transform (10) ----
    words_pool = [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
        "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
        "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
    ]

    def add_transform(tid, name, transform, prompt_tpl, keywords):
        rows = [(w, transform(w)) for w in words_pool]
        def make(r):
            src, out = r
            return Item(prompt_tpl.format(x=src), _exact(out), {"src": src, "out": out})
        tasks.append(_task(tid, name, "transform", name, keywords,
                           f"Reply with ONLY the transformed token. No punctuation.", rows, make))

    add_transform("upper_token", "Uppercase token", str.upper,
                  "Convert to UPPERCASE. Reply with only the word: {x}", ["uppercase", "upper", "MAYÚSCULAS"])
    add_transform("lower_token", "Lowercase token", str.lower,
                  "Convert to lowercase. Reply with only the word: {x}", ["lowercase", "lower"])
    add_transform("title_token", "Titlecase token", str.title,
                  "Convert to Title Case. Reply with only the word: {x}", ["titlecase", "title"])
    add_transform("reverse_token", "Reverse token", lambda s: s[::-1],
                  "Reverse the letters. Reply with only the reversed word: {x}", ["reverse", "invert"])
    add_transform("dup_token", "Duplicate token", lambda s: s + s,
                  "Duplicate the word (concat twice). Reply with only the result: {x}", ["duplicate", "repeat"])
    add_transform("first3", "First 3 chars", lambda s: s[:3],
                  "Reply with ONLY the first 3 characters of: {x}", ["prefix", "first3"])
    add_transform("last3", "Last 3 chars", lambda s: s[-3:],
                  "Reply with ONLY the last 3 characters of: {x}", ["suffix", "last3"])
    add_transform("len_token", "Length of token", lambda s: str(len(s)),
                  "How many characters? Reply with ONLY the integer: {x}", ["length", "count chars"])
    add_transform("wrap_brackets", "Wrap brackets", lambda s: f"[{s}]",
                  "Wrap the word in square brackets. Reply with only that: {x}", ["brackets", "wrap"])
    add_transform("snake_join", "Join with underscore", lambda s: f"{s}_{s}",
                  "Reply with word_word (underscore join of the same word twice): {x}", ["snake", "underscore"])

    # ---- family: format (10) ----
    fmt_topics = [
        ("cat", "meow"), ("dog", "bark"), ("sun", "hot"), ("moon", "cold"), ("rain", "wet"),
        ("snow", "white"), ("book", "read"), ("tree", "tall"), ("fish", "swim"), ("bird", "fly"),
        ("car", "drive"), ("ship", "sail"), ("fire", "burn"), ("wind", "blow"), ("stone", "hard"),
        ("river", "flow"), ("cloud", "soft"), ("star", "bright"),
    ]

    # json_ab
    def make_json_ab(r):
        a, b = r
        def chk(t, aa=a, bb=b):
            o = _parse_json(t)
            return isinstance(o, dict) and o.get("a") == aa and o.get("b") == bb
        return Item(f'Reply ONLY with JSON object with keys "a" and "b" set to "{a}" and "{b}" respectively.', chk, {"a": a, "b": b})
    tasks.append(_task("json_ab", "JSON a/b exact", "format", "Emit exact JSON keys a,b",
                       ["json", "object"], "Output ONLY valid JSON.", fmt_topics, make_json_ab))

    # json_list2
    def make_json_list(r):
        a, b = r
        def chk(t, aa=a, bb=b):
            o = _parse_json(t)
            return isinstance(o, list) and o == [aa, bb]
        return Item(f'Reply ONLY with a JSON array of exactly two strings: "{a}" then "{b}".', chk)
    tasks.append(_task("json_list2", "JSON list of 2", "format", "Emit JSON array length 2",
                       ["json", "array", "list"], "Output ONLY a JSON array.", fmt_topics, make_json_list))

    # csv_two
    def make_csv(r):
        a, b = r
        return Item(f'Reply ONLY with CSV line: {a},{b}', _exact(f"{a},{b}"))
    tasks.append(_task("csv_two", "CSV two fields", "format", "Emit a,b CSV",
                       ["csv"], "Output ONLY csv.", fmt_topics, make_csv))

    # kv_colon
    def make_kv(r):
        a, b = r
        return Item(f'Reply ONLY with key:value as {a}:{b}', _exact(f"{a}:{b}"))
    tasks.append(_task("kv_colon", "key:value pair", "format", "Emit key:value",
                       ["keyvalue", "kv"], "Output ONLY key:value.", fmt_topics, make_kv))

    # bullets_exact2
    def make_bul2(r):
        a, b = r
        def chk(t, aa=a, bb=b):
            lines = [ln.strip() for ln in t.strip().splitlines() if ln.strip()]
            return lines == [f"- {aa}", f"- {bb}"]
        return Item(f"Reply with EXACTLY two bullet lines: '- {a}' then '- {b}'. Nothing else.", chk)
    tasks.append(_task("bullets_two", "Exactly 2 bullets", "format", "Two '- ' bullets",
                       ["bullets", "list"], "Exact bullets only.", fmt_topics, make_bul2))

    # numbered_two
    def make_num2(r):
        a, b = r
        def chk(t, aa=a, bb=b):
            lines = [ln.strip() for ln in t.strip().splitlines() if ln.strip()]
            return lines == [f"1. {aa}", f"2. {bb}"]
        return Item(f"Reply with EXACTLY: '1. {a}' then '2. {b}' on two lines.", chk)
    tasks.append(_task("numbered_two", "Numbered 1. 2.", "format", "Numbered list 2",
                       ["numbered", "ordered"], "Exact numbered list.", fmt_topics, make_num2))

    # xml_tag
    def make_xml(r):
        a, b = r
        tag = f"<{a}>{b}</{a}>"
        return Item(f'Reply ONLY with XML element <{a}>{b}</{a}>', _exact(tag))
    tasks.append(_task("xml_wrap", "XML wrap", "format", "Emit <tag>val</tag>",
                       ["xml", "tag"], "Output ONLY the XML element.", fmt_topics, make_xml))

    # quoted
    def make_quoted(r):
        a, _ = r
        return Item(f'Reply ONLY with the word in double quotes: {a}', _exact(f'"{a}"'))
    tasks.append(_task("double_quote", "Double-quote word", "format", "Wrap in quotes",
                       ["quote", "quotes"], "Output ONLY quoted token.", fmt_topics, make_quoted))

    # paren_wrap
    def make_paren(r):
        a, _ = r
        return Item(f"Reply ONLY with the word in parentheses: {a}", _exact(f"({a})"))
    tasks.append(_task("paren_wrap", "Parentheses wrap", "format", "Wrap in ()",
                       ["paren", "parentheses"], "Output ONLY (word).", fmt_topics, make_paren))

    # pipe_join
    def make_pipe(r):
        a, b = r
        return Item(f"Reply ONLY with {a}|{b}", _exact(f"{a}|{b}"))
    tasks.append(_task("pipe_join", "Pipe join", "format", "Join with |",
                       ["pipe", "join"], "Output ONLY a|b.", fmt_topics, make_pipe))

    # ---- family: extract (10) ----
    extract_rows = [
        ("Order 42 shipped", "42"), ("Code AB-7 ready", "AB-7"), ("User id=99 ok", "99"),
        ("Room 12A north", "12A"), ("Ticket T-1001 closed", "T-1001"), ("Score was 87 points", "87"),
        ("Gate B2 open", "B2"), ("Invoice INV9 paid", "INV9"), ("Flight AA12 delayed", "AA12"),
        ("Box 3 of 5", "3"), ("Page 16 starts", "16"), ("Zone Z9 clear", "Z9"),
        ("Serial SN88 found", "SN88"), ("Dock 4 busy", "4"), ("Team X3 wins", "X3"),
        ("Slot 21 free", "21"), ("Key K5 used", "K5"),
    ]
    # extract first integer
    int_rows = [(s, n) for s, n in [
        ("There are 7 apples", "7"), ("I saw 12 birds", "12"), ("Add 3 spoons", "3"),
        ("Wait 45 minutes", "45"), ("Need 100 units", "100"), ("Only 1 left", "1"),
        ("Found 9 keys", "9"), ("Ran 26 miles", "26"), ("Has 8 sides", "8"),
        ("Took 15 days", "15"), ("Built 4 walls", "4"), ("Ate 2 pears", "2"),
        ("Lost 11 coins", "11"), ("Won 6 games", "6"), ("Cut 5 boards", "5"),
        ("Read 30 pages", "30"), ("Burned 90 calories", "90"),
    ]]
    def make_int(r):
        s, n = r
        return Item(f"Extract the integer from the sentence. Reply with ONLY the number.\n{s}", _exact(n))
    tasks.append(_task("extract_int", "Extract integer", "extract", "First integer token",
                       ["extract", "number", "integer"], "Output ONLY the integer.", int_rows, make_int))

    # extract email local
    email_rows = [(f"Contact {u}@example.com please", u) for u in
                  ["ana", "bob", "cara", "dan", "eva", "finn", "gina", "hugo", "ira", "jade",
                   "kai", "lea", "mia", "ned", "ola", "pia", "quin"]]
    def make_email_local(r):
        s, u = r
        return Item(f"Extract the email local-part (before @). Reply with ONLY that token.\n{s}", _exact(u))
    tasks.append(_task("extract_email_local", "Email local-part", "extract", "Local part of email",
                       ["email", "extract"], "Output ONLY local-part.", email_rows, make_email_local))

    # first word
    fw_rows = [(s, s.split()[0]) for s in [
        "Red cars go fast", "Blue sky today", "Green tea helps", "Silent night falls",
        "Quick fox jumps", "Brave knights ride", "Calm waters shine", "Dark clouds gather",
        "Early birds sing", "Fresh bread smells", "Golden leaves fall", "Happy kids play",
        "Icy winds blow", "Jolly tunes play", "Kind words heal", "Loud drums bang", "Mild days last",
    ]]
    def make_fw(r):
        s, w = r
        return Item(f"Reply with ONLY the first word of: {s}", _exact(w))
    tasks.append(_task("first_word", "First word", "extract", "First whitespace token",
                       ["first", "word"], "Output ONLY first word.", fw_rows, make_fw))

    # last word
    lw_rows = [(s, s.split()[-1]) for s, _ in fw_rows]
    def make_lw(r):
        s, w = r
        return Item(f"Reply with ONLY the last word of: {s}", _exact(w))
    tasks.append(_task("last_word", "Last word", "extract", "Last whitespace token",
                       ["last", "word"], "Output ONLY last word.", lw_rows, make_lw))

    # word count
    wc_rows = [(s, str(len(s.split()))) for s, _ in fw_rows]
    def make_wc(r):
        s, n = r
        return Item(f"How many words? Reply with ONLY the integer.\n{s}", _exact(n))
    tasks.append(_task("word_count", "Word count", "extract", "Count words",
                       ["count", "words"], "Output ONLY integer.", wc_rows, make_wc))

    # extract code token like AB-7
    code_rows = extract_rows
    def make_code(r):
        s, c = r
        # more specific prompts: ask for the code-like token matching pattern
        return Item(f"Extract the code token (digits and/or letters with optional hyphen) highlighted by context. Reply ONLY with that token.\n{s}", _exact(c))
    tasks.append(_task("extract_code", "Extract code token", "extract", "Code-like token",
                       ["code", "token", "extract"], "Output ONLY the code token.", code_rows, make_code))

    # yes if digit present
    dig_rows = []
    for s, has in [("hello world", False), ("room 2", True), ("no digits", False), ("A1", True),
                   ("plain text", False), ("99 bottles", True), ("abc", False), ("x7y", True),
                   ("none here", False), ("v2", True), ("test", False), ("42", True),
                   ("alpha", False), ("b2b", True), ("zzz", False), ("3d", True), ("void", False)]:
        dig_rows.append((s, "yes" if has else "no"))
    def make_has_digit(r):
        s, ans = r
        return Item(f"Does the text contain a digit? Reply ONLY yes or no.\n{s}", _exact(ans))
    tasks.append(_task("has_digit", "Has digit?", "extract", "Binary digit presence",
                       ["digit", "yes", "no"], "Output ONLY yes or no.", dig_rows, make_has_digit))

    # char at index 0
    cai_rows = [(w, w[0]) for w in words_pool[:17]]
    def make_c0(r):
        w, ch = r
        return Item(f"Reply with ONLY the first character of: {w}", _exact(ch))
    tasks.append(_task("char0", "First character", "extract", "Index 0 char",
                       ["character", "first"], "Output ONLY one character.", cai_rows, make_c0))

    # is_alpha_word
    al_rows = [(w, "yes") for w in words_pool[:9]] + [(w, "no") for w in ["a1", "x-y", "hi!", "2two", "ok?", "n0", "ab_c", "z9"]]
    def make_alpha(r):
        w, ans = r
        return Item(f"Is this an alphabetic word only? Reply ONLY yes or no: {w}", _exact(ans))
    tasks.append(_task("is_alpha", "Alphabetic?", "extract", "Alpha-only check",
                       ["alpha", "letters"], "Output ONLY yes or no.", al_rows, make_alpha))

    # count vowels
    def n_vowels(s):
        return str(sum(c in "aeiouAEIOU" for c in s))
    vow_rows = [(w, n_vowels(w)) for w in words_pool[:17]]
    def make_vow(r):
        w, n = r
        return Item(f"Count vowels (aeiou). Reply with ONLY the integer: {w}", _exact(n))
    tasks.append(_task("count_vowels", "Count vowels", "extract", "Vowel count",
                       ["vowels", "count"], "Output ONLY integer.", vow_rows, make_vow))

    # ---- family: math (10) ----
    def math_task(tid, name, opname, pairs, fn, keywords):
        rows = [(a, b, fn(a, b)) for a, b in pairs]
        def make(r):
            a, b, out = r
            return Item(f"Compute {opname}. Reply with ONLY the integer result.\n{a} ? {b}", _exact(str(out)), {"a": a, "b": b})
        tasks.append(_task(tid, name, "math", name, keywords, "Output ONLY an integer.", rows, make))

    pairs = [(i, j) for i in range(1, 8) for j in range(1, 6)][:17]
    math_task("add_int", "Integer add", "a+b", pairs, lambda a, b: a + b, ["add", "plus", "sum"])
    math_task("sub_int", "Integer sub", "a-b", pairs, lambda a, b: a - b, ["subtract", "minus"])
    math_task("mul_int", "Integer mul", "a*b", pairs, lambda a, b: a * b, ["multiply", "product"])
    math_task("max_int", "Integer max", "max(a,b)", pairs, lambda a, b: max(a, b), ["max", "maximum"])
    math_task("min_int", "Integer min", "min(a,b)", pairs, lambda a, b: min(a, b), ["min", "minimum"])
    math_task("mod_int", "Integer mod", "a%b", [(a, b) for a, b in pairs if b], lambda a, b: a % b, ["modulo", "mod", "remainder"])
    math_task("abs_diff", "Abs difference", "|a-b|", pairs, lambda a, b: abs(a - b), ["difference", "abs"])
    # compare
    cmp_rows = [(a, b, "yes" if a > b else "no") for a, b in pairs]
    def make_gt(r):
        a, b, ans = r
        return Item(f"Is {a} greater than {b}? Reply ONLY yes or no.", _exact(ans))
    tasks.append(_task("cmp_gt", "Compare greater", "math", "a>b yes/no",
                       ["greater", "compare"], "Output ONLY yes or no.", cmp_rows, make_gt))
    # parity
    par_rows = [(n, "even" if n % 2 == 0 else "odd") for n in range(1, 18)]
    def make_par(r):
        n, ans = r
        return Item(f"Is {n} even or odd? Reply ONLY even or odd.", _exact(ans))
    tasks.append(_task("parity", "Even or odd", "math", "Parity",
                       ["even", "odd", "parity"], "Output ONLY even or odd.", par_rows, make_par))
    # triple
    trip_rows = [(n, str(3 * n)) for n in range(1, 18)]
    def make_trip(r):
        n, out = r
        return Item(f"Multiply by 3. Reply with ONLY the integer: {n}", _exact(out))
    tasks.append(_task("triple", "Times three", "math", "n*3",
                       ["triple", "times three"], "Output ONLY integer.", trip_rows, make_trip))

    # ---- family: classify / label (10) ----
    # sentiment ternary
    sent_rows = [
        ("I love this", "positive"), ("This is awful", "negative"), ("It is a table", "neutral"),
        ("Best day ever", "positive"), ("I hate waiting", "negative"), ("The box is blue", "neutral"),
        ("Amazing work", "positive"), ("Terrible service", "negative"), ("Page number five", "neutral"),
        ("So happy today", "positive"), ("Worst meal", "negative"), ("Wooden chair", "neutral"),
        ("Fantastic news", "positive"), ("Disgusting smell", "negative"), ("Metal spoon", "neutral"),
        ("Joyful morning", "positive"), ("Painful injury", "negative"),
    ]
    def make_sent(r):
        s, lab = r
        return Item(f"Classify sentiment as positive, negative, or neutral. Reply with ONLY the label.\n{s}", _exact(lab))
    tasks.append(_task("sentiment3", "Sentiment 3-way", "classify", "pos/neg/neu",
                       ["sentiment", "positive", "negative", "neutral"], "Output ONLY one label.", sent_rows, make_sent))

    # lang en/es
    lang_rows = [
        ("The cat sleeps", "en"), ("El gato duerme", "es"), ("I eat bread", "en"), ("Como pan", "es"),
        ("Blue sky", "en"), ("Cielo azul", "es"), ("Good morning", "en"), ("Buenos dias", "es"),
        ("Red apple", "en"), ("Manzana roja", "es"), ("Fast car", "en"), ("Coche rapido", "es"),
        ("Cold water", "en"), ("Agua fria", "es"), ("Happy child", "en"), ("Nino feliz", "es"),
        ("Open door", "en"),
    ]
    def make_lang(r):
        s, lab = r
        return Item(f"Is the text English (en) or Spanish (es)? Reply ONLY en or es.\n{s}", _exact(lab))
    tasks.append(_task("lang_en_es", "Language en/es", "classify", "en vs es",
                       ["language", "english", "spanish"], "Output ONLY en or es.", lang_rows, make_lang))

    # yes/no question echo
    yn_rows = [("true", "yes"), ("false", "no"), ("yes", "yes"), ("no", "no"),
               ("1", "yes"), ("0", "no"), ("YES", "yes"), ("NO", "no"),
               ("True", "yes"), ("False", "no"), ("y", "yes"), ("n", "no"),
               ("on", "yes"), ("off", "no"), ("ok", "yes"), ("never", "no"), ("always", "yes")]
    def make_yn(r):
        s, ans = r
        return Item(f"Map to yes/no. Reply ONLY yes or no. Input: {s}", _exact(ans))
    tasks.append(_task("map_yes_no", "Map to yes/no", "classify", "Normalize yes/no",
                       ["yes", "no", "boolean"], "Output ONLY yes or no.", yn_rows, make_yn))

    # animal vs object (simple lexicon)
    ao_rows = [
        ("cat", "animal"), ("dog", "animal"), ("table", "object"), ("chair", "object"),
        ("bird", "animal"), ("fish", "animal"), ("stone", "object"), ("book", "object"),
        ("horse", "animal"), ("cow", "animal"), ("lamp", "object"), ("cup", "object"),
        ("lion", "animal"), ("tiger", "animal"), ("pen", "object"), ("desk", "object"), ("wolf", "animal"),
    ]
    def make_ao(r):
        s, lab = r
        return Item(f"Classify as animal or object. Reply ONLY with the label: {s}", _exact(lab))
    tasks.append(_task("animal_or_object", "Animal vs object", "classify", "Binary noun class",
                       ["animal", "object"], "Output ONLY animal or object.", ao_rows, make_ao))

    # food vs not
    food_rows = [
        ("apple", "food"), ("bread", "food"), ("car", "other"), ("river", "other"),
        ("cheese", "food"), ("rice", "food"), ("mountain", "other"), ("cloud", "other"),
        ("soup", "food"), ("fish", "food"), ("hammer", "other"), ("window", "other"),
        ("milk", "food"), ("egg", "food"), ("shoe", "other"), ("road", "other"), ("cake", "food"),
    ]
    def make_food(r):
        s, lab = r
        return Item(f"Is it food or other? Reply ONLY food or other: {s}", _exact(lab))
    tasks.append(_task("food_or_other", "Food vs other", "classify", "Food lexicon",
                       ["food", "other"], "Output ONLY food or other.", food_rows, make_food))

    # color word?
    color_set = {"red", "blue", "green", "yellow", "black", "white", "orange", "purple"}
    col_rows = [(w, "yes" if w in color_set else "no") for w in
                list(color_set) + ["cat", "dog", "table", "happy", "fast", "moon", "tree", "code", "alpha"]]
    def make_col(r):
        w, ans = r
        return Item(f"Is this an English color word? Reply ONLY yes or no: {w}", _exact(ans))
    tasks.append(_task("is_color", "Is color word?", "classify", "Color lexicon",
                       ["color", "yes", "no"], "Output ONLY yes or no.", col_rows, make_col))

    # plural heuristic ends with s
    pl_rows = [(w, "plural" if w.endswith("s") else "singular") for w in
               ["cats", "dog", "books", "pen", "cars", "tree", "birds", "cup", "keys", "map",
                "houses", "road", "apples", "stone", "boxes", "lamp", "ships"]]
    def make_pl(r):
        w, lab = r
        return Item(f"Guess singular/plural by ending. Reply ONLY singular or plural: {w}", _exact(lab))
    tasks.append(_task("sing_plur", "Singular/plural heuristic", "classify", "Ends with s",
                       ["plural", "singular"], "Output ONLY singular or plural.", pl_rows, make_pl))

    # short/long by len
    sl_rows = [(w, "short" if len(w) <= 4 else "long") for w in words_pool[:17]]
    def make_sl(r):
        w, lab = r
        return Item(f"Is the word short (<=4 chars) or long? Reply ONLY short or long: {w}", _exact(lab))
    tasks.append(_task("short_or_long", "Short vs long word", "classify", "Length threshold 4",
                       ["short", "long"], "Output ONLY short or long.", sl_rows, make_sl))

    # starts with vowel
    sv_rows = [(w, "yes" if w[0].lower() in "aeiou" else "no") for w in words_pool[:17]]
    def make_sv(r):
        w, ans = r
        return Item(f"Does it start with a vowel? Reply ONLY yes or no: {w}", _exact(ans))
    tasks.append(_task("starts_vowel", "Starts with vowel?", "classify", "Initial vowel",
                       ["vowel", "starts"], "Output ONLY yes or no.", sv_rows, make_sv))

    # pick label from {A,B}
    ab_rows = [(f"Choose A", "A"), (f"Choose B", "B")] * 9
    ab_rows = ab_rows[:17]
    # make distinct prompts
    ab_rows = []
    for i in range(17):
        lab = "A" if i % 2 == 0 else "B"
        ab_rows.append((f"Reply with option {lab} only.", lab))
    def make_ab(r):
        prompt, lab = r
        return Item(prompt + " Output ONLY the letter.", _exact(lab))
    tasks.append(_task("pick_ab", "Pick A or B", "classify", "Forced A/B",
                       ["option", "A", "B"], "Output ONLY A or B.", ab_rows, make_ab))

    assert len(tasks) == 50, len(tasks)
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    for t in tasks:
        assert len(t.heldout_items) >= 5, t.id
        assert len(t.tune_items) >= 3 and len(t.val_items) >= 3, t.id
    return tasks


_CATALOG: Optional[List[TaskSpec]] = None


def all_tasks() -> List[TaskSpec]:
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = _build_catalog()
    return _CATALOG


def task_by_id(tid: str) -> TaskSpec:
    for t in all_tasks():
        if t.id == tid:
            return t
    raise KeyError(tid)


def catalog_manifest() -> Dict:
    return {
        "n_tasks": len(all_tasks()),
        "families": sorted({t.family for t in all_tasks()}),
        "tasks": [
            {
                "id": t.id, "name": t.name, "family": t.family,
                "n_train": len(t.train_items), "n_tune": len(t.tune_items),
                "n_val": len(t.val_items), "n_heldout": len(t.heldout_items),
                "keywords": t.keywords,
            }
            for t in all_tasks()
        ],
    }
