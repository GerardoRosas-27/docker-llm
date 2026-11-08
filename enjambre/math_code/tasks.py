"""Tareas y validadores ejecutables para Matemáticas y Programación (splits disjuntos)."""
from __future__ import annotations
import ast
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Set, Tuple

from enjambre.bridges.data import all_historical_prompts as bridges_hist_prompts
from enjambre.bridges.data import make_splits as make_bridges_k12_splits
from enjambre.experts import Item

CheckFn = Callable[[str], bool]
Row = Tuple[str, str, CheckFn]

MATH_TASKS = ["math_arith", "math_mul_mod", "math_expr", "math_logic"]
CODE_TASKS = ["code_fn", "code_algo", "code_comp", "code_lambda"]
ALL_MC_TASKS = MATH_TASKS + CODE_TASKS

MATH_SYSTEM_PROMPT = (
    "You are an exact mathematical calculator. Compute step-by-step concisely in 1-2 lines, "
    "then write 'Final Answer: <integer>'."
)

CODE_SYSTEM_PROMPT = (
    "You are a pure Python code generator. Reply with ONLY valid, executable Python code. "
    "No markdown prose, no bullet points, no explanations after the code."
)


@dataclass
class MCTaskSpec:
    id: str
    domain: str  # "math" or "code"
    name: str
    system_prompt: str


MC_SPECS: List[MCTaskSpec] = [
    MCTaskSpec("math_arith", "math", "Aritmética 2-3 dígitos (+ / -)", MATH_SYSTEM_PROMPT),
    MCTaskSpec("math_mul_mod", "math", "Multiplicación, división entera y módulo", MATH_SYSTEM_PROMPT),
    MCTaskSpec("math_expr", "math", "Expresiones algebraicas y paréntesis", MATH_SYSTEM_PROMPT),
    MCTaskSpec("math_logic", "math", "Lógica numérica (max/min/abs/triple)", MATH_SYSTEM_PROMPT),
    MCTaskSpec("code_fn", "code", "Funciones Python puras (AST + exec)", CODE_SYSTEM_PROMPT),
    MCTaskSpec("code_algo", "code", "Algoritmos cortos condicionales (AST + unit tests)", CODE_SYSTEM_PROMPT),
    MCTaskSpec("code_comp", "code", "Comprensiones de listas Python (AST + eval)", CODE_SYSTEM_PROMPT),
    MCTaskSpec("code_lambda", "code", "Expresiones lambda ejecutables (AST + eval)", CODE_SYSTEM_PROMPT),
]


def extract_int_candidate(text: str) -> Optional[int]:
    """Extrae de manera robusta el entero de una respuesta matemática:
    1. Si existe 'Final Answer: X' o 'Answer: X'.
    2. Si existe '\\boxed{X}'.
    3. Si la última o primera línea es un entero (o 'entero.').
    4. El último entero aislado en el texto de cálculo.
    """
    s = text.strip()
    if not s:
        return None

    # 1. Final Answer: <int>
    m = re.search(r"(?:final\s+answer|answer)\s*[:=]?\s*([+-]?\d+)", s, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass

    # 2. \boxed{<int>}
    m_box = re.search(r"\\boxed\{\s*([+-]?\d+)\s*\}", s)
    if m_box:
        try:
            return int(m_box.group(1))
        except ValueError:
            pass

    # 3. Líneas individuales (última o primera)
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines:
        for candidate in [lines[-1], lines[0]]:
            cand_clean = re.sub(r"[\. \t]", "", candidate)
            if re.fullmatch(r"[+-]?\d+", cand_clean):
                try:
                    return int(cand_clean)
                except ValueError:
                    pass

    # 4. Último entero aislado
    all_ints = re.findall(r"(?<![a-zA-Z_])([+-]?\d+)(?![a-zA-Z_])", s)
    if all_ints:
        try:
            return int(all_ints[-1])
        except ValueError:
            pass

    return None


def check_exact_int(text: str, expected: int) -> bool:
    """Verifica que el entero predicho coincida con el entero esperado."""
    cand = extract_int_candidate(text)
    if cand is None:
        return False
    return cand == expected


def _strip_code_fences(text: str) -> str:
    s = text.strip()
    m = re.search(r"```(?:python)?\s*\n(.*?)(?:\n```|$)", s, re.DOTALL)
    if m:
        return m.group(1).strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def has_trailing_prose(text: str) -> bool:
    """Detecta viñetas ('- The function...') o párrafos explicativos fuera del código."""
    s = _strip_code_fences(text)
    for ln in s.splitlines():
        st = ln.strip()
        if st.startswith("- ") or st.startswith("* ") or st.startswith("• "):
            return True
        if st.lower().startswith(("explanation:", "note:", "this function", "the function", "here ")):
            return True
    return False


def check_python_function(text: str, fn_name: str, test_cases: List[Tuple[Tuple[Any, ...], Any]]) -> bool:
    """Verifica que:
    1) No haya viñetas ni párrafos explicativos fuera del código.
    2) Se extraiga el snippet Python limpio (fenced o directo).
    3) Compile con ast.parse y pase todos los `test_cases`.
    """
    if has_trailing_prose(text):
        return False

    code = _strip_code_fences(text)
    if not code or f"def {fn_name}" not in code:
        idx = text.find(f"def {fn_name}")
        if idx != -1:
            lines = []
            for ln in text[idx:].splitlines():
                if ln.strip().startswith("```"):
                    break
                lines.append(ln)
            code = "\n".join(lines).strip()
        else:
            return False

    try:
        tree = ast.parse(code)
    except Exception:
        # Intentar recortar hasta la última línea que cierre el bloque de la función
        try:
            lines = code.splitlines()
            valid_tree = None
            for end in range(len(lines), 0, -1):
                subcode = "\n".join(lines[:end])
                try:
                    t = ast.parse(subcode)
                    if any(isinstance(n, ast.FunctionDef) and n.name == fn_name for n in t.body):
                        valid_tree = t
                        break
                except Exception:
                    continue
            if valid_tree is not None:
                tree = valid_tree
            else:
                return False
        except Exception:
            return False

    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # docstring o comentario
        if not isinstance(node, (ast.FunctionDef, ast.Return, ast.Assign, ast.Import, ast.ImportFrom, ast.Pass)):
            return False
    ns: Dict[str, Any] = {
        "__builtins__": {
            "__import__": __import__,
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "range": range,
            "int": int,
            "bool": bool,
            "str": str,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "sorted": sorted,
            "reversed": reversed,
            "enumerate": enumerate,
            "zip": zip,
        }
    }
    try:
        exec(compile(tree, "<generated>", "exec"), ns)
        fn = ns.get(fn_name)
        if not callable(fn):
            return False
        for args, expected in test_cases:
            res = fn(*args)
            if res != expected:
                return False
        return True
    except Exception:
        return False


def check_python_expr(text: str, required_substr: str, expected_val: Any = None,
                      callable_tests: List[Tuple[Tuple[Any, ...], Any]] = None) -> bool:
    """Verifica expresiones Python (`[...]`, `lambda ...`, `{...}`) con ast.parse(mode='eval') y evaluación."""
    if has_trailing_prose(text):
        return False
    code = _strip_code_fences(text)
    if not code or required_substr not in code:
        return False
    # Si hay múltiples líneas, debe ser una única línea de expresión
    lines = [ln.strip() for ln in code.splitlines() if ln.strip()]
    if len(lines) != 1:
        return False
    expr_str = lines[0]
    # Permitir asignación simple `f = lambda ...` o `xs = [...]`
    if "=" in expr_str and not expr_str.strip().startswith("lambda"):
        parts = expr_str.split("=", 1)
        if parts[0].strip().isidentifier() and not parts[1].strip().startswith("="):
            expr_str = parts[1].strip()
    try:
        tree = ast.parse(expr_str, mode="eval")
        val = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {"abs": abs, "min": min, "max": max, "sum": sum, "len": len, "range": range, "str": str, "list": list, "dict": dict, "tuple": tuple}}, {})
        if callable_tests is not None:
            if not callable(val):
                return False
            for args, exp in callable_tests:
                if val(*args) != exp:
                    return False
            return True
        return val == expected_val
    except Exception:
        return False


def all_prior_mc_prompts() -> Set[str]:
    s = set(bridges_hist_prompts())
    tr, va, ho, _ = make_bridges_k12_splits(20, 10, 50, seed=0)
    for d in (tr, va):
        for rows in d.values():
            for p, _, _ in rows:
                s.add(p)
    for items in ho.values():
        for it in items:
            s.add(it.prompt)
    return s


def build_mc_pool() -> Dict[str, List[Row]]:
    pool: Dict[str, List[Row]] = {}

    # 1. math_arith: sumas y restas de 2-3 dígitos
    rows_arith: List[Row] = []
    for idx in range(70):
        a = 125 + idx * 3
        b = 42 + (idx * 7) % 85
        if idx % 2 == 0:
            ans = a + b
            p = f"What is {a}+{b}? Reply with ONLY the number."
        else:
            ans = a - b
            p = f"What is {a}-{b}? Reply with ONLY the number."
        rows_arith.append((p, str(ans), (lambda t, e=ans: check_exact_int(t, e))))
    pool["math_arith"] = rows_arith

    # 2. math_mul_mod: multiplicación, división exacta y módulo
    rows_mm: List[Row] = []
    for idx in range(70):
        if idx % 3 == 0:
            a = 6 + idx
            b = 4 + (idx % 9)
            ans = a * b
            p = f"Compute {a} * {b}. Reply with ONLY the integer."
        elif idx % 3 == 1:
            b = 3 + (idx % 8)
            q = 7 + idx
            a = b * q
            ans = q
            p = f"Compute {a} // {b}. Reply with ONLY the integer."
        else:
            a = 45 + idx * 3
            b = 5 + (idx % 9)
            ans = a % b
            p = f"Compute {a} % {b}. Reply with ONLY the integer."
        rows_mm.append((p, str(ans), (lambda t, e=ans: check_exact_int(t, e))))
    pool["math_mul_mod"] = rows_mm

    # 3. math_expr: expresiones de 2 pasos con paréntesis y evaluación lineal
    rows_expr: List[Row] = []
    for idx in range(70):
        a = 5 + idx
        b = 3 + ((idx * 2) % 13)
        c = 2 + (idx % 5)
        if idx % 3 == 0:
            ans = (a + b) * c
            p = f"Evaluate ({a} + {b}) * {c}. Reply with ONLY the integer."
        elif idx % 3 == 1:
            ans = a * c + b
            p = f"Evaluate {a} * {c} + {b}. Reply with ONLY the integer."
        else:
            ans = c * a - b
            p = f"If x = {a}, what is {c}*x - {b}? Reply with ONLY the integer."
        rows_expr.append((p, str(ans), (lambda t, e=ans: check_exact_int(t, e))))
    pool["math_expr"] = rows_expr

    # 4. math_logic: max, min, abs, triple/doble
    rows_logic: List[Row] = []
    for idx in range(70):
        a = 20 + idx * 2
        b = 15 + (idx * 7) % 95
        if a == b:
            b += 3
        if idx % 3 == 0:
            ans = abs(a - b)
            p = f"What is the absolute difference |{a} - {b}|? Reply with ONLY the integer."
        elif idx % 3 == 1:
            c = 10 + idx * 3
            ans = max(a, b, c)
            p = f"What is max({a}, {b}, {c})? Reply with ONLY the integer."
        else:
            ans = 3 * a + 2 * b
            p = f"Compute 3*{a} + 2*{b}. Reply with ONLY the integer."
        rows_logic.append((p, str(ans), (lambda t, e=ans: check_exact_int(t, e))))
    pool["math_logic"] = rows_logic

    # 5. code_fn: funciones Python puras verificadas con AST + exec (sin viñetas ni prosa)
    rows_fn: List[Row] = []
    for idx in range(70):
        fname = f"calc_{idx}"
        k = (idx % 9) + 2
        m = (idx % 5) + 1
        if idx % 3 == 0:
            body = f"return x * {k} + {m}"
            tests = [((0,), m), ((3,), 3 * k + m), ((-2,), -2 * k + m)]
            p = f"Write ONLY a Python function `{fname}(x)` with body `{body}`. No prose."
            tgt = f"def {fname}(x):\n    {body}"
        elif idx % 3 == 1:
            body = f"return a + b * {k}"
            tests = [((1, 2), 1 + 2 * k), ((5, 0), 5), ((-1, 3), -1 + 3 * k)]
            p = f"Write ONLY a Python function `{fname}(a, b)` with body `{body}`. No prose."
            tgt = f"def {fname}(a, b):\n    {body}"
        else:
            body = f"return (x - {m}) * {k}"
            tests = [((m,), 0), ((m + 2,), 2 * k), ((0,), -m * k)]
            p = f"Write ONLY a Python function `{fname}(x)` with body `{body}`. No prose."
            tgt = f"def {fname}(x):\n    {body}"
        rows_fn.append((p, tgt, (lambda t, fn=fname, tc=tests: check_python_function(t, fn, tc))))
    pool["code_fn"] = rows_fn

    # 6. code_algo: algoritmos cortos condicionales/numéricos en Python
    rows_algo: List[Row] = []
    for idx in range(70):
        fname = f"algo_{idx}"
        k = (idx % 7) + 2
        if idx % 4 == 0:
            body = f"return x if x > {k} else {k}"
            tests = [((k - 5,), k), ((k + 5,), k + 5), ((k,), k)]
        elif idx % 4 == 1:
            body = f"return x % {k} == 0"
            tests = [((k * 3,), True), ((k * 3 + 1,), False), ((0,), True)]
        elif idx % 4 == 2:
            body = f"return abs(a - b) + {k}"
            tests = [((10, 3), 7 + k), ((2, 9), 7 + k), ((5, 5), k)]
        else:
            body = f"return max(a, b) * {k}"
            tests = [((2, 5), 5 * k), ((8, 3), 8 * k), ((-1, -4), -1 * k)]
        args_sig = "x" if idx % 4 in (0, 1) else "a, b"
        p = f"Write ONLY a Python function `{fname}({args_sig})` with body `{body}`. No prose."
        tgt = f"def {fname}({args_sig}):\n    {body}"
        rows_algo.append((p, tgt, (lambda t, fn=fname, tc=tests: check_python_function(t, fn, tc))))
    pool["code_algo"] = rows_algo

    # 7. code_comp: comprensiones de listas Python ejecutables
    rows_comp: List[Row] = []
    for idx in range(70):
        n = 3 + (idx % 7)
        k = 2 + idx
        if idx % 3 == 0:
            expr = f"[x * {k} for x in range({n})]"
            expected = [x * k for x in range(n)]
        elif idx % 3 == 1:
            expr = f"[x + {k} for x in range({n})]"
            expected = [x + k for x in range(n)]
        else:
            expr = f"[x * x + {k} for x in range({n})]"
            expected = [x * x + k for x in range(n)]
        p = f"Write ONLY the Python list comprehension `{expr}`. No prose."
        rows_comp.append((p, expr, (lambda t, ex=expected: check_python_expr(t, "[", expected_val=ex))))
    pool["code_comp"] = rows_comp

    # 8. code_lambda: funciones lambda ejecutables
    rows_lam: List[Row] = []
    for idx in range(70):
        k = 2 + idx
        m = 1 + (idx % 9)
        if idx % 3 == 0:
            expr = f"lambda x: x * {k} + {m}"
            tests = [((0,), m), ((4,), 4 * k + m), ((-1,), -k + m)]
        elif idx % 3 == 1:
            expr = f"lambda a, b: a * {k} - b"
            tests = [((2, 1), 2 * k - 1), ((0, 5), -5), ((3, 3), 3 * k - 3)]
        else:
            expr = f"lambda x: (x + {m}) * {k}"
            tests = [((0,), m * k), ((2,), (2 + m) * k), ((-m,), 0)]
        p = f"Write ONLY the Python lambda expression `{expr}`. No prose."
        rows_lam.append((p, expr, (lambda t, tc=tests: check_python_expr(t, "lambda", callable_tests=tc))))
    pool["code_lambda"] = rows_lam

    return pool


def make_mc_splits(n_train: int = 16, n_val: int = 8, n_hold: int = 25, seed: int = 42):
    rng = random.Random(seed)
    hist = all_prior_mc_prompts()
    pool = build_mc_pool()
    train, val, hold_rows = {}, {}, {}
    seen_global: Set[str] = set()
    for tid in ALL_MC_TASKS:
        clean = []
        for r in pool[tid]:
            if r[0] not in hist and r[0] not in seen_global:
                seen_global.add(r[0])
                clean.append(r)
        rng.shuffle(clean)
        need = n_train + n_val + n_hold
        assert len(clean) >= need, (tid, len(clean), need)
        train[tid] = clean[:n_train]
        val[tid] = clean[n_train:n_train + n_val]
        hold_rows[tid] = clean[n_train + n_val:n_train + n_val + n_hold]

    # Verificar disjunción total
    b_tr = {p for rows in train.values() for p, _, _ in rows}
    b_va = {p for rows in val.values() for p, _, _ in rows}
    b_ho = {p for rows in hold_rows.values() for p, _, _ in rows}
    assert b_tr.isdisjoint(hist) and b_va.isdisjoint(hist) and b_ho.isdisjoint(hist)
    assert b_tr.isdisjoint(b_va) and b_tr.isdisjoint(b_ho) and b_va.isdisjoint(b_ho)

    def to_items(d):
        return {
            tid: [Item(prompt=p, check=chk, meta={"target": tgt, "task": tid}) for p, tgt, chk in rows]
            for tid, rows in d.items()
        }

    manifest = {
        "n_train": n_train,
        "n_val": n_val,
        "n_hold": n_hold,
        "seed": seed,
        "tasks": ALL_MC_TASKS,
        "prompt_sha256": hashlib.sha256(json.dumps({
            "train": {k: [r[0] for r in v] for k, v in train.items()},
            "val": {k: [r[0] for r in v] for k, v in val.items()},
            "hold": {k: [r[0] for r in v] for k, v in hold_rows.items()},
        }, sort_keys=True).encode()).hexdigest(),
    }
    return train, val, to_items(hold_rows), manifest


def get_legacy_k12_math_code_holdout(n_hold: int = 50):
    """Devuelve las 2 tareas (`arithmetic` y `code_python`) del holdout exacto de `bridges-k12` (seed=0)
    más un validador AST estricto para `code_python` para medir limpieza de código."""
    _, _, hold_items, _ = make_bridges_k12_splits(20, 10, n_hold, seed=0)
    arith_items = hold_items["arithmetic"]
    code_items_lax = hold_items["code_python"]

    code_items_strict = []
    for it in code_items_lax:
        # Extraer nombre de función `fn<N>` del prompt
        m = re.search(r"`(fn\d+)\(x\)` with body `return x \+ (\d+)`", it.prompt)
        if m:
            fname = m.group(1)
            add_v = int(m.group(2))
            tc = [((0,), add_v), ((5,), 5 + add_v), ((-3,), -3 + add_v)]
            chk = (lambda t, fn=fname, tests=tc: check_python_function(t, fn, tests))
        else:
            chk = (lambda t, orig=it.check: orig(t) and not has_trailing_prose(t))
        code_items_strict.append(Item(prompt=it.prompt, check=chk, meta=dict(it.meta)))

    return {
        "legacy_arithmetic": arith_items,
        "legacy_code_lax": code_items_lax,
        "legacy_code_strict_ast": code_items_strict,
    }


# ─── Benchmark Extendido de Matemáticas y Programación (Nuevos Tests) ────────
EXT_MATH_TASKS = ["math_poly", "math_seq"]
EXT_CODE_TASKS = ["code_list_fn", "code_dict_str"]
EXT_MC_TASKS = EXT_MATH_TASKS + EXT_CODE_TASKS

EXT_MC_SPECS: List[MCTaskSpec] = [
    MCTaskSpec("math_poly", "math", "Polinomios cuadráticos y potencias (x^2 + c)", MATH_SYSTEM_PROMPT),
    MCTaskSpec("math_seq", "math", "Sucesiones aritméticas y sumas finitas", MATH_SYSTEM_PROMPT),
    MCTaskSpec("code_list_fn", "code", "Funciones de listas y agregación (AST + unit tests)", CODE_SYSTEM_PROMPT),
    MCTaskSpec("code_dict_str", "code", "Comprensiones de diccionarios y cadenas (AST + eval)", CODE_SYSTEM_PROMPT),
]


def build_ext_mc_pool() -> Dict[str, List[Row]]:
    """Construye el pool de 4 tareas adicionales de Matemáticas y Programación."""
    pool: Dict[str, List[Row]] = {}

    # 9. math_poly: cuadrados y polinomios cuadráticos
    rows_poly: List[Row] = []
    for idx in range(70):
        x = 3 + (idx % 12)
        c = 2 + idx
        if idx % 3 == 0:
            ans = x * x + c
            p = f"Compute {x}*{x} + {c}. Reply with ONLY the integer."
        elif idx % 3 == 1:
            ans = x * x + (idx % 9)
            p = f"If x = {x}, what is x*x + {idx % 9}? Reply with ONLY the integer."
        else:
            k = 2 + (idx % 3)
            ans = k * (x * x) - (idx % 5)
            p = f"Compute {k} * ({x} * {x}) - {idx % 5}. Reply with ONLY the integer."
        rows_poly.append((p, str(ans), (lambda t, e=ans: check_exact_int(t, e))))
    pool["math_poly"] = rows_poly

    # 10. math_seq: progresiones aritméticas y término siguiente
    rows_seq: List[Row] = []
    for idx in range(70):
        a0 = 5 + idx * 2
        d = 3 + (idx % 9)
        if idx % 2 == 0:
            t1, t2, t3, t4 = a0, a0 + d, a0 + 2 * d, a0 + 3 * d
            ans = a0 + 4 * d
            p = f"What is the next integer in the arithmetic sequence {t1}, {t2}, {t3}, {t4}, ? Reply with ONLY the integer."
        else:
            t1, t2, t3 = a0, a0 + d, a0 + 2 * d
            ans = t1 + t2 + t3
            p = f"Compute the sum {t1} + {t2} + {t3}. Reply with ONLY the integer."
        rows_seq.append((p, str(ans), (lambda t, e=ans: check_exact_int(t, e))))
    pool["math_seq"] = rows_seq

    # 11. code_list_fn: funciones sobre listas Python verificadas con AST + exec
    rows_lfn: List[Row] = []
    for idx in range(70):
        fname = f"list_op_{idx}"
        k = (idx % 8) + 2
        if idx % 3 == 0:
            body = f"return xs[0] * {k} + xs[-1]"
            tests = [(([2, 5, 3],), 2 * k + 3), (([1, 9],), 1 * k + 9), (([0, 4, 7],), 7)]
        elif idx % 3 == 1:
            body = f"return sum(xs) + {k}"
            tests = [(([1, 2, 3],), 6 + k), (([10],), 10 + k), (([],), k)]
        else:
            body = f"return len(xs) * {k}"
            tests = [(([1, 2, 3],), 3 * k), (([],), 0), (([9, 8],), 2 * k)]
        p = f"Write ONLY a Python function `{fname}(xs)` with body `{body}`. No prose."
        tgt = f"def {fname}(xs):\n    {body}"
        rows_lfn.append((p, tgt, (lambda t, fn=fname, tc=tests: check_python_function(t, fn, tc))))
    pool["code_list_fn"] = rows_lfn

    # 12. code_dict_str: comprensiones de diccionarios y funciones de strings
    rows_ds: List[Row] = []
    for idx in range(70):
        n = 3 + (idx % 5)
        k = 2 + idx
        if idx % 2 == 0:
            expr = f"{{x: x * {k} for x in range({n})}}"
            expected = {x: x * k for x in range(n)}
            p = f"Write ONLY the Python dict comprehension `{expr}`. No prose."
            rows_ds.append((p, expr, (lambda t, ex=expected: check_python_expr(t, "{", expected_val=ex))))
        else:
            expr = f"lambda s: s[:{n}] + '{k}'"
            tests = [(("abcdefg",), "abcdefg"[:n] + str(k)), (("xy",), "xy"[:n] + str(k))]
            p = f"Write ONLY the Python lambda expression `{expr}`. No prose."
            rows_ds.append((p, expr, (lambda t, tc=tests: check_python_expr(t, "lambda", callable_tests=tc))))
    pool["code_dict_str"] = rows_ds

    return pool


def make_ext_mc_splits(n_train: int = 12, n_val: int = 6, n_hold: int = 10, seed: int = 142):
    """Genera splits disjuntos para las 4 tareas extendidas (`math_poly`, `math_seq`, `code_list_fn`, `code_dict_str`)."""
    rng = random.Random(seed)
    hist = all_prior_mc_prompts()
    pool = build_ext_mc_pool()
    train, val, hold_rows = {}, {}, {}
    seen_global: Set[str] = set()
    for tid in EXT_MC_TASKS:
        clean = []
        for r in pool[tid]:
            if r[0] not in hist and r[0] not in seen_global:
                seen_global.add(r[0])
                clean.append(r)
        rng.shuffle(clean)
        need = n_train + n_val + n_hold
        assert len(clean) >= need, (tid, len(clean), need)
        train[tid] = clean[:n_train]
        val[tid] = clean[n_train:n_train + n_val]
        hold_rows[tid] = clean[n_train + n_val:n_train + n_val + n_hold]

    b_tr = {p for rows in train.values() for p, _, _ in rows}
    b_va = {p for rows in val.values() for p, _, _ in rows}
    b_ho = {p for rows in hold_rows.values() for p, _, _ in rows}
    assert b_tr.isdisjoint(hist) and b_va.isdisjoint(hist) and b_ho.isdisjoint(hist)
    assert b_tr.isdisjoint(b_va) and b_tr.isdisjoint(b_ho) and b_va.isdisjoint(b_ho)

    def to_items(d):
        return {
            tid: [Item(prompt=p, check=chk, meta={"target": tgt, "task": tid}) for p, tgt, chk in rows]
            for tid, rows in d.items()
        }

    manifest = {
        "n_train": n_train,
        "n_val": n_val,
        "n_hold": n_hold,
        "seed": seed,
        "tasks": EXT_MC_TASKS,
        "prompt_sha256": hashlib.sha256(json.dumps({
            "train": {k: [r[0] for r in v] for k, v in train.items()},
            "val": {k: [r[0] for r in v] for k, v in val.items()},
            "hold": {k: [r[0] for r in v] for k, v in hold_rows.items()},
        }, sort_keys=True).encode()).hexdigest(),
    }
    return train, val, to_items(hold_rows), manifest

