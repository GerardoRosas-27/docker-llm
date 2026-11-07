"""Suite de pruebas y tareas complejas de Matemáticas y Programación (Algoritmos, Bucles, Recursión y Ecuaciones)."""
from __future__ import annotations
import ast
import hashlib
import json
import math
import random
import re
from typing import Any, Callable, Dict, List, Set, Tuple

from enjambre.experts import Item
from .tasks import (
    CODE_SYSTEM_PROMPT,
    MATH_SYSTEM_PROMPT,
    CheckFn,
    MCTaskSpec,
    Row,
    all_prior_mc_prompts,
    check_exact_int,
    check_python_function,
)

COMPLEX_MATH_TASKS = [
    "complex_math_gcd_lcm",
    "complex_math_fib_recur",
    "complex_math_modular_exp",
    "complex_math_solve_linear",
    "complex_math_combinatorics",
]

COMPLEX_CODE_TASKS = [
    "complex_code_loop_filter",
    "complex_code_string_algo",
    "complex_code_recursion",
    "complex_code_dict_freq",
    "complex_code_search_nested",
]

ALL_COMPLEX_TASKS = COMPLEX_MATH_TASKS + COMPLEX_CODE_TASKS

COMPLEX_SPECS: List[MCTaskSpec] = [
    MCTaskSpec("complex_math_gcd_lcm", "math", "MCD y MCM multi-paso", MATH_SYSTEM_PROMPT),
    MCTaskSpec("complex_math_fib_recur", "math", "Fibonacci y relaciones de recurrencia", MATH_SYSTEM_PROMPT),
    MCTaskSpec("complex_math_modular_exp", "math", "Exponenciación modular a^b mod m", MATH_SYSTEM_PROMPT),
    MCTaskSpec("complex_math_solve_linear", "math", "Resolución de ecuaciones lineales a*x + b = c", MATH_SYSTEM_PROMPT),
    MCTaskSpec("complex_math_combinatorics", "math", "Combinatoria C(n, k) y P(n, k)", MATH_SYSTEM_PROMPT),
    MCTaskSpec("complex_code_loop_filter", "code", "Bucles condicionales y filtros de listas (AST + unit tests)", CODE_SYSTEM_PROMPT),
    MCTaskSpec("complex_code_string_algo", "code", "Algoritmos de cadenas (palíndromos, conteos, orden) (AST + tests)", CODE_SYSTEM_PROMPT),
    MCTaskSpec("complex_code_recursion", "code", "Funciones recursivas o acumuladores (AST + tests)", CODE_SYSTEM_PROMPT),
    MCTaskSpec("complex_code_dict_freq", "code", "Mapas de frecuencias y agregación de diccionarios (AST + tests)", CODE_SYSTEM_PROMPT),
    MCTaskSpec("complex_code_search_nested", "code", "Navegación y búsqueda en matrices/estructuras anidadas (AST + tests)", CODE_SYSTEM_PROMPT),
]


def build_complex_pool() -> Dict[str, List[Row]]:
    pool: Dict[str, List[Row]] = {}

    # 1. complex_math_gcd_lcm: Máximo común divisor y mínimo común múltiplo
    rows_gcd: List[Row] = []
    for idx in range(70):
        k = 2 + (idx % 7)
        a = k * (4 + idx)
        b = k * (7 + idx * 2)
        if idx % 2 == 0:
            ans = math.gcd(a, b)
            p = f"Compute gcd({a}, {b}). Reply with ONLY the integer."
        else:
            ans = (a * b) // math.gcd(a, b)
            p = f"Compute lcm({a}, {b}). Reply with ONLY the integer."
        rows_gcd.append((p, str(ans), (lambda t, e=ans: check_exact_int(t, e))))
    pool["complex_math_gcd_lcm"] = rows_gcd

    # 2. complex_math_fib_recur: Fibonacci y recurrencias afines
    def fib(n: int) -> int:
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    rows_fib: List[Row] = []
    for idx in range(70):
        if idx % 2 == 0:
            n = 4 + (idx // 2)
            ans = fib(n)
            p = f"What is the {n}-th Fibonacci number F({n}) where F(0)=0 and F(1)=1? Reply with ONLY the integer."
        else:
            c = 1 + (idx % 5)
            steps = 3 + (idx // 10)
            mult = 2 + (idx % 3)
            curr = c
            for _ in range(steps):
                curr = mult * curr + 1
            ans = curr
            p = f"Given a(0) = {c} and a(n) = {mult}*a(n-1) + 1 for episode {idx}, what is a({steps})? Reply with ONLY the integer."
        rows_fib.append((p, str(ans), (lambda t, e=ans: check_exact_int(t, e))))
    pool["complex_math_fib_recur"] = rows_fib

    # 3. complex_math_modular_exp: Exponenciación modular
    rows_modexp: List[Row] = []
    for idx in range(70):
        base = 2 + (idx % 8)
        exp = 3 + (idx // 5)
        mod = 7 + idx
        ans = pow(base, exp, mod)
        p = f"Compute {base}^{exp} mod {mod}. Reply with ONLY the integer."
        rows_modexp.append((p, str(ans), (lambda t, e=ans: check_exact_int(t, e))))
    pool["complex_math_modular_exp"] = rows_modexp

    # 4. complex_math_solve_linear: Resolución de ecuaciones algebraicas
    rows_solve: List[Row] = []
    for idx in range(70):
        x_val = 3 + idx
        a = 2 + (idx % 7)
        b = 5 + (idx * 3)
        if idx % 2 == 0:
            c = a * x_val + b
            p = f"Solve for x: {a}*x + {b} = {c}. Reply with ONLY the integer."
        else:
            c = a * x_val - b
            p = f"Solve for x: {a}*x - {b} = {c}. Reply with ONLY the integer."
        rows_solve.append((p, str(x_val), (lambda t, e=x_val: check_exact_int(t, e))))
    pool["complex_math_solve_linear"] = rows_solve

    # 5. complex_math_combinatorics: Combinatoria C(n, k) y P(n, k)
    rows_comb: List[Row] = []
    for idx in range(70):
        n = 5 + (idx // 2)
        k = 2 + (idx % 3)
        if k >= n:
            k = 2
        if idx % 2 == 0:
            ans = math.comb(n, k)
            p = f"Compute the combination C({n}, {k}). Reply with ONLY the integer."
        else:
            ans = math.perm(n, k)
            p = f"Compute the permutation P({n}, {k}). Reply with ONLY the integer."
        rows_comb.append((p, str(ans), (lambda t, e=ans: check_exact_int(t, e))))
    pool["complex_math_combinatorics"] = rows_comb

    # 6. complex_code_loop_filter: Funciones con bucles condicionales sobre listas
    rows_loop: List[Row] = []
    for idx in range(65):
        fname = f"filter_op_{idx}"
        k = 2 + (idx % 6)
        if idx % 3 == 0:
            p = f"Write ONLY a Python function `{fname}(xs)` that returns the count of even numbers in `xs`. No prose."
            tgt = f"def {fname}(xs):\n    count = 0\n    for x in xs:\n        if x % 2 == 0:\n            count += 1\n    return count"
            tc = [(([1, 2, 3, 4, 5, 6],), 3), (([1, 3, 5],), 0), (([],), 0), (([0, 8, -2],), 3)]
        elif idx % 3 == 1:
            p = f"Write ONLY a Python function `{fname}(xs)` that returns the sum of numbers strictly greater than {k} in `xs`. No prose."
            tgt = f"def {fname}(xs):\n    total = 0\n    for x in xs:\n        if x > {k}:\n            total += x\n    return total"
            tc = [(([k - 1, k, k + 2, k + 5],), 2 * k + 7), (([k - 2, k],), 0), (([],), 0)]
        else:
            p = f"Write ONLY a Python function `{fname}(xs)` that returns a new list containing only positive numbers from `xs`. No prose."
            tgt = f"def {fname}(xs):\n    res = []\n    for x in xs:\n        if x > 0:\n            res.append(x)\n    return res"
            tc = [(([-2, 0, 5, -1, 3],), [5, 3]), (([-5, -1],), []), (([1, 2],), [1, 2])]
        rows_loop.append((p, tgt, (lambda t, fn=fname, tests=tc: check_python_function(t, fn, tests))))
    pool["complex_code_loop_filter"] = rows_loop

    # 7. complex_code_string_algo: Algoritmos de cadenas (palíndromos, reversas, conteos)
    rows_str: List[Row] = []
    for idx in range(65):
        fname = f"str_algo_{idx}"
        if idx % 3 == 0:
            p = f"Write ONLY a Python function `{fname}(s)` that checks if string `s` is a palindrome (case-insensitive). Return True or False. No prose."
            tgt = f"def {fname}(s):\n    cleaned = s.lower()\n    return cleaned == cleaned[::-1]"
            tc = [(("Racecar",), True), (("hello",), False), (("a",), True), (("",), True)]
        elif idx % 3 == 1:
            p = f"Write ONLY a Python function `{fname}(s)` that counts the number of vowels (a, e, i, o, u, case-insensitive) in `s`. No prose."
            tgt = f"def {fname}(s):\n    vowels = set('aeiouAEIOU')\n    return sum(1 for c in s if c in vowels)"
            tc = [(("hello world",), 3), (("sky",), 0), (("Antigravity",), 4), (("",), 0)]
        else:
            p = f"Write ONLY a Python function `{fname}(s)` that reverses the words in string `s` separated by a single space. No prose."
            tgt = f"def {fname}(s):\n    return ' '.join(s.split()[::-1])"
            tc = [(("hello world",), "world hello"), (("apple",), "apple"), (("one two three",), "three two one")]
        rows_str.append((p, tgt, (lambda t, fn=fname, tests=tc: check_python_function(t, fn, tests))))
    pool["complex_code_string_algo"] = rows_str

    # 8. complex_code_recursion: Recursión o acumuladores multi-paso
    rows_rec: List[Row] = []
    for idx in range(65):
        fname = f"recur_fn_{idx}"
        if idx % 2 == 0:
            p = f"Write ONLY a Python function `{fname}(n)` that computes the factorial of non-negative integer `n`. No prose."
            tgt = f"def {fname}(n):\n    if n <= 1:\n        return 1\n    return n * {fname}(n - 1)"
            tc = [((0,), 1), ((1,), 1), ((4,), 24), ((5,), 120)]
        else:
            p = f"Write ONLY a Python function `{fname}(base, exp)` that computes base raised to integer power exp using a loop or recursion. No prose."
            tgt = f"def {fname}(base, exp):\n    res = 1\n    for _ in range(exp):\n        res *= base\n    return res"
            tc = [((2, 3), 8), ((5, 0), 1), ((3, 4), 81), ((7, 2), 49)]
        rows_rec.append((p, tgt, (lambda t, fn=fname, tests=tc: check_python_function(t, fn, tests))))
    pool["complex_code_recursion"] = rows_rec

    # 9. complex_code_dict_freq: Mapeo de frecuencias y diccionarios
    rows_freq: List[Row] = []
    for idx in range(65):
        fname = f"dict_fn_{idx}"
        if idx % 2 == 0:
            p = f"Write ONLY a Python function `{fname}(xs)` that returns a dictionary mapping each element in `xs` to its frequency count. No prose."
            tgt = f"def {fname}(xs):\n    d = {{}}\n    for x in xs:\n        d[x] = d.get(x, 0) + 1\n    return d"
            tc = [(([1, 2, 2, 3, 1, 1],), {1: 3, 2: 2, 3: 1}), (([],), {}), ((['a', 'b', 'a'],), {'a': 2, 'b': 1})]
        else:
            p = f"Write ONLY a Python function `{fname}(d)` that returns a new dictionary with keys and values inverted. No prose."
            tgt = f"def {fname}(d):\n    return {{v: k for k, v in d.items()}}"
            tc = [(( {'a': 1, 'b': 2} ,), {1: 'a', 2: 'b'}), (({},), {})]
        rows_freq.append((p, tgt, (lambda t, fn=fname, tests=tc: check_python_function(t, fn, tests))))
    pool["complex_code_dict_freq"] = rows_freq

    # 10. complex_code_search_nested: Matrices y estructuras anidadas
    rows_nested: List[Row] = []
    for idx in range(65):
        fname = f"matrix_op_{idx}"
        if idx % 2 == 0:
            p = f"Write ONLY a Python function `{fname}(matrix)` that flattens a 2D list into a single 1D list. No prose."
            tgt = f"def {fname}(matrix):\n    res = []\n    for row in matrix:\n        for val in row:\n            res.append(val)\n    return res"
            tc = [(([[1, 2], [3, 4]],), [1, 2, 3, 4]), (([[], [5]],), [5]), (([],), [])]
        else:
            p = f"Write ONLY a Python function `{fname}(mat)` that returns the trace (sum of diagonal elements mat[i][i]) of a square matrix. No prose."
            tgt = f"def {fname}(mat):\n    return sum(mat[i][i] for i in range(len(mat)))"
            tc = [(([[1, 2], [3, 4]],), 5), (([[2, 0, 1], [0, 5, 0], [1, 2, 3]],), 10), (([],), 0)]
        rows_nested.append((p, tgt, (lambda t, fn=fname, tests=tc: check_python_function(t, fn, tests))))
    pool["complex_code_search_nested"] = rows_nested

    return pool


def make_complex_mc_splits(n_train: int = 14, n_val: int = 6, n_hold: int = 15, seed: int = 242):
    """Genera splits disjuntos para las 10 tareas de la suite Compleja."""
    rng = random.Random(seed)
    hist = all_prior_mc_prompts()
    pool = build_complex_pool()
    train, val, hold_rows = {}, {}, {}
    seen_global: Set[str] = set()

    for tid in ALL_COMPLEX_TASKS:
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
        "tasks": ALL_COMPLEX_TASKS,
        "prompt_sha256": hashlib.sha256(json.dumps({
            "train": {k: [r[0] for r in v] for k, v in train.items()},
            "val": {k: [r[0] for r in v] for k, v in val.items()},
            "hold": {k: [r[0] for r in v] for k, v in hold_rows.items()},
        }, sort_keys=True).encode()).hexdigest(),
    }
    return train, val, to_items(hold_rows), manifest
