"""Cargador y evaluador para el benchmark MATH-500 de Artificial Analysis.

Referencia:
- Artificial Analysis MATH-500: https://artificialanalysis.ai/evaluations/math-500
- Dataset oficial: https://huggingface.co/datasets/HuggingFaceH4/MATH-500
"""
from __future__ import annotations
import json
import os
import random
import re
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "math500")
MATH500_URL = "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/raw/main/test.jsonl"


def ensure_math500_downloaded() -> str:
    """Descarga test.jsonl de MATH-500 si no existe localmente."""
    os.makedirs(DATA_DIR, exist_ok=True)
    local_path = os.path.join(DATA_DIR, "test.jsonl")
    if not os.path.isfile(local_path) or os.path.getsize(local_path) < 1000:
        req = urllib.request.Request(MATH500_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(local_path, "wb") as f:
            f.write(resp.read())
    return local_path


def load_math500(
    sample_size: Optional[int] = None,
    seed: int = 42,
    subjects: Optional[List[str]] = None,
    levels: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Carga problemas de MATH-500 con filtrado opcional por materia o nivel y muestreo balanceado."""
    path = ensure_math500_downloaded()
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    if subjects:
        items = [it for it in items if it.get("subject") in subjects]
    if levels:
        items = [it for it in items if int(it.get("level", 0)) in levels]

    if sample_size and sample_size < len(items):
        rng = random.Random(seed)
        # Muestreo estratificado balanceado por materia
        by_sub: Dict[str, List[Dict[str, Any]]] = {}
        for it in items:
            by_sub.setdefault(it.get("subject", "other"), []).append(it)
        sampled = []
        per_sub = max(1, sample_size // len(by_sub))
        for sub, sub_items in by_sub.items():
            rng.shuffle(sub_items)
            sampled.extend(sub_items[:per_sub])
        if len(sampled) < sample_size:
            rem = [it for it in items if it not in sampled]
            rng.shuffle(rem)
            sampled.extend(rem[: sample_size - len(sampled)])
        items = sampled[:sample_size]

    return items


def extract_boxed(text: str) -> Optional[str]:
    """Extrae el contenido del último \\boxed{...} respetando llaves anidadas."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def extract_answer_fallback(text: str) -> Optional[str]:
    """Búsqueda de respaldo si no hay \\boxed{...} (e.g. 'The answer is X', 'Answer: X')."""
    patterns = [
        r"(?:the\s+answer\s+is|final\s+answer\s+is|answer:)\s*[:\$]?\s*([^\n\$\.]+)",
        r"(?:es|igual\s+a)\s*[:\$]?\s*([0-9\/\-\+\.]+[a-zA-Z]?)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            if cand:
                return cand
    return None


def normalize_math(s: str) -> str:
    """Normaliza expresiones matemáticas en LaTeX para comparación robusta."""
    if not s:
        return ""
    s = str(s).strip()
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\$|\\\(|\\\)", "", s)
    s = re.sub(r"\\dfrac", r"\\frac", s)
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    # Normalizar paréntesis exteriores redundantes
    if s.startswith("(") and s.endswith(")") and s.count("(") == 1 and s.count(")") == 1:
        s = s[1:-1]
    return s


def math_equal(pred: Optional[str], target: str) -> bool:
    """Compara si la predicción matemática coincide con el objetivo ground truth."""
    if pred is None:
        return False
    np = normalize_math(pred)
    nt = normalize_math(target)
    if np == nt:
        return True

    # Comparación numérica directa si ambos son números / decimales
    try:
        fp = float(np)
        ft = float(nt)
        if abs(fp - ft) < 1e-4:
            return True
    except Exception:
        pass

    # Comparación de fracciones simples a/b vs c/d
    def parse_frac(x: str) -> Optional[float]:
        m = re.match(r"^\\?frac\{(-?\d+)\}\{(-?\d+)\}$", x)
        if m:
            num, den = float(m.group(1)), float(m.group(2))
            return num / den if den != 0 else None
        m2 = re.match(r"^(-?\d+)\/(-?\d+)$", x)
        if m2:
            num, den = float(m2.group(1)), float(m2.group(2))
            return num / den if den != 0 else None
        return None

    fp_frac = parse_frac(np)
    ft_frac = parse_frac(nt)
    if fp_frac is not None and ft_frac is not None:
        if abs(fp_frac - ft_frac) < 1e-4:
            return True
    if fp_frac is not None:
        try:
            if abs(fp_frac - float(nt)) < 1e-4:
                return True
        except Exception:
            pass
    if ft_frac is not None:
        try:
            if abs(float(np) - ft_frac) < 1e-4:
                return True
        except Exception:
            pass

    return False


def format_math500_prompt(problem: str) -> str:
    """Formatea la consulta para respuesta concisa directa en MATH-500 evitando verborrea discursiva."""
    return (
        f"Problem: {problem.strip()}\n\n"
        "Solve concisely. Show 1-2 lines of direct calculation and write your final answer in \\boxed{}."
    )
