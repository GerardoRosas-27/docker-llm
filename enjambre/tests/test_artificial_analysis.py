"""Tests unitarios para la integración de benchmarks de Artificial Analysis."""
import os
import pytest
from enjambre.artificial_analysis.math500 import (
    extract_answer_fallback,
    extract_boxed,
    load_math500,
    math_equal,
    normalize_math,
)
from enjambre.artificial_analysis.perf import aggregate_perf_stats


def test_extract_boxed_nested_brackets():
    text1 = r"The answer is \boxed{\frac{14}{3}}."
    assert extract_boxed(text1) == r"\frac{14}{3}"

    text2 = r"Step 1: x=2. Therefore $\boxed{\left( 3, \frac{\pi}{2} \right)}$."
    assert extract_boxed(text2) == r"\left( 3, \frac{\pi}{2} \right)"

    text3 = r"Nested \boxed{a + \sqrt{b + \{c\}} + d} done."
    assert extract_boxed(text3) == r"a + \sqrt{b + \{c\}} + d"

    text4 = "No boxed here at all."
    assert extract_boxed(text4) is None


def test_extract_answer_fallback():
    text1 = "After calculation, the final answer is: 42."
    assert extract_answer_fallback(text1) == "42"

    text2 = "Answer: 3/4"
    assert extract_answer_fallback(text2) == "3/4"


def test_normalize_math_and_equality():
    # Paréntesis y left/right
    assert math_equal(r"\left( 3, \frac{\pi}{2} \right)", r"(3, \frac{\pi}{2})")
    # Fracciones idénticas
    assert math_equal(r" \frac{14}{3} ", r"\frac{14}{3}")
    # Equivalencia fraccionaria simple
    assert math_equal(r"\frac{1}{2}", "0.5")
    assert math_equal("2/4", r"\frac{1}{2}")
    # Espacios y puntuación
    assert math_equal("$15$", "15.")
    # Negativos
    assert not math_equal("15", "16")
    assert not math_equal(None, "15")


def test_aggregate_perf_stats():
    runs = [
        {"ttft_ms": 100.0, "tps": 20.0, "total_ms": 1000.0},
        {"ttft_ms": 120.0, "tps": 25.0, "total_ms": 1200.0},
        {"ttft_ms": 140.0, "tps": 30.0, "total_ms": 1400.0},
    ]
    stats = aggregate_perf_stats(runs)
    assert stats["n_runs"] == 3
    assert stats["ttft_ms"]["p50"] == 120.0
    assert stats["output_tps"]["mean"] == 25.0
    assert stats["latency_ms"]["p50"] == 1200.0


def test_load_math500_sampling():
    items = load_math500(sample_size=14, seed=123)
    assert len(items) == 14
    for it in items:
        assert "problem" in it
        assert "answer" in it
        assert "subject" in it
        assert "level" in it
