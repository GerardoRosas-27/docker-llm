"""Módulo de evaluación y benchmarking de Artificial Analysis.

Conecta:
- Benchmark MATH-500 (https://artificialanalysis.ai/evaluations/math-500)
- Métricas de rendimiento TTFT, TPS y Latencia (https://artificialanalysis.ai/methodology/performance-benchmarking)
"""
from .math500 import ensure_math500_downloaded, extract_boxed, load_math500, math_equal
from .perf import aggregate_perf_stats, benchmark_generation_perf
from .bench import ArtificialAnalysisHarness

__all__ = [
    "ensure_math500_downloaded",
    "extract_boxed",
    "load_math500",
    "math_equal",
    "aggregate_perf_stats",
    "benchmark_generation_perf",
    "ArtificialAnalysisHarness",
]
