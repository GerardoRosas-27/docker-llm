"""Fusión de los 10 micro-expertos originales (bias steering) sobre Qwen2.5-0.5B.

Importante: los deltas previos son vectores de SESGO en mlp.down_proj, NO ediciones
low-rank de pesos (ΔW=uv^T). Fusionarlos no aumenta el número de parámetros del
Transformer base; solo combina vectores de bias (o materializa bias si no existía).
"""
from .orchestrator import FusionHarness
__all__ = ["FusionHarness"]
