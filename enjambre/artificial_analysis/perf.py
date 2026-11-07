"""Medición de rendimiento de inferencia siguiendo la metodología de Artificial Analysis.

Métricas oficiales de Artificial Analysis:
- Time to First Token (TTFT, ms): tiempo hasta emitir el primer token.
- Output Tokens per Second (TPS, tok/s): velocidad de generación durante la fase de decodificación.
- Latency (s): tiempo total por tarea.
- Distribución P50 y P95 de latencia.

Referencia: https://artificialanalysis.ai/methodology/performance-benchmarking
"""
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def benchmark_generation_perf(
    model: torch.nn.Module,
    tok: Any,
    prompt: str,
    max_new_tokens: int = 32,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Ejecuta una inferencia y mide TTFT, TPS y Latencia total con precisión de milisegundos."""
    if device is None:
        device = next(model.parameters()).device

    inputs = tok(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs.get("attention_mask")
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)

    # 1. Fase de Prefill: Medición de Time to First Token (TTFT)
    t0 = time.perf_counter()
    with torch.no_grad():
        out0 = model(input_ids, attention_mask=attn_mask, use_cache=True)
        next_token = torch.argmax(out0.logits[:, -1, :], dim=-1, keepdim=True)
        past_key_values = out0.past_key_values
    t1 = time.perf_counter()
    ttft_ms = (t1 - t0) * 1000.0

    # 2. Fase de Decode: Generación autoregresiva token por token
    generated = [next_token]
    cur_token = next_token
    eos_id = tok.eos_token_id

    tokens_generated = 1
    for _ in range(max_new_tokens - 1):
        if eos_id is not None and int(cur_token[0, 0]) == eos_id:
            break
        with torch.no_grad():
            out = model(cur_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            cur_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(cur_token)
            tokens_generated += 1
    t2 = time.perf_counter()

    decode_s = t2 - t1
    decode_tokens = tokens_generated - 1
    tps = decode_tokens / decode_s if decode_s > 0 and decode_tokens > 0 else 0.0
    total_ms = (t2 - t0) * 1000.0

    full_tokens = torch.cat(generated, dim=-1)
    text = tok.decode(full_tokens[0], skip_special_tokens=True)

    return {
        "ttft_ms": round(ttft_ms, 2),
        "tps": round(tps, 2),
        "total_ms": round(total_ms, 2),
        "total_secs": round(total_ms / 1000.0, 3),
        "tokens_generated": tokens_generated,
        "output_text": text,
    }


def aggregate_perf_stats(runs: List[Dict[str, float]]) -> Dict[str, Any]:
    """Calcula estadísticas agregadas: media, P50, P95 y desviación estándar."""
    if not runs:
        return {}
    ttfts = [r["ttft_ms"] for r in runs]
    tpss = [r["tps"] for r in runs if r["tps"] > 0]
    totals = [r["total_ms"] for r in runs]

    return {
        "n_runs": len(runs),
        "ttft_ms": {
            "mean": round(float(np.mean(ttfts)), 2),
            "p50": round(float(np.percentile(ttfts, 50)), 2),
            "p95": round(float(np.percentile(ttfts, 95)), 2),
            "min": round(float(np.min(ttfts)), 2),
            "max": round(float(np.max(ttfts)), 2),
        },
        "output_tps": {
            "mean": round(float(np.mean(tpss)), 2) if tpss else 0.0,
            "p50": round(float(np.percentile(tpss, 50)), 2) if tpss else 0.0,
            "p95": round(float(np.percentile(tpss, 95)), 2) if tpss else 0.0,
        },
        "latency_ms": {
            "mean": round(float(np.mean(totals)), 2),
            "p50": round(float(np.percentile(totals, 50)), 2),
            "p95": round(float(np.percentile(totals, 95)), 2),
        },
    }
