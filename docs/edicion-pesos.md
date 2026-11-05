# Experimento: edición directa de pesos (sin entrenamiento) para seguir instrucciones

**Modelo:** SmolLM2-135M-Instruct (base: SmolLM2-135M). **Sin gradientes, sin fine-tuning, sin LoRA**: todas las ediciones son fórmulas cerradas sobre los pesos. CPU (8 núcleos).

## Métodos
1. **Aritmética de tareas / extrapolación**: `W = W_base + α·(W_instruct − W_base)`, α ∈ {0.5, 1.25, 1.5, 2.0} (α=1 = baseline).
2. **Dirección de activación horneada en pesos (rank-1, sesgo)**: dirección `d_L` = media(residual del último token del prompt **con** system prompt "sigue instrucciones EXACTAMENTE") − media(**sin** él), calculada sólo con el set `dir` (96 prompts, temas distintos). Se añade `c·d_L` como sesgo de `mlp.down_proj` de la capa L → el residual queda desplazado permanentemente (equivale a "llevar el system prompt" dentro de los pesos).
3. **Dirección cumple−viola**: igual, pero contrastando prompts del set `dir` que el baseline cumplió (28) vs violó (68): (a) adición como sesgo; (b) ortogonalización (`W ← (I − r rᵀ)W` en `o_proj` y `down_proj` de todas las capas, estilo *abliteration*) de la dirección de "no cumplimiento".

## Benchmark
`edicion_pesos/bench.py`: 108 prompts de evaluación (12 tipos × 9), verificables automáticamente, con temas **disjuntos** del set usado para calcular direcciones. Decodificación greedy, max 100 tokens, semilla 0. Perplejidad sobre un texto fijo (en/es).

## Resultados (strict accuracy, n=108)
| Método | Acc | PPL |
|---|---|---|
| baseline_instruct | 0.250 | 12.65 |
| task_arith α=0.5 | 0.259 | 11.55 |
| task_arith α=1.25 | 0.241 | 16.23 |
| task_arith α=1.5 | 0.194 | 30.07 |
| task_arith α=2.0 | 0.111 | 710.5 |
| actadd_strict L10 c=1 | 0.333 | 13.53 |
| **actadd_strict L10 c=2** | **0.352** | 16.39 |
| actadd_strict L15 c=1 | 0.213 | 14.53 |
| actadd_strict L15 c=2 | 0.157 | 18.72 |
| actadd_comply L10 c=1 | 0.241 | 12.79 |
| actadd_comply L15 c=1 | 0.185 | 13.21 |
| orth_noncomply L15 | 0.287 | 14.89 |

Detalle por tipo en `edicion_pesos/metrics.json` (incluye todas las salidas). Mejora del mejor (baseline → L10 c=2): `empieza_con` 0.22→0.89, `en_ingles` 0.33→0.78, `sin_comas` 0.11→0.44; empeora `sin_palabra` 0.33→0.11. Tipos en 0 para todos: viñetas exactas, mayúsculas, minúsculas, termina_con, máx. palabras, una palabra (capacidad insuficiente del modelo de 135M, sobre todo en español).

## Conclusiones honestas
- **Sí es factible** mejorar el seguimiento de instrucciones editando pesos sin entrenar: +10 puntos (25%→35%) con una edición de 1 vector, a costa de +30% de perplejidad (12.6→16.4). c=1 da +8 pts con sólo +7% PPL.
- La extrapolación del "vector instruct" **no ayuda** (α>1 degrada y α=2 colapsa).
- n=9 por tipo: diferencias de ±1–2 aciertos son ruido. Sólo un modelo, un benchmark propio; no es IFEval oficial.
- Las ganancias se concentran en restricciones "fáciles"; no crea capacidades nuevas.

## Exportación GGUF
`export_best.py` guarda el modelo con `mlp_bias=True` (sesgos en cero salvo capa 10); `llama.cpp/convert_hf_to_gguf.py --outtype f16` lo convierte y `llama-completion` reproduce la salida editada ("Indeed, cats are indeed fascinating creatures."). El GGUF (270 MB) no se sube al repo.

## Reproducir
```
pip install torch transformers
cd edicion_pesos && python run.py      # ~7 min en CPU
python export_best.py ./hf_out && python llama.cpp/convert_hf_to_gguf.py ./hf_out --outtype f16
```
