# Barrido cantidad/posición de capas — fusión bias Qwen2.5-0.5B

Pedido: repartir la edición de los 10 expertos originales entre **más bloques existentes**
del Transformer (24 capas) y medir en cuántas capas el conjunto funciona mejor.
**No se añaden bloques nuevos entrenados**; solo se editan sesgos `mlp.down_proj.bias`
en capas ya presentes (misma familia de edición directa que la fusión previa).

Rama: `exp/fusion-barrido-capas-0.5b` (desde `exp/fusion-expertos-10` @ `c4845d1`).

## Alcance (qué es / qué no es)

| Sí | No |
|----|----|
| Redistribuir / estimar direcciones en K∈{1,2,3,4,6,8,12,24} bloques existentes | Añadir capas/bloques nuevos o fine-tuning |
| Presupuesto L2 total comparable entre K (`scale = B/√K` por capa) | Confundir “más capas” con “más edición total” |
| Selección en tune + `fusion_val` (1ª mitad eval histórica) | Elegir ganador con el holdout final |
| Confirmación en **holdout NUEVO** ≥20/tarea | Reutilizar el confirm histórico (~10) como holdout independiente |
| Variante principal: dirección **por capa** (contraste tune) | Copiar un único residual a cualquier profundidad sin etiquetar |

La fusión previa (`multi_avg` en capas 6/8/10) sigue siendo baseline; su B L2
(`√Σ scale² ≈ 3.2807`) fija el presupuesto del barrido.

## Diseño del grid

- Layouts controlados: `early`, `mid`, `late`, `uniform` (32 configs = 8×4).
- Tras coarse (PPL×1.5 + tune): mejor layout por K → `scale_mult ∈ {0.5,1.0,1.5}` en val.
- Criterio de selección (macro fijo): `ok_ppl`, mayor mean, **menos K**, menos L2, layout.
- Ablación aparte (no entra en selección): `redistribute_prior_bias` (reubica el sesgo
  previo, **sin** estimar dirección por capa).

## Comandos

```bash
# barrido completo (reanudable; 1 modelo residente + SnapshotActivator)
 /workspace/ftenv/bin/python -m enjambre.fusion_sweep --model qwen2.5-0.5b --max-new 64 run

# manifiesto / curva / export / generate
 /workspace/ftenv/bin/python -m enjambre.fusion_sweep manifest
 /workspace/ftenv/bin/python -m enjambre.fusion_sweep curve
 /workspace/ftenv/bin/python -m enjambre.fusion_sweep export-info
 /workspace/ftenv/bin/python -m enjambre.fusion_sweep generate "…" --mode selected

# tests
 /workspace/ftenv/bin/python -m pytest enjambre/tests/test_fusion_sweep.py enjambre/tests/test_fusion_core.py -q
```

Artefactos: `enjambre/artifacts/fusion-sweep-0.5b/`
(`coarse_grid.json`, `val_grid.json`, `curve_by_K.json`, `final_holdout.json`,
`best_delta.pt`, `sweep_summary.json`, …). No se suben pesos HF completos.

## Resultado seleccionado (por VAL, congelado antes del final)

| Campo | Valor |
|-------|-------|
| K | **4** |
| layout | `uniform` |
| capas | `[0, 8, 15, 23]` |
| scale_mult | `1.0` (B≈3.2807 → scale/capa ≈1.640) |
| kind | `per_layer_direction_mean` |
| val_mean | **0.70** |
| PPL | 21.95 (base 15.91; guard 1.5×) |
| total_l2 | 3.2807 (igual presupuesto que fusión previa) |

Baselines en la misma `fusion_val`: base **0.54**, fusión previa **0.69**,
single mid-dir (K=1) **0.57**.

## Curva por K (mejor layout/escala por K en val; final = holdout nuevo)

El **ganador no se elige con final**. La columna final es solo reporte de las
configs ya preseleccionadas por val.

| K | layout | capas | scale_mult | val | final (n=20/tarea) |
|---|--------|-------|------------|-----|---------------------|
| 1 | mid | [12] | 0.5 | 0.65 | 0.755 |
| 2 | uniform | [0,23] | 1.0 | 0.58 | 0.685 |
| 3 | uniform | [0,12,23] | 0.5 | 0.63 | 0.755 |
| **4** | **uniform** | **[0,8,15,23]** | **1.0** | **0.70** | **0.71** |
| 6 | uniform | [0,5,9,14,18,23] | 1.0 | 0.69 | 0.74 |
| 8 | late | [16..23] | 1.5 | 0.63 | 0.765 |
| 12 | late | [12..23] | 0.5 | 0.65 | 0.77 |
| 24 | early(=all) | [0..23] | 0.5 | 0.69 | 0.75 |

**Lectura honesta:** en val, K=4 uniform empató/superó a la fusión previa (0.70 vs 0.69)
y bate a K=24. En el holdout nuevo, base ya está alto (**0.705**); selected queda en
**0.71**, prior fusion **0.735**, y varias K mayores marcan final ≥ selected.
Eso **no** autoriza coronar K=12/8: esas configs perdieron en val frente a K=4.
Más capas **no** ganan de forma monótona; K=24 no domina.

## Confirmación final (holdout nuevo, n=20/tarea, 10 tareas)

| condición | mean_acc |
|-----------|----------|
| base | 0.705 |
| prior_fusion (6/8/10) | 0.735 |
| selected K=4 uniform | 0.710 |

Selected vs base (pareado por tarea): **1 gain / 6 loss / 3 flat**.
Ganancia fuerte solo en `sentiment` (+0.55). Empeoran: `code_python` (−0.20),
`translate_en` (−0.10), y leves (−0.05) en json/arithmetic/summarize/uppercase.
`bullets` sigue en ~0 (cobertura sintética limitada).

Prior fusion vs base en el mismo holdout: 3 gain / 5 loss / 2 flat (mejor macro
que selected aquí, coherente con que selected ≈ prior en val pero no idéntico).

## Ablación redistribute (fuera de selección)

Reubicar el vector de sesgo previo (sin dirección por capa): K=3 mid val=0.59,
K=6 uniform val=0.62, K=3 uniform val=0.54 — peores que la variante por-capa
en las mismas K. Confirma que la hipótesis principal no es “pegar el bias viejo
en más sitios”.

## Rechazos PPL

3 configs mid (K=3/4/6) cortadas por `ppl > 1.5 × base` sin gastar eval completa.

## Recursos / reproducibilidad

- 1 modelo residente fp32, restore exacto vía `SnapshotActivator` (no 24 copias).
- RAM pico ~2.4 GiB RSS del proceso; wall ~3411 s (~57 min) en CPU `/workspace/ftenv`.
- Semilla 0; splits train/tune/val/final disjuntos; holdout prompts hasheados en
  `holdout_manifest.json`.
- Eval greedy `max_new=64`, mismos prompts por condición.

## Límites (no sobreclamar)

1. Cobertura acotada a **diez tareas sintéticas**; no inteligencia general.
2. Holdout nuevo usa templates distintos pero dominio similar → no fingir
   independencia total respecto al eval histórico.
3. Base en holdout nuevo (0.705) ≠ base en confirm histórico (0.57): **no**
   comparar números entre splits distintos.
4. Si más capas empeoran o no ayudan en val, es un resultado válido.
5. No hay router; el export es un delta estático único.

## Tests

`test_fusion_sweep.py`: índices/layouts, L2 comparable entre K, restore/zero,
holdout sin solape de prompts vs tune/eval, redistribute etiquetado.
Sin regresión en `test_fusion_core` / `test_adaptive_core`.
