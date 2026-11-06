# Confirmación K=12 late — holdout independiente (Qwen2.5-0.5B)

Pedido: probar la configuración **exacta** K=12 del barrido (que marcó ~0.77 en el
holdout n=20 previo) **sin reajustarla**, con preguntas nuevas, contra base y fusión previa.

Rama: `exp/fusion-12capas-confirmacion` (desde `exp/fusion-barrido-capas-0.5b` @ `a3b7a16`).

## Configuración congelada (NO reoptimizada)

| Campo | Valor |
|-------|-------|
| K | 12 |
| layout | `late` |
| capas | `[12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]` |
| scale_mult | `0.5` |
| B / B_eff | 3.280703 / 1.640351 |
| scale/capa | 0.473529 |
| total_l2 | 1.640351 |
| kind | `per_layer_direction_mean` |
| delta_sha256 | `b8cf061d4dbdf427b411cf54ef23cab9d694b013301347d2d28fa93efaa07c06` |
| directions_sha256 | `f5cfee64548909b40a4bd5d6c9e1e27a753b8b056c94ffdb1c29b99095dabea2` |
| sweep_val_key | `K12_late_L12-13-14-15-16-17-18-19-20-21-22-23_s0.5` |

`best_delta.pt` del barrido era **K=4**; aquí se exporta `k12_frozen_delta.pt`
reconstruido de forma determinista (mismas direcciones + params congelados).
No se usaron respuestas nuevas para ajustar nada.

## Protocolo

1. Congelar K12 + prior (capas 6/8/10) **antes** de generar el holdout.
2. Holdout nuevo: **n=50/tarea** (500 ítems), split `confirm12_holdout`,
   `prompt_sha256=d655f9a9714c11f4a0e74cc4f46140e5c0dc86ee90ac9ca629d6e99905bcfbab`.
3. Disjunto de tune/eval históricos y del holdout n=20 del barrido (assert en tests).
4. Mismos prompts/`max_new=64`/presupuesto entre base, prior y K12; 1 modelo + snapshot.
5. Sin router, sin fine-tuning, sin elegir otro K.

**No** comparar el 0.77 del holdout n=20 previo con la base de este holdout.

## Resultados (n=50/tarea, 500 prompts)

| condición | mean_acc (macro=global) | PPL |
|-----------|-------------------------|-----|
| base | **0.674** | 15.911 |
| prior_fusion (6/8/10) | **0.696** | 16.648 |
| **K12 late s=0.5** | **0.738** | 17.185 |

### Pareado global (por ítem)

| contraste | Δ acc | CI95 bootstrap (2000) | McNemar p (exact) | better/worse |
|-----------|-------|------------------------|-------------------|--------------|
| K12 − base | +0.064 | [0.04, 0.09] | 3e-06 | 40/8 |
| K12 − prior | +0.042 | [0.014, 0.068] | 0.003088 | 34/13 |
| prior − base | +0.022 | [-0.01, 0.052] | 0.207368 | 37/26 |

**Lectura:** K12 **supera a base** (Δ+0.064, CI95 [0.04, 0.09], McNemar p≈3e-6) y
**también a prior** (Δ+0.042, CI95 [0.014, 0.068], p≈0.003) en este holdout.
Prior vs base: Δ+0.022 con CI que cruza 0 → no significativo aquí.

### Por tarea (K12 vs base: gains/losses de ítems)

| tarea | base | prior | K12 | Δ(K12−base) | #gain/#loss |
|-------|------|-------|-----|-------------|-------------|
| json | 0.76 | 0.98 | 0.94 | +0.18 | 11/2 |
| spanish | 1.00 | 1.00 | 1.00 | +0.00 | 0/0 |
| translate_en | 0.22 | 0.26 | 0.20 | -0.02 | 0/1 |
| arithmetic | 0.90 | 0.80 | 0.94 | +0.04 | 2/0 |
| summarize | 0.94 | 0.84 | 1.00 | +0.06 | 3/0 |
| uppercase | 0.42 | 0.38 | 0.42 | +0.00 | 3/3 |
| sentiment | 0.50 | 0.70 | 0.90 | +0.40 | 21/1 |
| entities | 1.00 | 1.00 | 1.00 | +0.00 | 0/0 |
| bullets | 0.00 | 0.00 | 0.00 | +0.00 | 0/0 |
| code_python | 1.00 | 1.00 | 0.98 | -0.02 | 0/1 |

Ganancia fuerte en `sentiment` (+0.40) y `json` (+0.18). Regresiones leves:
`translate_en` (−0.02), `code_python` (−0.02). `bullets` permanece en 0 en las tres
condiciones (límite del validador/tarea sintética).

## Recursos

- Wall ≈ 699.3 s; RSS ≈ 2338.1 MiB; CPU `/workspace/ftenv`.
- Artefactos: `enjambre/artifacts/fusion-confirm12/` (`results_*.json` con output/ok por ejemplo,
  `confirm_summary.json`, `k12_frozen_delta.pt`).

## Comandos

```bash
/workspace/ftenv/bin/python -m enjambre.fusion_confirm12 --holdout-n 50 run
/workspace/ftenv/bin/python -m enjambre.fusion_confirm12 generate "…" --mode k12
/workspace/ftenv/bin/python -m enjambre.fusion_confirm12 export-info
/workspace/ftenv/bin/python -m pytest enjambre/tests/test_fusion_confirm12.py -q
```

## Límites

1. Diez tareas sintéticas; **no** razonamiento general ni inteligencia universal.
2. Tradeoffs por tarea; macro ↑ no implica mejora en todas.
3. Templates nuevos pero dominio similar al histórico → independencia acotada.
4. El 0.77 previo (n=20) y estos números (n=50) no son el mismo split.

## Tests

`test_fusion_confirm12.py`: capas/escala congeladas, rebuild determinista + L2,
holdout sin solape, bootstrap/McNemar. Suite previa de fusión/sweep sin regresión.
