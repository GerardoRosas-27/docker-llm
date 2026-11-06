# Extensión K12→K20 (literal vs L2-normalizado) — Qwen2.5-0.5B

Pedido: fusionar **más capas** encima de la K=12 congelada (12..23) hasta **K=20**
(capas **4..23** de 24 existentes) y ver si mejora. **No** se insertan bloques nuevos;
solo sesgos `mlp.down_proj.bias` (steering horneado), nunca ΔW ni fine-tuning.

Rama: `exp/fusion-20capas` desde `exp/fusion-12capas-confirmacion` @ `c5e801f`.

## Diseños congelados (antes de evaluar)

| ID | Capas | Escala/capa | L2 total | Idea |
|----|-------|-------------|----------|------|
| **K12** (baseline) | 12..23 | 0.473529 | 1.640351 | exacto confirmación previa |
| **K20-A literal** | 4..23 | misma que K12 | **2.117685** (>K12) | conserva 12..23 **bit-iguales** + 8 nuevas (4..11) |
| **K20-B normalizado** | 4..23 | 0.366794 | **1.640351** (=K12) | mismas dirs; presupuesto L2 constante |

Hashes delta:
- k12: `b8cf061d4dbdf427b411cf54ef23cab9d694b013301347d2d28fa93efaa07c06`
- k20_A: `899581c1f208ceb352aa61770a59d2a7e29c64f75a00c0b834d887497338bb0c`
- k20_B: `ae0d9df28ef7a86a88b81f95cd17ffccdc797c30e54f103d0d29aa2f2055be44`

Aplicación siempre sobre **base fresca** + `SnapshotActivator` (restore exacto).
**No** sumar un delta encima de otro ya aplicado.

## Holdout nuevo

n=**50**/tarea (500), split `k20_holdout`, disjunto de tune/eval + holdout barrido n=20
+ confirm K12 n=50. `prompt_sha256=614418bf6f600bd5ad810781776456ba39572353f725f23b16bb2621e3e8c72a`.

**No** reutilizar el score 0.738 del confirm K12 previo: aquí se **reevalúa** K12
en este conjunto (obtuvo 0.706).

## Resultados (macro = global, n=500)

| condición | mean_acc | PPL | L2 |
|-----------|----------|-----|----|
| base | **0.638** | 15.911 | 0 |
| K12 | **0.706** | 17.185 | 1.6404 |
| K20-A literal | **0.63** | 17.266 | 2.1177 |
| K20-B normalizado | **0.678** | 16.75 | 1.6404 |

### Contrastes pareados (bootstrap CI95, McNemar)

| k20A_vs_k12 | -0.076 | [-0.106, -0.046] | 1e-06 | 11/49 |
| k20B_vs_k12 | -0.028 | [-0.052, -0.004] | 0.024307 | 10/24 |
| k20A_vs_base | -0.008 | [-0.048, 0.03] | 0.766604 | 49/53 |
| k20B_vs_base | +0.040 | [0.006, 0.072] | 0.026517 | 47/27 |
| k12_vs_base | +0.068 | [0.038, 0.098] | 1.7e-05 | 48/14 |
| k20A_vs_k20B | -0.048 | [-0.07, -0.026] | 8e-06 | 3/27 |

### Por tarea

| tarea | base | K12 | A | B |
|-------|------|-----|---|---|
| json | 0.86 | 0.96 | 0.52 | 0.86 |
| spanish | 1.00 | 1.00 | 1.00 | 1.00 |
| translate_en | 0.20 | 0.20 | 0.20 | 0.22 |
| arithmetic | 0.52 | 0.68 | 0.70 | 0.66 |
| summarize | 0.92 | 0.98 | 1.00 | 1.00 |
| uppercase | 0.56 | 0.52 | 0.28 | 0.36 |
| sentiment | 0.40 | 0.84 | 0.78 | 0.80 |
| entities | 0.96 | 0.94 | 0.88 | 0.92 |
| bullets | 0.00 | 0.00 | 0.00 | 0.00 |
| code_python | 0.96 | 0.94 | 0.94 | 0.96 |

## Lectura honesta / recomendación

1. **Más capas no mejora de forma monótona.** K12 sigue siendo el mejor en este holdout.
2. **K20-A** (más presupuesto L2) **empeora** vs K12 (Δ−0.076, CI totalmente negativa)
   y no supera a base (CI cruza 0). Daña especialmente `json` (0.96→0.52) y `uppercase`.
3. **K20-B** (mismo L2 que K12) también **pierde** frente a K12 (Δ−0.028, CI negativa,
   McNemar p≈0.024), aunque sigue por encima de base (Δ+0.040).
4. Separar **cobertura** (más capas) de **fuerza** (L2): ni ampliar cobertura con la
   misma intensidad por capa (A) ni repartir el mismo presupuesto (B) bate a K12 aquí.
5. Tradeoffs: A/B ayudan un poco en arithmetic/summarize vs base, pero no compensan
   regresiones. `bullets`=0 en todas (límite de tarea/validador).

**Recomendación práctica:** quedarse con **K12 late** para este ensamble de 10 tareas;
no adoptar K20 sin nueva evidencia. Sesgos ≠ pesos completos; no implica inteligencia general.

## Comandos

```bash
/workspace/ftenv/bin/python -m enjambre.fusion_20 --holdout-n 50 run
/workspace/ftenv/bin/python -m enjambre.fusion_20 generate "…" --mode k12
/workspace/ftenv/bin/python -m enjambre.fusion_20 report
/workspace/ftenv/bin/python -m pytest enjambre/tests/test_fusion_20.py -q
```

Artefactos: `enjambre/artifacts/fusion-20capas/` (`k20_A_literal.pt`, `k20_B_normalized.pt`,
`summary.json`, `results_*.json`).

## Recursos / límites

- ~862.6 s wall · RSS ~2343.7 MiB · CPU ftenv · 1 modelo residente.
- Diez tareas sintéticas; templates nuevos pero dominio similar; independencia acotada.
