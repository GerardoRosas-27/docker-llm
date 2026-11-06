# Plan reproducible — K12 + puentes residuales (t125)

## Objetivo
Conservar Qwen2.5-0.5B K12 congelado (hash `b8cf061d…`, late12..23, scale≈0.4735, L2≈1.6404)
y entrenar **solo** 3 puentes residuales forward-only `12→16`, `16→20`, `20→23` con gate
input-dependent (sigmoid, bias init −4) y proyección `down` zero-init (identidad exacta al inicio).

## Presupuesto CPU (caja compartida)
| Fase | Límite |
|------|--------|
| RSS pico | ≤ ~4–5 GiB (dejar margen; available ≥7 GiB al inicio) |
| Train wall | ≤ 28 min (hard deadline en harness) |
| Pilot | 20 steps → estima sec/step y reduce `max_steps` si hace falta |
| Rank | 16 por defecto (8/32 solo si piloto lo justifica) |
| Batch | 2 ejemplos TF; `use_cache=False` en train |
| Threads | `torch.set_num_threads(min(8, ncpu))` |

## Protocolo
1. `identity` — greedy K12 == K12+puentes(init); grad down ≠ 0; checksum base.
2. `pilot` — 20 steps; reinicia bank; ajusta `max_steps`.
3. `train` — AdamW solo params bank; selección por VAL (no holdout); checkpoint `bridges_best.pt`.
4. Holdout nuevo n=50/tarea (500), disjunto de históricos: base / K12 / K12+zero / K12+trained.
5. Bootstrap pareado + McNemar; no declarar inteligencia general.

## Artefactos
`enjambre/artifacts/bridges-k12/` — split_manifest, identity, pilot, train_log,
bridges_best.pt (solo puentes), results_*.json, summary.json.

## CLI
```
/workspace/ftenv/bin/python -m enjambre.bridges identity
/workspace/ftenv/bin/python -m enjambre.bridges pilot
/workspace/ftenv/bin/python -m enjambre.bridges run
```

## No tocar
`finetune/`, `main`, otras ramas exp, neuro-geometrica, PRs, cloud agents.
