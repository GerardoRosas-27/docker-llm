# K12 + puentes residuales (t125) — resultados

Rama: `exp/k12-puentes-residuales` (desde `exp/fusion-20capas` @ `0dc3afd`).
Base congelada Qwen2.5-0.5B + K12 exacto (`b8cf061d…`, late12..23, scale≈0.4735, L2≈1.6404).
**Solo se entrenan** 3 puentes residuales; matrices/bias base+K12 con `requires_grad=False`.

## Arquitectura

| Campo | Valor |
|-------|-------|
| Pares | `12→16`, `16→20`, `20→23` (forward-only, mismo token) |
| Módulo | `up: D→r`, GELU, `down: r→D` (zero-init) + `gate=σ(Linear(h_tgt))` bias init −4 |
| Rank | 16 |
| Params nuevos | **88 707** |
| Identidad init | `down.weight=0` ⇒ contribución nula; greedy K12 ≡ K12+puentes(zero) |
| Grad init | no nulo en `down.*` (up/gate arrancan tras primer update de down) |
| Optimizer | AdamW lr=1e−3 solo params del bank; clip grad 1.0 |
| Loss | CE teacher-forced **solo tokens de respuesta** |

Hooks: buffer de estados limpiado al inicio de cada forward (pre-hook capa 0);
no se mezclan tokens futuros; `use_cache=False` en train; restauración de hooks sin acumulación.

## Splits (disjuntos de históricos)

| Split | n/tarea | total |
|-------|---------|-------|
| train | 20 | 200 |
| val | 10 | 100 |
| holdout | 50 | **500** |

`prompt_sha256=97913f5005916549…` (manifest en `enjambre/artifacts/bridges-k12/split_manifest.json`).
Selección de checkpoint por **VAL**; holdout no usado para tuning ni targets de train.

## Presupuesto / recursos (CPU ftenv)

| Métrica | Valor |
|---------|-------|
| Pilot | 20 steps, 60.1 s, ~3.0 s/step, est. 400 steps ≈ 20 min |
| Train wall | **1019 s (~17 min)** / 400 steps; best step=280 val=0.75 |
| RSS pico | ~2.56 GiB (≪ 4–5 GiB) |
| Checkpoint | `bridges_best.pt` (~360 KB, solo puentes) sha256 `832f5c60…` |

## Holdout n=50/tarea — mean acc

| Condición | mean | PPL | secs |
|-----------|------|-----|------|
| base | **0.594** | 15.911 | 233 |
| K12 | **0.624** | 17.185 | 218 |
| K12 + puentes zero (init) | **0.624** | 17.185 | 230 |
| K12 + puentes entrenados | **0.740** | 16.247 | 372 |

Identidad holdout: zero ≡ K12 exacto (Δ=0, 0 discordantes).
Checksum parámetros base: OK antes/después.

### Por tarea (trained vs K12)

| Tarea | base | K12 | zero | trained |
|-------|------|-----|------|---------|
| json | 1.00 | 1.00 | 1.00 | 1.00 |
| spanish | 1.00 | 1.00 | 1.00 | 1.00 |
| translate_en | 0.00 | 0.00 | 0.00 | **1.00** |
| arithmetic | 0.92 | 0.92 | 0.92 | **0.44** ↓ |
| summarize | 0.00 | 0.08 | 0.08 | 0.00 |
| uppercase | 0.52 | 0.36 | 0.36 | **1.00** |
| sentiment | 0.50 | 0.88 | 0.88 | **0.96** |
| entities | 1.00 | 1.00 | 1.00 | 1.00 |
| bullets | 0.00 | 0.00 | 0.00 | 0.00 |
| code_python | 1.00 | 1.00 | 1.00 | 1.00 |

## Estadística pareada (holdout 500)

| Contraste | Δ | IC95 bootstrap | McNemar |
|-----------|---|----------------|---------|
| trained vs K12 | **+0.116** | [0.074, 0.160] | n01=90 n10=32 p≈0 |
| trained vs base | +0.146 | [0.102, 0.188] | — |
| zero vs K12 | 0.000 | [0, 0] | identidad |
| K12 vs base | +0.030 | [0.004, 0.058] | — |

Interpretación: mejora media significativa en este holdout sintético, **con regresión clara en arithmetic**
y sin ganar summarize/bullets. No es inteligencia general; plantillas sintéticas + validadores con
debilidades conocidas. No se añadió set OOD de formatos distintos (declarado).

## Tests

`pytest enjambre/tests/test_bridges.py` — identidad init, grad down, no-mezcla futura,
hooks restore, splits disjuntos, pares late. Smoke real ftenv: identity CLI + run completo.

## Limitaciones

1. 10 tareas sintéticas; independencia acotada respecto al dominio histórico.
2. Trade-off arithmetic (−0.48 abs) pese a ganancia media.
3. Sin barrido rank/gates constantes vs input-conditioned (rank=16 fijo tras piloto).
4. Sin set externo de plantillas/formatos distintos.
5. Val n=10/tarea ruidoso (dip a 0.30 en step 40); best por val=0.75 @280.
6. No replay/regularización de normas más allá de clip; PPL trained intermedia.

## Repro

```bash
cd /workspace/docker-llm
/workspace/ftenv/bin/python -m enjambre.bridges --out enjambre/artifacts/bridges-k12 run
```

Artefactos: `enjambre/artifacts/bridges-k12/{summary,identity,pilot,train_log,results_*,bridges_best.pt,split_manifest}.json|pt`
