# Experimento: fusión de 10 expertos (Qwen2.5-0.5B)

**Rama:** `exp/fusion-expertos-10`.  
**Base:** `Qwen/Qwen2.5-0.5B-Instruct` (una copia).  
**Deltas:** los 10 originales en `enjambre/artifacts/qwen2.5-0.5b/delta_*.pt`.

## Qué se fusiona (y qué no)

Los artefactos previos son **steering por SESGO** en `mlp.down_proj`
(`scale * vec` unitario), **no** ediciones low-rank de pesos (`ΔW = u v^T`).
Al aplicarlos, si `Linear.bias` era `None`, se **materializa** un bias nuevo
(+`hidden_size` parámetros por capa tocada). Eso no profundiza el Transformer.

**No** se mezclan las 50 tareas del arnés adaptativo: solo `all_expert_specs()`.

## Hipótesis

1. **Multi-capa (origen):** sumar/promediar sesgos en sus capas 6/8/10 puede
   interferir; el selector oráculo (tarea→su delta) es cota superior de utilidad
   de los deltas individuales, no “inteligencia de router”.
2. **Una capa común:** reubicar todos los vectores a una sola capa es una
   **hipótesis**, no equivalencia con aplicarlos en origen.
3. **Crecimiento:** fusionar deltas en la misma arquitectura **no** crea un
   modelo más grande ni garantiza habilidades nuevas. Una variante MLP paralelo
   con N ramas (mismo `W_down`, biases distintos, promedio de salidas) **sí**
   infla parámetros si se materializan N copias, pero por linealidad equivale a
   un único bias promedio — se demuestra en `parallel_equiv.py`.

## Protocolo de evaluación

| Split | Uso |
|---|---|
| `tune` (6/tarea) | grid compacto de hiperparámetros de fusión |
| `fusion_val` (1ª mitad de eval, ~10) | elige entre top-K del grid |
| `fusion_confirm` (2ª mitad, ~10) | **reporte final** (no usado para elegir) |
| `full` 20 | **exploratorio** reutilizado del scaling previo (no holdout independiente) |

Condiciones: `base`, `oracle`, `multi_avg_g1`, `multi_sum_g1`, `single_avg_ref`, `selected`.

## Comandos

```bash
cd docker-llm
export PYTHONPATH=.
PY=/workspace/ftenv/bin/python

$PY -m pytest enjambre/tests/test_fusion_core.py -q
$PY -m enjambre.fusion manifest
$PY -m enjambre.fusion --model qwen2.5-0.5b --max-new 48 run
$PY -m enjambre.fusion --model qwen2.5-0.5b generate "..." --mode selected
$PY -m enjambre.fusion export-info
```

Artefactos: `enjambre/artifacts/fusion-0.5b/`
(`manifest.json`, `fusion_summary.json`, `fused_selected.pt`, `fused_selected_biases.pt`).

## Límites

- Mejoras en confirm n≈10 son ruidosas; reportar gains/losses por tarea.
- Fusión no sustituye aprendizaje; interferencia entre tareas es esperable.
- No se suben pesos HF completos a git.

## Resultados (confirm n≈10/tarea, greedy, seed 0, max_new=48)

Fuente: `enjambre/artifacts/fusion-0.5b/fusion_summary.json`.

### Medias confirm

| Condición | mean_acc |
|---|---:|
| base | 0.57 |
| oracle (tarea→su delta) | 0.60 |
| multi_sum g=1 (ablation) | 0.56 |
| single_avg capa ref | 0.59 |
| **multi_avg g=1 (= selected)** | **0.70** |

Selected por tune→val: `multi_avg`, `global_scale=1.0`, capas de origen (no reubicación).
VAL mean selected 0.67; PPL 16.65 (guard OK vs base≈15.9).

### selected vs base (confirm)

gains **7** / losses **0** / flat **3**. Mejoras notables: sentiment +0.5, entities/code_python +0.2.
Ninguna tarea empeoró en este split (n pequeño: no extrapolar).

oracle vs base: +3 / −1 / 6 (media solo +0.03) — los deltas individuales aportan poco en esta mitad;
la fusión promedio ayudó más que el oráculo en **esta** muestra (no implica superioridad general).

### Exploratorio full-20

Ver `fusion_summary.json` → `full20_exploratory_means` (ítems reutilizados del scaling; no holdout nuevo).

### Parámetros / “crecimiento”

- Base Transformer: **494,032,768** params.
- Fusión bias: materializa hasta **3** biases × 896 = **+2,688** params si no existían (no crece profundidad/ancho).
- Prueba MLP paralelo 10 ramas (capa 8): equivalencia numérica `max_abs_err≈5e-7`; materializar 10 copias de down_proj añadiría ~**39.2M** params **sin** cambiar la función (≡ bias promedio).

### Export

- `fused_selected.pt` / `fused_selected_biases.pt` (sha256 en `fused_selected_export.json`).
- Reload smoke: genera con biases fusionados y tras limpiar (`ok: true`).
