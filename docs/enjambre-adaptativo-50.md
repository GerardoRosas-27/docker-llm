# Arnés adaptativo: ~50 especialistas sobre Qwen2.5-0.5B

**Rama:** `exp/enjambre-adaptativo-50`. **Base:** `Qwen/Qwen2.5-0.5B-Instruct` (UNA copia en RAM).
Cada experto es un **delta de sesgo** rank-1 en `mlp.down_proj`, no una copia del modelo.

## Qué es (y qué no)

| Afirmación correcta | No afirmar |
|---|---|
| 50 tareas/deltas con checks deterministas | 50 “capacidades generales” adquiridas |
| Router **heurístico** keywords / `task:<id>` | Razonamiento LLM del router |
| Memoria = **grafo asociativo SQLite** | Red neuronal de memoria |
| Auto-mejora acotada por N episodios | Mejora garantizada al ejecutar |
| Promote atómico solo si VAL mejora | Usar heldout para seleccionar |
| Snapshot exacto de capas tocadas | 50 modelos residentes |

MEMIT/ROME del paquete `enjambre/memit.py` es **opcional** y **no** es la verdad de este arnés;
aquí la memoria útil son experiencias verificadas + versiones de deltas con métricas/provenance.

## Arquitectura

```
prompt → router heurístico → task_id
              ↓
     ExpertBank (versión activa .pt)
              ↓
 SnapshotActivator (lock) → genera → restore exacto
              ↓
 MemoryNet SQLite (éxitos recuperables / fallos auditados)
              ↓
 AutoImprover: propose → VAL → promote|reject (heldout intacto)
```

Artefactos: `enjambre/artifacts/adaptive-0.5b/`
(`deltas/*.pt`, `bank_meta.json`, `memory_net.sqlite`, checkpoints de build/bench).

## Catálogo

50 tareas en familias `transform|format|extract|math|classify`.
Splits: train / tune / **val** (gating) / **heldout** (≥5, solo reporte).
Manifest: `GET /adaptive/catalog` o `python -m enjambre adaptive-catalog --model qwen2.5-0.5b`.

## CLI

```bash
cd docker-llm
export PYTHONPATH=.
PY=/workspace/ftenv/bin/python   # o tu venv

# tests unitarios (sin HF)
$PY -m pytest enjambre/tests/test_adaptive_core.py -q

# construir 50 deltas (reanudable; horas en CPU)
$PY -m enjambre adaptive-build --model qwen2.5-0.5b

# heldout base vs experto activo
$PY -m enjambre adaptive-bench --model qwen2.5-0.5b

# resolver (activa delta y restaura)
$PY -m enjambre adaptive-solve "task:upper_token Convert to UPPERCASE. Reply with only the word: alpha" --model qwen2.5-0.5b

# auto-mejora acotada (ej. 2 tareas × 1 episodio) — puede rechazar candidatos
$PY -m enjambre adaptive-auto-improve --model qwen2.5-0.5b --max-tasks 2 --episodes 1

$PY -m enjambre adaptive-improve-history --model qwen2.5-0.5b
$PY -m enjambre adaptive-status --model qwen2.5-0.5b
```

## API / tools

Montado en el mismo FastAPI (`:8090`) bajo `/adaptive/*`.
Schema: `enjambre/tool_schema_adaptive.json`.
Compose profile `enjambre` define `ENJAMBRE_ADAPTIVE_MODEL` / `ENJAMBRE_ADAPTIVE_ROOT`
(sin romper el servicio de 10 expertos previo).

Ejemplo tool calls (orquestador del contenedor):

1. `adaptive_catalog` → elige `task_id`
2. `adaptive_solve` con `task_id` o prompt `task:<id> ...`
3. `adaptive_feedback` si hay verificación externa
4. `adaptive_auto_improve` con `episodes`/`max_tasks` presupuesto
5. `adaptive_improve_history` → ver accepted/rejected reales

## Auto-mejora

1. Recupera experiencias **ok=1** de MemoryNet (calibración).
2. Calcula dirección + grid compacto capa/rel (tune).
3. Evalúa candidato vs incumbente/base en **VAL** (+ guardia PPL 1.5×).
4. Si gana por margen (`min_val_margin`, default 0.05) → `promote_atomic` (prev → retired, archivo conservado).
5. Si no → `reject` / quarantine / experimental; **rollback implícito** (activo no cambia).

No es una rutina de app: es un ciclo invocado con presupuesto explícito.

## Límites honestos

- n heldout ≈5/tarea: ruido alto.
- Muchos deltas pueden quedar `experimental` (sin ganancia VAL) → fallback base honesto.
- Steering rank-1 no crea habilidades nuevas complejas.
- Docker smoke solo si hay daemon; pytest/TestClient valida API sin Docker.
