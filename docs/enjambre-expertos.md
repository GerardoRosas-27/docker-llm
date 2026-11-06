# Enjambre de micro-expertos + memoria compartida

**Modelo base:** SmolLM2-135M-Instruct. **Sin LoRA / sin fine-tuning de matrices.**
Las especializaciones son ediciones de pesos (sesgo rank-1 y ΔW rank-1), reutilizando
el enfoque de [`docs/edicion-pesos.md`](edicion-pesos.md).

**Rama:** `exp/enjambre-escalado` (antes `exp/enjambre-expertos`). Código: `enjambre/`.

## Arquitectura

```
                 ┌──────────────────────────────────┐
  prompt ──────► │ Router (keyword / BoW / híbrido) │
                 └──────────────┬───────────────────┘
                                ▼
              ┌─────────────────────────────────────┐
              │  EditableModel (1× SmolLM2 en RAM)  │
              │  + delta sesgo del experto activo   │
              │  + rank-1 de hechos de memoria      │
              └─────────────────────────────────────┘
                     ▲                    ▲
         10 deltas   │                    │  SharedMemory
     (bias down_proj)│                    │  (ROME: Adam solo sobre v)
```

- **Un solo modelo en RAM** (~540 MB fp32). Cada experto es un **delta** (~2 KB:
  vector de sesgo en `mlp.down_proj`). Once copias serían ~6 GB; con deltas el
  coste extra es despreciable (`Swarm.status()` reporta `delta_bytes`).
- **10 micro-expertos:** dirección = media(h con system+few-shot) − media(h sin)
  + 0.5·(cumple−viola) si aplica; vector **normalizado** por capa; se elige
  (capa, magnitud) en set de *tuning* (temas disjuntos del *eval*) y se hornea
  como sesgo `c · û` en `layers[L].mlp.down_proj`.
- **Memoria compartida (ROME):**
  - `k*` = activación a la entrada de `down_proj` en el último token del sujeto
    (prompt alineado al chat template de lectura).
  - `v*` = init logit-lens del token distintivo del objeto + **≤40 pasos Adam
    solo sobre el vector `v`** (pesos `W` congelados; documentado).
  - `ΔW = (v* − W k̂*) ⊗ k̂*` con `‖Δv‖ ≤ 35`. **Un hecho por capa** (14/16/18/20…).
  - Métrica primaria: escribir/releer **un hecho a la vez**. El lote multi-hecho
    muestra interferencia (motivo por el que existe MEMIT).
- **Arnés:** CLI (`python -m enjambre …`) + FastAPI `:8090` + `tool_schema.json`
  + `sample_orchestrator_prompt.md` para el LLM del contenedor docker-llm.

### Expertos

| id | Tarea |
|---|---|
| `json` | JSON con claves exactas |
| `spanish` | Respuestas solo en español |
| `translate_en` | Traducción al inglés |
| `arithmetic` | Aritmética, solo el número |
| `summarize` | Resumen con tope de palabras |
| `uppercase` | TODA LA RESPUESTA EN MAYÚSCULAS |
| `sentiment` | positive / negative / neutral |
| `entities` | Extracción person/place/date → JSON |
| `bullets` | Exactamente 3 viñetas `- ` |
| `code_python` | Snippets Python sin prosa |

## Cómo lo usa un LLM del contenedor

```bash
docker compose --profile enjambre up -d --build enjambre
# API en http://enjambre:8090  (desde la red compose) o localhost:8090
```

| Método | Ruta | Uso |
|---|---|---|
| GET | `/experts` | listar expertos / deltas |
| GET | `/status` | modelo único + bytes de deltas + hechos |
| POST | `/route` | enrutar + aplicar delta + generar |
| POST | `/generate` | generar con `expert_id` concreto |
| POST | `/experts/create_or_modify` | recalcular dirección y hornear sesgo |
| POST | `/experts/apply` | aplicar delta (opcional `keep_memory`) |
| POST | `/memory/write` | escribir hecho (ROME) |
| POST | `/memory/read` | leer hecho |
| GET | `/memory/list` | listar hechos |
| GET | `/ppl` | perplejidad |
| POST | `/bench/quick` | bench corto |

Schema de function-calling: `enjambre/tool_schema.json`.
Prompt sugerido: `enjambre/sample_orchestrator_prompt.md`.

CLI:

```bash
pip install -r enjambre/requirements.txt
python -m enjambre.build   # o: python -m enjambre run_all
python -m enjambre route "What is 7+5? Reply with ONLY the number."
python -m enjambre memory-write BlergCorp robotics
python -m enjambre memory-read BlergCorp
python -m enjambre serve --port 8090
```

## Resultados (CPU 8 hilos, greedy, seed=0, ~7.7 min)

Fuente: `enjambre/artifacts/summary.json` / `bench_report.json`.

### Por tarea (held-out, n=12)

| Tarea | Base | Experto | Swarm+router | L | c | PPL experto |
|---|---:|---:|---:|---:|---:|---:|
| json | 0.000 | 0.000 | 0.000 | 12 | 12 | 22.97 |
| spanish | 1.000 | 1.000 | 1.000 | 12 | 3 | 23.41 |
| translate_en | 0.250 | 0.250 | 0.250 | 12 | 3 | 23.13 |
| arithmetic | 0.667 | 0.667 | 0.667 | 12 | 3 | 23.50 |
| summarize | 0.083 | 0.083 | 0.083 | 8 | 6 | 22.79 |
| uppercase | 0.000 | 0.000 | 0.000 | 10 | 3 | 23.41 |
| sentiment | 0.500 | **0.417** ↓ | 0.417 | 12 | 6 | 23.66 |
| entities | 1.000 | 1.000 | 1.000 | 12 | 6 | 24.06 |
| bullets | 0.000 | 0.000 | 0.000 | 12 | 3 | 23.52 |
| code_python | 1.000 | **0.917** ↓ | 0.917 | 8 | 12 | 22.87 |

**Router accuracy:** **1.000** (120/120 prompts al experto correcto).

**PPL base:** 23.37 (texto fijo en/es; distinto del bench `edicion-pesos`, no comparable 1:1).

En *tuning*, sí hubo movimiento (p.ej. sentiment 0.50→0.67, entities 0.67→0.83,
code 0.83→1.00) que **no generalizó** al held-out o incluso degradó: el steering
rank-1 no crea capacidades nuevas en 135M (json/uppercase/bullets siguen en 0,
igual que en `edicion-pesos`).

### Memoria

| Métrica | Valor |
|---|---:|
| Recall base (1.er token) | 0.25 |
| Recall **un hecho** (1.er token) | **0.50** |
| Recall **un hecho** (texto) | 0.25 |
| Recall **4 hechos a la vez** (1.er token) | 0.00 |
| Unrelated OK con 1 hecho (avg Paris) | 0.25 |
| Unrelated OK con 4 hechos | 0.00 |
| PPL base / multi-memoria | 23.37 / 27.52 |
| Capas usadas | 14, 16, 18, 20 |

Conclusión memoria: ROME con Adam-sobre-`v` **sí mueve el primer token** del
sujeto editado (p.ej. `BlergCorp` → `robot`), pero **apilar hechos interfiere**
y daña conocimiento no relacionado. Esperable en 135M sin MEMIT multi-capa.

### Timing

`total_seconds`: **459.9** (~7.7 min) — dentro del presupuesto.

## Caveats

- SmolLM2-135M no cumple formatos estrictos (viñetas exactas, MAYÚSCULAS totales,
  JSON perfecto) ni con steering; coherente con `edicion-pesos`.
- n=12 por tarea: ±1 acierto es ruido.
- El swarm iguala al experto porque el router acierta siempre; no hay ganancia
  extra de orquestación cuando el experto ≈ base.
- Memoria multi-hecho sin covarianza tipo MEMIT colapsa; la API permite escribir
  de uno en uno y olvidar.
- Deltas `.pt` no se versionan (`.gitignore`); regenerar con `python -m enjambre build`.

## Reproducir

```bash
export HF_HOME=~/.cache/huggingface
cd docker-llm
python -m enjambre.run_all
# artefactos en enjambre/artifacts/{summary,bench_report,experts_report}.json
```

## Escalado (135M → 360M → Qwen2.5-0.5B → Qwen2.5-1.5B)

Resumen canónico: `enjambre/artifacts/scaling_summary.json` (expertos/swarm) +
`enjambre/artifacts/memory_fix_summary.json` (memoria post-fix).
Held-out **n=20 por tarea**, greedy, seed 0, CPU. El router es **keyword/BoW**, no un
juicio de inteligencia general: acierto 1.0 solo dice que las plantillas del bench
tienen pistas léxicas suficientes.

### Fix MEMIT / ROME (estabilidad >135M)

Síntoma previo: en 360M/0.5B, MEMIT recall=0 y `unrelated_kept=0` (ΔW explotaba).
Causa: covarianza empírico con pocas muestras mete **autovalores negativos**;
`λ·C` los amplifica y `solve()` produce `‖ΔW‖ ≫ ‖W‖`.

Cambios en `enjambre/memit.py`:

- **MEMIT:** `C` → simetrización + proyección PSD (`eigh`, floor relativo a la mediana
  de eigs ≥0). Residuales clampeados vs `‖h‖`; `‖ΔW‖ ≤ dw_clamp·‖W‖`. Álgebra en float64.
- **ROME:** sigue con **C bruta + ridge** (`eps·mean(diag)·I`); la PSD le perjudicaba.
- Revertir ediciones resta `U@V` casteado al dtype de `W`: casi exacto en fp32;
  **no bit-exacto en fp16** (tests con tolerancia).
- `tune_no_gain` en scaling es solo etiqueta (el cand. no superó tune); **no** implica
  delta identidad — el delta elegido igual se aplica/guarda.

### Expertos / swarm (fuente: `*/scaling_report.json`)

| Modelo | dtype | base_ppl | router | rss MB | total_s | notas |
|---|---|---:|---:|---:|---:|---|
| smollm2-135m | fp32 | 23.371 | 1.0 | 1333 | ~580 | expertos ≈ base o peores |
| smollm2-360m | fp32 | 14.617 | 1.0 | 2411 | 1610 | igual patrón |
| qwen2.5-0.5b | fp32 | 15.911 | 1.0 | 3420 | 1448 | json 0.80→1.00; entities 0.35→0.50; summarize ↓ |
| qwen2.5-1.5b | **fp16** | 15.088 | 1.0 | 11226 | 12991 | json 0.75→0.90; bullets 0.65→0.70; resto plano |

Qwen2.5-1.5B por tarea (n=20):

| Tarea | Base | Experto | Swarm | tune_no_gain | s |
|---|---:|---:|---:|---|---:|
| json | 0.75 | **0.90** | 0.90 | no | 177 |
| spanish | 1.00 | 1.00 | 1.00 | sí | **10436** † |
| translate_en | 0.80 | 0.80 | 0.80 | no | 66 |
| arithmetic | 1.00 | 1.00 | 1.00 | sí | 31 |
| summarize | 1.00 | 1.00 | 1.00 | sí | 83 |
| uppercase | 1.00 | 1.00 | 1.00 | sí | 175 |
| sentiment | 1.00 | 1.00 | 1.00 | no | 25 |
| entities | 0.95 | 0.95 | 0.95 | sí | 93 |
| bullets | 0.65 | **0.70** | 0.70 | sí | 168 |
| code_python | 0.85 | 0.85 | 0.85 | sí | 140 |

† **Outlier de wall-clock** (posible cuelgue/reanudación del entorno). No extrapolar
coste por experto a partir de ese número; el resto está en 25–177 s.

### Memoria post-fix (canónica)

Fuentes por modelo (no mezclar con `scaling_report.memory` viejo de 360M/0.5B):

| Modelo | Fuente canónica | rome_avg | memit_4 | memit_8 | para (mejor) |
|---|---|---:|---:|---:|---:|
| 135m | `smollm2-135m/memory_fix.json` | 0.75 | 1.00 | 1.00 | 0.875 (memit_4) |
| 360m | `smollm2-360m/memory_fix.json` | 0.875 | 1.00 | 0.875 | 0.25 |
| 0.5b | `qwen2.5-0.5b/memory_fix.json` (rerun left-pad) | **1.00** | **1.00** | **1.00** | 0.0625 |
| 1.5b | `qwen2.5-1.5b/scaling_report.json` → `memory` (fp16) | 0.875 | 0.75 | 0.875 | 0.0625 |

En todos los post-fix medidos: `unrelated_kept ≈ 1.0` (ya no se borra París/Tokio).
**Generalización a paráfrasis sigue baja** (a menudo 0–0.25 fuera de 135M).

Histórico explícito (no canónico):

- `qwen2.5-0.5b/memory_fix_intermediate_psd_memit_rome_broken.json` — MEMIT OK, ROME roto.
- `qwen2.5-0.5b/memory_fix_rome_restored.txt` — log parcial 14:49 con ROME 1.0.
- `qwen2.5-0.5b/memory_fix_invalid_rightpad.json` — rerun mal configurado (padding right); no usar.
- `*/scaling_report.json` de 360M/0.5B (madrugada): expertos OK; **memoria MEMIT obsoleta** (pre-fix).

### Limitaciones del bench

- n=20/tarea: ruido ±1 acierto ≈ 0.05.
- Router keyword ≠ capacidad de routing aprendida.
- Memoria: recall de objeto en prompt canónico; paráfrasis held-out suele fallar.
- fp16 en ≥1.5B por RAM (~15 GB box); álgebra MEMIT en f32/f64.
- Deltas `.pt` no versionados (`.gitignore`); sí los JSON de métricas.

### Comandos reproducibles

```bash
cd docker-llm
pip install -r enjambre/requirements.txt pytest

# tests numéricos (sin descargar HF)
python -m pytest enjambre/tests/test_memit_stability.py -q

# scaling completo (largo; 1.5B ~horas en CPU, usa fp16)
python -m enjambre.scaling --model qwen2.5-1.5b --max-eval 20 --max-new 64

# solo memoria 0.5B (~7–10 min CPU); ver también scripts en el historial del agente
PYTHONPATH=. python -m enjambre.scaling --model qwen2.5-0.5b --max-eval 20 --max-new 64 --skip-memory
# (para memoria sola: llamar enjambre.membench.run_memory_bench tras cargar el modelo)
```

