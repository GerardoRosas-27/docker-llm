# Enjambre de micro-expertos + memoria compartida

**Modelo base:** SmolLM2-135M-Instruct. **Sin LoRA / sin fine-tuning de matrices.**
Las especializaciones son ediciones de pesos (sesgo rank-1 y ΔW rank-1), reutilizando
el enfoque de [`docs/edicion-pesos.md`](edicion-pesos.md).

**Rama:** `exp/enjambre-expertos`. Código: `enjambre/`.

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
