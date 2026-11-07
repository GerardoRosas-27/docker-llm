# Conectores Neuro-Geométricos sobre K24_mc (24L+6B)

Informe experimental de integración de arquitecturas de conectores inter-capa extraídas de [`neuro-geometrica_multi-agente`](https://github.com/GerardoRosas-27/neuro-geometrica_multi-agente) (rama `main`) sobre la configuración compuesta ganadora **`K24_mc (24L+6B)`** (24 capas editadas `0..23` + 6 puentes conectores `0→4, 4→8, 8→12, 12→16, 16→20, 20→23`) para tareas de Matemáticas y Programación.

---

## 1. Principio y Reglas del Ensamblaje

1. **Ensamblaje por Capas Editadas Intacto**: El modelo base cuenta con sus **24 capas (`0..23`)** editadas mediante `k24_mc.pt` (proyecciones `down_proj` con vectores de contraste ortogonalizados mediante Gram-Schmidt contra prosa y viñetas). Estas 24 capas y los pesos troncales de Qwen permanecen **100% congelados (`requires_grad=False`)**, verificados con checksum SHA-256.
2. **Entrenamiento Exclusivo en Conectores Inter-Capa**: El entrenamiento supervisado actúa **única y exclusivamente sobre los 6 conectores inter-capa** que transfieren información entre bloques.
3. **Paridad de Paso 0 (`atlas-paso0`)**: Todos los conectores neuro-geométricos se inicializan de forma que en `t=0` reproducen **bit a bit con error absoluto 0.0** la inferencia exacta de `K24_mc (24L+6B)` (`paso0_parity.json`).

---

## 2. Las 4 Arquitecturas Portadas de `neuro-geometrica_multi-agente`

| Arquitectura | Origen en `neuro-geometrica` | Mecanismo Matemático en el Conector | Parámetros Extra |
|---|---|---|---|
| **1. CONSTELACIÓN v4** (`constelacion_v4`) | `orbita/src/constelacion4/mod.rs` | Bloques $2\times 2$ de rotación-escala de Koopman $K(\rho, \omega)$, levantamiento monótono $s = \sigma(R K \mu + b)$, acoplamiento de innovación con precisiones normalizadas $\hat{\Pi}$ y modulación en el subespacio de rango $r$. | +175,506 |
| **2. ATLAS-FW** (`atlas_fw`) | `orbita/src/constelacion4/atlas.rs` (`M1+M2+M6`) | Biblioteca M1 de 17 generadores no lineales Koopman/SINDy (polinomios grado $\le 3$, Fourier $\sin/\cos(2\cdot)$, $\tanh(2\cdot)$, radiales $r\cdot x$ con escala $\text{LIB\_S}=0.3$), proyección M2 y compuerta contextual precuencial CLS M6. | +137,166 |
| **3. BQ10 DA-STR** (`bq10_dastr`) | `atlas.rs` (`Mech::bq(10, false, true)`) + `docs/campo/` | Ganadora global BQ10: Núcleo ATLAS-FW + regularización Tikhonov M7 DA-STR con penalización por grado $da=10$ ($s_j = da^{-(deg_j-1)/2}$) + interacción bilineal de grado 2 atenuada + innovación causal diferencial $\text{RMSNorm}(h_{\text{tgt}} - h_{\text{src}})$ de `gemma_campo` + compuerta neuromodulada por canal. | +310,446 |
| **4. Soft-Bellman Simplicial 3D** (`soft_bellman_simplicial`) | `soft_bellman_simplicial/src/main.rs` | Complejo simplicial de $K=4$ vértices (tetraedro $\Delta^3$) con métrica Riemanniana SPD $3\times 3$ aprendida (`transform` con $0.5 + \text{softplus}(w)$ y $0.8\tanh(w)$), distancias geodésicas y coordenadas baricéntricas $\text{softmin}(q/\tau)$ agregando valores nodales. | +174,474 |

---

## 3. Banco de Pruebas Ampliado (235 Ítems de Test)

Para medir rigurosamente las mejoras y evitar sesgos, se evaluaron **3 suites de test independientes**:
- **Core Holdout (8 tareas $\times$ 15 = 120 ítems):** Aritmética 2-3 dígitos (`math_arith`), Multiplicación/División/Módulo (`math_mul_mod`), Expresiones multi-paso (`math_expr`), Lógica numérica (`math_logic`), Funciones Python AST (`code_fn`), Algoritmos condicionales (`code_algo`), Comprensiones de listas (`code_comp`), Lambdas ejecutables (`code_lambda`).
- **Extended Holdout (4 tareas nuevas $\times$ 10 = 40 ítems):** Polinomios y potencias cuadráticas (`math_poly`), Sucesiones aritméticas (`math_seq`), Funciones sobre listas (`code_list_fn`), Comprensiones de diccionarios y strings (`code_dict_str`).
- **Legacy Holdout (3 suites $\times$ 25 = 75 ítems):** `legacy_arithmetic`, `legacy_code_lax`, `legacy_code_strict_ast`.

---

## 4. Tabla Comparativa de Resultados

Evaluación sobre los 160 ítems holdout disjuntos (12 tareas de Matemáticas y Programación) más los 75 ítems legacy:

| Condición / Arquitectura | Core Acc (120) | Ext Acc (40) | **All-12 Acc (160)** | Math Acc (80) | Code Acc (80) | Legacy Holdout (75) | Extra Params |
|---|---|---|---|---|---|---|---|
| **`1_base_k24_mc_residual`** (Línea Base K24_mc) | 0.7667 | 0.6750 | **0.7438** | 0.4875 | 1.0000 | Arit: 0.88 / Code: 0.96 | 0 |
| **`2_k24_constelacion_v4`** (CONSTELACIÓN v4) | 0.7500 | 0.7750 | **0.7562** (+0.0125) | 0.5125 | 1.0000 | Arit: 0.96 / Code: 1.00 | +175,506 |
| **`3_k24_atlas_fw`** (ATLAS-FW ALL sin M7) | 0.7583 | 0.6750 | **0.7375** (-0.0063) | 0.4750 | 1.0000 | Arit: 0.88 / Code: 1.00 | +137,166 |
| **`4_k24_bq10_dastr`** (BQ10 DA-STR) | **0.7750** | 0.7000 | **0.7562** (+0.0125) | 0.5125 | 1.0000 | Arit: 0.92 / Code: 1.00 | +310,446 |
| **`5_k24_soft_bellman`** (Soft-Bellman Simplicial) | 0.7667 | 0.7250 | **0.7562** (+0.0125) | 0.5125 | 1.0000 | Arit: 0.92 / Code: 1.00 | +174,474 |

---

## 5. Desglose Detallado por Tarea (12 Tareas)

| Tarea | Dominio | Base K24_mc | Constelación v4 | ATLAS-FW | **BQ10 DA-STR** | **Soft-Bellman** |
|---|---|---|---|---|---|---|
| `math_arith` | Math | 0.8000 | 0.7333 | 0.7333 | **0.8000** | **0.8000** |
| `math_mul_mod` | Math | 0.6000 | 0.5333 | 0.6000 | **0.6000** | **0.6000** |
| `math_expr` | Math | 0.2000 | 0.2000 | 0.2000 | **0.2000** | **0.2000** |
| `math_logic` | Math | 0.5333 | 0.5333 | 0.5333 | **0.6000** (+0.067) | **0.5333** |
| `code_fn` | Code | 1.0000 | 1.0000 | 1.0000 | **1.0000** | **1.0000** |
| `code_algo` | Code | 1.0000 | 1.0000 | 1.0000 | **1.0000** | **1.0000** |
| `code_comp` | Code | 1.0000 | 1.0000 | 1.0000 | **1.0000** | **1.0000** |
| `code_lambda` | Code | 1.0000 | 1.0000 | 1.0000 | **1.0000** | **1.0000** |
| `math_poly` | Math | 0.1000 | **0.2000** (+0.100) | 0.1000 | **0.1000** | **0.1000** |
| `math_seq` | Math | 0.6000 | **0.9000** (+0.300) | 0.6000 | **0.7000** (+0.100) | **0.8000** (+0.200) |
| `code_list_fn` | Code | 1.0000 | 1.0000 | 1.0000 | **1.0000** | **1.0000** |
| `code_dict_str` | Code | 1.0000 | 1.0000 | 1.0000 | **1.0000** | **1.0000** |

---

## 6. Análisis Diagnóstico: ¿Por Qué Gana Cada Arquitectura?

1. **BQ10 DA-STR es la Ganadora en el Banco Principal (`core_acc = 0.7750`)**:
   - Tal como se documentó en `neuro-geometrica_multi-agente` (`docs/bio_quantum_v10/ANALISIS_POR_QUE_GANA.md`), la regularización M7 DA-STR ($da=10$) penaliza cuadrática y cúbicamente los términos no lineales de orden superior en la biblioteca de 17 funciones Koopman/SINDy.
   - Sin M7, ATLAS-FW (`ALL`) sobreajusta en horizontes multi-paso (cayendo a `0.7375`).
   - Con M7, BQ10 protege intacto el subespacio base de `K24_mc`, elimina sobreajuste y eleva `math_logic` a **0.6000**, manteniendo **100% en todas las 6 tareas de código** y subiendo el holdout legacy de código estricto a **1.0000**.
2. **Soft-Bellman Simplicial 3D Ofrece Cero Regresiones Netas**:
   - En la comparación pareada elemento a elemento (`all12_flat_ok`), **Soft-Bellman obtuvo 2 aciertos nuevos y CERO regresiones (0 fallos sobre respuestas previamente correctas)**. La compresión de curvatura baricéntrica en el 3-símplice ($\Delta^3$) estabiliza las transiciones entre capas intermedias sin distorsionar el espacio latente.
3. **CONSTELACIÓN v4 Sobresale en Sucesiones Numéricas y Polinomios**:
   - El bloque de rotación-escala $2\times 2$ de Koopman $K(\rho, \omega)$ modela de forma natural progresiones y recurrencias, disparando `math_seq` de **0.6000 a 0.9000** y `math_poly` de **0.1000 a 0.2000**.

---

## 7. Verificación de Código y Tests

- Tests de regresión y nuevas arquitecturas: `11/11 passed` (`enjambre/tests/test_neuro_connectors.py` y `enjambre/tests/test_math_code.py`).
- Integridad de pesos base: El SHA-256 de los pesos troncales de Qwen y las 24 capas editadas de `K24_mc` se mantuvo idéntico e inmutable durante todo el proceso.
