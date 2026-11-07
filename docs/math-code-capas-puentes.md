# Ensamblaje por Capas Editadas + Puentes Conectores (Matemáticas y Programación)

Partiendo de la base **K12 + Puentes Entrenados** (`bridges_best.pt`: capas `12..23` + 3 puentes `12→16`, `16→20`, `20→23`), se ensambló un modelo enfocado en **Matemáticas** y **Programación**:
1. **Edición de pesos por capas (`BiasEdit` en `mlp.down_proj.bias`)**: Extracción de direcciones contrastivas por capa (`0..23`) para cálculo matemático exacto y síntesis de código Python limpio, con ortogonalización Gram-Schmidt frente al subespacio de interferencia de viñetas/prosa (`bullets`, `spanish`, `summarize`).
2. **Expansión progresiva de capas editadas**: Se evaluó pasar de **12 capas** (`12..23`) a **16 capas** (`8..23`), **20 capas** (`4..23`) y **24 capas** (`0..23`, el modelo completo ensamblado por rol funcional de cada capa).
3. **Entrenamiento exclusivo de los conectores entre capas (`BridgeBank`)**: Todas las matrices y sesgos de las capas permanecen congelados (`requires_grad=False`, verificado por checksum SHA-256). Al agregar más capas editadas se agregan puentes residuales conectores (`8→12`, `4→8`, `0→4`) inicializados en identidad (`down.weight=0`) junto con *warm-start* de los 3 puentes entrenados base (`12→16`, `16→20`, `20→23`).

---

## Arquitectura por Número de Capas Integradas

| Configuración | Capas Editadas | Rango | Puentes Conectores | Params Puentes |
| :--- | :---: | :---: | :--- | :---: |
| `1_base_k12_bridges_trained` | 12 | `12..23` | `12→16, 16→20, 20→23` (base `bridges_best.pt`) | 88,707 |
| `2_k12_mc_layer_edit_only` | 12 | `12..23` | `12→16, 16→20, 20→23` (sin reentrenar puentes) | 88,707 |
| `3_k12_mc_bridges_3` | 12 | `12..23` | `12→16, 16→20, 20→23` | 88,707 |
| `4_k16_mc_bridges_4` | **16** (+4) | `8..23` | `8→12, 12→16, 16→20, 20→23` | 118,276 |
| `5_k20_mc_bridges_5` | **20** (+8) | `4..23` | `4→8, 8→12, 12→16, 16→20, 20→23` | 147,845 |
| `6_k24_mc_bridges_6` | **24** (+12) | `0..23` | `0→4, 4→8, 8→12, 12→16, 16→20, 20→23` | 177,414 |

---

## Resultados Globales (Holdout Disjunto Math & Code + Holdout Legacy `bridges-k12`)

| Configuración | Capas | Puentes | Math Acc | Code Acc (AST+Exec) | **Media Compuesta** | Legacy Arith | Legacy Code (AST) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1_base_k12_bridges_trained` | 12 (`12..23`) | 3 | 31.67% | 0.00% | **15.83%** | 40.0% | 0.0% |
| `2_k12_mc_layer_edit_only` | 12 (`12..23`) | 3 | 35.00% | 0.00% | **17.50%** | **80.0%** | 0.0% |
| `3_k12_mc_bridges_3` | 12 (`12..23`) | 3 | 31.67% | **100.0%** | **65.83%** | **92.0%** | **100.0%** |
| `4_k16_mc_bridges_4` | 16 (`8..23`) | 4 | 41.67% | **100.0%** | **70.83%** | **92.0%** | **100.0%** |
| `5_k20_mc_bridges_5` | 20 (`4..23`) | 5 | 38.33% | **100.0%** | **69.17%** | 88.0% | **100.0%** |
| **`6_k24_mc_bridges_6`** 🏆 | **24 (`0..23`)** | **6** | **53.33%** | **100.0%** | **76.67%** | 88.0% | 96.0% |

---

## Desglose por Sub-tarea (4 Matemáticas + 4 Programación Ejecutable)

| Tarea | Dominio | `1_base` (12L) | `2_edit_only` (12L) | `3_k12_mc` (12L+3B) | `4_k16_mc` (16L+4B) | `5_k20_mc` (20L+5B) | **`6_k24_mc` (24L+6B)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `math_arith` (+ / - 3 dígitos) | Math | 66.67% | 66.67% | 60.00% | 66.67% | 60.00% | **80.00%** |
| `math_mul_mod` (`*`, `//`, `%`) | Math | 46.67% | 53.33% | 53.33% | 53.33% | 46.67% | **60.00%** |
| `math_expr` (`(a+b)*c`, `c*x-b`) | Math | 0.00% | 6.67% | 13.33% | **20.00%** | 6.67% | **20.00%** |
| `math_logic` (`|a-b|`, `max`, `3a+2b`) | Math | 13.33% | 13.33% | 0.00% | 26.67% | 40.00% | **53.33%** |
| `code_fn` (`def` + `ast.parse` + `exec`) | Code | 0.00% | 0.00% | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| `code_algo` (condicionales + unit tests) | Code | 0.00% | 0.00% | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| `code_comp` (list comprehension + `eval`) | Code | 0.00% | 0.00% | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| `code_lambda` (`lambda` + `eval` tests) | Code | 0.00% | 0.00% | **100.0%** | **100.0%** | **100.0%** | **100.0%** |

---

## Estadística Pareada (Holdout $N=120$)

| Contraste | $\Delta$ Global | IC95 Bootstrap | $\Delta$ Math | $\Delta$ Code | McNemar $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `3_k12_mc` (12L+3B) vs Base | **+50.00%** | `[+39.17%, +60.83%]` | +0.00% | **+100.0%** | $p \approx 0.0$ (65 vs 5) |
| `4_k16_mc` (16L+4B) vs Base | **+55.00%** | `[+45.00%, +65.00%]` | **+10.00%** | **+100.0%** | $p \approx 0.0$ (69 vs 3) |
| `5_k20_mc` (20L+5B) vs Base | **+53.33%** | `[+43.33%, +62.50%]` | +6.66% | **+100.0%** | $p \approx 0.0$ (68 vs 4) |
| **`6_k24_mc` (24L+6B) vs Base** | **+60.83%** | `[+51.67%, +70.00%]` | **+21.66%** | **+100.0%** | $p \approx 0.0$ (74 vs 1) |
| **`6_k24_mc` (24L) vs `3_k12_mc` (12L)** | **+10.83%** | `[+4.17%, +18.33%]` | **+21.66%** | +0.00% | **$p = 0.0072$** (17 vs 4) |

---

## Reproducción

```bash
PYTHONPATH=. python3 -m enjambre.math_code --train-steps 28 --n-hold 15 --n-legacy-hold 25 run
PYTHONPATH=. python3 -m enjambre.math_code report
PYTHONPATH=. pytest enjambre/tests/test_math_code.py -v
```
