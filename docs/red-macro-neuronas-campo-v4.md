# Red de Macro-Neuronas y Campo de Interconexión CONSTELACIÓN v4 (K32, K36 y K66)

Este documento presenta el diseño, formulación matemática y resultados empíricos del nuevo paradigma arquitectónico propuesto: **tratar cada capa del Transformer como una macro-neurona individual**, interconectadas mediante un **campo geométrico multiescala** basado en la arquitectura **CONSTELACIÓN v4** de [`GerardoRosas-27/neuro-geometrica_multi-agente`](https://github.com/GerardoRosas-27/neuro-geometrica_multi-agente).

---

## 1. Fundamento Conceptual: De Capas Secuenciales a Red de Macro-Neuronas

En los modelos de lenguaje Transformer convencionales, el procesamiento se concibe como una pila vertical estrictamente secuencial:
$$h_{l} = \text{Capa}_l(h_{l-1})$$

En nuestro paradigma:
1. **La Capa como Macro-Neurona ($N_i$)**:
   Cada bloque decodificador (Self-Attention + MLP + LayerNorm) actúa como una **macro-neurona** autónoma dotada de un subespacio receptivo especializado.
2. **El Campo Geométrico v4 (`ConstelacionV4Field`)**:
   En lugar de puentes dispersos aislados o una cadena rígida, el espacio inter-neuronal es gobernado por un **campo dinámico continuo** que modula, rota e inyecta innovación entre macro-neuronas a múltiples escalas temporales y espaciales.
3. **Escalamiento Modular**:
   Hemos evaluado empíricamente el comportamiento del sistema al escalar el número de macro-neuronas y puentes del campo:
   - **K32_macro**: 32 macro-neuronas (24 base + 8 añadidas), Campo de 12 puentes (705K params).
   - **K36_macro**: 36 macro-neuronas (24 base + 12 añadidas en 2 clusters de matemáticas), Hiper-Campo de 18 puentes (1.05M params).
   - **K66_macro**: 66 macro-neuronas (24 base + 42 añadidas en 11 etapas de 6 macro-neuronas), Hiper-Campo de 28 puentes (1.64M params en `bfloat16`).

---

## 2. Topología de las Redes de Macro-Neuronas

### 2.1 Red K32_macro (32 Macro-Neuronas, 12 Puentes)
- $N_0..N_{11}$: Percepción Temprana y Sintaxis (12 Base).
- $N_{12}..N_{15}$: Especialización Matemática Simbólica (4 clonadas + $d_{\text{math}}$).
- $N_{16}..N_{21}$: Razonamiento Semántico (6 Base).
- $N_{22}..N_{25}$: Especialización Algorítmica Python (4 clonadas + $d_{\text{code}}$).
- $N_{26}..N_{31}$: Síntesis Superior y Decodificación de Salida (6 Base).

### 2.2 Red K36_macro (36 Macro-Neuronas, 18 Puentes)
- Descompone el dominio matemático en Álgebra Simbólica ($N_{10}..N_{13}$) y Aritmética/Recurrencia ($N_{20}..N_{23}$), interconectados por la autopista armónica $(10 \to 20)$.

### 2.3 Red K66_macro (66 Macro-Neuronas, 28 Puentes)
Estructurada en **11 etapas simétricas de 6 macro-neuronas cada una**:
- **Etapa 0 ($N_0..N_5$)**: Percepción Sintáctica Temprana (Base 0..5).
- **Etapa 1 ($N_6..N_{11}$)**: Álgebra Simbólica y Sistemas Lineales (clonadas 6..11, damping 0.35 + $d_{\text{math}}$).
- **Etapa 2 ($N_{12}..N_{17}$)**: Teoría de Números y Algoritmo de Euclides (clonadas 8..13, damping 0.35 + $d_{\text{math}}$).
- **Etapa 3 ($N_{18}..N_{23}$)**: Núcleo Semántico Inicial (Base 6..11).
- **Etapa 4 ($N_{24}..N_{29}$)**: Aritmética Modular y Exponenciación Rápida (clonadas 10..15, damping 0.35 + $d_{\text{math}}$).
- **Etapa 5 ($N_{30}..N_{35}$)**: Combinatoria, Probabilidad y Recurrencias (clonadas 12..17, damping 0.35 + $d_{\text{math}}$).
- **Etapa 6 ($N_{36}..N_{41}$)**: Núcleo Semántico Intermedio (Base 12..17).
- **Etapa 7 ($N_{42}..N_{47}$)**: Estructuras de Datos, Tablas Hash y Diccionarios (clonadas 14..19, damping 0.35 + $d_{\text{code}}$).
- **Etapa 8 ($N_{48}..N_{53}$)**: Búsqueda, Recursión y Grafos (clonadas 16..21, damping 0.35 + $d_{\text{code}}$).
- **Etapa 9 ($N_{54}..N_{59}$)**: Algoritmia Python, Filtrado y Bucles Complejos (clonadas 18..23, damping 0.35 + $d_{\text{code}}$).
- **Etapa 10 ($N_{60}..N_{65}$)**: Síntesis Superior y Decodificación de Salida (Base 18..23).

#### Topología de 28 Puentes del Hiper-Campo para K66
1. **Tronco Continuo (11 puentes)**: $(0,5), (5,11), (11,17), (17,23), (23,29), (29,35), (35,41), (41,47), (47,53), (53,59), (59,65)$.
2. **Autopistas Armónicas Trans-Campo (8 puentes)**: $(5,17), (11,23), (17,29), (23,35), (29,41), (35,47), (41,53), (47,59)$.
3. **Resonadores de Largo Alcance (6 puentes)**: $(11,29), (17,47), (29,59), (35,65), (47,65), (53,65)$.
4. **Canales Causales de Innovación Cruzada (3 puentes)**: $(5,29), (11,47), (23,59)$.

---

## 3. Dinámica del Campo de Interconexión CONSTELACIÓN v4

Cada enlace del campo está implementado mediante un conector `ConstelacionV4Bridge` derivado de `orbita/src/constelacion4/mod.rs`:

1. **Operador de Koopman $2 \times 2$ ($K(\rho, \omega)$)**:
   Aplica rotaciones y escalados en subespacios 2D emparejados de rango $r=16$:
   $$K(\rho_k, \omega_k) = \rho_k \begin{pmatrix} \cos\omega_k & -\sin\omega_k \\ \sin\omega_k & \cos\omega_k \end{pmatrix}, \quad \rho_k \in [0.5, 1.05]$$
2. **Levantamiento Monótono ($s$)**:
   $$s = \sigma(R \cdot K(z_{\text{src}}) + b)$$
3. **Innovación Latente Ponderada por Precisión ($\Pi$)**:
   $$\text{innov} = z_{\text{tgt}} - K(z_{\text{src}})$$
   $$\mu = \text{GELU}\left( K(z_{\text{src}}) + \eta_b s + \kappa (\hat{\Pi} \odot \text{innov}) \right)$$
4. **Compuerta Neuromodulada Vectorial**:
   $$g_{\text{total}} = \sigma\left(\text{gate}(h_{\text{tgt}}) + \text{neuro\_gate}(\mu)\right)$$
   $$h_{\text{tgt}} \leftarrow h_{\text{tgt}} + g_{\text{total}} \cdot W_{\text{down}}(\phi_{\text{total}}) + g_{\text{total}} \cdot W_{\text{neuro\_down}}(\mu)$$

---

## 4. Comparativa Global en Artificial Analysis (MATH-500 & Inferencia)

Evaluación oficial en la muestra balanceada de **MATH-500** (`HuggingFaceH4/MATH-500`, 28 problemas olímpicos):

| Arquitectura | Capas / Macro-Neuronas | Puentes del Campo | Precisión MATH-500 | TPS (tok/s) | TTFT P50 | Latencia P50 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Qwen-0.5B** | 24L Secuencial | 0 | 0.00% (0/28) | **7.88** | **684.9 ms** | 6,747 ms |
| **K24 CONSTELACIÓN v4** | 24L Secuencial | 6 | 0.00% (0/28) | 7.67 | 693.7 ms | 6,818 ms |
| **K32_macro + Campo v4** | **32 Macro-Neuronas** | **12** | **25.00% (7/28)** | 6.06 | 781.3 ms | 20,373 ms |
| **K36_macro + Campo v4** | **36 Macro-Neuronas** | **18** | **17.86% (5/28)** | 5.30 | 869.5 ms | **12,729 ms** |
| **K66_macro + Campo v4** | **66 Macro-Neuronas** | **28** | **14.29% (4/28)** | 4.47 | 1304.1 ms | 15,190 ms |

---

## 5. Comparativa en la Suite Compleja (150 Ítems de Holdout)

| Tarea Compleja | Base K24 | K32_macro (12P) | K36_macro (18P) | K66_macro (28P) | Hallazgo Clave |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`complex_math_gcd_lcm`** | 0.0% | 6.7% | **40.0%** | 0.0% | K36 alcanza el máximo en Euclides |
| **`complex_math_modular_exp`** | 0.0% | 0.0% | **13.3%** | 0.0% | K36 rompe el cero en modular |
| **`complex_math_solve_linear`** | 0.0% | 0.0% | 0.0% | **6.7%** | K66 resuelve ecuación lineal |
| **`complex_math_combinatorics`** | 6.7% | 0.0% | **13.3%** | 0.0% | K36 lidera en conteo |
| **`complex_code_dict_freq`** | 0.0% | 46.7% | **53.3%** | 0.0% | K36 récord histórico absoluto |
| **`complex_code_loop_filter`** | **100%** | 73.3% | **100%** | 20.0% | K36 mantiene perfección |
| **`complex_code_recursion`** | 60.0% | **100%** | 40.0% | 40.0% | K32 récord perfecto |
| **`complex_code_string_algo`** | 93.3% | **100%** | 26.7% | 0.0% | K32 récord perfecto |
| **`complex_code_search_nested`** | 53.3% | **100%** | 0.0% | 0.0% | K32 récord perfecto |
| **Precisión Matemáticas Global** | 2.67% | 2.67% | **13.33%** | 1.33% | **K36 quintuplica el rendimiento base** |
| **Precisión Código Global** | 61.33% | **84.00%** | 44.00% | 12.00% | **K32 lidera en algoritmia** |
| **Precisión Global (150 ítems)** | 32.00% | **43.33%** | 28.67% | 6.67% | **K32 y K36 frontera de Pareto** |

---

## 6. Curva de Escalamiento y Conclusiones Científicas

1. **La Frontera de Pareto del Enjambre (K32 y K36)**:
   - **K32** es la configuración óptima para **programación y razonamiento algorítmico** (84.0% en código, 100% en recursión/strings/búsqueda y 25.0% en MATH-500).
   - **K36** es la configuración óptima para **matemáticas multi-paso** (13.33% global, 40.0% en Euclides MCD/MCM, 13.33% en exponenciación modular y récord de 53.33% en diccionarios).
2. **Límite de Saturación por Profundidad (K66)**:
   - Al escalar a 66 capas manteniendo el 100% de los pesos de las capas congelados, el modelo sufre atenuación de señal sintáctica en tareas de código complejo con AST estricto.
   - Sin embargo, **K66 demostró retener 14.29% en MATH-500**, superando con contundencia el 0.00% del modelo base de 24 capas, demostrando que los puentes del campo v4 preservan los flujos latentes incluso a través de 66 capas.
3. **Eficiencia en Memoria (`bfloat16`)**:
   - Gracias a la calibración de amortiguamiento $\alpha = 0.35$ y el uso de precisión `bfloat16`, K66 operó consumiendo únicamente **2.89 GB de RAM** (frente a los >5.6 GB que requería en float32), permitiendo inferencia fluida a 4.47 tok/s.
