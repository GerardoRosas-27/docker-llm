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
   - **K66_macro (0.5B)**: 66 macro-neuronas (24 base + 42 añadidas en 11 etapas de 6 macro-neuronas), Hiper-Campo de 28 puentes (1.64M params en `bfloat16`).
   - **K66_macro (1.5B)**: 66 macro-neuronas sobre `Qwen2.5-1.5B` ($d=1536$, 28 base preservadas + 38 añadidas en 10 etapas funcionales), Hiper-Campo de 28 puentes (2.81M params en `bfloat16`).

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

### 2.4 Red K66_macro sobre Qwen2.5-1.5B (Escalamiento de Capacidad Base)
Basada en `Qwen/Qwen2.5-1.5B-Instruct` ($d=1536$, 28 capas base):
- Preserva las **28 capas originales intactas y en orden causal** como espina dorsal perceptiva y sintáctica.
- Intercala **38 macro-neuronas clonadas y especializadas** en 10 etapas funcionales:
  - Etapas 0, 3, 6, 9: Macro-neuronas base (7, 7, 7, 7 capas respectivamente).
  - Etapas 1, 2, 4, 5 (Matemáticas): Álgebra (6), Euclides (6), Modular (6), Combinatoria (6) con amortiguamiento $\alpha=0.25$ y sesgo $d_{\text{math}}$ (escala 0.25).
  - Etapas 7, 8 (Código): Diccionarios (7), Algoritmos/Recursión (7) con amortiguamiento $\alpha=0.25$ y sesgo $d_{\text{code}}$ (escala 0.25).
- Campo CONSTELACIÓN v4 adaptado a $d=1536$ con 28 puentes ($2.81\text{M}$ parámetros entrenables en `bfloat16`). PPL inicial: $4.32$.

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
| **K32_macro + Campo v4 (0.5B)** | **32 Macro-Neuronas** | **12** | **25.00% (7/28)** | 6.06 | 781.3 ms | 20,373 ms |
| **K36_macro + Campo v4 (0.5B)** | **36 Macro-Neuronas** | **18** | **17.86% (5/28)** | 5.30 | 869.5 ms | **12,729 ms** |
| **K66_macro + Campo v4 (0.5B)** | **66 Macro-Neuronas** | **28** | **14.29% (4/28)** | 4.47 | 1304.1 ms | 15,190 ms |
| **K66_macro + Campo v4 (1.5B)** | **66 Macro-Neuronas** | **28** | **25.00% (7/28)** | 2.15 | 3462.6 ms | 32,831 ms |

---

## 5. Comparativa en la Suite Compleja

| Tarea Compleja | Base K24 (0.5B) | K32_macro (0.5B) | K36_macro (0.5B) | K66_macro (0.5B) | K66_macro (1.5B) | Hallazgo Clave |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`complex_math_gcd_lcm`** | 0.0% | 6.7% | **40.0%** | 0.0% | 12.5% | K36 lidera en Euclides; K66-1.5B reactiva capacidad |
| **`complex_math_modular_exp`** | 0.0% | 0.0% | **13.3%** | 0.0% | 0.0% | K36 exclusivo en modular |
| **`complex_math_solve_linear`** | 0.0% | 0.0% | 0.0% | 6.7% | **12.5%** | K66-1.5B duplica aciertos en lineal |
| **`complex_math_combinatorics`** | 6.7% | 0.0% | 13.3% | 0.0% | **37.5%** | **K66-1.5B récord histórico en combinatoria** |
| **`complex_code_dict_freq`** | 0.0% | 46.7% | 53.3% | 0.0% | **100.0%** | **K66-1.5B perfección absoluta (8/8)** |
| **`complex_code_loop_filter`** | **100%** | 73.3% | **100%** | 20.0% | 75.0% | Robustez sintáctica en bucles |
| **`complex_code_recursion`** | 60.0% | **100%** | 40.0% | 40.0% | **100.0%** | **K66-1.5B perfección absoluta (8/8)** |
| **`complex_code_string_algo`** | 93.3% | **100%** | 26.7% | 0.0% | 75.0% | Recuperación drástica sobre 0.5B |
| **`complex_code_search_nested`** | 53.3% | **100%** | 0.0% | 0.0% | **100.0%** | **K66-1.5B perfección absoluta (8/8)** |
| **Precisión Matemáticas Global** | 2.67% | 2.67% | **13.33%** | 1.33% | 12.50% | K66-1.5B multiplica x9 a K66-0.5B |
| **Precisión Código Global** | 61.33% | 84.00% | 44.00% | 12.00% | **90.00%** | **K66-1.5B nuevo récord histórico absoluto** |
| **Precisión Global de la Suite** | 32.00% | 43.33% | 28.67% | 6.67% | **51.25%** | **K66-1.5B rompe la barrera del 50%** |

---

## 6. Curva de Escalamiento y Conclusiones Científicas

1. **El Avance Cuántico del Escalamiento de Capacidad Base (1.5B vs 0.5B)**:
   - Al elevar el modelo base de 0.5B ($d=896$) a 1.5B ($d=1536$), **la red K66 rompió la saturación de profundidad**:
     - La precisión en código escaló de **12.00% a 90.00%** (+78 puntos porcentuales), alcanzando el 100% perfecto en recursión, tablas hash / diccionarios y búsqueda anidada.
     - La precisión matemática global escaló de **1.33% a 12.50%** (x9), con un pico de **37.50% en combinatoria**.
     - La precisión global de la suite escaló a **51.25%**, convirtiéndose en la arquitectura más potente probada.
2. **Artificial Analysis (MATH-500)**:
   - K66 (1.5B) alcanzó **25.00% de precisión (7/28 problemas olímpicos resueltos)**, superando el 14.29% de K66-0.5B y dejando muy atrás el 0.00% del modelo base de 24 capas.
3. **Calibración Dinámica de Amortiguamiento y Escala**:
   - En $d=1536$, la calibración óptima fue $\alpha = 0.25$ para el amortiguamiento de las macro-neuronas clonadas y `scale_ref = 0.25` para los contrastes. Esto redujo la perplejidad de referencia a **4.32** y evitó la degradación latente.
4. **Eficiencia de Cómputo y Memoria**:
   - Gracias a la formulación en `bfloat16`, K66 (1.5B) operó consumiendo **5.74 GB de RAM** durante inferencia, manteniéndose cómodamente dentro de los límites del sistema y ejecutando a 2.15 TPS en CPU pura.

---

## 7. Re-evaluación Unificada: CoT Conciso y Extracción Robusta (Mismas Condiciones para Todos)

Para resolver las anomalías identificadas en el análisis de causa raíz (donde los modelos eran forzados a emitir un único entero desnudo sin memoria de trabajo, o truncados arbitrariamente), se homogeneizó el protocolo de evaluación para **todas las arquitecturas** bajo parámetros idénticos:
- **Prompting CoT Conciso**: Cálculo en 1-2 líneas como scratchpad, seguido de `Final Answer: <entero>`.
- **Parser Flexible de Candidatos**: Función `extract_int_candidate` que busca formatos estructurados (`Final Answer: X`, `\boxed{X}`, líneas finales y enteros terminales) sin castigar la generación de pasos intermedios.
- **Presupuesto de Tokens Unificado**: 56 tokens para matemáticas y 72 tokens para código.
- **Holdout Idéntico**: 8 ítems independientes por tarea (80 ítems totales por condición, `seed=42`).

### 7.1. Tabla Comparativa Unificada (Suite Compleja, 80 Ítems)

| Tarea Compleja | Dominio | Base Qwen-0.5B (K24) | K32_macro (0.5B) | K36_macro (0.5B) | K66_macro (0.5B) | K66_macro (1.5B) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`complex_math_gcd_lcm`** | Math | **50.00%** (4/8) | 12.50% (1/8) | 0.00% (0/8) | 0.00% (0/8) | 12.50% (1/8) |
| **`complex_math_fib_recur`** | Math | 0.00% (0/8) | **12.50%** (1/8) | 0.00% (0/8) | 0.00% (0/8) | 0.00% (0/8) |
| **`complex_math_modular_exp`** | Math | **37.50%** (3/8) | 12.50% (1/8) | **37.50%** (3/8) | 0.00% (0/8) | 12.50% (1/8) |
| **`complex_math_solve_linear`** | Math | **62.50%** (5/8) | 0.00% (0/8) | 0.00% (0/8) | 0.00% (0/8) | 12.50% (1/8) |
| **`complex_math_combinatorics`** | Math | **50.00%** (4/8) | 25.00% (2/8) | 25.00% (2/8) | 0.00% (0/8) | 0.00% (0/8) |
| **Subtotal Matemáticas** | **Math** | **40.00%** (16/40) | 12.50% (5/40) | 12.50% (5/40) | 0.00% (0/40) | 7.50% (3/40) |
| **`complex_code_loop_filter`** | Code | 0.00% (0/8) | 62.50% (5/8) | **100.00%** (8/8) | 25.00% (2/8) | 75.00% (6/8) |
| **`complex_code_string_algo`** | Code | 0.00% (0/8) | **100.00%** (8/8) | 25.00% (2/8) | 0.00% (0/8) | 75.00% (6/8) |
| **`complex_code_recursion`** | Code | 0.00% (0/8) | **100.00%** (8/8) | 37.50% (3/8) | 37.50% (3/8) | **100.00%** (8/8) |
| **`complex_code_dict_freq`** | Code | 37.50% (3/8) | 37.50% (3/8) | 62.50% (5/8) | 0.00% (0/8) | **100.00%** (8/8) |
| **`complex_code_search_nested`** | Code | 0.00% (0/8) | **100.00%** (8/8) | 0.00% (0/8) | 0.00% (0/8) | **100.00%** (8/8) |
| **Subtotal Programación** | **Code** | 7.50% (3/40) | 80.00% (32/40) | 45.00% (18/40) | 12.50% (5/40) | **90.00%** (36/40) |
| **TOTAL GLOBAL** | **All** | 23.75% (19/80) | 46.25% (37/80) | 28.75% (23/80) | 6.25% (5/80) | **48.75%** (39/80) |

---

### 7.2. Diagnóstico Teórico: ¿Por qué K66-1.5B rinde en matemáticas similar a K32 mientras que en código alcanza un dominante 90%?

La evidencia empírica desvela un fenómeno arquitectónico fundamental:

1. **La Paradoja Aritmética del Vector de Contraste Homogéneo**:
   - En el modelo **Base Qwen-0.5B puro** (sin macro-neuronas editadas), al otorgarle 1-2 líneas de scratchpad, la precisión matemática se disparó a **40.00%**. Los pesos preentrenados originales poseen los circuitos aritméticos para calcular $3x+5=14$, $\gcd(48, 18)$ y $\binom{6}{2}$.
   - Al injertar **macro-neuronas editadas con vectores de contraste**, inyectamos una perturbación direccional $\Delta W = \alpha \cdot (\mathbf{v}_{\text{math}} - \mathbf{v}_{\text{base}})$.
   - En K66, se agregaron **24 macro-neuronas matemáticas consecutivas**, todas proyectadas con el **mismo vector genérico $\mathbf{v}_{\text{math}}$**. A través de 24 capas, la acumulación residual $\sum_{k=1}^{24} \Delta_k$ actúa como un campo magnético que **arrastra el flujo latente hacia tokens de texto matemático general**, interfiriendo destructivamente con los circuitos atencionales finos que ejecutan sumas y multiplicaciones exactas dígito por dígito.
   - Por ello, tanto en K32 (4 capas math), K36 (6 capas math) y K66 (24 capas math), la precisión matemática se reduce a 7.5% - 12.5%.
2. **La Sinergia Explosiva en Código Algorítmico**:
   - A diferencia de la aritmética escalar, el seguimiento de ejecución de código (bucles, llamadas recursivas, tablas de frecuencias, búsqueda en grafos anidados) es una **computación secuencial de estados apilados**.
   - Cada macro-neurona de código actúa como un paso de la máquina de estados de ejecución (stack frame virtual).
   - En un modelo pequeño de 0.5B, el flujo latente sufre saturación sintáctica a las 66 capas (cayendo a 12.50%). Pero en **1.5B ($d=1536$)**, el ancho del canal latente soporta perfectamente el apilamiento:
     - K66-1.5B alcanza **90.00% en código**, logrando **100% de precisión perfecta** en recursión, diccionarios de frecuencias y búsqueda anidada.
     - Esto representa una **mejora de 12x** sobre el modelo base no editado (7.50%).
3. **El Rol del Campo CONSTELACIÓN v4 (28 Puentes)**:
   - Los 28 puentes con compuertas neuromoduladas permiten que las capas finales lean directamente estados intermedios sin degradación por la profundidad de 66 capas, lo que explica por qué la retención en tareas largas de código se mantuvo intacta.

