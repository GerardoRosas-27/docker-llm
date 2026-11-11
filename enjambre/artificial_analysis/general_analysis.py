"""Suite de Evaluación de Artificial Analysis para Conocimiento General y Análisis de IA.

Evalúa:
1. Precisión en 50 problemas multi-disciplinares rigurosos distribuidos en 5 áreas:
   - Análisis de Inteligencia Artificial y Modelos de Lenguaje
   - Ciencias Físicas, Químicas y Cosmología
   - Historia Universal y Geopolítica
   - Filosofía, Lógica Formal y Epistemología
   - Ciencias Biológicas, Genética y Medicina
2. Comparación directa del Modelo Base vs Ultra-Híbrido Ganador con Capas Editadas (K66_general).
3. Métricas oficiales de rendimiento de Artificial Analysis:
   - Time to First Token (TTFT, ms)
   - Output Speed (TPS, tokens/s)
   - Latencia P50 y P95 (ms)
"""
from __future__ import annotations
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GeneralQuestion:
    id: str
    domain: str
    category: str
    question: str
    gold_answer: str
    keywords: List[str]


def check_general_answer(text: str, gold: str, keywords: List[str]) -> bool:
    """Verifica si la respuesta contiene el concepto o término clave."""
    s = text.lower()
    # Buscar en 'Final Answer: ...' si existe
    m = re.search(r"(?:final\s+answer|answer|respuesta\s+final)\s*[:=]?\s*([^\n\.]+)", s)
    cand = m.group(1).strip() if m else s

    gold_lower = gold.lower()
    if gold_lower in cand or gold_lower in s:
        return True

    # Verificar si al menos el 60% de las palabras clave principales coinciden
    matched = sum(1 for kw in keywords if kw.lower() in s)
    return matched >= max(1, int(len(keywords) * 0.6))


def build_general_knowledge_dataset() -> List[GeneralQuestion]:
    """Catálogo balanceado de 50 problemas de evaluación en Conocimiento General y Análisis de IA."""
    items: List[GeneralQuestion] = []

    # 1. AI Analysis & Neural Architectures (10 problemas)
    ai_questions = [
        ("ai_01", "ai_analysis", "AI Architectures",
         "In standard multi-head self-attention, what is the mathematical formula for scaled dot-product attention?",
         "softmax(Q*K^T / sqrt(d_k)) * V",
         ["softmax", "q", "k", "v", "sqrt"]),
        ("ai_02", "ai_analysis", "AI Architectures",
         "What is the time and space computational complexity of standard full self-attention with respect to sequence length N?",
         "O(N^2)",
         ["o(n^2)", "n^2", "quadratic", "cuadrática"]),
        ("ai_03", "ai_analysis", "AI Architectures",
         "What mechanism allows Grouped-Query Attention (GQA) to reduce memory bandwidth during LLM inference?",
         "Sharing key and value heads across multiple query heads",
         ["sharing", "key", "value", "heads", "kv", "query"]),
        ("ai_04", "ai_analysis", "AI Architectures",
         "What is RoPE in modern LLMs such as LLaMA and Qwen?",
         "Rotary Position Embedding",
         ["rotary", "position", "embedding", "rotación", "posicional"]),
        ("ai_05", "ai_analysis", "AI Architectures",
         "What hardware bottleneck does FlashAttention optimize by tiling SRAM memory blocks?",
         "High Bandwidth Memory (HBM) memory access latency and IO bandwidth",
         ["hbm", "sram", "memory", "bandwidth", "io", "tiling"]),
        ("ai_06", "ai_analysis", "AI Architectures",
         "In Mixture of Experts (MoE) models, what component routes tokens to specific expert networks?",
         "Gating network or router",
         ["gating", "router", "enrutador", "gate", "red de compuerta"]),
        ("ai_07", "ai_analysis", "AI Architectures",
         "What is the primary benefit of KV caching in autoregressive transformer decoding?",
         "Avoids recomputing past key and value vectors at each token step",
         ["recomputing", "past", "key", "value", "cache", "kv"]),
        ("ai_08", "ai_analysis", "AI Architectures",
         "What does the acronym LoRA stand for in parameter-efficient fine-tuning?",
         "Low-Rank Adaptation",
         ["low-rank", "adaptation", "baja", "rango", "adaptación"]),
        ("ai_09", "ai_analysis", "AI Architectures",
         "In transformer MLPs, what is the role of the down-projection layer?",
         "Projects intermediate expanded features back to hidden model dimension",
         ["projects", "down", "hidden", "dimension", "intermedio"]),
        ("ai_10", "ai_analysis", "AI Architectures",
         "What numerical precision format uses 1 sign bit, 8 exponent bits, and 7 mantissa bits?",
         "bfloat16 (Brain Floating Point 16)",
         ["bfloat16", "brain", "floating", "exponent", "mantissa"]),
    ]
    for qid, dom, cat, q, ans, kws in ai_questions:
        items.append(GeneralQuestion(qid, dom, cat, q, ans, kws))

    # 2. Physical Sciences & Cosmology (10 problemas)
    physics_questions = [
        ("phys_01", "physical_sciences", "Thermodynamics",
         "What fundamental law states that the total entropy of an isolated thermodynamic system can never decrease over time?",
         "Second Law of Thermodynamics",
         ["second law", "segunda ley", "termodinámica", "entropía", "entropy"]),
        ("phys_02", "physical_sciences", "Relativity",
         "What phenomenon describes the shift of light emitted from a strong gravitational field toward longer wavelengths?",
         "Gravitational Redshift",
         ["gravitational redshift", "corrimiento al rojo", "gravitacional", "redshift"]),
        ("phys_03", "physical_sciences", "Electromagnetism",
         "Which of Maxwell's equations states that there are no isolated magnetic monopoles in nature?",
         "Gauss's law for magnetism (div B = 0)",
         ["gauss", "magnetism", "div b", "monopole", "monopolo"]),
        ("phys_04", "physical_sciences", "Quantum Mechanics",
         "What principle asserts that one cannot simultaneously know with arbitrary precision both the position and momentum of a particle?",
         "Heisenberg Uncertainty Principle",
         ["heisenberg", "uncertainty", "incertidumbre", "principio"]),
        ("phys_05", "physical_sciences", "Quantum Mechanics",
         "What is the mathematical term for a quantum state existing simultaneously in multiple basis states until measurement?",
         "Quantum Superposition",
         ["superposition", "superposición", "cuántica", "quantum"]),
        ("phys_06", "physical_sciences", "Astrophysics",
         "What boundary around a black hole marks the threshold beyond which nothing, not even light, can escape?",
         "Event Horizon",
         ["event horizon", "horizonte de sucesos", "horizonte de eventos", "horizon"]),
        ("phys_07", "physical_sciences", "Classical Mechanics",
         "In the absence of external torque, what mechanical quantity remains strictly conserved in a rotating system?",
         "Angular Momentum",
         ["angular momentum", "momento angular", "conserved", "conserva"]),
        ("phys_08", "physical_sciences", "Optics",
         "What physical law describes the relationship between angles of incidence and refraction for light passing between media?",
         "Snell's Law",
         ["snell", "ley de snell", "refraction", "refracción"]),
        ("phys_09", "physical_sciences", "Cosmology",
         "What is the residual thermal radiation left over from the early stage of the universe after the Big Bang?",
         "Cosmic Microwave Background (CMB)",
         ["cosmic microwave background", "cmb", "fondo cósmico de microondas", "radiación de fondo"]),
        ("phys_10", "physical_sciences", "Nuclear Physics",
         "What nuclear reaction powers the energy output of the Sun in main-sequence stellar evolution?",
         "Nuclear fusion (proton-proton chain / hydrogen to helium)",
         ["fusion", "fusión", "nuclear", "proton", "hydrogen", "helium"]),
    ]
    for qid, dom, cat, q, ans, kws in physics_questions:
        items.append(GeneralQuestion(qid, dom, cat, q, ans, kws))

    # 3. World History & Geopolitics (10 problemas)
    history_questions = [
        ("hist_01", "world_history", "Geopolitics",
         "Which 1648 peace treaties established the principle of national state sovereignty in European international law?",
         "Peace of Westphalia",
         ["peace of westphalia", "paz de westfalia", "westfalia", "westphalia"]),
        ("hist_02", "world_history", "Ancient History",
         "Which conflict between Athens and Sparta from 431 to 404 BC reshaped the ancient Greek world?",
         "Peloponnesian War",
         ["peloponnesian", "peloponeso", "guerra del peloponeso"]),
        ("hist_03", "world_history", "Medieval History",
         "In which year did the Ottoman Empire under Sultan Mehmed II conquer Constantinople, ending the Byzantine Empire?",
         "1453",
         ["1453", "mehmed", "constantinopla", "constantinople"]),
        ("hist_04", "world_history", "Modern Institutions",
         "Which 1944 conference established the International Monetary Fund (IMF) and the World Bank?",
         "Bretton Woods Conference",
         ["bretton woods", "imf", "fmi", "banco mundial", "world bank"]),
        ("hist_05", "world_history", "Constitutional History",
         "What charter signed by King John of England in 1215 established that the sovereign was subject to the rule of law?",
         "Magna Carta",
         ["magna carta", "carta magna", "1215", "king john"]),
        ("hist_06", "world_history", "Industrial Revolution",
         "Which Scottish inventor developed the separate condenser in 1769, massively improving the efficiency of the steam engine?",
         "James Watt",
         ["james watt", "watt", "steam engine", "máquina de vapor"]),
        ("hist_07", "world_history", "Diplomatic History",
         "Which 1814-1815 diplomatic congress reorganized Europe following the defeat of Napoleon Bonaparte?",
         "Congress of Vienna",
         ["congress of vienna", "congreso de viena", "vienna", "viena"]),
        ("hist_08", "world_history", "World War II",
         "Which international military tribunal held between 1945 and 1946 prosecuted prominent leaders of Nazi Germany?",
         "Nuremberg Trials",
         ["nuremberg", "juicios de núremberg", "tribunal de nuremberg"]),
        ("hist_09", "world_history", "Colonial History",
         "Which 1494 treaty divided newly discovered non-European lands between the Portuguese and Spanish Empires?",
         "Treaty of Tordesillas",
         ["treaty of tordesillas", "tratado de tordesillas", "tordesillas"]),
        ("hist_10", "world_history", "Cold War",
         "What 1948 US economic initiative provided over $13 billion to rebuild Western European economies after World War II?",
         "Marshall Plan",
         ["marshall plan", "plan marshall", "marshall"]),
    ]
    for qid, dom, cat, q, ans, kws in history_questions:
        items.append(GeneralQuestion(qid, dom, cat, q, ans, kws))

    # 4. Philosophy, Formal Logic & Epistemology (10 problemas)
    phil_questions = [
        ("phil_01", "philosophy_logic", "Epistemology",
         "What classical tripartite definition defines knowledge as a cognitive attitude meeting three conditions?",
         "Justified True Belief (JTB)",
         ["justified true belief", "creencia verdadera justificada", "jtb", "justificada"]),
        ("phil_02", "philosophy_logic", "Epistemology",
         "What philosophical counterexamples demonstrated that Justified True Belief does not always constitute knowledge?",
         "Gettier problems (Edmund Gettier)",
         ["gettier", "problema de gettier", "contraejemplos"]),
        ("phil_03", "philosophy_logic", "Formal Logic",
         "What logical rule of inference concludes Q from the premises 'If P then Q' and 'P'?",
         "Modus Ponens",
         ["modus ponens", "ponens", "afirmación del antecedente"]),
        ("phil_04", "philosophy_logic", "Formal Logic",
         "What logical rule of inference concludes 'Not P' from the premises 'If P then Q' and 'Not Q'?",
         "Modus Tollens",
         ["modus tollens", "tollens", "negación del consecuente"]),
        ("phil_05", "philosophy_logic", "Ethics",
         "What ethical theory founded by Jeremy Bentham and John Stuart Mill argues actions are right if they maximize overall happiness?",
         "Utilitarianism",
         ["utilitarianism", "utilitarismo", "bentham", "mill"]),
        ("phil_06", "philosophy_logic", "Ethics",
         "What moral framework developed by Immanuel Kant evaluates actions based on duty and the Categorical Imperative?",
         "Deontology (Deontological Ethics)",
         ["deontology", "deontología", "kant", "imperativo categórico"]),
        ("phil_07", "philosophy_logic", "Philosophy of Mind",
         "What famous thought experiment by John Searle argues that syntactic symbol manipulation cannot produce genuine semantic understanding?",
         "Chinese Room Argument",
         ["chinese room", "habitación china", "searle", "cuarto chino"]),
        ("phil_08", "philosophy_logic", "Philosophy of Mind",
         "What operational behavioral benchmark was proposed in 1950 by Alan Turing to determine whether a machine can exhibit intelligent behavior?",
         "Turing Test (Imitation Game)",
         ["turing test", "test de turing", "imitation game", "juego de imitación"]),
        ("phil_09", "philosophy_logic", "Causal Inference",
         "In Pearl's causal graphical models, what graphical criterion determines whether a set of variables d-separates two nodes conditional on Z?",
         "d-separation (Directional Separation)",
         ["d-separation", "separación d", "d-separación", "pearl", "causal"]),
        ("phil_10", "philosophy_logic", "Philosophy of Science",
         "What criterion did Karl Popper propose as the demarcation boundary distinguishing empirical science from non-science?",
         "Falsifiability (Falsificationism)",
         ["falsifiability", "falsabilidad", "falsacionismo", "refutabilidad", "popper"]),
    ]
    for qid, dom, cat, q, ans, kws in phil_questions:
        items.append(GeneralQuestion(qid, dom, cat, q, ans, kws))

    # 5. Life Sciences, Genetics & Medicine (10 problemas)
    bio_questions = [
        ("bio_01", "life_sciences", "Genetics",
         "What bacterial adaptive immune mechanism utilizing a Cas endonuclease was adapted into a revolutionary gene-editing tool?",
         "CRISPR-Cas9",
         ["crispr", "cas9", "crispr-cas9", "edición genética"]),
        ("bio_02", "life_sciences", "Molecular Biology",
         "During DNA replication, what are the short newly synthesized DNA fragments formed on the lagging template strand called?",
         "Okazaki fragments",
         ["okazaki", "fragmentos de okazaki", "lagging"]),
        ("bio_03", "life_sciences", "Cellular Biology",
         "What enzyme uses an electrochemical proton gradient across the inner mitochondrial membrane to synthesize adenosine triphosphate?",
         "ATP Synthase",
         ["atp synthase", "atp sintasa", "sintasa de atp", "atp"]),
        ("bio_04", "life_sciences", "Neuroscience",
         "Which insulating lipid-rich sheath surrounds nerve cell axons to enable rapid saltatory electrical signal conduction?",
         "Myelin Sheath",
         ["myelin", "mielina", "vaina de mielina", "saltatoria"]),
        ("bio_05", "life_sciences", "Molecular Biology",
         "What biological dogma describes the flow of genetic information from DNA to RNA to Protein?",
         "Central Dogma of Molecular Biology",
         ["central dogma", "dogma central", "dna to rna", "transcripción"]),
        ("bio_06", "life_sciences", "Immunology",
         "What Y-shaped immunoglobulin proteins are secreted by plasma B cells to identify and neutralize foreign pathogens?",
         "Antibodies (Immunoglobulins)",
         ["antibodies", "anticuerpos", "immunoglobulin", "inmunoglobulina"]),
        ("bio_07", "life_sciences", "Cellular Biology",
         "What programmed, non-inflammatory cellular death mechanism removes damaged or unnecessary cells during organism development?",
         "Apoptosis",
         ["apoptosis", "muerte celular programada", "programmed cell death"]),
        ("bio_08", "life_sciences", "Evolutionary Biology",
         "What evolutionary process driven by differential reproductive success of organisms was proposed by Charles Darwin in 1859?",
         "Natural Selection",
         ["natural selection", "selección natural", "darwin"]),
        ("bio_09", "life_sciences", "Genetics",
         "What biological study investigates heritable phenotype changes in gene expression that do not involve alterations to the underlying DNA sequence?",
         "Epigenetics",
         ["epigenetics", "epigenética", "metilación", "histonas"]),
        ("bio_10", "life_sciences", "Biochemistry",
         "What anaerobic metabolic pathway breaks down glucose into two pyruvate molecules, yielding a net gain of 2 ATP?",
         "Glycolysis",
         ["glycolysis", "glucólisis", "glicólisis", "pyruvate", "piruvato"]),
    ]
    for qid, dom, cat, q, ans, kws in bio_questions:
        items.append(GeneralQuestion(qid, dom, cat, q, ans, kws))

    return items
