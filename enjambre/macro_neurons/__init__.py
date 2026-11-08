"""Red de Macro-Neuronas y Campo de Interconexión CONSTELACIÓN v4.

Paradigma:
- Cada capa del Transformer es conceptualizada como una 'macro-neurona' individual.
- La red se expande agregando macro-neuronas editadas (e.g. 24 -> 32 capas).
- La interconexión entre macro-neuronas es gobernada por un Campo Geométrico v4
  (bloques Koopman 2x2, levantamiento monótono, precisión e innovación).
"""
from .network import (
    MacroNeuronNetwork,
    build_k32_macro_network,
    build_k36_macro_network,
    build_k66_macro_network,
    build_k66_macro_network_1_5b,
)
from .field_v4 import ConstelacionV4Field, FIELD_PAIRS_32, FIELD_PAIRS_36, FIELD_PAIRS_66

__all__ = [
    "MacroNeuronNetwork",
    "build_k32_macro_network",
    "build_k36_macro_network",
    "build_k66_macro_network",
    "build_k66_macro_network_1_5b",
    "ConstelacionV4Field",
    "FIELD_PAIRS_32",
    "FIELD_PAIRS_36",
    "FIELD_PAIRS_66",
]
