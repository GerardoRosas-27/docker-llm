"""Campo de Interconexión CONSTELACIÓN v4 para la Red de Macro-Neuronas.

Implementa un campo geométrico multiescala sobre las 32 macro-neuronas de K32_macro,
utilizando conectores `ConstelacionV4Bridge` (bloques 2x2 Koopman, levantamiento
monótono, precisión normalizada Π̂ e innovación acoplada).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from enjambre.math_code.neuro_connectors import ConstelacionV4Bridge


# Topología de 12 Puentes del Campo CONSTELACIÓN v4 (Red de 32 Macro-Neuronas):
# 1. Puentes Troncales (8): flujo continuo a lo largo de las 32 capas
# 2. Puentes Trans-Campo (4): autopistas armónicas de largo alcance entre clusters funcionales
FIELD_PAIRS_32: List[Tuple[int, int]] = [
    # Tronco (8)
    (0, 4),
    (4, 8),
    (8, 12),    # Enlace hacia Cluster Math
    (12, 16),   # Salida de Cluster Math hacia Núcleo
    (16, 20),   # Enlace de Núcleo
    (20, 24),   # Enlace en Cluster Code
    (24, 28),   # Salida de Cluster Code hacia Síntesis
    (28, 31),   # Síntesis hacia salida final
    # Autopistas Armónicas Trans-Campo (4)
    (4, 12),    # Sintaxis temprana -> Cluster Math
    (12, 22),   # Cluster Math -> Cluster Code (acoplamiento cruzado)
    (16, 26),   # Núcleo de Razonamiento -> Bloque de Síntesis
    (22, 31),   # Cluster Code -> Salida Final
]

# Topología Ampliada del Hiper-Campo CONSTELACIÓN v4 (Red de 36 Macro-Neuronas, 18 Puentes):
# 1. Tronco Principal (9): Flujo continuo a través de las 36 capas
# 2. Autopistas Armónicas (6): Resonancia inter-cluster (Álgebra, Aritmética, Código y Salida)
# 3. Canales Causales Cruzados (3): Inyección de innovación directa entre polos funcionales
FIELD_PAIRS_36: List[Tuple[int, int]] = [
    # Tronco Principal (9)
    (0, 4),
    (4, 8),
    (8, 12),    # Enlace hacia Cluster Álgebra
    (12, 16),   # Salida de Álgebra hacia Núcleo
    (16, 20),   # Núcleo hacia Cluster Aritmética
    (20, 24),   # Aritmética hacia Cluster Código
    (24, 28),   # Código hacia Síntesis
    (28, 32),   # Síntesis intermedia
    (32, 35),   # Salida Final
    # Autopistas Armónicas Trans-Campo (6)
    (4, 10),    # Sintaxis temprana -> Cluster Álgebra
    (10, 20),   # Álgebra -> Aritmética y Exponenciación
    (14, 24),   # Núcleo Semántico -> Cluster Código
    (20, 28),   # Aritmética -> Síntesis
    (24, 32),   # Código -> Síntesis avanzada
    (20, 35),   # Aritmética directa -> Salida Final
    # Canales de Innovación Causal Cruzada (3)
    (10, 24),   # Álgebra -> Código (coordinación simbólica)
    (16, 28),   # Núcleo Semántico -> Síntesis
    (28, 35),   # Síntesis -> Salida Final
]


class ConstelacionV4Field(nn.Module):
    """Campo Geométrico de Conexión de Macro-Neuronas basado en CONSTELACIÓN v4."""

    def __init__(
        self,
        hidden: int,
        rank: int = 16,
        pairs: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()
        self.hidden = hidden
        self.rank = rank
        self.pairs = list(pairs or FIELD_PAIRS_32)
        self.bridges = nn.ModuleDict()

        for s, t in self.pairs:
            key = f"{s}_{t}"
            b = ConstelacionV4Bridge(hidden=hidden, rank=rank)
            b.src_layer, b.tgt_layer = s, t
            self.bridges[key] = b

        self._buf: Dict[int, torch.Tensor] = {}
        self._hooks: List = []
        self.enabled: bool = True

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def clear_buf(self) -> None:
        self._buf.clear()

    def attach(self, model: nn.Module) -> None:
        """Registra pre-hook y hooks en las capas para gestionar el flujo del campo."""
        self.remove_hooks()
        layers = model.model.layers

        src_set = {s for s, _ in self.pairs}
        tgt_sources: Dict[int, List[int]] = {}
        for s, t in self.pairs:
            tgt_sources.setdefault(t, []).append(s)

        def pre_clear(_mod, _inp):
            self.clear_buf()

        self._hooks.append(layers[0].register_forward_pre_hook(pre_clear))

        def make_hook(idx: int):
            def hook(_mod, _inp, out):
                if not self.enabled:
                    return out
                h = out

                # 1) Si idx es target de uno o más puentes del campo: aplicar en orden topológico
                if idx in tgt_sources:
                    for s in tgt_sources[idx]:
                        if s in self._buf:
                            h_src = self._buf[s]
                            if h_src.shape[1] != h.shape[1]:
                                T = min(h_src.shape[1], h.shape[1])
                                h_src = h_src[:, -T:]
                                h = h[:, -T:]
                            key = f"{s}_{idx}"
                            h_new = self.bridges[key](h_src, h)
                            if h_new.shape[1] == out.shape[1]:
                                h = h_new
                            else:
                                out2 = out.clone()
                                out2[:, -h_new.shape[1] :, :] = h_new
                                h = out2

                # 2) Si idx es fuente de uno o más puentes: almacenar su activación en el buffer
                if idx in src_set:
                    self._buf[idx] = h

                return h if idx in tgt_sources else out

            return hook

        needed = sorted(src_set | set(tgt_sources))
        for idx in needed:
            self._hooks.append(layers[idx].register_forward_hook(make_hook(idx)))

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.clear_buf()
