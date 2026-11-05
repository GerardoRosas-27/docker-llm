"""Registro de modelos + escalado de índices de capa por profundidad (model-agnostic)."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, List

# Fracciones de profundidad calibradas en SmolLM2-135M (30 capas):
#   expertos: capas 8/10/12 -> 0.27/0.33/0.40 ; ROME: 0.47..0.80 ; MEMIT: 0.15..0.35
EXPERT_LAYER_FRACS = [0.27, 0.33, 0.40]
ROME_LAYER_FRAC = 0.30          # capa única de ROME (mid-early, la misma familia que MEMIT)
MEMIT_LAYER_FRACS = [0.15, 0.20, 0.25, 0.30, 0.35]
# escala del sesgo de steering RELATIVA a la norma mediana del residual en esa capa
STEER_REL_SCALES = [0.02, 0.08, 0.25]
STEER_MULTI_REL_SCALES = [0.06, 0.12]   # multi-capa: misma dirección en las 3 capas


@dataclass
class ModelCfg:
    key: str
    hf_id: str
    note: str = ""


MODELS: Dict[str, ModelCfg] = {
    "smollm2-135m": ModelCfg("smollm2-135m", "HuggingFaceTB/SmolLM2-135M-Instruct", "30 capas, d=576"),
    "smollm2-360m": ModelCfg("smollm2-360m", "HuggingFaceTB/SmolLM2-360M-Instruct", "32 capas, d=960"),
    "qwen2.5-0.5b": ModelCfg("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct", "24 capas, d=896"),
}
DEFAULT_MODEL = os.environ.get("ENJAMBRE_MODEL_KEY", "smollm2-135m")


def resolve(model: str) -> ModelCfg:
    """Acepta clave corta o id de HF."""
    if model in MODELS:
        return MODELS[model]
    for m in MODELS.values():
        if m.hf_id == model:
            return m
    return ModelCfg(model.split("/")[-1].lower(), model, "no registrado")


def scale_layers(fracs: List[float], n_layers: int) -> List[int]:
    out = []
    for f in fracs:
        L = min(n_layers - 1, max(0, round(f * n_layers)))
        if L not in out:
            out.append(L)
    return out


def artifacts_dir(key: str) -> str:
    base = os.environ.get("ENJAMBRE_ARTIFACTS_ROOT", os.path.join(os.path.dirname(__file__), "artifacts"))
    return os.path.join(base, key)
