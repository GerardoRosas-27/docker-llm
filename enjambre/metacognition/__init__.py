"""Módulo de Arquitectura Multi-Agente Metacognitiva Bidireccional (S1 <-> S2)."""
from .agents import (
    AgentObservationResult,
    BaseMetacognitiveAgent,
    BidirectionalMetacognitiveSystem,
    MetacognitiveObserverAgent,
    RealTimeHookCollector,
    ReflexivePrimaryAgent,
    S1_TO_S2_PROMPT,
    S2_TO_S1_PROMPT,
    build_weight_shared_k32_0_5b,
    build_weight_shared_k66_1_5b,
)
from .evaluator import evaluate_bidirectional_system
from .modules import (
    AttributePredictionHeads,
    ManifoldSoftPromptProjector,
    MetaKoopmanMonitorBridge,
    WeightIntrospectionModule,
    WeightLatentEncoder,
    build_vocab_anchors,
)
from .spectrometer import (
    DampedLinear,
    LayerWeightSnapshot,
    LayerWeightSpectrometer,
    ROLE_LABELS,
    ROLE_TO_ID,
    SPECTRAL_VECTOR_DIM,
    apply_reversible_layer_perturbation,
)
from .trainer import BidirectionalMetaTrainer, DEFAULT_META_DIR

__all__ = [
    "AgentObservationResult",
    "AttributePredictionHeads",
    "BaseMetacognitiveAgent",
    "BidirectionalMetaTrainer",
    "BidirectionalMetacognitiveSystem",
    "DampedLinear",
    "DEFAULT_META_DIR",
    "LayerWeightSnapshot",
    "LayerWeightSpectrometer",
    "ManifoldSoftPromptProjector",
    "MetaKoopmanMonitorBridge",
    "MetacognitiveObserverAgent",
    "ROLE_LABELS",
    "ROLE_TO_ID",
    "RealTimeHookCollector",
    "ReflexivePrimaryAgent",
    "S1_TO_S2_PROMPT",
    "S2_TO_S1_PROMPT",
    "SPECTRAL_VECTOR_DIM",
    "WeightIntrospectionModule",
    "WeightLatentEncoder",
    "apply_reversible_layer_perturbation",
    "build_vocab_anchors",
    "build_weight_shared_k32_0_5b",
    "build_weight_shared_k66_1_5b",
    "evaluate_bidirectional_system",
]
