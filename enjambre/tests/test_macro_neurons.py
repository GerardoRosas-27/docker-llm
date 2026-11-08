"""Tests para la Red de Macro-Neuronas y el Campo CONSTELACIÓN v4."""
import pytest
import torch

from enjambre.macro_neurons.network import build_k32_macro_network, macro_param_checksum
from enjambre.macro_neurons.field_v4 import ConstelacionV4Field, FIELD_PAIRS_32


def test_build_k32_macro_network():
    model, tok, delta = build_k32_macro_network()
    assert len(model.model.layers) == 32
    assert model.config.num_hidden_layers == 32
    assert len(model.config.layer_types) == 32
    assert delta.meta["n_macro"] == 32
    assert delta.meta["math_cluster"] == [12, 13, 14, 15]
    assert delta.meta["code_cluster"] == [22, 23, 24, 25]

    # Verificar que layer_idx está correctamente indexado
    for i, l in enumerate(model.model.layers):
        assert l.self_attn.layer_idx == i

    # Verificar que todos los parámetros del modelo base están congelados
    for p in model.parameters():
        assert not p.requires_grad


def test_constelacion_v4_field_attachment_and_step0_identity():
    model, tok, delta = build_k32_macro_network()
    hidden = model.config.hidden_size
    field = ConstelacionV4Field(hidden=hidden, rank=16, pairs=FIELD_PAIRS_32)

    assert len(field.bridges) == 12
    assert field.n_params() > 0

    # Tokenizar un prompt
    inputs = tok("Calcula: 2 + 2 =", return_tensors="pt")

    # Forward sin campo
    with torch.no_grad():
        out_base = model(**inputs).logits

    # Acoplar el campo
    field.attach(model)
    with torch.no_grad():
        out_field = model(**inputs).logits

    # En step 0, el campo debe ser idéntico al modelo base (zero-init en salida)
    diff = (out_base - out_field).abs().max().item()
    assert diff < 1e-4, f"Diferencia en step 0: {diff} (debe ser casi 0)"

    field.remove_hooks()
