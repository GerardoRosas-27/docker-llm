"""Tests para la Red de Macro-Neuronas y el Campo CONSTELACIÓN v4 (K32 y K36)."""
import pytest
import torch

from enjambre.macro_neurons.network import (
    build_k32_macro_network,
    build_k36_macro_network,
    build_k66_macro_network,
    build_k66_macro_network_1_5b,
    macro_param_checksum,
)
from enjambre.macro_neurons.field_v4 import (
    ConstelacionV4Field,
    FIELD_PAIRS_32,
    FIELD_PAIRS_36,
    FIELD_PAIRS_66,
)


def test_build_k32_macro_network():
    model, tok, delta = build_k32_macro_network()
    assert len(model.model.layers) == 32
    assert model.config.num_hidden_layers == 32
    assert len(model.config.layer_types) == 32
    assert delta.meta["n_macro"] == 32
    assert delta.meta["math_cluster"] == [12, 13, 14, 15]
    assert delta.meta["code_cluster"] == [22, 23, 24, 25]

    for i, l in enumerate(model.model.layers):
        assert l.self_attn.layer_idx == i

    for p in model.parameters():
        assert not p.requires_grad


def test_build_k36_macro_network():
    model, tok, delta = build_k36_macro_network()
    assert len(model.model.layers) == 36
    assert model.config.num_hidden_layers == 36
    assert len(model.config.layer_types) == 36
    assert delta.meta["n_macro"] == 36
    assert delta.meta["algebra_cluster"] == [10, 11, 12, 13]
    assert delta.meta["arithmetic_cluster"] == [20, 21, 22, 23]
    assert delta.meta["code_cluster"] == [24, 25, 26, 27]

    for i, l in enumerate(model.model.layers):
        assert l.self_attn.layer_idx == i

    for p in model.parameters():
        assert not p.requires_grad


def test_constelacion_v4_field_36_attachment_and_step0_identity():
    model, tok, delta = build_k36_macro_network()
    hidden = model.config.hidden_size
    field = ConstelacionV4Field(hidden=hidden, rank=16, pairs=FIELD_PAIRS_36)

    assert len(field.bridges) == 18
    assert field.n_params() > 1_000_000

    inputs = tok("Calcula: 2 + 2 =", return_tensors="pt")

    with torch.no_grad():
        out_base = model(**inputs).logits

    field.attach(model)
    with torch.no_grad():
        out_field = model(**inputs).logits

    diff = (out_base - out_field).abs().max().item()
    assert diff < 1e-4, f"Diferencia en step 0: {diff} (debe ser casi 0)"

    field.remove_hooks()


def test_build_k66_macro_network():
    model, tok, delta = build_k66_macro_network()
    assert len(model.model.layers) == 66
    assert model.config.num_hidden_layers == 66
    assert len(model.config.layer_types) == 66
    assert delta.meta["n_macro"] == 66
    assert delta.meta["stages"] == 11
    assert delta.meta["stage_size"] == 6
    assert len(delta.meta["algebra_cluster"]) == 6
    assert len(delta.meta["euclid_cluster"]) == 6
    assert len(delta.meta["arithmetic_cluster"]) == 6
    assert len(delta.meta["combinatorics_cluster"]) == 6
    assert len(delta.meta["dicts_cluster"]) == 6
    assert len(delta.meta["recursion_cluster"]) == 6
    assert len(delta.meta["algorithms_cluster"]) == 6

    for i, l in enumerate(model.model.layers):
        assert l.self_attn.layer_idx == i

    for p in model.parameters():
        assert not p.requires_grad


def test_constelacion_v4_field_66_attachment_and_step0_identity():
    model, tok, delta = build_k66_macro_network()
    hidden = model.config.hidden_size
    field = ConstelacionV4Field(hidden=hidden, rank=16, pairs=FIELD_PAIRS_66).to(model.dtype)

    assert len(field.bridges) == 28
    assert field.n_params() > 1_600_000

    inputs = tok("Calcula: 2 + 2 =", return_tensors="pt")

    with torch.no_grad():
        out_base = model(**inputs).logits

    field.attach(model)
    with torch.no_grad():
        out_field = model(**inputs).logits

    diff = (out_base.float() - out_field.float()).abs().max().item()
    assert diff < 1e-3, f"Diferencia en step 0: {diff} (debe ser casi 0)"

    field.remove_hooks()


def test_build_k66_macro_network_1_5b():
    model, tok, delta = build_k66_macro_network_1_5b()
    assert len(model.model.layers) == 66
    assert model.config.num_hidden_layers == 66
    assert model.config.hidden_size == 1536
    assert delta.meta["n_macro"] == 66
    assert delta.meta["base_size"] == "1.5b"
    assert delta.meta["base_layers"] == 28
    assert delta.meta["added_layers"] == 38

    for i, l in enumerate(model.model.layers):
        assert l.self_attn.layer_idx == i

    for p in model.parameters():
        assert not p.requires_grad


def test_constelacion_v4_field_66_1_5b_attachment_and_step0_identity():
    model, tok, delta = build_k66_macro_network_1_5b()
    hidden = model.config.hidden_size
    assert hidden == 1536
    field = ConstelacionV4Field(hidden=hidden, rank=16, pairs=FIELD_PAIRS_66).to(model.dtype)

    assert len(field.bridges) == 28
    assert field.n_params() > 2_800_000

    inputs = tok("Calcula: 2 + 2 =", return_tensors="pt")

    with torch.no_grad():
        out_base = model(**inputs).logits

    field.attach(model)
    with torch.no_grad():
        out_field = model(**inputs).logits

    diff = (out_base.float() - out_field.float()).abs().max().item()
    assert diff < 1e-3, f"Diferencia en step 0: {diff} (debe ser casi 0)"

    field.remove_hooks()

