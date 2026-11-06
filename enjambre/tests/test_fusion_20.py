"""Tests K20: conteo capas, invarianza 12 literal, L2 normalizado, overlap."""
from __future__ import annotations
import torch
from enjambre.fusion_20.build import (
    K20_LAYERS, K20_NEW_LAYERS, K20_KEEP_LAYERS, build_all_deltas,
)
from enjambre.fusion_20.holdout import assert_no_overlap, build_k20_holdout


def test_layer_counts():
    assert len(K20_LAYERS) == 20
    assert len(K20_NEW_LAYERS) == 8
    assert len(K20_KEEP_LAYERS) == 12
    assert K20_LAYERS == list(range(4, 24))


def test_A_literal_keeps_k12_bits_and_more_l2():
    b = build_all_deltas()
    k12 = b["k12"]["delta"]
    a = b["k20_A"]["delta"]
    assert len(a.bias_edits) == 20
    by_a = {e.layer: e for e in a.bias_edits}
    for e in k12.bias_edits:
        assert torch.equal(by_a[e.layer].vec.cpu(), e.vec.cpu())
        assert abs(by_a[e.layer].scale - e.scale) < 1e-12
    # new layers present
    for L in K20_NEW_LAYERS:
        assert L in by_a
        assert abs(by_a[L].scale - b["scale_k12"]) < 1e-9
    assert b["k20_A"]["meta"]["total_l2"] > b["l2_k12"] + 0.1


def test_B_same_total_l2():
    b = build_all_deltas()
    assert abs(b["k20_B"]["meta"]["total_l2"] - b["l2_k12"]) < 1e-5
    assert len(b["k20_B"]["delta"].bias_edits) == 20


def test_holdout_no_overlap():
    h = build_k20_holdout(50)
    assert all(len(v) >= 50 for v in h.values())
    assert_no_overlap(h)
