"""Tests fusión bias (sin HF)."""
from __future__ import annotations
import torch
from enjambre.edits import BiasEdit, EditDelta
from enjambre.fusion.combine import fuse_deltas
from enjambre.fusion.parallel_equiv import prove_average_bias_equivalence


def _d(name, layer, scale=1.0, dim=8):
    v = torch.zeros(dim); v[0] = 1.0
    return EditDelta(name, [BiasEdit(layer, v, scale)])


def test_multi_sum_same_layer_adds():
    a = _d("a", 2, 1.0)
    b = _d("b", 2, 2.0)
    f = fuse_deltas([a, b], mode="multi_sum", global_scale=1.0)
    assert len(f.bias_edits) == 1 and f.bias_edits[0].layer == 2
    # effective = 1*e0 + 2*e0 = 3 e0 → scale≈3
    assert abs(f.bias_edits[0].scale - 3.0) < 1e-4


def test_multi_avg_same_layer():
    a = _d("a", 1, 2.0)
    b = _d("b", 1, 4.0)
    f = fuse_deltas([a, b], mode="multi_avg")
    assert abs(f.bias_edits[0].scale - 3.0) < 1e-4  # mean of 2 and 4


def test_single_relocates_all_to_target():
    a = _d("a", 1, 1.0)
    b = _d("b", 5, 1.0)
    f = fuse_deltas([a, b], mode="single_sum", target_layer=7)
    assert [e.layer for e in f.bias_edits] == [7]


def test_order_invariance_sum():
    a, b, c = _d("a", 3, 1.0), _d("b", 3, 2.0), _d("c", 4, 1.0)
    f1 = fuse_deltas([a, b, c], mode="multi_sum")
    f2 = fuse_deltas([c, a, b], mode="multi_sum")
    for e1, e2 in zip(sorted(f1.bias_edits, key=lambda x: x.layer), sorted(f2.bias_edits, key=lambda x: x.layer)):
        assert e1.layer == e2.layer
        assert torch.allclose(e1.scale * e1.vec, e2.scale * e2.vec, atol=1e-5)


def test_zero_deltas_emptyish():
    f = fuse_deltas([], mode="multi_avg")
    assert f.bias_edits == []


def test_parallel_bias_average_equivalence():
    torch.manual_seed(0)
    W = torch.randn(16, 32)
    biases = [torch.randn(16) for _ in range(5)]
    x = torch.randn(3, 32)
    proof = prove_average_bias_equivalence(W, biases, x, atol=1e-5)
    assert proof["equivalent"] is True
    assert proof["extra_params_if_materialized"] > 0
