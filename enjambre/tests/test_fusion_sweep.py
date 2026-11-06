"""Tests barrido capas: layouts, presupuesto L2, restore, holdout leakage, zero."""
from __future__ import annotations
import torch
import torch.nn as nn

from enjambre.adaptive.snapshot import SnapshotActivator
from enjambre.edits import BiasEdit, EditDelta
from enjambre.experts import all_expert_specs
from enjambre.fusion_sweep.build_delta import fuse_directions_on_layers, redistribute_prior_biases, total_l2
from enjambre.fusion_sweep.holdout import build_new_holdout
from enjambre.fusion_sweep.layouts import all_configs, clamp_unique, default_Ks, layout_layers


def test_default_Ks_fit_24():
    assert default_Ks(24) == [1, 2, 3, 4, 6, 8, 12, 24]


def test_layouts_unique_sorted_in_range():
    nL = 24
    for cfg in all_configs(nL):
        layers = cfg["layers"]
        assert len(layers) == cfg["K"]
        assert layers == sorted(layers)
        assert len(set(layers)) == len(layers)
        assert all(0 <= i < nL for i in layers)


def test_uniform_K1_is_mid():
    assert layout_layers(24, 1, "uniform") == [12]


def test_clamp_unique_pads():
    out = clamp_unique([5, 5, 5], 24)
    assert len(out) == 3 and len(set(out)) == 3 and 5 in out


def test_budget_L2_comparable_across_K():
    # fake directions: 10 experts, 25 rows (embed+24), dim=8
    dirs = {}
    for i in range(10):
        t = torch.randn(25, 8)
        t = t / (t.norm(dim=-1, keepdim=True) + 1e-8)
        dirs[f"e{i}"] = t
    B = 3.0
    for K in [1, 3, 6, 12, 24]:
        layers = layout_layers(24, K, "uniform")
        d = fuse_directions_on_layers(dirs, layers, B=B, name=f"k{K}")
        assert abs(total_l2(d) - B) < 1e-4
        assert len(d.bias_edits) == K
        # each scale ~= B/sqrt(K)
        for b in d.bias_edits:
            assert abs(b.scale - B / (K ** 0.5)) < 1e-4


def test_redistribute_marked_distinct():
    prior = EditDelta("p", [BiasEdit(6, torch.ones(4), 2.0), BiasEdit(8, torch.ones(4), 1.0)])
    d = redistribute_prior_biases([prior], [3, 9, 15], B=2.0, name="r")
    assert d.meta["kind"] == "redistribute_prior_bias"
    assert abs(total_l2(d) - 2.0) < 1e-4


def test_snapshot_restore_exact():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            class L(nn.Module):
                def __init__(self):
                    super().__init__()
                    class M(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.down_proj = nn.Linear(4, 4, bias=False)
                    self.mlp = M()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([L(), L()])
    m = Tiny()
    act = SnapshotActivator(m)
    delta = EditDelta("t", [BiasEdit(0, torch.tensor([1., 0, 0, 0]), 0.5)])
    assert m.model.layers[0].mlp.down_proj.bias is None
    with act.session(delta):
        assert m.model.layers[0].mlp.down_proj.bias is not None
        assert torch.allclose(m.model.layers[0].mlp.down_proj.bias, torch.tensor([0.5, 0, 0, 0]))
    assert m.model.layers[0].mlp.down_proj.bias is None


def test_zero_delta_session():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            class L(nn.Module):
                def __init__(self):
                    super().__init__()
                    class M(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.down_proj = nn.Linear(4, 4, bias=True)
                    self.mlp = M()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([L()])
    m = Tiny()
    bak = m.model.layers[0].mlp.down_proj.bias.detach().clone()
    act = SnapshotActivator(m)
    with act.session(None):
        pass
    assert torch.allclose(m.model.layers[0].mlp.down_proj.bias, bak)


def test_holdout_size_and_no_prompt_leak_vs_eval():
    hold = build_new_holdout(20)
    specs = all_expert_specs()
    assert set(hold) == {s.id for s in specs}
    for s in specs:
        assert len(hold[s.id]) >= 20
        hist = {it.prompt for it in list(s.tune_items) + list(s.eval_items)}
        newp = {it.prompt for it in hold[s.id]}
        assert hist.isdisjoint(newp), s.id
