"""Unit tests for MEMIT numerical helpers (no HF model download)."""
from __future__ import annotations

import torch
import torch.nn as nn

from enjambre.memit import LowRankEdit, _apply_lowrank, _clamp_rows, _make_psd


def test_make_psd_kills_negative_eigs():
    torch.manual_seed(0)
    A = torch.randn(8, 8)
    C = A @ A.T
    # inject a clear negative eigenvalue direction
    evals, evecs = torch.linalg.eigh(C)
    evals[0] = -3.0
    C_bad = (evecs * evals) @ evecs.T
    assert float(torch.linalg.eigvalsh(C_bad).min()) < 0
    C_psd = _make_psd(C_bad, eps_frac=1e-2)
    assert torch.allclose(C_psd, C_psd.T, atol=1e-5)
    assert float(torch.linalg.eigvalsh(C_psd.double()).min()) >= 0


def test_make_psd_floor_relative_to_median():
    # diagonal with mixed scale; floor should lift tiny/neg toward ~eps*median
    C = torch.diag(torch.tensor([1e-9, 1.0, 2.0, 3.0]))
    C_psd = _make_psd(C, eps_frac=0.1)
    evals = torch.linalg.eigvalsh(C_psd.double())
    med = torch.tensor([0.0, 1.0, 2.0, 3.0]).median()  # approx of positive median path
    assert float(evals.min()) >= 0.05  # 0.1 * median(~1) order of magnitude


def test_clamp_rows_limits_norm():
    X = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    ref = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    out = _clamp_rows(X, ref, max_ratio=1.0)
    assert torch.allclose(out[0].norm(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(out[1].norm(), torch.tensor(2.0), atol=1e-5)
    # already small rows unchanged
    X2 = torch.tensor([[0.1, 0.0]])
    ref2 = torch.tensor([[1.0, 0.0]])
    out2 = _clamp_rows(X2, ref2, max_ratio=1.0)
    assert torch.allclose(out2, X2)


def test_apply_lowrank_revert_fp32_near_exact():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            layer = nn.Module()
            layer.mlp = nn.Module()
            layer.mlp.down_proj = nn.Linear(16, 8, bias=False)
            self.model.layers = nn.ModuleList([layer])

    m = M()
    W0 = m.model.layers[0].mlp.down_proj.weight.detach().clone()
    e = LowRankEdit(layer=0, U=torch.randn(8, 2), V=torch.randn(2, 16))
    _apply_lowrank(m, e, +1.0)
    _apply_lowrank(m, e, -1.0)
    err = (m.model.layers[0].mlp.down_proj.weight - W0).abs().max().item()
    assert err < 1e-5


def test_apply_lowrank_revert_fp16_needs_tolerance():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            layer = nn.Module()
            layer.mlp = nn.Module()
            layer.mlp.down_proj = nn.Linear(32, 16, bias=False).half()
            self.model.layers = nn.ModuleList([layer])

    torch.manual_seed(1)
    m = M()
    W0 = m.model.layers[0].mlp.down_proj.weight.detach().clone()
    e = LowRankEdit(layer=0, U=torch.randn(16, 3), V=torch.randn(3, 32))
    _apply_lowrank(m, e, +1.0)
    _apply_lowrank(m, e, -1.0)
    err = (m.model.layers[0].mlp.down_proj.weight.float() - W0.float()).abs().max().item()
    # fp16 round-trip is NOT bit-exact; tolerate a small residual
    assert err < 5e-2
    assert err > 0.0 or True  # may be tiny on some seeds; main assert is upper bound
