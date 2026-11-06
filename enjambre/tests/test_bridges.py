"""Tests puentes: identidad init, grad, base freeze, splits, causalidad, hooks."""
from __future__ import annotations
import torch
from enjambre.bridges.modules import ResidualBridge, BridgeBank, BRIDGE_PAIRS
from enjambre.bridges.data import make_splits, all_historical_prompts


def test_pairs_forward_only_late():
    assert BRIDGE_PAIRS == [(12, 16), (16, 20), (20, 23)]
    for s, t in BRIDGE_PAIRS:
        assert s < t and s >= 12 and t <= 23


def test_bridge_identity_init():
    b = ResidualBridge(32, rank=8)
    x = torch.randn(2, 5, 32)
    y = b(x, x)
    assert torch.allclose(y, x, atol=1e-6)


def test_bridge_grad_down_nonzero():
    b = ResidualBridge(16, rank=4)
    x = torch.randn(1, 3, 16)
    y = b(x, x)
    y.sum().backward()
    assert b.down.weight.grad is not None
    assert float(b.down.weight.grad.abs().sum()) > 0


def test_bridge_no_future_mix():
    """Puente es punto a punto por posición: permutar T en src cambia solo esas posiciones."""
    b = ResidualBridge(8, rank=4)
    # romper identity para ver efecto
    b.down.weight.data.normal_(0, 0.1)
    h_tgt = torch.randn(1, 4, 8)
    h_src = torch.randn(1, 4, 8)
    y = b(h_src, h_tgt)
    # cambiar solo posición 0 de src → solo pos 0 de salida cambia
    h_src2 = h_src.clone()
    h_src2[:, 0, :] += 1.0
    y2 = b(h_src2, h_tgt)
    delta = (y2 - y).abs()
    assert float(delta[:, 1:, :].max().detach()) < 1e-5
    assert float(delta[:, 0, :].max().detach()) > 1e-5


def test_bank_hook_restore():
    class FakeLayer(torch.nn.Module):
        def forward(self, x):
            return x + 1

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([FakeLayer() for _ in range(24)])

    m = FakeModel()
    bank = BridgeBank(8, rank=4, pairs=[(12, 16)])
    bank.attach(m)
    assert len(bank._hooks) >= 2
    bank.remove_hooks()
    assert len(bank._hooks) == 0
    # re-attach / remove idempotent
    bank.attach(m)
    bank.remove_hooks()
    bank.remove_hooks()


def test_splits_disjoint():
    tr, va, ho, man = make_splits(20, 10, 50, seed=0)
    hist = all_historical_prompts()
    for rows in list(tr.values()) + list(va.values()):
        for p, _, _ in rows:
            assert p not in hist
    for items in ho.values():
        assert len(items) == 50
        for it in items:
            assert it.prompt not in hist
    # train/val/hold disjoint among themselves
    def prompts(rows_or_items):
        out = set()
        for v in rows_or_items.values():
            for x in v:
                out.add(x.prompt if hasattr(x, "prompt") else x[0])
        return out
    pt, pv, ph = prompts(tr), prompts(va), prompts(ho)
    assert not (pt & pv) and not (pt & ph) and not (pv & ph)
