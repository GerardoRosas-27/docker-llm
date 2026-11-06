"""Tests del arnés adaptativo (sin descargar HF cuando hay mock)."""
from __future__ import annotations
import json
import threading
from pathlib import Path

import torch
import torch.nn as nn

from enjambre.adaptive.bank import ACTIVE, CANDIDATE, EXPERIMENTAL, ExpertBank
from enjambre.adaptive.catalog import all_tasks, catalog_manifest, task_by_id
from enjambre.adaptive.memory_net import MemoryNet
from enjambre.adaptive.router_heuristic import route
from enjambre.adaptive.snapshot import SnapshotActivator
from enjambre.edits import BiasEdit, EditDelta


def _tiny_model(n_layers=3, hidden=16):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Module()
            self.mlp.down_proj = nn.Linear(hidden, hidden, bias=False)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Block() for _ in range(n_layers)])
            self.config = type("C", (), {"hidden_size": hidden})()

    return M()


def test_catalog_50_unique_and_splits():
    m = catalog_manifest()
    assert m["n_tasks"] == 50
    ids = [t["id"] for t in m["tasks"]]
    assert len(ids) == len(set(ids))
    for t in all_tasks():
        assert len(t.heldout_items) >= 5
        assert len(t.val_items) >= 3
        assert len(t.tune_items) >= 3
        # checks are callable and reject empty nonsense often
        assert callable(t.heldout_items[0].check)


def test_snapshot_restore_exact_and_exception_safe():
    m = _tiny_model()
    act = SnapshotActivator(m)
    layer = 1
    lin = m.model.layers[layer].mlp.down_proj
    # set a known bias baseline None
    assert lin.bias is None
    vec = torch.ones(16)
    delta = EditDelta("t", [BiasEdit(layer=layer, vec=vec, scale=2.0)])
    import pytest
    with pytest.raises(RuntimeError):
        with act.session(delta):
            assert lin.bias is not None
            assert torch.allclose(lin.bias, torch.ones(16) * 2)
            raise RuntimeError("boom")
    # restored even after exception
    assert lin.bias is None
    assert act.active() is None


def test_no_cross_task_leakage_sequential():
    m = _tiny_model()
    act = SnapshotActivator(m)
    d1 = EditDelta("a", [BiasEdit(0, torch.ones(16), 1.0)])
    d2 = EditDelta("b", [BiasEdit(0, torch.ones(16) * 3, 1.0)])
    with act.session(d1):
        b1 = m.model.layers[0].mlp.down_proj.bias.detach().clone()
    with act.session(d2):
        b2 = m.model.layers[0].mlp.down_proj.bias.detach().clone()
    assert not torch.allclose(b1, b2)
    assert m.model.layers[0].mlp.down_proj.bias is None


def test_concurrent_activation_serialized():
    m = _tiny_model()
    act = SnapshotActivator(m)
    errors = []

    def worker(scale):
        try:
            d = EditDelta(f"s{scale}", [BiasEdit(0, torch.ones(16), float(scale))])
            with act.session(d):
                b = m.model.layers[0].mlp.down_proj.bias
                # while holding session, bias should equal scale * ones
                if not torch.allclose(b, torch.ones(16) * scale):
                    errors.append("mismatch")
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=worker, args=(s,)) for s in (1.0, 2.0, 3.0, 4.0)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert m.model.layers[0].mlp.down_proj.bias is None


def test_memory_good_only_and_audit_failures(tmp_path):
    mem = MemoryNet(str(tmp_path / "m.sqlite"))
    mem.record_experience("t1", "p1", "good", True, version_id="v1", source="auto_check")
    mem.record_experience("t1", "p2", "bad", False, version_id="v1", source="auto_check", reason="check_failed")
    good = mem.retrieve_good("t1")
    assert len(good) == 1 and good[0]["output"] == "good"
    fails = mem.failures("t1")
    assert len(fails) == 1 and fails[0]["reason"] == "check_failed"
    mem.reset_task("t1", keep_audit=True)
    assert mem.retrieve_good("t1") == []
    assert len(mem.failures("t1")) == 1


def test_bank_promote_rollback_keeps_files(tmp_path):
    bank = ExpertBank(str(tmp_path / "bank"))
    d1 = EditDelta("taskx", [BiasEdit(0, torch.randn(8), 1.0)], meta={"layer": 0})
    d2 = EditDelta("taskx", [BiasEdit(1, torch.randn(8), 1.0)], meta={"layer": 1})
    v1 = bank.register("taskx", d1, status=ACTIVE, make_active=True, metrics={"val": 0.2})
    v2 = bank.register("taskx", d2, status=CANDIDATE, parent_version_id=v1.version_id, metrics={"val": 0.5})
    bank.promote_atomic("taskx", v2.version_id)
    assert bank.get_active("taskx").version_id == v2.version_id
    assert bank.versions[v1.version_id].status == "retired"
    assert Path(v1.path).is_file() and Path(v2.path).is_file()
    bank.rollback("taskx", v1.version_id)
    assert bank.get_active("taskx").version_id == v1.version_id
    assert bank.versions[v2.version_id].status == "quarantine"


def test_router_explicit_and_keyword():
    tid, info = route("task:upper_token please")
    assert tid == "upper_token" and info["method"] == "explicit_task_id"
    tid2, info2 = route("please convert to UPPERCASE only")
    assert tid2 == "upper_token"
    tid3, info3 = route("asdf qwer zxcv")
    assert tid3 is None and info3.get("fallback") == "base"


def test_task_checks_not_substring_lax():
    t = task_by_id("add_int")
    it = t.tune_items[0]
    assert it.check("") is False
    # verbose prose without the exact lone integer token should fail _exact
    assert it.check("The answer is maybe right") is False
