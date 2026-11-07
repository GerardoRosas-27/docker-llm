"""Tests unitarios para el ensamblaje por capas editadas + puentes en Matemáticas y Programación."""
from __future__ import annotations
import torch

from enjambre.math_code.bridges_ext import PAIRS_BY_CONFIG, build_warm_bridge_bank
from enjambre.math_code.layer_edits import (
    build_math_code_layer_deltas,
    orthogonalize_against_subspace,
    project_out,
)
from enjambre.math_code.tasks import (
    ALL_MC_TASKS,
    check_exact_int,
    check_python_expr,
    check_python_function,
    has_trailing_prose,
    make_mc_splits,
)


def test_validators_strict_math_and_code():
    assert check_exact_int("213", 213)
    assert not check_exact_int("213.", 213)  # rechaza punto final
    assert not check_exact_int("213\nExplanation", 213)

    clean_code = "def f(x):\n    return x * 2 + 1"
    dirty_code = "def f(x):\n    return x * 2 + 1\n- The function f takes x."
    assert not has_trailing_prose(clean_code)
    assert has_trailing_prose(dirty_code)
    assert check_python_function(clean_code, "f", [((0,), 1), ((3,), 7)])
    assert not check_python_function(dirty_code, "f", [((0,), 1), ((3,), 7)])

    assert check_python_expr("[x * 2 for x in range(3)]", "[", expected_val=[0, 2, 4])
    assert check_python_expr("lambda x: x + 5", "lambda", callable_tests=[((2,), 7), ((-1,), 4)])


def test_orthogonalization_removes_interference():
    v = torch.tensor([1.0, 2.0, 3.0, 0.0])
    u = torch.tensor([1.0, 0.0, 0.0, 0.0])
    vp = project_out(v, u, strength=1.0)
    assert abs(float(torch.dot(vp, u))) < 1e-6

    clean = orthogonalize_against_subspace(v, [u], strength=1.0)
    assert abs(float(torch.norm(clean)) - 1.0) < 1e-5
    assert abs(float(torch.dot(clean, u))) < 1e-6


def test_layer_deltas_progression_counts():
    deltas = build_math_code_layer_deltas()
    assert len(deltas["k12_orig"].bias_edits) == 12
    assert len(deltas["k12_mc"].bias_edits) == 12
    assert len(deltas["k16_mc"].bias_edits) == 16
    assert len(deltas["k20_mc"].bias_edits) == 20
    assert len(deltas["k24_mc"].bias_edits) == 24
    assert [b.layer for b in deltas["k16_mc"].bias_edits] == list(range(8, 24))
    assert [b.layer for b in deltas["k20_mc"].bias_edits] == list(range(4, 24))
    assert [b.layer for b in deltas["k24_mc"].bias_edits] == list(range(0, 24))


def test_warm_bridge_bank_preserves_pretrained_and_zero_inits_new():
    bank3 = build_warm_bridge_bank(896, PAIRS_BY_CONFIG["k12"], rank=16)
    bank6 = build_warm_bridge_bank(896, PAIRS_BY_CONFIG["k24"], rank=16)
    for k in ("12_16", "16_20", "20_23"):
        assert torch.equal(bank3.bridges[k].down.weight, bank6.bridges[k].down.weight)
        assert float(bank6.bridges[k].down.weight.norm()) > 0.1
    for k in ("0_4", "4_8", "8_12"):
        assert torch.allclose(bank6.bridges[k].down.weight, torch.zeros_like(bank6.bridges[k].down.weight))


def test_mc_splits_disjoint():
    tr, va, ho, man = make_mc_splits(12, 6, 15)
    assert set(tr.keys()) == set(ALL_MC_TASKS)
    pt = {r[0] for rows in tr.values() for r in rows}
    pv = {r[0] for rows in va.values() for r in rows}
    ph = {it.prompt for items in ho.values() for it in items}
    assert pt.isdisjoint(pv) and pt.isdisjoint(ph) and pv.isdisjoint(ph)
