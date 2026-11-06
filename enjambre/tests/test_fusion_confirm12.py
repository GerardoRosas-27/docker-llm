"""Tests confirmación K12: freeze, overlap, L2, stats."""
from __future__ import annotations
from enjambre.fusion_confirm12.confirm_holdout import (
    assert_no_overlap, build_confirm_holdout, holdout_manifest,
)
from enjambre.fusion_confirm12.frozen import FROZEN_K12, rebuild_k12_delta, delta_content_sha256
from enjambre.fusion_confirm12.stats import bootstrap_delta_ci, mcnemar_exact, paired_flags


def test_frozen_layers_exact():
    assert FROZEN_K12["K"] == 12
    assert FROZEN_K12["layers"] == list(range(12, 24))
    assert FROZEN_K12["scale_mult"] == 0.5
    assert FROZEN_K12["layout"] == "late"


def test_rebuild_deterministic_and_l2():
    d1, m1 = rebuild_k12_delta()
    d2, m2 = rebuild_k12_delta()
    assert m1["delta_sha256"] == m2["delta_sha256"]
    assert abs(m1["total_l2"] - 1.6404) < 1e-3
    assert [b.layer for b in d1.bias_edits] == FROZEN_K12["layers"]
    assert delta_content_sha256(d1) == m1["delta_sha256"]


def test_confirm_holdout_size_and_no_overlap():
    h = build_confirm_holdout(50)
    assert all(len(v) >= 50 for v in h.values())
    assert_no_overlap(h)
    man = holdout_manifest(h)
    assert len(man["prompt_sha256"]) == 64


def test_paired_and_mcnemar_bootstrap():
    a = [True, True, False, False, True]
    b = [True, False, True, False, True]
    pf = paired_flags(a, b)
    assert pf["n_other_better"] == 1 and pf["n_other_worse"] == 1
    m = mcnemar_exact(1, 1)
    assert m["p_two_sided"] == 1.0
    ci = bootstrap_delta_ci(a, b, n_boot=200, seed=0)
    assert "ci95" in ci and len(ci["ci95"]) == 2
