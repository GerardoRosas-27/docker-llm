"""Test unitario para las 10 tareas de la suite Compleja de Matemáticas y Programación."""
from __future__ import annotations
from enjambre.math_code.complex_tasks import (
    ALL_COMPLEX_TASKS,
    COMPLEX_CODE_TASKS,
    COMPLEX_MATH_TASKS,
    build_complex_pool,
    make_complex_mc_splits,
)


def test_complex_tasks_pool_and_validators():
    pool = build_complex_pool()
    assert len(pool) == 10
    for tid in ALL_COMPLEX_TASKS:
        assert tid in pool
        assert len(pool[tid]) >= 60

    # Probar que los targets sintéticos generados pasan sus propios validadores
    for tid in ALL_COMPLEX_TASKS:
        for prompt, target, chk in pool[tid][:3]:
            assert chk(target), f"El target de {tid} falló su propio check: {target}"
            if tid in COMPLEX_CODE_TASKS:
                assert not chk(target + "\n- Viñeta explicativa colateral"), "Debe rechazar prosa colateral"


def test_complex_splits_disjointness():
    tr, va, ho, man = make_complex_mc_splits(n_train=10, n_val=5, n_hold=15, seed=242)
    assert len(man["prompt_sha256"]) == 64
    all_tr = {r[0] for rows in tr.values() for r in rows}
    all_va = {r[0] for rows in va.values() for r in rows}
    all_ho = {it.prompt for items in ho.values() for it in items}

    assert all_tr.isdisjoint(all_va)
    assert all_tr.isdisjoint(all_ho)
    assert all_va.isdisjoint(all_ho)
    assert len(ho) == 10
    for tid, items in ho.items():
        assert len(items) == 15
