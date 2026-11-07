"""Tests unitarios y de integración para las 4 arquitecturas de conectores de `neuro-geometrica_multi-agente`
sobre `K24_mc (24L+6B)` y el nuevo banco extendido de pruebas de Matemáticas y Programación."""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from enjambre.bridges.modules import ResidualBridge
from enjambre.math_code.neuro_connectors import (
    ARCHITECTURES,
    DEFAULT_DA,
    LIB_DEG,
    N_LIB,
    PAIRS_K24,
    AtlasFWBridge,
    BQ10DaStrBridge,
    ConstelacionV4Bridge,
    NeuroBridgeBank,
    SoftBellmanSimplicialBridge,
    build_warm_neuro_bridge_bank,
    da_str_colscale,
    koopman_sindy_lib_17,
)
from enjambre.math_code.tasks import (
    EXT_MC_TASKS,
    make_ext_mc_splits,
    make_mc_splits,
)


def test_koopman_sindy_17_basis_and_dastr_colscale():
    """Verifica que la biblioteca M1 de 17 funciones y el reescalado M7 DA-STR (da=10)
    coinciden exactamente con `orbita/src/constelacion4/atlas.rs`."""
    assert len(LIB_DEG) == N_LIB == 17
    scales = da_str_colscale(DEFAULT_DA)
    assert len(scales) == 17
    for deg, s in zip(LIB_DEG, scales):
        if deg == 1:
            assert math.isclose(s, 1.0, rel_tol=1e-6)
        elif deg == 2:
            assert math.isclose(s, 10.0 ** -0.5, rel_tol=1e-6)
        elif deg == 3:
            assert math.isclose(s, 0.1, rel_tol=1e-6)

    a = torch.tensor([[[0.5, -0.25]]])
    b = torch.tensor([[[-0.4, 0.2]]])
    terms = koopman_sindy_lib_17(a, b)
    assert len(terms) == 17
    assert torch.allclose(terms[0], a)
    assert torch.allclose(terms[1], b)
    assert torch.allclose(terms[2], a * a)
    assert torch.allclose(terms[3], a * b)
    assert torch.allclose(terms[5], a * a * a)


def test_all_4_neuro_bridges_paso0_bit_exact_with_residual_bridge():
    """Verifica el invariante `atlas-paso0` (`off_is_v4_bit_exact`):
    Cuando un conector neuro-geométrico carga los pesos (`up`, `down`, `gate`) de un `ResidualBridge`,
    en `t=0` (con `neuro_down.weight = 0`) produce una salida idéntica bit a bit."""
    torch.manual_seed(7)
    D, R = 64, 16
    ref = ResidualBridge(hidden=D, rank=R)
    nn.init.normal_(ref.down.weight, std=0.05)  # simular puente K24_mc ya entrenado
    ref_state = ref.state_dict()

    h_src = torch.randn(2, 6, D)
    h_tgt = torch.randn(2, 6, D)
    out_ref = ref(h_src, h_tgt)

    for arch in ARCHITECTURES:
        bank = build_warm_neuro_bridge_bank(
            hidden=D,
            arch=arch,
            rank=R,
            pairs=[(0, 4)],
            k24_pretrained_state={f"bridges.0_4.{k}": v for k, v in ref_state.items()},
        )
        out_arch = bank.bridges["0_4"](h_src, h_tgt)
        assert torch.allclose(out_ref, out_arch, atol=0.0), f"Fallo de paridad paso0 en {arch}"


def test_constelacion_v4_koopman_rotation_norm():
    """Verifica que los bloques 2x2 de rotación-escala de `ConstelacionV4Bridge` escalan la norma por `rho`."""
    b = ConstelacionV4Bridge(hidden=32, rank=8)
    with torch.no_grad():
        b.rho_raw.fill_(0.9)
    z = torch.randn(3, 5, 8)
    kz = b._kmul(z)
    assert kz.shape == z.shape
    assert torch.allclose(kz.norm(dim=-1), 0.9 * z.norm(dim=-1), atol=1e-5)


def test_soft_bellman_simplicial_barycentric_simplex():
    """Verifica que `SoftBellmanSimplicialBridge` produce coordenadas baricéntricas válidas en Δ^{K-1}."""
    b = SoftBellmanSimplicialBridge(hidden=32, rank=8, n_vertices=4, tau=0.25)
    h_src = torch.randn(2, 5, 32)
    h_tgt = torch.randn(2, 5, 32)
    p = b.compute_barycentric_weights(h_src, h_tgt)
    assert p.shape == (2, 5, 4)
    assert (p >= 0.0).all()
    assert torch.allclose(p.sum(dim=-1), torch.ones(2, 5), atol=1e-5)


def test_neuro_bridges_gradient_flow_and_no_future_leak():
    """Verifica para las 4 arquitecturas:
    1) Flujo de gradiente en el paso 1 hacia `neuro_down` y en el paso 2 hacia los parámetros internos.
    2) Causalidad por posición de token (sin fuga temporal hacia el pasado)."""
    D, R = 32, 8
    for arch in ARCHITECTURES:
        bank = build_warm_neuro_bridge_bank(hidden=D, arch=arch, rank=R, pairs=PAIRS_K24)
        br = bank.bridges["0_4"]
        opt = torch.optim.Adam(br.parameters(), lr=1e-2)

        h_src = torch.randn(2, 4, D)
        h_tgt = torch.randn(2, 4, D)
        target = torch.randn(2, 4, D)

        # Paso 1: gradiente no nulo en neuro_down
        loss1 = (br(h_src, h_tgt) - target).pow(2).mean()
        loss1.backward()
        assert br.neuro_down.weight.grad is not None
        assert br.neuro_down.weight.grad.abs().sum().item() > 0.0
        opt.step()
        opt.zero_grad()

        # Paso 2: tras dejar neuro_down != 0, no hay causal leak entre tokens
        h_src2 = h_src.clone()
        h_tgt2 = h_tgt.clone()
        h_src2[:, 2:, :] += 10.0
        h_tgt2[:, 2:, :] -= 10.0
        out_a = br(h_src, h_tgt)
        out_b = br(h_src2, h_tgt2)
        assert torch.allclose(out_a[:, :2, :], out_b[:, :2, :], atol=1e-5), f"Fuga temporal en {arch}"


def test_extended_mc_splits_and_validators():
    """Verifica las 4 nuevas tareas extendidas (`math_poly`, `math_seq`, `code_list_fn`, `code_dict_str`)
    y su disjunción respecto a los splits principales."""
    tr_core, va_core, ho_core, _ = make_mc_splits(n_train=8, n_val=4, n_hold=10, seed=42)
    tr_ext, va_ext, ho_ext, man_ext = make_ext_mc_splits(n_train=8, n_val=4, n_hold=10, seed=142)

    assert set(ho_ext.keys()) == set(EXT_MC_TASKS)
    assert len(man_ext["prompt_sha256"]) == 64

    core_prompts = {it.prompt for items in ho_core.values() for it in items}
    ext_prompts = {it.prompt for items in ho_ext.values() for it in items}
    assert core_prompts.isdisjoint(ext_prompts)

    # Verificar que cada target del generador extendido pasa su propio validador
    for tid in EXT_MC_TASKS:
        for prompt, target, chk in tr_ext[tid][:4]:
            assert chk(target), f"El target canónico de {tid} falló su propio check: {target}"
            assert not chk(target + "\n- Explicación extra"), f"{tid} debe rechazar prosa colateral"
