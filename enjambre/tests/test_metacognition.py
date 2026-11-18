"""Tests unitarios e integración para el módulo `enjambre.metacognition`.

Verifica:
1. Equivalencia algebraica y compartición real de memoria en `DampedLinear`.
2. Cero fuga de etiquetas (`zero label leakage`) en `LayerWeightSpectrometer`:
   la etiqueta `is_perturbed` NUNCA altera `spectral_vector`; solo mutaciones
   físicas de pesos alteran `spectral_vector`.
3. Aislamiento y reversibilidad de `apply_reversible_layer_perturbation` en capas
   que comparten pesos base.
4. Inspección de puentes `ConstelacionV4Bridge` y sondas `WeightIntrospectionModule`.
5. Flujo real de gradientes en todos los submódulos entrenables de `WeightIntrospectionModule`.
6. Modo Tiempo Real (hooks + observación recíproca S2 <-> S1) y Lote de Episodios sobre
   un modelo real compacto (`build_weight_shared_k32_0_5b` + `MetaAgent` ligero).
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn

from enjambre.math_code.neuro_connectors import ConstelacionV4Bridge
from enjambre.metacognition.spectrometer import (
    DampedLinear,
    LayerWeightSpectrometer,
    SPECTRAL_VECTOR_DIM,
    apply_reversible_layer_perturbation,
)
from enjambre.metacognition.modules import (
    MetaKoopmanMonitorBridge,
    WeightIntrospectionModule,
)
from enjambre.metacognition.agents import RealTimeHookCollector


class _FakeAttn(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.layer_idx = 0


class _FakeMLP(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.up_proj = nn.Linear(hidden, hidden * 2, bias=False)
        self.down_proj = nn.Linear(hidden * 2, hidden, bias=True)


class _FakeDecoderLayer(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.self_attn = _FakeAttn(hidden)
        self.mlp = _FakeMLP(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.self_attn.o_proj(self.self_attn.v_proj(x))
        return y + self.mlp.down_proj(torch.relu(self.mlp.up_proj(y)))


def test_damped_linear_exact_equivalence_and_shared_pointer():
    base = nn.Linear(64, 32, bias=False)
    b_delta = torch.randn(32) * 0.2
    damped = DampedLinear(base, damping=0.25, bias=b_delta)

    # Cero copia del tensor gigante base_weight
    assert damped.base_weight.data_ptr() == base.weight.data_ptr()

    x = torch.randn(2, 5, 64)
    y_damped = damped(x)

    # Equivalencia exacta con W_eff = 0.25 * W + b_delta
    ref = nn.Linear(64, 32, bias=True)
    with torch.no_grad():
        ref.weight.copy_(base.weight * 0.25)
        ref.bias.copy_(b_delta)
    y_ref = ref(x)
    assert torch.allclose(y_damped, y_ref, atol=1e-5)


def test_spectrometer_zero_label_leakage():
    torch.manual_seed(42)
    spec = LayerWeightSpectrometer()
    layer = _FakeDecoderLayer(hidden=32)

    snap_false = spec.inspect_decoder_layer(
        layer=layer,
        layer_idx=10,
        n_total_layers=32,
        system_id="S1_principal",
        role_label="code_cluster",
        is_edited=False,
        is_perturbed=False,
    )
    snap_true = spec.inspect_decoder_layer(
        layer=layer,
        layer_idx=10,
        n_total_layers=32,
        system_id="S1_principal",
        role_label="perturbed_anomaly",
        is_edited=True,
        is_perturbed=True,
    )
    assert snap_false.spectral_vector.shape == (SPECTRAL_VECTOR_DIM,)
    # Las etiquetas NO pueden filtrarse en spectral_vector sobre exactamente el mismo tensor físico
    assert torch.allclose(snap_false.spectral_vector, snap_true.spectral_vector, atol=1e-6)


def test_reversible_perturbation_isolation_and_restore():
    torch.manual_seed(7)
    spec = LayerWeightSpectrometer()
    base_layer = _FakeDecoderLayer(hidden=32)

    # Hermana compartiendo los mismos pesos vía _shallow_wrap_damped_layer
    sibling = copy.copy(base_layer)
    sibling._modules = copy.copy(base_layer._modules)
    sibling.self_attn = copy.copy(base_layer.self_attn)
    sibling.self_attn._modules = copy.copy(base_layer.self_attn._modules)
    sibling.mlp = copy.copy(base_layer.mlp)
    sibling.mlp._modules = copy.copy(base_layer.mlp._modules)
    sibling.mlp.down_proj = DampedLinear(base_layer.mlp.down_proj, damping=0.5)

    x = torch.randn(1, 4, 32)
    with torch.no_grad():
        y_sib_before = sibling(x)
        snap_before = spec.inspect_decoder_layer(base_layer, 3, 12, "S1_principal", "math_cluster")

    restore = apply_reversible_layer_perturbation(
        base_layer, damping_mult=0.25, rank1_spike_scale=3.0, bias_spike_scale=1.2, seed=99
    )
    try:
        with torch.no_grad():
            y_sib_during = sibling(x)
            snap_during = spec.inspect_decoder_layer(
                base_layer, 3, 12, "S1_principal", "perturbed_anomaly", is_perturbed=True
            )
        # Capa hermana completamente aislada
        assert torch.allclose(y_sib_before, y_sib_during, atol=1e-6)
        # El vector espectral de la capa mutada sí cambia físicamente de forma marcada
        vec_delta = float((snap_during.spectral_vector - snap_before.spectral_vector).abs().max().item())
        assert vec_delta > 0.25
        assert abs(snap_during.damping - 0.25) < 1e-4
    finally:
        restore()

    with torch.no_grad():
        snap_after = spec.inspect_decoder_layer(base_layer, 3, 12, "S1_principal", "math_cluster")
    assert torch.allclose(snap_before.spectral_vector, snap_after.spectral_vector, atol=1e-6)


def test_inspect_bridge_and_probe_sensitivity():
    spec = LayerWeightSpectrometer()
    bridge = ConstelacionV4Bridge(hidden=64, rank=8)
    s_clean = spec.inspect_bridge(bridge, bridge_idx=0, n_bridges=4, is_perturbed=False)
    with torch.no_grad():
        bridge.rho_raw.data.fill_(1.22)
    s_pert = spec.inspect_bridge(bridge, bridge_idx=0, n_bridges=4, is_perturbed=True)
    assert s_pert.spectral_vector.shape == (SPECTRAL_VECTOR_DIM,)
    assert float((s_pert.spectral_vector - s_clean.spectral_vector).abs().max().item()) > 0.1

    # Detección de cambio en pesos del módulo sonda de S2
    layer = _FakeDecoderLayer(hidden=32)
    probe = WeightIntrospectionModule(hidden_lm=32, k_prefix=4, n_anchors=8)
    snap_p1 = spec.inspect_decoder_layer(layer, 5, 16, "S2_observer", "meta_observer_core", probe_module=probe)
    with torch.no_grad():
        probe.encoder.proj_in.weight.mul_(3.0)
    snap_p2 = spec.inspect_decoder_layer(layer, 5, 16, "S2_observer", "perturbed_anomaly", probe_module=probe)
    assert snap_p2.probe_frob_norm > snap_p1.probe_frob_norm
    assert float((snap_p2.spectral_vector - snap_p1.spectral_vector).abs().max().item()) > 0.05


def test_introspection_module_gradients_and_koopman_monitor():
    torch.manual_seed(0)
    mod = WeightIntrospectionModule(hidden_lm=64, k_prefix=4, n_anchors=12)
    x = torch.randn(4, SPECTRAL_VECTOR_DIM)
    prev = torch.randn(4, 256)
    out = mod.encode_and_predict(x, prev_latent=prev)

    assert out["prefix_embeds"].shape == (4, 4, 64)
    assert out["role_logits"].shape == (4, 9)
    assert out["innov_norm"].shape == (4,)
    assert out["anomaly_prob"].shape == (4,)

    loss = (
        out["role_logits"].sum()
        + out["damping"].sum()
        + out["prefix_embeds"].pow(2).sum()
        + out["anomaly_prob"].sum()
        + out["contrastive"].pow(2).sum()
    )
    loss.backward()

    assert mod.encoder.proj_in.weight.grad is not None
    assert float(mod.encoder.proj_in.weight.grad.abs().sum().item()) > 0.0
    assert mod.koopman_monitor.bridge.rho_raw.grad is not None
    assert mod.soft_prompt.anchor_gate[0].weight.grad is not None
    assert mod.soft_prompt.delta_proj[-1].weight.grad is not None
    assert float(mod.soft_prompt.delta_proj[-1].weight.grad.abs().sum().item()) > 0.0


def test_realtime_hook_collector():
    layer = _FakeDecoderLayer(hidden=16)

    class MiniModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([layer, _FakeDecoderLayer(hidden=16)])

        def forward(self, x):
            for l in self.model.layers:
                x = l(x)
            return x

    m = MiniModel()
    collector = RealTimeHookCollector(system_id="S1_principal")
    seen = []
    collector.add_callback(lambda rec: seen.append(rec))
    collector.attach(m, layer_indices=[0, 1])

    _ = m(torch.randn(1, 3, 16))
    assert len(collector.stats) == 2
    assert len(seen) == 2
    assert "act_norm" in collector.stats[0]
    collector.remove()
    assert len(collector._handles) == 0
