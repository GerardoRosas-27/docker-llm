"""Pruebas para los endpoints metacognitivos bidireccionales en app.hybrid_server."""
from __future__ import annotations

from starlette.testclient import TestClient

from app.hybrid_server import app as hybrid_app


def test_metacognition_status_endpoint():
    client = TestClient(hybrid_app)
    resp = client.get("/api/metacognition/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "Multi-Agente Metacognitiva Bidireccional" in data["architecture"]
    assert "S1_principal" in data["systems"]
    assert "S2_observer" in data["systems"]
    assert "artifacts" in data


def test_metacognition_realtime_observe_clean_and_perturbed():
    client = TestClient(hybrid_app)

    clean = client.post(
        "/api/metacognition/observe",
        json={
            "prompt": "Compute gcd(48, 18).",
            "s1_layer_idx": 48,
            "s2_layer_idx": 13,
            "simulate_s1_perturbation": False,
            "simulate_s2_probe_perturbation": False,
            "execute_primary_task": True,
        },
    )
    assert clean.status_code == 200
    c_data = clean.json()
    assert c_data["mode"] == "realtime_bidirectional"
    assert c_data["primary_execution"]["domain"] == "math"
    assert "Final Answer: 6" in c_data["primary_execution"]["reply"]

    s2_clean = c_data["s2_observes_s1"]
    s1_clean = c_data["s1_observes_s2"]
    assert s2_clean["predicted"]["is_perturbed_pred"] is False
    assert s1_clean["predicted"]["is_perturbed_pred"] is False
    assert "Rol:" in s2_clean["generated_description"]
    assert "Rol:" in s1_clean["generated_description"]
    assert s1_clean["ground_truth"]["probe_frob_norm"] > 0.0

    # Contrafactual: perturbar capa de S1 y sonda de S2 simultáneamente
    pert = client.post(
        "/api/metacognition/observe",
        json={
            "prompt": "Explain self-attention in transformers.",
            "s1_layer_idx": 48,
            "s2_layer_idx": 13,
            "simulate_s1_perturbation": True,
            "simulate_s2_probe_perturbation": True,
            "execute_primary_task": True,
        },
    )
    assert pert.status_code == 200
    p_data = pert.json()
    assert p_data["primary_execution"]["domain"] == "general"
    assert p_data["s2_observes_s1"]["predicted"]["is_perturbed_pred"] is True
    assert p_data["s1_observes_s2"]["predicted"]["is_perturbed_pred"] is True
    assert p_data["s2_observes_s1"]["anomaly_score"] > s2_clean["anomaly_score"]
    assert p_data["s1_observes_s2"]["anomaly_score"] > s1_clean["anomaly_score"]
    assert "editada_perturbada" in p_data["s2_observes_s1"]["generated_description"]
    assert "editada_perturbada" in p_data["s1_observes_s2"]["generated_description"]


def test_metacognition_episodic_batch_endpoint():
    client = TestClient(hybrid_app)

    # Lote por defecto de 12 episodios cruzando K66 y K32
    batch_default = client.post("/api/metacognition/episode-batch", json={})
    assert batch_default.status_code == 200
    bd = batch_default.json()
    assert bd["mode"] == "episodic_batch"
    assert bd["batch_size"] == 12
    assert len(bd["episodes"]) == 12
    assert bd["s2_to_s1_role_acc"] >= 0.75
    assert bd["s1_to_s2_role_acc"] >= 0.75

    # Lote personalizado mixto sano + perturbado
    batch_mixed = client.post(
        "/api/metacognition/episode-batch",
        json={
            "episodes": [
                {"prompt": "ep0", "s1_layer_idx": 10, "s2_layer_idx": 13, "simulate_s1_perturbation": False},
                {"prompt": "ep1", "s1_layer_idx": 48, "s2_layer_idx": 19, "simulate_s1_perturbation": True, "simulate_s2_probe_perturbation": True},
                {"prompt": "ep2", "s1_layer_idx": 62, "s2_layer_idx": 24, "simulate_s1_perturbation": False},
            ],
            "execute_primary_tasks": True,
        },
    )
    assert batch_mixed.status_code == 200
    bm = batch_mixed.json()
    assert bm["batch_size"] == 3
    assert bm["episodes"][1]["s2_observes_s1"]["predicted"]["is_perturbed_pred"] is True
    assert bm["episodes"][1]["s1_observes_s2"]["predicted"]["is_perturbed_pred"] is True
    assert bm["episodes"][0]["primary_execution"] is not None
