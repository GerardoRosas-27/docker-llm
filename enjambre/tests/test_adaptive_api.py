"""API /adaptive con harness mock (sin cargar HF). No afirma smoke Docker."""
from __future__ import annotations
from fastapi.testclient import TestClient

from enjambre import api as api_mod


class _Fake:
    def status(self):
        return {"model": "fake", "router": "heuristic_keywords (not LLM reasoning)", "bank": {"n_versions": 0}}

    def catalog(self):
        from enjambre.adaptive.catalog import catalog_manifest
        return catalog_manifest()

    def list_experts(self):
        return []

    def route(self, prompt):
        return {"task_id": None, "method": "keyword", "fallback": "base"}

    def solve(self, prompt, task_id=None, use_expert=True, max_new=32):
        return {"task_id": task_id, "output": "ALPHA", "ok": None, "used_expert": False}

    def auto_improve(self, task_ids=None, episodes=1, max_tasks=None):
        return {"n_tasks": 0, "episodes": episodes, "accepted": 0, "rejected": 0, "results": []}

    def improve_history(self, task_id=None, limit=50):
        return []


def test_adaptive_routes_with_fake(monkeypatch):
    monkeypatch.setattr(api_mod, "_swarm", None)
    from enjambre.adaptive import api_routes
    monkeypatch.setattr(api_routes, "_h", _Fake())
    client = TestClient(api_mod.app)
    r = client.get("/adaptive/status")
    assert r.status_code == 200
    assert "heuristic" in r.json()["router"]
    r = client.get("/adaptive/catalog")
    assert r.status_code == 200 and r.json()["n_tasks"] == 50
    r = client.post("/adaptive/solve", json={"prompt": "hi"})
    assert r.status_code == 200 and r.json()["output"] == "ALPHA"
    r = client.post("/adaptive/auto_improve", json={"episodes": 1, "max_tasks": 1})
    assert r.status_code == 200 and "accepted" in r.json()
