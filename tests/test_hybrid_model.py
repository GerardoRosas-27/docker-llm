"""Pruebas para el modelo Ultra-Híbrido 3B + K66 Campo v4 en Obrador."""

import pytest
from starlette.testclient import TestClient

from app import config, db
from app.hybrid_server import (
    MODEL_ID,
    classify_prompt_domain,
    generate_hybrid_reply,
    app as hybrid_app,
)
from app.main import app, _preset


def test_classify_prompt_domain():
    # Tareas matemáticas
    assert classify_prompt_domain("Calcula gcd(48, 18)") == "math"
    assert classify_prompt_domain("Compute lcm(12, 15)") == "math"
    assert classify_prompt_domain("Solve for x: 3*x + 5 = 20") == "math"
    assert classify_prompt_domain("Compute combination C(10, 2)") == "math"

    # Tareas de código
    assert classify_prompt_domain("Escribe una función def reverse_string(s):") == "code"
    assert classify_prompt_domain("Implementa un loop con filter en Python") == "code"
    assert classify_prompt_domain("Calcula la frecuencia en un dict") == "code"
    assert classify_prompt_domain("Escribe un algoritmo con recursion") == "code"


def test_hybrid_server_reply_math_and_code():
    # Math: gcd
    math_reply = generate_hybrid_reply([{"role": "user", "content": "Compute gcd(48, 18)."}])
    assert "Base Qwen2.5-3B" in math_reply
    assert "Final Answer: 6" in math_reply

    # Math: linear solve
    lin_reply = generate_hybrid_reply([{"role": "user", "content": "Solve for x: 3*x + 5 = 20."}])
    assert "Final Answer: 5" in lin_reply

    # Code: Fibonacci
    code_reply = generate_hybrid_reply([{"role": "user", "content": "Escribe una función en Python para fibonacci"}])
    assert "K66 Macro-Neuronas" in code_reply
    assert "def fibonacci" in code_reply


def test_hybrid_server_endpoints():
    client = TestClient(hybrid_app)

    # Health
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["model"] == MODEL_ID
    assert health.json()["accuracy_global"] == "73.75%"

    # List models
    models = client.get("/v1/models")
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == MODEL_ID

    # Chat completion (no stream)
    chat = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Compute gcd(48, 18)."}]},
    )
    assert chat.status_code == 200
    data = chat.json()
    assert "choices" in data
    assert "Final Answer: 6" in data["choices"][0]["message"]["content"]

    # Chat completion (stream)
    stream_resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hola"}], "stream": True},
    )
    assert stream_resp.status_code == 200
    assert "text/event-stream" in stream_resp.headers["content-type"]
    text = stream_resp.text
    assert "data: " in text
    assert "[DONE]" in text


def test_preset_exposes_hybrid_winning_model():
    preset = _preset()
    assert preset["filename"] == config.HYBRID_FILE
    assert preset["label"] == config.HYBRID_LABEL
    assert "73.75%" in preset["note"]


def test_hybrid_registered_in_obrador(client):
    # Registrar el modelo híbrido
    db.ensure_hybrid_registered()
    models = client.get("/api/models").json()["models"]
    hybrid = next((m for m in models if m["slug"] == config.HYBRID_SLUG), None)
    assert hybrid is not None
    assert hybrid["status"] == "ready"
    assert hybrid["is_default"] is True
    assert hybrid["blocked"] is None
    assert hybrid["ram_warning"] is None


def test_end_to_end_chat_with_hybrid_runner(client):
    db.ensure_hybrid_registered()
    # Arrancar el modelo a través de la API de administración
    start_resp = client.post(f"/api/models/{config.HYBRID_SLUG}/start")
    assert start_resp.status_code == 200
    assert start_resp.json()["slug"] == config.HYBRID_SLUG

    # Ejecutar inferencia matemática completa a través de Obrador (/v1/models/{slug}/chat/completions)
    chat_resp = client.post(
        f"/v1/models/{config.HYBRID_SLUG}/chat/completions",
        json={"messages": [{"role": "user", "content": "Compute gcd(60, 24)."}]},
    )
    assert chat_resp.status_code == 200
    data = chat_resp.json()
    assert "choices" in data
    content = data["choices"][0]["message"]["content"]
    assert "Base Qwen2.5-3B" in content
    assert "Final Answer: 12" in content

    # Ejecutar inferencia de código completa a través de Obrador
    code_resp = client.post(
        f"/v1/models/{config.HYBRID_SLUG}/chat/completions",
        json={"messages": [{"role": "user", "content": "def fibonacci(n): in python"}]},
    )
    assert code_resp.status_code == 200
    code_content = code_resp.json()["choices"][0]["message"]["content"]
    assert "K66 Macro-Neuronas" in code_content
    assert "def fibonacci" in code_content

    # Inferencia en streaming a través del proxy de Obrador
    stream_resp = client.post(
        f"/v1/models/{config.HYBRID_SLUG}/chat/completions",
        json={"messages": [{"role": "user", "content": "Hola"}], "stream": True},
    )
    assert stream_resp.status_code == 200
    assert "text/event-stream" in stream_resp.headers["content-type"]
    assert "Ultra-Híbrido" in stream_resp.text
    assert "[DONE]" in stream_resp.text

    # Detener el modelo limpiamente
    stop_resp = client.post(f"/api/models/{config.HYBRID_SLUG}/stop")
    assert stop_resp.status_code == 200
