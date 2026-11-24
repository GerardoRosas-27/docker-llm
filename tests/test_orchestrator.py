"""Pruebas completas del modo Orquestador por objetivos, sandbox compartido y verificador."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from app import config, db, orchestrator
from app.sandbox import SharedWorkspace
from tests.conftest import TEST_SECRET


def _register_fake_ready(tmp_path: Path, repo: str, filename: str) -> str:
    p = tmp_path / filename
    p.write_bytes(b"GGUF-TEST")
    rec = db.upsert_download(repo, filename, 1_000_000)
    db.mark_downloading(rec["slug"])
    db.mark_ready(rec["slug"], str(p), 1_000_000)
    return rec["slug"]


def _wait_status(client, run_id: str, wanted: set[str], timeout: float = 12.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        resp = client.get(f"/api/orchestrator/runs/{run_id}")
        assert resp.status_code == 200, resp.text
        last = resp.json()
        if last["run"]["status"] in wanted:
            return last
        time.sleep(0.08)
    raise AssertionError(f"La misión {run_id} no alcanzó {wanted} a tiempo; último estado: {last['run'] if last else None}")


# ------------------------------------------------------------------ Seguridad y acceso

def test_orquestador_exige_sesion_y_falla_cerrado_sin_master_secret(anon, monkeypatch):
    # Sin sesión -> 401
    assert anon.get("/api/orchestrator/config").status_code == 401
    assert anon.get("/api/orchestrator/runs").status_code == 401
    assert anon.post("/api/orchestrator/runs", json={"objective": "prueba"}).status_code == 401

    # API key (obk1) no puede administrar el orquestador -> 401
    session = anon.post("/api/auth/login", json={"secret": TEST_SECRET}).json()["token"]
    key = anon.post(
        "/api/keys",
        json={"label": "app"},
        headers={"Authorization": f"Bearer {session}"},
    ).json()["key"]
    assert anon.get("/api/orchestrator/config", headers={"Authorization": f"Bearer {key}"}).status_code == 401
    assert anon.get("/api/orchestrator/runs", headers={"Authorization": f"Bearer {key}"}).status_code == 401

    # Sin MASTER_SECRET -> 503 fail-closed
    monkeypatch.delenv("MASTER_SECRET", raising=False)
    r = anon.get("/api/orchestrator/config", headers={"Authorization": f"Bearer {session}"})
    assert r.status_code == 503
    assert r.json()["code"] == "master_secret_not_configured"


# ------------------------------------------------------------------ Sandbox y espacio compartido

def test_sandbox_aislamiento_secretos_timeout_y_resultados(tmp_path, monkeypatch):
    monkeypatch.setenv("MASTER_SECRET", "super-secret-value-never-leak")
    monkeypatch.setenv("HF_TOKEN", "hf_secret_never_leak")

    ws = SharedWorkspace(tmp_path / "run-sandbox")
    ws.initialize(
        "Probar ejecución aislada",
        ["No filtrar secretos", "Guardar archivo en results/"],
        {
            "orchestrator_model": "orch",
            "verifier_model": "ver",
            "workers": [{"id": "worker-1", "name": "Dev", "model": "w1", "role": "código"}],
            "max_rounds": 3,
            "execution_enabled": True,
        },
    )
    assert ws.notes_path.is_file()
    assert "Probar ejecución aislada" in ws.read_notes_full()

    # 1. Ejecución limpia que lee NOTES.md, verifica ausencia de secretos y escribe en ORCH_RESULTS_DIR
    code = (
        "import json, os, pathlib\n"
        "assert 'MASTER_SECRET' not in os.environ, 'Fuga de MASTER_SECRET!'\n"
        "assert 'HF_TOKEN' not in os.environ, 'Fuga de HF_TOKEN!'\n"
        "notes = pathlib.Path(os.environ['ORCH_NOTES_FILE']).read_text(encoding='utf-8')\n"
        "assert 'Probar ejecución aislada' in notes\n"
        "out = pathlib.Path(os.environ['ORCH_RESULTS_DIR']) / 'salida_sandbox.json'\n"
        "out.write_text(json.dumps({'valor': 42}), encoding='utf-8')\n"
        "print('OK_SANDBOX_42')\n"
    )
    res = ws.execute_python(code, step_tag="r1_worker1", timeout_sec=5, author="Dev", round_num=1)
    assert res["ok"] is True
    assert res["exit_code"] == 0
    assert "OK_SANDBOX_42" in res["stdout"]
    assert "salida_sandbox.json" in res["created_files"]

    listed = {item["filename"]: item for item in ws.list_results()}
    assert "salida_sandbox.json" in listed
    assert res["transcript_file"] in listed
    assert "INDEX.md" in listed
    text, _meta = ws.read_result("salida_sandbox.json")
    assert json.loads(text)["valor"] == 42

    # 2. Timeout mata procesos colgados
    slow = ws.execute_python("import time; time.sleep(10)", step_tag="r1_slow", timeout_sec=2, author="Dev", round_num=1)
    assert slow["ok"] is False
    assert slow["timed_out"] is True
    assert slow["exit_code"] == 124

    # 3. Path traversal rechazado
    with pytest.raises(ValueError):
        ws.read_result("../../etc/passwd")


# ------------------------------------------------------------------ Configuración por defecto

def test_configuracion_por_defecto_orquestador_workers_y_verificador(client, tmp_path):
    m_orch = _register_fake_ready(tmp_path, "org/orch-GGUF", "orch-Q4_K_M.gguf")
    m_code = _register_fake_ready(tmp_path, "org/code-GGUF", "code-Q4_K_M.gguf")
    m_rev = _register_fake_ready(tmp_path, "org/rev-GGUF", "rev-Q4_K_M.gguf")
    m_ver = _register_fake_ready(tmp_path, "org/ver-GGUF", "ver-Q4_K_M.gguf")

    initial = client.get("/api/orchestrator/config")
    assert initial.status_code == 200
    assert initial.json()["saved"] is False
    assert len(initial.json()["available_models"]) == 4

    # Guardar configuración personalizada
    saved = client.put(
        "/api/orchestrator/config",
        json={
            "orchestrator_model": m_orch,
            "verifier_model": m_ver,
            "workers": [
                {"id": "dev", "name": "Especialista Código", "role": "Escribir y ejecutar Python", "model": m_code},
                {"id": "qa", "name": "Revisor QA", "role": "Verificar outputs de dev en results/", "model": m_rev},
            ],
            "pacing_mode": "step",
            "pace_seconds": 0.5,
            "max_rounds": 5,
            "execution_enabled": True,
            "execution_timeout": 12,
            "temperature": 0.3,
            "max_tokens": 512,
        },
    )
    assert saved.status_code == 200, saved.text
    body = saved.json()
    assert body["saved"] is True
    assert body["config"]["orchestrator_model"] == m_orch
    assert body["config"]["verifier_model"] == m_ver
    assert [w["model"] for w in body["config"]["workers"]] == [m_code, m_rev]
    assert body["config"]["pacing_mode"] == "step"
    assert body["config"]["max_rounds"] == 5

    # Rechaza modelo inexistente
    bad = client.put(
        "/api/orchestrator/config",
        json={
            "orchestrator_model": "modelo-fantasma",
            "verifier_model": m_ver,
            "workers": [{"name": "Dev", "role": "code", "model": m_code}],
        },
    )
    assert bad.status_code == 400


# ------------------------------------------------------------------ Flujo completo multi-ronda + colaboradores

def test_orquestacion_multimodelo_con_notas_sandbox_resultados_y_verificador(client, tmp_path, monkeypatch):
    m_orch = _register_fake_ready(tmp_path, "org/orch-GGUF", "orch-Q4_K_M.gguf")
    m_w1 = _register_fake_ready(tmp_path, "org/w1-GGUF", "worker1-Q4_K_M.gguf")
    m_w2 = _register_fake_ready(tmp_path, "org/w2-GGUF", "worker2-Q4_K_M.gguf")
    m_ver = _register_fake_ready(tmp_path, "org/ver-GGUF", "verifier-Q4_K_M.gguf")

    client.put(
        "/api/orchestrator/config",
        json={
            "orchestrator_model": m_orch,
            "verifier_model": m_ver,
            "workers": [
                {"id": "w1", "name": "Calculador", "role": "Ejecutar cálculo y dejar JSON", "model": m_w1},
                {"id": "w2", "name": "Auditor", "role": "Leer JSON de w1 y validar", "model": m_w2},
            ],
            "pacing_mode": "auto",
            "pace_seconds": 0.0,
            "max_rounds": 4,
            "execution_enabled": True,
            "execution_timeout": 10,
        },
    )

    calls: list[tuple[str, str]] = []

    async def fake_call_model(slug, messages, temperature=0.4, max_tokens=768):
        sys_txt = messages[0]["content"]
        usr_txt = messages[1]["content"]
        calls.append((slug, sys_txt.splitlines()[0]))

        if "[OBRADOR_ROLE:ORCHESTRATOR]" in sys_txt:
            assert slug == m_orch
            r = 1 if "RONDA ACTUAL: 1" in usr_txt else 2
            return (
                f"<plan>Plan de ronda {r}: producir datos en w1 y auditarlos en w2.</plan>\n"
                f"<note>Orquestador marca el ritmo de ronda {r}.</note>\n"
                f"<pace>{'continue' if r == 1 else 'converge'}</pace>\n"
                f'<dispatch worker="w1">Calcula suma de primos menores de 20 y guarda metricas_r{r}.json en ORCH_RESULTS_DIR.</dispatch>\n'
                f'<dispatch worker="w2">Lee metricas_r{r}.json de ORCH_RESULTS_DIR y verifica el valor.</dispatch>'
            )

        if "[OBRADOR_ROLE:WORKER]" in sys_txt and slug == m_w1:
            r = 1 if "RONDA: 1" in usr_txt else 2
            return (
                f"<note>Calculador terminó ronda {r} y dejó metricas_r{r}.json.</note>\n"
                '<exec lang="python">\n'
                "import json, os, pathlib\n"
                "primes = [2, 3, 5, 7, 11, 13, 17, 19]\n"
                f"out = pathlib.Path(os.environ['ORCH_RESULTS_DIR']) / 'metricas_r{r}.json'\n"
                "out.write_text(json.dumps({'primes': primes, 'total': sum(primes)}), encoding='utf-8')\n"
                "print('SUM_PRIMES=', sum(primes))\n"
                "</exec>\n"
                f'<result file="calculo_r{r}.md">Suma calculada: 77 (primos menores de 20).</result>'
            )

        if "[OBRADOR_ROLE:WORKER]" in sys_txt and slug == m_w2:
            # Worker 2 DEBE ver en su prompt lo que Worker 1 acaba de escribir en NOTES.md y results/
            assert "Calculador terminó ronda" in usr_txt
            r = 1 if "RONDA: 1" in usr_txt else 2
            return (
                f"<note>Auditor leyó metricas_r{r}.json y confirmó total == 77.</note>\n"
                '<exec lang="python">\n'
                "import json, os, pathlib\n"
                f"data = json.loads((pathlib.Path(os.environ['ORCH_RESULTS_DIR']) / 'metricas_r{r}.json').read_text())\n"
                "assert data['total'] == 77, f'Esperaba 77, obtuve {data}'\n"
                "print('AUDIT_PASSED_77')\n"
                "</exec>\n"
                f'<result file="auditoria_r{r}.md">Auditoría exitosa: suma verificada = 77.</result>'
            )

        if "[OBRADOR_ROLE:VERIFIER]" in sys_txt:
            assert slug == m_ver
            if "RONDA EVALUADA: 1" in usr_txt:
                return (
                    "<verdict>in_progress</verdict>\n"
                    "<progress>60</progress>\n"
                    "<summary>Ronda 1 generó y auditó 77; falta ronda de confirmación final.</summary>\n"
                    "<criteria>\n- [PASS] Cálculo 77 ejecutado\n- [FAIL] Confirmación en ronda 2\n</criteria>\n"
                    "<feedback>Ejecuta una segunda ronda de confirmación para cerrar.</feedback>"
                )
            assert "AUDIT_PASSED_77" in usr_txt
            return (
                "<verdict>achieved</verdict>\n"
                "<progress>100</progress>\n"
                "<summary>Objetivo cumplido al 100% con evidencia en sandbox y results/.</summary>\n"
                "<criteria>\n- [PASS] Cálculo 77 ejecutado\n- [PASS] Confirmación en ronda 2\n</criteria>\n"
                "<feedback>Listo para consolidar FINAL_RESULT.md.</feedback>"
            )

        if "[OBRADOR_ROLE:SYNTHESIZER]" in sys_txt:
            assert slug == m_orch
            return "# Resultado Final\n\nLa suma de los primos menores de 20 es **77**, ejecutada y auditada."

        raise AssertionError(f"Rol inesperado: {sys_txt}")

    monkeypatch.setattr(orchestrator, "call_model", fake_call_model)

    created = client.post(
        "/api/orchestrator/runs",
        json={
            "objective": "Calcular y auditar la suma de números primos menores de 20",
            "criteria": ["Cálculo 77 ejecutado", "Confirmación en ronda 2"],
        },
    )
    assert created.status_code == 200, created.text
    run_id = created.json()["run"]["id"]

    final = _wait_status(client, run_id, {"completed", "error", "max_rounds"})
    assert final["run"]["status"] == "completed"
    assert final["run"]["current_round"] == 2
    assert final["run"]["progress_pct"] == 100
    assert "77" in (final["run"]["final_result"] or "")

    # Verificar NOTES.md con notas de todos los participantes
    notes = final["notes"]
    assert "Orquestador" in notes
    assert "Calculador" in notes
    assert "Auditor" in notes
    assert "Verificador" in notes

    # Verificar archivos compartidos en results/
    filenames = {item["filename"] for item in final["results"]}
    assert "metricas_r1.json" in filenames
    assert "metricas_r2.json" in filenames
    assert "calculo_r1.md" in filenames
    assert "auditoria_r2.md" in filenames
    assert "verificacion_ronda_1.md" in filenames
    assert "verificacion_ronda_2.md" in filenames
    assert "FINAL_RESULT.md" in filenames
    assert "INDEX.md" in filenames

    # Lectura de archivo y protección traversal en endpoint HTTP
    f_resp = client.get(f"/api/orchestrator/runs/{run_id}/files/FINAL_RESULT.md")
    assert f_resp.status_code == 200
    assert "77" in f_resp.json()["content"]
    assert client.get(f"/api/orchestrator/runs/{run_id}/files/..%2F..%2Fetc%2Fpasswd").status_code == 400

    # SSE stream en misión ya terminada emite snapshot y [DONE]
    sse = client.get(f"/api/orchestrator/runs/{run_id}/stream")
    assert sse.status_code == 200
    assert '"type": "snapshot"' in sse.text
    assert "data: [DONE]" in sse.text

    # Verificar que efectivamente se llamó a los 4 modelos configurados en sus roles
    models_called = {slug for slug, _role in calls}
    assert models_called == {m_orch, m_w1, m_w2, m_ver}


# ------------------------------------------------------------------ Ritmo paso a paso, nota humana y archivo usuario

def test_ritmo_paso_a_paso_nota_humana_y_archivo_compartido(client, tmp_path, monkeypatch):
    m = _register_fake_ready(tmp_path, "org/m-GGUF", "m-Q4_K_M.gguf")
    client.put(
        "/api/orchestrator/config",
        json={
            "orchestrator_model": m,
            "verifier_model": m,
            "workers": [{"id": "w1", "name": "Agente 1", "role": "resolver", "model": m}],
            "pacing_mode": "step",
            "pace_seconds": 0.0,
            "max_rounds": 3,
            "execution_enabled": False,
        },
    )

    seen_human_note = {"ok": False}

    async def step_model(slug, messages, temperature=0.4, max_tokens=768):
        sys_txt = messages[0]["content"]
        usr_txt = messages[1]["content"]
        if "USA_CLAVE_SECRETA_XYZ" in usr_txt:
            seen_human_note["ok"] = True
        if "[OBRADOR_ROLE:ORCHESTRATOR]" in sys_txt:
            return '<plan>Avanzar ronda</plan><note>Nota orq</note><pace>continue</pace><dispatch worker="w1">Trabaja</dispatch>'
        if "[OBRADOR_ROLE:WORKER]" in sys_txt:
            return '<note>Worker entregó resultado</note><result file="paso.md">Contenido parcial</result>'
        if "[OBRADOR_ROLE:VERIFIER]" in sys_txt:
            if "RONDA EVALUADA: 1" in usr_txt:
                return "<verdict>in_progress</verdict><progress>50</progress><summary>Falta paso 2</summary><feedback>Seguir</feedback>"
            return "<verdict>achieved</verdict><progress>100</progress><summary>Completado tras nota humana</summary><feedback>OK</feedback>"
        return "# Final OK"

    monkeypatch.setattr(orchestrator, "call_model", step_model)

    created = client.post("/api/orchestrator/runs", json={"objective": "Probar ritmo manual"})
    run_id = created.json()["run"]["id"]

    # En modo paso a paso, tras la ronda 1 debe quedar pausada esperando al usuario
    paused = _wait_status(client, run_id, {"paused"})
    assert paused["run"]["current_round"] == 1

    # El humano deja una nota en NOTES.md y un archivo en results/
    note_resp = client.post(
        f"/api/orchestrator/runs/{run_id}/notes",
        json={"text": "Prioridad del operador: USA_CLAVE_SECRETA_XYZ", "author": "Gerardo"},
    )
    assert note_resp.status_code == 200
    assert "USA_CLAVE_SECRETA_XYZ" in note_resp.json()["notes"]

    file_resp = client.post(
        f"/api/orchestrator/runs/{run_id}/files",
        json={"filename": "entrada.txt", "content": "dato=123", "summary": "Archivo de entrada"},
    )
    assert file_resp.status_code == 200
    assert any(f["filename"] == "entrada.txt" for f in file_resp.json()["results"])

    # Avanzar 1 ronda manualmente -> debe leer la nota humana y completar
    step_resp = client.post(f"/api/orchestrator/runs/{run_id}/step")
    assert step_resp.status_code == 200
    done = _wait_status(client, run_id, {"completed"})
    assert done["run"]["current_round"] == 2
    assert seen_human_note["ok"] is True

    # Borrar misión limpia disco y DB
    ws_dir = Path(done["run"]["workspace_dir"])
    assert ws_dir.exists()
    deleted = client.delete(f"/api/orchestrator/runs/{run_id}")
    assert deleted.status_code == 200
    assert not ws_dir.exists()
    assert client.get(f"/api/orchestrator/runs/{run_id}").status_code == 404


# ------------------------------------------------------------------ End-to-end real con Ultra-Híbrido + UI

def test_orquestador_e2e_con_modelo_hibrido_y_ui(client):
    # Verificar que la interfaz sirve la pestaña y el script del orquestador
    html = client.get("/")
    assert html.status_code == 200
    assert "Orquestador" in html.text
    assert "view-orchestrator" in html.text
    assert "/static/orchestrator.js" in html.text
    js = client.get("/static/orchestrator.js")
    assert js.status_code == 200
    assert "ObradorOrch" in js.text

    # Registrar el modelo híbrido por defecto y ejecutar una misión real E2E (sin mocks!)
    db.ensure_hybrid_registered()
    cfg = client.get("/api/orchestrator/config")
    assert cfg.status_code == 200
    assert cfg.json()["config"]["orchestrator_model"] == config.HYBRID_SLUG

    created = client.post(
        "/api/orchestrator/runs",
        json={
            "objective": "Calcular gcd(60, 24) en Python y validar los resultados en results/",
            "criteria": [
                "Ejecución Python exitosa en sandbox",
                "Resultados compartidos en results/",
            ],
            "pace_seconds": 0.0,
            "max_rounds": 3,
        },
    )
    assert created.status_code == 200, created.text
    run_id = created.json()["run"]["id"]

    try:
        done = _wait_status(client, run_id, {"completed", "error", "max_rounds"}, timeout=25.0)
        assert done["run"]["status"] == "completed", done["run"]
        assert done["run"]["progress_pct"] == 100
        filenames = {f["filename"] for f in done["results"]}
        assert "FINAL_RESULT.md" in filenames
        assert any(name.startswith("datos_") and name.endswith(".json") for name in filenames)
        assert any(name.startswith("validacion_cruzada_") for name in filenames)
        assert any(ev["kind"] == "execution" and ev["metadata"].get("ok") for ev in done["events"])
        assert any(ev["kind"] == "verifier" and ev["metadata"].get("achieved") for ev in done["events"])
    finally:
        client.post(f"/api/models/{config.HYBRID_SLUG}/stop")
