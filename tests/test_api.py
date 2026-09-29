from app import db


def test_salud_y_panel(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["max_model_bytes"] == 9_000_000_000
    page = client.get("/")
    assert page.status_code == 200
    assert "9 GB" in page.text
    assert "Descargas en curso" in page.text
    assert "Obrador" in page.text
    assert "Qwen" in page.text


def test_no_descarga_mas_de_9gb(client, monkeypatch):
    monkeypatch.setattr("app.hfclient.file_size", lambda repo, filename: 9_804_541_984)
    monkeypatch.setattr("app.hfclient.head_size", lambda repo, filename: None)
    response = client.post(
        "/api/models/download",
        json={
            "repo_id": "bartowski/Qwen_Qwen3.5-9B-GGUF",
            "filename": "Qwen_Qwen3.5-9B-Q8_0.gguf",
        },
    )
    assert response.status_code == 400
    assert "9.00 GB" in response.json()["detail"]
    assert db.list_models() == []


def test_encola_el_q4_dentro_del_limite(client, monkeypatch):
    queued = []
    monkeypatch.setattr("app.hfclient.file_size", lambda repo, filename: 6_169_341_984)
    monkeypatch.setattr("app.hfclient.head_size", lambda repo, filename: None)
    monkeypatch.setattr(
        "app.main.shutil.disk_usage",
        lambda path: type("Usage", (), {"free": 80_000_000_000})(),
    )
    monkeypatch.setattr("app.downloader.service.enqueue", lambda slug: queued.append(slug))
    response = client.post(
        "/api/models/download",
        json={
            "repo_id": "bartowski/Qwen_Qwen3.5-9B-GGUF",
            "filename": "Qwen_Qwen3.5-9B-Q4_K_M.gguf",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["slug"] == "qwen-qwen3.5-9b-q4-k-m"
    assert body["status"] == "queued"
    assert body["api"]["chat"].endswith("/v1/models/qwen-qwen3.5-9b-q4-k-m/chat/completions")
    assert queued == ["qwen-qwen3.5-9b-q4-k-m"]


def test_el_catalogo_no_lista_partes_sueltas(client, monkeypatch):
    def files(_repo):
        parts = [
            {"filename": f"MiMo.Q2_K-{index:05d}-of-00008.gguf", "size_bytes": 2_734_034_240}
            for index in range(1, 9)
        ]
        parts.append({"filename": "MiMo.Q2_K-00008-of-00099.gguf", "size_bytes": 1000})
        return parts

    monkeypatch.setattr("app.hfclient.repo_gguf_files", files)
    response = client.get(
        "/api/catalog/files",
        params={"repo_id": "DevQuasar/XiaomiMiMo.MiMo-V2.5-GGUF"},
    )
    assert response.status_code == 200
    listed = response.json()["files"]
    assert len(listed) == 1
    assert listed[0]["split"] is True
    assert listed[0]["part_total"] == 8
    assert listed[0]["allowed"] is False
    assert "00008-of-00008" not in listed[0]["label"]
    assert "modelo completo" in listed[0]["label"]


def test_un_modelo_partido_de_mas_de_9gb_no_se_encola(client, monkeypatch):
    monkeypatch.setattr(
        "app.hfclient.repo_gguf_files",
        lambda repo: [
            {"filename": f"MiMo-Q2_K-{index:05d}-of-00008.gguf", "size_bytes": 2_734_034_240}
            for index in range(1, 9)
        ],
    )
    response = client.post(
        "/api/models/download",
        json={
            "repo_id": "DevQuasar/XiaomiMiMo.MiMo-V2.5-GGUF",
            "filename": "MiMo-Q2_K-00008-of-00008.gguf",
        },
    )
    assert response.status_code == 400
    assert "9.00 GB" in response.json()["detail"]
    assert db.list_models() == []


def test_no_descarga_un_imatrix(client, monkeypatch):
    monkeypatch.setattr(
        "app.hfclient.file_size",
        lambda repo, filename: 5_000_000,
    )
    response = client.post(
        "/api/models/download",
        json={
            "repo_id": "DevQuasar/XiaomiMiMo.MiMo-V2-Flash-GGUF",
            "filename": "XiaomiMiMo_MiMo-V2-Flash-imatrix.gguf",
        },
    )
    assert response.status_code == 400
    assert "imatrix" in response.json()["detail"]
    assert db.list_models() == []


def test_rechaza_nombres_peligrosos(client):
    response = client.post(
        "/api/models/download",
        json={"repo_id": "bartowski/Qwen_Qwen3.5-9B-GGUF", "filename": "../otro.gguf"},
    )
    assert response.status_code == 400


def test_api_por_modelo_desconocido(client):
    missing = client.post("/v1/models/no-existe/chat/completions", json={"messages": []})
    assert missing.status_code == 404
    listed = client.get("/v1/models")
    assert listed.status_code == 200
    assert listed.json()["data"] == []


def test_catalogo_marca_archivos_grandes(client, monkeypatch):
    monkeypatch.setattr(
        "app.hfclient.repo_gguf_files",
        lambda repo: [
            {"filename": "chico-Q4_K_M.gguf", "size_bytes": 6_169_341_984},
            {"filename": "medio-Q6_K_L.gguf", "size_bytes": 8_451_485_728},
            {"filename": "grande-Q8_0.gguf", "size_bytes": 9_804_541_984},
        ],
    )
    response = client.get("/api/catalog/files", params={"repo_id": "alguien/modelo-gguf"})
    assert response.status_code == 200
    files = {item["filename"]: item for item in response.json()["files"]}
    assert files["chico-Q4_K_M.gguf"]["allowed"] is True
    assert files["medio-Q6_K_L.gguf"]["allowed"] is True
    assert files["grande-Q8_0.gguf"]["allowed"] is False
