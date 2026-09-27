from app.policy import evaluate, format_bytes, slugify, validate_filename, validate_repo, PolicyError
import pytest


def test_qwen_q4_cabe_y_los_pesados_no():
    # Tamaños reales de bartowski/Qwen_Qwen3.5-9B-GGUF.
    assert evaluate(6_169_341_984)[0] is True  # Q4_K_M, 6.17 GB
    assert evaluate(6_924_234_784)[0] is True  # Q4_K_L, 6.92 GB
    assert evaluate(8_451_485_728)[0] is True  # Q6_K_L, 8.45 GB, bajo 9 GB
    assert evaluate(9_000_000_000)[0] is True
    assert evaluate(9_000_000_001)[0] is False
    assert evaluate(9_804_541_984)[0] is False  # Q8_0, 9.80 GB
    assert evaluate(None)[0] is False


def test_el_mensaje_nombra_el_limite():
    allowed, reason = evaluate(9_804_541_984)
    assert allowed is False
    assert "9.00 GB" in reason
    assert format_bytes(6_169_341_984) == "6.17 GB"


def test_parte_suelta_no_arranca_y_el_conjunto_se_suma():
    from app.policy import group_gguf_entries, split_load_reason

    reason = split_load_reason("XiaomiMiMo.MiMo-V2.5.Q2_K-00008-of-00008.gguf")
    assert reason
    assert "parte 8 de 8" in reason
    grouped = group_gguf_entries([
        {"filename": "modelo-Q4_K-00001-of-00002.gguf", "size_bytes": 4_000_000_000},
        {"filename": "modelo-Q4_K-00002-of-00002.gguf", "size_bytes": 6_000_000_000},
        {"filename": "chico-Q4_K_M.gguf", "size_bytes": 1_000_000_000},
    ])
    split = next(item for item in grouped if item["split"])
    assert split["size_bytes"] == 10_000_000_000
    assert split["parts"][0].endswith("00001-of-00002.gguf")
    assert evaluate(split["size_bytes"])[0] is False
    small = group_gguf_entries([
        {"filename": "mini-00001-of-00002.gguf", "size_bytes": 2_000_000_000},
        {"filename": "mini-00002-of-00002.gguf", "size_bytes": 2_000_000_000},
    ])[0]
    assert evaluate(small["size_bytes"])[0] is True


def test_imatrix_no_es_un_modelo():
    from app.policy import non_model_reason, split_load_reason

    reason = non_model_reason("XiaomiMiMo_MiMo-V2-Flash-imatrix.gguf")
    assert reason
    assert "imatrix" in reason
    assert split_load_reason("XiaomiMiMo_MiMo-V2-Flash-imatrix.gguf") == reason
    assert non_model_reason("Qwen3.5-9B-Q4_K_M.gguf") is None


def test_slug_del_qwen():
    assert slugify("Qwen_Qwen3.5-9B-Q4_K_M.gguf") == "qwen-qwen3.5-9b-q4-k-m"


def test_rechaza_rutas_y_archivos_que_no_son_gguf():
    with pytest.raises(PolicyError):
        validate_filename("../secreto.gguf")
    with pytest.raises(PolicyError):
        validate_filename("modelo.bin")
    with pytest.raises(PolicyError):
        validate_repo("no-es-un-repo")
    assert validate_filename("Qwen_Qwen3.5-9B-Q4_K_M.gguf").endswith(".gguf")
    assert validate_repo("bartowski/Qwen_Qwen3.5-9B-GGUF")
