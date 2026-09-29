"""Configuración leída del entorno en cada acceso, para tests y Railway."""

from __future__ import annotations

import os
from pathlib import Path

# Qwen 3.5 2B cuantizado Q4_K_M publicado por bartowski.
# El archivo pesa 1.40 GB (1_396_198_496 bytes) y en CPU ocupa ~1.5 GB de RAM.
# El 9B anterior (6.17 GB) pedía ~7-8 GB de RAM y en CPU va 4-5 veces más lento:
# en Docker Desktop o en un plan pequeño el chat se quedaba sin responder.
DEFAULT_REPO = "bartowski/Qwen_Qwen3.5-2B-GGUF"
DEFAULT_FILE = "Qwen_Qwen3.5-2B-Q4_K_M.gguf"
DEFAULT_SIZE = 1_396_198_496
# sha256 publicado por Hugging Face (lfs.oid). La descarga se comprueba contra él.
DEFAULT_SHA256 = "57a1085840f497d764a7fc5d346922dbde961efb54cc792ea81d694fd846a1d8"
DEFAULT_LABEL = "Qwen 3.5 2B · Q4_K_M"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def data_dir() -> Path:
    return Path(_env("DATA_DIR", "/data"))


def models_dir() -> Path:
    return data_dir() / "models"


def db_path() -> Path:
    return data_dir() / "obrador.sqlite"


def ctx_size() -> int:
    return int(_env("CTX_SIZE", "2048"))


def n_gpu_layers() -> int:
    return int(_env("N_GPU_LAYERS", "0"))


def llama_threads() -> int:
    return int(_env("LLAMA_THREADS", "0"))


def llama_repack() -> bool:
    # El repack de llama.cpp copia los pesos a memoria anónima y casi duplica
    # la RAM (2.8 GB frente a 1.45 GB con el 2B). Apagado por defecto.
    return _env("LLAMA_REPACK", "0") == "1"


def cache_ram_mib() -> int:
    # llama-server guarda hasta 8 GiB de caché de prompts si no se le dice nada.
    return int(_env("CACHE_RAM_MIB", "0"))


def generation_timeout() -> int:
    """Segundos máximos para una respuesta completa, con el modelo ya cargado."""
    return max(1, int(_env("GENERATION_TIMEOUT", "180")))


def stream_idle_timeout() -> int:
    """Segundos máximos sin recibir nada de llama-server mientras genera."""
    return max(1, int(_env("STREAM_IDLE_TIMEOUT", "90")))


def ram_check() -> bool:
    return _env("RAM_CHECK", "1") == "1"


def preload_default() -> bool:
    return _env("PRELOAD_DEFAULT", "1") == "1"


def max_loaded_models() -> int:
    return max(1, int(_env("MAX_LOADED_MODELS", "1")))


def api_key() -> str:
    return _env("API_KEY", "").strip()


def admin_token() -> str:
    return _env("ADMIN_TOKEN", "").strip()


def hf_token() -> str:
    return _env("HF_TOKEN", "").strip()


def llama_server_bin() -> str:
    return _env("LLAMA_SERVER", "llama-server")


def load_timeout() -> int:
    # En Docker Desktop el GGUF se lee por el sistema de archivos de Windows.
    # El 2B carga en segundos en Linux y en pocos minutos en un bind mount lento.
    return int(_env("LOAD_TIMEOUT", "600"))


def auto_download() -> bool:
    return _env("AUTO_DOWNLOAD_DEFAULT", "1") == "1"


def default_repo() -> str:
    return _env("DEFAULT_REPO", DEFAULT_REPO)


def default_file() -> str:
    return _env("DEFAULT_FILE", DEFAULT_FILE)


def default_sha256() -> str:
    """sha256 esperado del modelo por defecto (vacío si no se conoce)."""
    explicit = _env("DEFAULT_SHA256", "").strip().lower()
    if explicit:
        return explicit
    if default_repo() == DEFAULT_REPO and default_file() == DEFAULT_FILE:
        return DEFAULT_SHA256
    return ""


def ensure_dirs() -> None:
    models_dir().mkdir(parents=True, exist_ok=True)
    cache = data_dir() / "hf-home"
    cache.mkdir(parents=True, exist_ok=True)
    # El caché de Hugging Face tiene que vivir en el volumen, no en el disco del sistema.
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache))
