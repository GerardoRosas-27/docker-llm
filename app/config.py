"""Configuración leída del entorno en cada acceso, para tests y Railway."""

from __future__ import annotations

import os
from pathlib import Path

# Qwen 3.5 9B cuantizado Q4_K_M publicado por bartowski.
# El archivo pesa 6.17 GB (6_169_341_984 bytes). Es el Q4 recomendado
# y cabe bajo el límite de 8 GB. Q4_K_L del mismo repo pesa 6.92 GB.
DEFAULT_REPO = "bartowski/Qwen_Qwen3.5-9B-GGUF"
DEFAULT_FILE = "Qwen_Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_SIZE = 6_169_341_984


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
    # En Docker Desktop el GGUF se lee por el sistema de archivos de Windows
    # y un 9B puede tardar más de diez minutos en quedar listo.
    return int(_env("LOAD_TIMEOUT", "1200"))


def auto_download() -> bool:
    return _env("AUTO_DOWNLOAD_DEFAULT", "1") == "1"


def default_repo() -> str:
    return _env("DEFAULT_REPO", DEFAULT_REPO)


def default_file() -> str:
    return _env("DEFAULT_FILE", DEFAULT_FILE)


def ensure_dirs() -> None:
    models_dir().mkdir(parents=True, exist_ok=True)
    cache = data_dir() / "hf-home"
    cache.mkdir(parents=True, exist_ok=True)
    # El caché de Hugging Face tiene que vivir en el volumen, no en el disco del sistema.
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache))
