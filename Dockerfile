FROM ghcr.io/ggml-org/llama.cpp:server

USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt /srv/requirements.txt
RUN python3 -m pip install --no-cache-dir --break-system-packages -r /srv/requirements.txt \
    || python3 -m pip install --no-cache-dir -r /srv/requirements.txt

COPY app /srv/app
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && mkdir -p /data/models

# MALLOC_ARENA_MAX=2 evita que glibc abra una arena por hilo y baja la RSS.
# LLAMA_REPACK=0 y CACHE_RAM_MIB=0 impiden que llama-server duplique los pesos
# o guarde hasta 8 GiB de caché de prompts. GENERATION_TIMEOUT corta cualquier
# respuesta que se quede colgada y el chat enseña el error.
ENV DATA_DIR=/data \
    PORT=8080 \
    AUTO_DOWNLOAD_DEFAULT=1 \
    PRELOAD_DEFAULT=1 \
    CTX_SIZE=2048 \
    MAX_LOADED_MODELS=1 \
    N_GPU_LAYERS=0 \
    LLAMA_REPACK=0 \
    CACHE_RAM_MIB=0 \
    LOAD_TIMEOUT=600 \
    GENERATION_TIMEOUT=180 \
    STREAM_IDLE_TIMEOUT=90 \
    MALLOC_ARENA_MAX=2 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/data/hf-home \
    HUGGINGFACE_HUB_CACHE=/data/hf-home \
    LD_LIBRARY_PATH=/app \
    PATH=/app:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LLAMA_SERVER=/app/llama-server

# Sin VOLUME: Railway rechaza esa instrucción y el build fallaba entero.
# El volumen se monta desde fuera (docker compose o Railway Volumes en /data).
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8080}/health" || exit 1
ENTRYPOINT ["/entrypoint.sh"]
