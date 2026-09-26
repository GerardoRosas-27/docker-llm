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

ENV DATA_DIR=/data \
    PORT=8080 \
    AUTO_DOWNLOAD_DEFAULT=1 \
    CTX_SIZE=2048 \
    MAX_LOADED_MODELS=1 \
    N_GPU_LAYERS=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/data/hf-home \
    HUGGINGFACE_HUB_CACHE=/data/hf-home \
    LD_LIBRARY_PATH=/app \
    PATH=/app:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LLAMA_SERVER=/app/llama-server

VOLUME ["/data"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8080}/health" || exit 1
ENTRYPOINT ["/entrypoint.sh"]
