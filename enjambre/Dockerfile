FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/data/hf-home \
    HUGGINGFACE_HUB_CACHE=/data/hf-home \
    ENJAMBRE_DELTAS=/data/enjambre/artifacts \
    ENJAMBRE_PORT=8090 \
    ENJAMBRE_HOST=0.0.0.0 \
    OMP_NUM_THREADS=8

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY enjambre/requirements.txt /srv/enjambre/requirements.txt
RUN pip install --no-cache-dir -r /srv/enjambre/requirements.txt

COPY enjambre /srv/enjambre
ENV PYTHONPATH=/srv

EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
  CMD curl -fsS "http://127.0.0.1:${ENJAMBRE_PORT}/health" || exit 1

CMD ["python", "-m", "enjambre", "serve", "--host", "0.0.0.0", "--port", "8090"]
