# Imagen de entrenamiento. CPU por defecto; para GPU:
#   docker build --build-arg BASE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime --build-arg TORCH_INDEX=skip ...
ARG BASE=python:3.12-slim
FROM ${BASE}
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HOME=/data/hf-home MALLOC_ARENA_MAX=2
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /ft
COPY finetune/requirements.txt /ft/requirements.txt
RUN if [ "$TORCH_INDEX" != "skip" ]; then pip install --no-cache-dir torch --index-url "$TORCH_INDEX"; fi \
    && pip install --no-cache-dir -r requirements.txt \
    && git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp \
    && pip install --no-cache-dir -r /opt/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt \
       --extra-index-url https://download.pytorch.org/whl/cpu
COPY finetune /ft
COPY app /ft/app
ENTRYPOINT ["/ft/pipeline.sh"]
