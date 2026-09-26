#!/bin/sh
set -eu
mkdir -p "${DATA_DIR:-/data}/models" "${DATA_DIR:-/data}/hf-home"
export LD_LIBRARY_PATH="/app${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ -x /app/llama-server ]; then
  export PATH="/app:${PATH}"
fi
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}"
