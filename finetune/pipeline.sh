#!/usr/bin/env sh
# datos -> LoRA -> fusión -> GGUF -> registro en el volumen del panel.
set -eu
cd "$(dirname "$0")"
BASE_MODEL="${BASE_MODEL:-HuggingFaceTB/SmolLM2-135M-Instruct}"
OUT="${OUT_DIR:-/data/finetune}"
LLAMA_CPP="${LLAMA_CPP:-/opt/llama.cpp}"
python3 generar_datos.py --seed "${SEED:-27}" --out "$OUT/data"
python3 entrenar.py --base "$BASE_MODEL" --data "$OUT/data" --out "$OUT" --epochs "${EPOCHS:-8}" ${TRAIN_ARGS:-}
python3 exportar_gguf.py --modelo "$OUT/fusionado" --llama-cpp "$LLAMA_CPP" --registrar
