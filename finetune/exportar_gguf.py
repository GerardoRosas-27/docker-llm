"""Convierte el modelo fusionado a GGUF (llama.cpp) y lo registra en el panel.

El panel solo sirve GGUF con llama-server, así que el resultado se deja en
$DATA_DIR/models/<slug>/ y se marca como 'ready' en la misma SQLite que usa
la app. Tras reiniciar (o al pulsar Arrancar) aparece en Instalados.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ID = "local/mochilas-rosas-ft"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True, help="carpeta HF del modelo fusionado")
    ap.add_argument("--llama-cpp", default=os.environ.get("LLAMA_CPP", "/opt/llama.cpp"))
    ap.add_argument("--outtype", default="q8_0", help="f16, bf16 o q8_0 (convert_hf_to_gguf)")
    ap.add_argument("--nombre", default="mochilas-rosas-SmolLM2-135M-Q8_0.gguf")
    ap.add_argument("--registrar", action="store_true", help="dar de alta en la DB del panel ($DATA_DIR)")
    a = ap.parse_args()

    data_dir = Path(os.environ.get("DATA_DIR", "/data"))
    dest = (data_dir / "models" / "local-mochilas-rosas") if a.registrar else Path(a.modelo).parent
    dest.mkdir(parents=True, exist_ok=True)
    gguf = dest / a.nombre
    subprocess.run([sys.executable, str(Path(a.llama_cpp) / "convert_hf_to_gguf.py"), a.modelo,
                    "--outfile", str(gguf), "--outtype", a.outtype], check=True)
    print(f"GGUF: {gguf} ({gguf.stat().st_size} bytes)")
    if not a.registrar:
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app import db  # misma SQLite que el panel
    db.init()
    row = db.upsert_download(REPO_ID, a.nombre, gguf.stat().st_size)
    db.mark_ready(row["slug"], str(gguf), gguf.stat().st_size)
    print(f"registrado en el panel como '{row['slug']}'")


if __name__ == "__main__":
    main()
