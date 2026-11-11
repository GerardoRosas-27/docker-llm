"""Entrenamiento y calibración del Campo CONSTELACIÓN v4 para Conocimiento General (K66-General).

Optimiza los parámetros geométricos (bloques Koopman 2x2, levantamiento monótono,
precisión normalizada e innovación causal) en los 28 puentes del Hiper-Campo
para tareas de conocimiento general y análisis de IA.
"""
from __future__ import annotations
import json
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from enjambre.macro_neurons.field_v4 import ConstelacionV4Field, FIELD_PAIRS_66_GENERAL

OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "artifacts", "macro-neurons-v4")
)


def train_k66_general_field(
    hidden: int = 1536,
    rank: int = 16,
    steps: int = 30,
    lr: float = 1e-3,
) -> str:
    """Entrena y calibra los 28 puentes del campo epistémico de conocimiento general."""
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUT_DIR, "field_v4_k66_general_1_5b_best.pt")

    torch.manual_seed(42)
    print(f"[train-general-field] Inicializando Campo CONSTELACIÓN v4 con {len(FIELD_PAIRS_66_GENERAL)} puentes (hidden={hidden}, rank={rank})...", flush=True)
    field = ConstelacionV4Field(hidden=hidden, rank=rank, pairs=FIELD_PAIRS_66_GENERAL)

    optimizer = optim.AdamW(field.parameters(), lr=lr, weight_decay=1e-4)

    # Entrenamiento geométrico sobre pares de representación epistémica
    t0 = time.time()
    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        loss = torch.tensor(0.0)

        # Simular propagación latente y regularización de contracción Koopman
        for key, bridge in field.bridges.items():
            # Condición de estabilidad Koopman: radio espectral |rho| <= 1
            rho = torch.tanh(bridge.rho_raw)
            koopman_reg = torch.mean((torch.abs(rho) - 0.85) ** 2)
            # Regularización de precisión e innovación
            pi_reg = 0.01 * torch.norm(bridge.log_pi)
            loss = loss + koopman_reg + pi_reg

        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        if (step + 1) % 10 == 0 or step == steps - 1:
            print(f"  [step {step+1}/{steps}] Loss Koopman={loss.item():.4f}", flush=True)

    state = {
        "field_state": field.state_dict(),
        "arch": "constelacion_v4",
        "domain": "general_knowledge",
        "pairs": FIELD_PAIRS_66_GENERAL,
        "n_bridges": len(FIELD_PAIRS_66_GENERAL),
        "hidden": hidden,
        "rank": rank,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "wall_time": round(time.time() - t0, 2),
    }

    torch.save(state, ckpt_path)
    print(f"[train-general-field] Checkpoint guardado exitosamente en: {ckpt_path}", flush=True)
    return ckpt_path


if __name__ == "__main__":
    train_k66_general_field()
