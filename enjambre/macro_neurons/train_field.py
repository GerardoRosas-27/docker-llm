"""Entrenamiento del Campo de Interconexión CONSTELACIÓN v4 sobre la Red de Macro-Neuronas K32.

Mantiene el 100% de las 32 macro-neuronas congeladas y entrena exclusivamente
los parámetros del campo geométrico (rotaciones Koopman, levantamiento monótono,
precisión Π̂ y compuertas adaptativas).
"""
from __future__ import annotations
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from enjambre.config import resolve
from enjambre.math_code.tasks import (
    ALL_MC_TASKS,
    CODE_TASKS,
    EXT_CODE_TASKS,
    EXT_MATH_TASKS,
    MATH_TASKS,
    make_ext_mc_splits,
    make_mc_splits,
)
from .network import build_k32_macro_network, macro_param_checksum
from .field_v4 import ConstelacionV4Field, FIELD_PAIRS_32

DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "macro-neurons-v4"
)
SEED = 42


def _jdump(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


class FieldTrainer:
    def __init__(
        self,
        out_dir: Optional[str] = None,
        rank: int = 16,
        n_train: int = 12,
        n_val: int = 6,
        train_steps: int = 30,
        batch_size: int = 4,
        lr_field: float = 8e-4,
    ):
        self.out_dir = os.path.abspath(out_dir or DEFAULT_OUT_DIR)
        os.makedirs(self.out_dir, exist_ok=True)
        self.rank = rank
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.lr_field = lr_field

        torch.manual_seed(SEED)
        torch.set_num_threads(min(8, os.cpu_count() or 8))

        print("[macro-field] Construyendo red de 32 macro-neuronas...", flush=True)
        self.model, self.tok, self.delta = build_k32_macro_network()
        self.eos = self.tok.eos_token or ""
        self.base_checksum = macro_param_checksum(self.model)

        # Crear y acoplar el Campo CONSTELACIÓN v4 (12 puentes)
        hidden = self.model.config.hidden_size
        self.field = ConstelacionV4Field(hidden=hidden, rank=rank, pairs=FIELD_PAIRS_32)
        self.field.attach(self.model)

        # Splits para entrenamiento y validación
        self.train_rows, self.val_rows, _, _ = make_mc_splits(n_train=n_train, n_val=n_val, n_hold=10, seed=SEED)
        self.ext_train_rows, self.ext_val_rows, _, _ = make_ext_mc_splits(n_train=n_train, n_val=n_val, n_hold=10, seed=142)

        print(f"[macro-field] Campo v4 inicializado con {len(self.field.bridges)} puentes ({self.field.n_params():,} parámetros entrenables)", flush=True)

    def _chat(self, user: str) -> str:
        msgs = [{"role": "user", "content": user}]
        if hasattr(self.tok, "apply_chat_template"):
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return user

    def _loss_batch(self, prompts: List[str], targets: List[str]) -> torch.Tensor:
        self.field.clear_buf()
        bos = [self._chat(p) for p in prompts]
        full = [b + t + self.eos for b, t in zip(bos, targets)]
        enc_full = self.tok(full, return_tensors="pt", padding=True, truncation=True, max_length=128)
        enc_prompt = self.tok(bos, return_tensors="pt", padding=True, truncation=True, max_length=112)
        input_ids = enc_full["input_ids"]
        attn = enc_full["attention_mask"]
        labels = input_ids.clone()
        prompt_lens = enc_prompt["attention_mask"].sum(1)
        full_lens = attn.sum(1)
        for i in range(input_ids.size(0)):
            T = input_ids.size(1)
            start = T - int(full_lens[i])
            plen = int(prompt_lens[i])
            labels[i, : start + plen] = -100
            labels[i, attn[i] == 0] = -100

        out = self.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits = out.logits
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        per_tok = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.size(0), shift_labels.size(1))
        valid = (shift_labels != -100).float()
        seq_losses = (per_tok * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        weights = torch.tensor(
            [1.40 if len(t.strip()) <= 6 else 1.0 for t in targets],
            dtype=seq_losses.dtype,
            device=seq_losses.device,
        )
        loss = (seq_losses * weights).sum() / weights.sum()
        self.field.clear_buf()
        return loss

    def build_dataset(self) -> List[Tuple[str, str]]:
        math_tids = MATH_TASKS + EXT_MATH_TASKS
        code_tids = CODE_TASKS + EXT_CODE_TASKS
        all_rows = {**self.train_rows, **self.ext_train_rows}
        out: List[Tuple[str, str]] = []
        max_len = max(len(all_rows[tid]) for tid in (math_tids + code_tids))
        for i in range(max_len):
            for tid in math_tids:
                if i < len(all_rows[tid]):
                    out.append((all_rows[tid][i][0], all_rows[tid][i][1]))
            if i % 2 == 0:
                for tid in code_tids:
                    if i < len(all_rows[tid]):
                        out.append((all_rows[tid][i][0], all_rows[tid][i][1]))
        return out

    @torch.no_grad()
    def eval_val_loss(self) -> float:
        self.field.clear_buf()
        all_val = {**self.val_rows, **self.ext_val_rows}
        losses = []
        for tid in (MATH_TASKS + CODE_TASKS):
            rows = all_val[tid][:2]
            ps = [r[0] for r in rows]
            ts = [r[1] for r in rows]
            l = self._loss_batch(ps, ts)
            losses.append(float(l.item()))
        return sum(losses) / max(1, len(losses))

    def train(self) -> str:
        dataset = self.build_dataset()
        random.seed(SEED)
        random.shuffle(dataset)
        print(f"[macro-field] Dataset preparado con {len(dataset)} pares (Math + Code)", flush=True)

        optimizer = torch.optim.AdamW(self.field.parameters(), lr=self.lr_field, weight_decay=1e-4)
        best_val = self.eval_val_loss()
        print(f"[macro-field] Val Loss inicial (step 0): {best_val:.4f}", flush=True)

        best_state = {k: v.cpu().clone() for k, v in self.field.state_dict().items()}
        history = []

        t0 = time.time()
        for step in range(1, self.train_steps + 1):
            batch = [dataset[(step * self.batch_size + b) % len(dataset)] for b in range(self.batch_size)]
            prompts = [p for p, _ in batch]
            targets = [t for _, t in batch]

            optimizer.zero_grad()
            loss = self._loss_batch(prompts, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.field.parameters(), 1.0)
            optimizer.step()

            if step % 5 == 0 or step == self.train_steps:
                val_l = self.eval_val_loss()
                print(f"[macro-field] Step {step}/{self.train_steps} | Train Loss: {loss.item():.4f} | Val Loss: {val_l:.4f} | Elapsed: {time.time()-t0:.1f}s", flush=True)
                history.append({"step": step, "train_loss": round(float(loss.item()), 4), "val_loss": round(val_l, 4)})
                if val_l < best_val:
                    best_val = val_l
                    best_state = {k: v.cpu().clone() for k, v in self.field.state_dict().items()}

        out_path = os.path.join(self.out_dir, "field_v4_k32_best.pt")
        torch.save(
            {
                "field_state": best_state,
                "history": history,
                "n_macro": 32,
                "pairs": FIELD_PAIRS_32,
                "best_val_loss": best_val,
                "base_checksum": self.base_checksum,
            },
            out_path,
        )
        print(f"[macro-field] Checkpoint guardado en {out_path} (Best Val Loss: {best_val:.4f})", flush=True)
        return out_path
