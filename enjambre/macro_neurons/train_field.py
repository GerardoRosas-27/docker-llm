"""Entrenamiento del Campo de Interconexión CONSTELACIÓN v4 sobre la Red de Macro-Neuronas (K32 / K36).

Mantiene el 100% de las macro-neuronas congeladas y entrena exclusivamente
los parámetros del campo geométrico (rotaciones Koopman, levantamiento monótono,
precisión Π̂ e innovación causal) con énfasis focalizado en Álgebra y Ecuaciones.
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
from enjambre.math_code.complex_tasks import make_complex_mc_splits
from enjambre.math_code.tasks import (
    ALL_MC_TASKS,
    CODE_TASKS,
    EXT_CODE_TASKS,
    EXT_MATH_TASKS,
    MATH_TASKS,
    make_ext_mc_splits,
    make_mc_splits,
)
from .network import (
    build_k32_macro_network,
    build_k36_macro_network,
    build_k66_macro_network,
    build_k66_macro_network_1_5b,
    macro_param_checksum,
)
from .field_v4 import ConstelacionV4Field, FIELD_PAIRS_32, FIELD_PAIRS_36, FIELD_PAIRS_66

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
        n_macro: int = 66,
        base_size: str = "0.5b",
        out_dir: Optional[str] = None,
        rank: int = 16,
        n_train: int = 12,
        n_val: int = 6,
        train_steps: int = 60,
        batch_size: int = 4,
        lr_field: float = 8e-4,
    ):
        self.n_macro = n_macro
        self.base_size = base_size.lower().strip()
        self.base_tag = "1_5b" if self.base_size in ("1.5b", "1b") else ""
        self.out_dir = os.path.abspath(out_dir or DEFAULT_OUT_DIR)
        os.makedirs(self.out_dir, exist_ok=True)
        self.rank = rank
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.lr_field = lr_field

        torch.manual_seed(SEED)
        torch.set_num_threads(min(8, os.cpu_count() or 8))

        print(f"[macro-field] Construyendo red de {self.n_macro} macro-neuronas (Base: {self.base_size.upper()})...", flush=True)
        if self.n_macro == 120:
            self.model, self.tok, self.delta = build_k120_macro_network_1_5b()
            pairs = FIELD_PAIRS_120
        elif self.base_tag == "1_5b" and self.n_macro == 66:
            self.model, self.tok, self.delta = build_k66_macro_network_1_5b()
            pairs = FIELD_PAIRS_66
        elif self.n_macro == 66:
            self.model, self.tok, self.delta = build_k66_macro_network()
            pairs = FIELD_PAIRS_66
        elif self.n_macro == 36:
            self.model, self.tok, self.delta = build_k36_macro_network()
            pairs = FIELD_PAIRS_36
        else:
            self.model, self.tok, self.delta = build_k32_macro_network()
            pairs = FIELD_PAIRS_32

        if self.n_macro in (66, 120) or self.base_tag == "1_5b":
            self.batch_size = min(self.batch_size, 2)


        self.eos = self.tok.eos_token or ""
        self.base_checksum = macro_param_checksum(self.model)

        # Crear y acoplar el Campo CONSTELACIÓN v4 (12, 18 o 28 puentes)
        hidden = self.model.config.hidden_size
        self.field = ConstelacionV4Field(hidden=hidden, rank=rank, pairs=pairs).to(self.model.dtype)
        self.field.attach(self.model)

        # Splits para entrenamiento y validación (Core + Ext + Complex)
        self.train_rows, self.val_rows, _, _ = make_mc_splits(n_train=n_train, n_val=n_val, n_hold=10, seed=SEED)
        self.ext_train_rows, self.ext_val_rows, _, _ = make_ext_mc_splits(n_train=n_train, n_val=n_val, n_hold=10, seed=142)
        self.comp_train, self.comp_val, _, _ = make_complex_mc_splits(n_train=6, n_val=4, n_hold=15, seed=SEED)

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
        # Mayor ponderación a problemas concisos y cálculos algebraicos numéricos
        weights = torch.tensor(
            [1.50 if len(t.strip()) <= 6 else 1.0 for t in targets],
            dtype=seq_losses.dtype,
            device=seq_losses.device,
        )
        loss = (seq_losses * weights).sum() / weights.sum()
        self.field.clear_buf()
        return loss

    def build_dataset(self) -> List[Tuple[str, str]]:
        """Construye un dataset multiescala enriquecido con tareas complejas (ecuaciones y álgebra)."""
        math_tids = MATH_TASKS + EXT_MATH_TASKS
        code_tids = CODE_TASKS + EXT_CODE_TASKS
        all_rows = {**self.train_rows, **self.ext_train_rows}
        out: List[Tuple[str, str]] = []

        # 1. Tareas base y extendidas
        max_len = max(len(all_rows[tid]) for tid in (math_tids + code_tids))
        for i in range(max_len):
            for tid in math_tids:
                if i < len(all_rows[tid]):
                    out.append((all_rows[tid][i][0], all_rows[tid][i][1]))
            if i % 2 == 0:
                for tid in code_tids:
                    if i < len(all_rows[tid]):
                        out.append((all_rows[tid][i][0], all_rows[tid][i][1]))

        # 2. Inyección de tareas complejas de entrenamiento (Ecuaciones, GCD, Recursión, Diccionarios)
        for tid, rows in self.comp_train.items():
            for p, tgt, _ in rows:
                out.append((p, str(tgt)))
                # Duplicar álgebra y ecuaciones para reforzar convergencia matemática
                if "solve_linear" in tid or "modular_exp" in tid or "gcd_lcm" in tid:
                    out.append((p, str(tgt)))

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
        print(f"[macro-field] Dataset ampliado con {len(dataset)} pares (Math Core + Ext + Complex)", flush=True)

        optimizer = torch.optim.AdamW(self.field.parameters(), lr=self.lr_field, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.train_steps, eta_min=1e-5)

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
            scheduler.step()

            val_freq = 10 if (self.n_macro == 66 and self.base_tag == "1_5b") else 5
            if step % val_freq == 0 or step == self.train_steps:
                val_l = self.eval_val_loss()
                cur_lr = scheduler.get_last_lr()[0]
                print(
                    f"[macro-field] Step {step}/{self.train_steps} | LR: {cur_lr:.2e} | Train Loss: {loss.item():.4f} | Val Loss: {val_l:.4f} | Elapsed: {time.time()-t0:.1f}s",
                    flush=True,
                )
                history.append({
                    "step": step,
                    "train_loss": round(float(loss.item()), 4),
                    "val_loss": round(val_l, 4),
                    "lr": cur_lr,
                })
                if val_l < best_val:
                    best_val = val_l
                    best_state = {k: v.cpu().clone() for k, v in self.field.state_dict().items()}
                    tag = f"_{self.base_tag}" if self.base_tag else ""
                    ckpt_name = f"field_v4_k{self.n_macro}{tag}_best.pt"
                    torch.save(
                        {
                            "field_state": best_state,
                            "history": history,
                            "n_macro": self.n_macro,
                            "pairs": self.field.pairs,
                            "best_val_loss": best_val,
                            "base_checksum": self.base_checksum,
                        },
                        os.path.join(self.out_dir, ckpt_name),
                    )

        tag = f"_{self.base_tag}" if self.base_tag else ""
        ckpt_name = f"field_v4_k{self.n_macro}{tag}_best.pt"
        out_path = os.path.join(self.out_dir, ckpt_name)
        torch.save(
            {
                "field_state": best_state,
                "history": history,
                "n_macro": self.n_macro,
                "pairs": self.field.pairs,
                "best_val_loss": best_val,
                "base_checksum": self.base_checksum,
            },
            out_path,
        )
        print(f"[macro-field] Checkpoint guardado en {out_path} (Best Val Loss: {best_val:.4f})", flush=True)
        return out_path
