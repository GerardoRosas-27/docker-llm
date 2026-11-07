"""Arnés de ensamblaje por capas editadas + entrenamiento exclusivo de puentes conectores (Math & Code)."""
from __future__ import annotations
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from enjambre.adaptive.snapshot import SnapshotActivator
from enjambre.bridges.train_eval import freeze_model, param_checksum
from enjambre.config import resolve
from enjambre.edits import EditDelta
from enjambre.experts import Item
from enjambre.fusion_confirm12.stats import bootstrap_delta_ci, mcnemar_exact, paired_flags
from enjambre.generate import generate, perplexity
from enjambre.scaling import save_delta

from .bridges_ext import (
    DEFAULT_PRETRAINED_BRIDGES,
    PAIRS_BY_CONFIG,
    build_warm_bridge_bank,
    load_pretrained_k12_bridges_state,
)
from .layer_edits import build_math_code_layer_deltas, compute_mc_contrast_directions
from .tasks import (
    ALL_MC_TASKS,
    MC_SPECS,
    get_legacy_k12_math_code_holdout,
    make_mc_splits,
)

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "artifacts", "math-code-k12plus")
SEED = 42


def _jdump(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


class MathCodeLayerHarness:
    def __init__(
        self,
        out_dir: Optional[str] = None,
        rank: int = 16,
        n_train: int = 12,
        n_val: int = 6,
        n_hold: int = 15,
        n_legacy_hold: int = 25,
        max_new: int = 22,
        train_steps: int = 28,
        batch_size_train: int = 4,
        lr_existing: float = 5e-4,
        lr_new: float = 1.5e-3,
    ):
        self.cfg = resolve("qwen2.5-0.5b")
        self.out_dir = os.path.abspath(out_dir or DEFAULT_OUT)
        os.makedirs(self.out_dir, exist_ok=True)
        self.rank = rank
        self.n_train, self.n_val, self.n_hold = n_train, n_val, n_hold
        self.n_legacy_hold = n_legacy_hold
        self.max_new = max_new
        self.train_steps = train_steps
        self.batch_size_train = batch_size_train
        self.lr_existing = lr_existing
        self.lr_new = lr_new

        torch.manual_seed(SEED)
        torch.set_num_threads(min(8, os.cpu_count() or 8))

        self.tok = AutoTokenizer.from_pretrained(self.cfg.hf_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.eos = self.tok.eos_token or ""

        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.hf_id, dtype=torch.float32).eval()
        freeze_model(self.model)
        self.base_checksum = param_checksum(self.model)
        self.act = SnapshotActivator(self.model)

        self.train_rows, self.val_rows, self.hold_items, self.split_man = make_mc_splits(
            n_train=n_train, n_val=n_val, n_hold=n_hold, seed=SEED
        )
        self.legacy_hold = get_legacy_k12_math_code_holdout(n_hold=n_legacy_hold)
        _jdump(os.path.join(self.out_dir, "split_manifest.json"), self.split_man)

        self.pretrained_bridges_state = load_pretrained_k12_bridges_state(DEFAULT_PRETRAINED_BRIDGES)

        fresh_path = os.path.join(self.out_dir, "mc_fresh_directions_v2.pt")
        if os.path.isfile(fresh_path):
            self.fresh_dirs = torch.load(fresh_path, map_location="cpu", weights_only=False)
        else:
            print("[math_code] Calculando direcciones de contraste por capa (0..23) para Math y Code...", flush=True)
            self.fresh_dirs = compute_mc_contrast_directions(self.model, self.tok, self.train_rows, batch_size=8)
            torch.save(self.fresh_dirs, fresh_path)

        self.deltas = build_math_code_layer_deltas(fresh_dirs=self.fresh_dirs)
        deltas_dir = os.path.join(self.out_dir, "deltas")
        os.makedirs(deltas_dir, exist_ok=True)
        for dname, d_obj in self.deltas.items():
            save_delta(os.path.join(deltas_dir, f"{dname}.pt"), d_obj)

        self.bank = build_warm_bridge_bank(
            self.model.config.hidden_size,
            PAIRS_BY_CONFIG["k12"],
            rank=self.rank,
            pretrained_state=self.pretrained_bridges_state,
        )
        self.bank.attach(self.model)

    def _chat(self, user: str) -> str:
        msgs = [{"role": "user", "content": user}]
        if hasattr(self.tok, "apply_chat_template"):
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return user

    def _loss_batch(self, prompts: List[str], targets: List[str]) -> torch.Tensor:
        """Teacher-forced CE loss normalizada por secuencia (para que respuestas cortas de
        matemáticas tengan igual peso que respuestas largas de código) + EOS (`<|im_end|>`)."""
        self.bank.clear_buf()
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
        # Dar peso 1.25x a secuencias numéricas cortas (math) para preservar precisión aritmética
        weights = torch.tensor(
            [1.25 if len(t.strip()) <= 6 else 1.0 for t in targets],
            dtype=seq_losses.dtype,
            device=seq_losses.device,
        )
        loss = (seq_losses * weights).sum() / weights.sum()
        self.bank.clear_buf()
        return loss

    def flat_supervised(self, rows_by_task: Dict[str, List]) -> List[Tuple[str, str]]:
        max_len = max(len(rows_by_task[tid]) for tid in ALL_MC_TASKS)
        out: List[Tuple[str, str]] = []
        for i in range(max_len):
            for tid in ALL_MC_TASKS:
                if i < len(rows_by_task[tid]):
                    p, tgt, _ = rows_by_task[tid][i]
                    out.append((p, tgt))
        return out

    @torch.no_grad()
    def eval_val_mean(self, delta: EditDelta) -> float:
        items = {
            tid: [Item(prompt=p, check=chk, meta={"target": tgt}) for p, tgt, chk in rows[:3]]
            for tid, rows in self.val_rows.items()
        }
        self.bank.enabled = True
        self.bank.clear_buf()
        oks = []
        with self.act.session(delta):
            for tid in ALL_MC_TASKS:
                its = items[tid]
                outs = generate(self.model, self.tok, [it.prompt for it in its],
                                max_new_tokens=20, batch_size=8)
                for it, o in zip(its, outs):
                    oks.append(bool(it.check(o)))
                self.bank.clear_buf()
        return sum(oks) / max(1, len(oks))

    def train_bridges_for_config(
        self,
        config_tag: str,
        delta: EditDelta,
        pairs: List[Tuple[int, int]],
    ) -> Dict[str, Any]:
        """Entrena EXCLUSIVAMENTE los puentes conectores (`BridgeBank`) sobre el modelo con capas editadas `delta`."""
        ckpt_path = os.path.join(self.out_dir, f"bridges_{config_tag}_best.pt")
        log_path = os.path.join(self.out_dir, f"train_log_{config_tag}.json")
        if os.path.isfile(ckpt_path) and os.path.isfile(log_path):
            print(f"[math_code:{config_tag}] Reutilizando checkpoint {ckpt_path}", flush=True)
            return json.load(open(log_path, encoding="utf-8"))

        self.bank.remove_hooks()
        self.bank = build_warm_bridge_bank(
            self.model.config.hidden_size,
            pairs=pairs,
            rank=self.rank,
            pretrained_state=self.pretrained_bridges_state,
        )
        self.bank.attach(self.model)
        freeze_model(self.model)

        pretrained_keys = {"12_16", "16_20", "20_23"}
        params_new, params_existing = [], []
        for bkey, bridge in self.bank.bridges.items():
            for p in bridge.parameters():
                p.requires_grad_(True)
                if bkey in pretrained_keys:
                    params_existing.append(p)
                else:
                    params_new.append(p)

        param_groups = [{"params": params_existing, "lr": self.lr_existing}]
        if params_new:
            param_groups.append({"params": params_new, "lr": self.lr_new})
        opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)

        data = self.flat_supervised(self.train_rows)
        rng = random.Random(SEED)
        t0 = time.time()

        best_val = -1.0
        best_state = {k: v.detach().cpu().clone() for k, v in self.bank.state_dict().items()}
        best_step = 0
        history = []
        print(f"[math_code:{config_tag}] start train steps={self.train_steps} pairs={pairs} n_params={self.bank.n_params()}", flush=True)

        self.model.train()
        step = 0
        bs = self.batch_size_train
        while step < self.train_steps:
            rng.shuffle(data)
            for i in range(0, len(data), bs):
                if step >= self.train_steps:
                    break
                batch = data[i : i + bs]
                if not batch:
                    continue
                ps, ts = zip(*batch)
                with self.act.session(delta):
                    loss = self._loss_batch(list(ps), list(ts))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.bank.parameters(), 1.0)
                opt.step()
                self.bank.clear_buf()
                step += 1

                if step == self.train_steps:
                    self.model.eval()
                    assert param_checksum(self.model) == self.base_checksum, "Base model weights must remain frozen!"
                    v = self.eval_val_mean(delta)
                    self.model.train()
                    row = {
                        "step": step,
                        "loss": round(float(loss.detach()), 4),
                        "val": round(v, 4),
                        "secs": round(time.time() - t0, 1),
                    }
                    history.append(row)
                    print(f"[math_code:{config_tag}] step={step} loss={row['loss']:.4f} val={v:.4f} ({row['secs']}s)", flush=True)
                    best_val = v
                    best_step = step
                    best_state = {k: v_t.detach().cpu().clone() for k, v_t in self.bank.state_dict().items()}

        self.model.eval()
        self.bank.load_state_dict(best_state)
        torch.save(
            {
                "state": best_state,
                "step": best_step,
                "val": best_val,
                "rank": self.rank,
                "pairs": pairs,
                "delta_name": delta.name,
                "layers_edited": [b.layer for b in delta.bias_edits],
            },
            ckpt_path,
        )
        info = {
            "config_tag": config_tag,
            "pairs": pairs,
            "n_bridge_params": self.bank.n_params(),
            "steps": step,
            "best_step": best_step,
            "best_val": round(best_val, 4),
            "secs": round(time.time() - t0, 1),
            "history": history,
            "ckpt_path": ckpt_path,
        }
        _jdump(log_path, info)
        return info

    @torch.no_grad()
    def evaluate_condition(
        self,
        label: str,
        delta: EditDelta,
        pairs: List[Tuple[int, int]],
        bridge_state: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """Evalúa una configuración (capas editadas `delta` + puentes `bridge_state`) en:
        1) Las 8 tareas de Holdout Math & Code (n_hold por tarea).
        2) Las tareas legacy de `bridges-k12` (`arithmetic`, `code_python` lax y `code_python` AST estricto).
        """
        out_path = os.path.join(self.out_dir, f"results_{label}.json")
        if os.path.isfile(out_path):
            cached = json.load(open(out_path, encoding="utf-8"))
            if len(cached.get("flat_ok", [])) == self.n_hold * len(MC_SPECS):
                print(f"[eval:{label}] Reutilizando resultado en caché ({out_path}) mean_acc={cached['mean_acc']}", flush=True)
                return cached

        self.bank.remove_hooks()
        self.bank = build_warm_bridge_bank(
            self.model.config.hidden_size,
            pairs=pairs,
            rank=self.rank,
            pretrained_state=bridge_state,
        )
        self.bank.attach(self.model)
        self.bank.enabled = True
        self.bank.clear_buf()

        t0 = time.time()
        task_reports = []
        math_oks: List[bool] = []
        code_oks: List[bool] = []
        all_oks: List[bool] = []

        with self.act.session(delta):
            ppl = perplexity(self.model, self.tok)
            for spec in MC_SPECS:
                items = self.hold_items[spec.id]
                outs = generate(
                    self.model, self.tok, [it.prompt for it in items],
                    max_new_tokens=self.max_new, batch_size=8,
                )
                oks = []
                rows = []
                for it, o in zip(items, outs):
                    ok = bool(it.check(o))
                    oks.append(ok)
                    rows.append({"prompt": it.prompt, "output": o, "ok": ok})
                all_oks.extend(oks)
                if spec.domain == "math":
                    math_oks.extend(oks)
                else:
                    code_oks.extend(oks)
                acc = round(sum(oks) / len(oks), 4)
                task_reports.append({
                    "task": spec.id,
                    "domain": spec.domain,
                    "n": len(items),
                    "n_ok": int(sum(oks)),
                    "acc": acc,
                    "examples": rows[:2],
                })
                print(f"[eval:{label}] {spec.id} ({spec.domain}) acc={acc:.4f}", flush=True)
                self.bank.clear_buf()

            legacy_reports = {}
            for lkey, litems in self.legacy_hold.items():
                outs = generate(
                    self.model, self.tok, [it.prompt for it in litems],
                    max_new_tokens=self.max_new, batch_size=8,
                )
                oks = [bool(it.check(o)) for it, o in zip(litems, outs)]
                legacy_reports[lkey] = {
                    "n": len(litems),
                    "n_ok": int(sum(oks)),
                    "acc": round(sum(oks) / len(oks), 4),
                    "examples": [
                        {"prompt": it.prompt, "output": o, "ok": ok}
                        for it, o, ok in list(zip(litems, outs, oks))[:2]
                    ],
                }
                self.bank.clear_buf()

        math_mean = round(sum(math_oks) / len(math_oks), 4)
        code_mean = round(sum(code_oks) / len(code_oks), 4)
        total_mean = round(sum(all_oks) / len(all_oks), 4)

        rep = {
            "label": label,
            "delta_name": delta.name,
            "n_layers_edited": len(delta.bias_edits),
            "layers_edited": [b.layer for b in delta.bias_edits],
            "bridge_pairs": pairs,
            "n_bridges": len(pairs),
            "n_bridge_params": self.bank.n_params(),
            "math_acc": math_mean,
            "code_acc": code_mean,
            "mean_acc": total_mean,
            "legacy_bridges_k12_holdout": {
                k: v["acc"] for k, v in legacy_reports.items()
            },
            "legacy_details": legacy_reports,
            "ppl": round(ppl, 3),
            "secs": round(time.time() - t0, 2),
            "tasks": task_reports,
            "flat_ok": all_oks,
            "math_flat_ok": math_oks,
            "code_flat_ok": code_oks,
        }
        _jdump(out_path, rep)
        return rep

    def verify_warm_extension_identity(self) -> Dict[str, Any]:
        prompt = "What is 145+68? Reply with ONLY the number."
        delta = self.deltas["k12_orig"]

        self.bank.remove_hooks()
        b3 = build_warm_bridge_bank(
            self.model.config.hidden_size, PAIRS_BY_CONFIG["k12"],
            rank=self.rank, pretrained_state=self.pretrained_bridges_state,
        )
        b3.attach(self.model)
        with self.act.session(delta):
            out3 = generate(self.model, self.tok, [prompt], max_new_tokens=16, batch_size=1)[0]
        b3.remove_hooks()

        b6 = build_warm_bridge_bank(
            self.model.config.hidden_size, PAIRS_BY_CONFIG["k24"],
            rank=self.rank, pretrained_state=self.pretrained_bridges_state,
        )
        b6.attach(self.model)
        with self.act.session(delta):
            out6 = generate(self.model, self.tok, [prompt], max_new_tokens=16, batch_size=1)[0]
        b6.remove_hooks()

        return {
            "warm_extension_exact_match": out3 == out6,
            "out_3_bridges": out3,
            "out_6_bridges_init": out6,
        }

    def run_all(self) -> Dict[str, Any]:
        t_start = time.time()
        warm_id = self.verify_warm_extension_identity()
        print(f"[math_code] Warm extension identity check: {warm_id}", flush=True)

        # 1) Base Solicitud: K12 + Puentes Entrenados (12 capas 12..23 + 3 puentes de bridges_best.pt)
        rep_base = self.evaluate_condition(
            "1_base_k12_bridges_trained",
            self.deltas["k12_orig"],
            PAIRS_BY_CONFIG["k12"],
            self.pretrained_bridges_state,
        )

        # 2) Edición de Pesos en 12 Capas (K12_mc) + 3 puentes base sin reentrenar
        rep_k12_edit_only = self.evaluate_condition(
            "2_k12_mc_layer_edit_only",
            self.deltas["k12_mc"],
            PAIRS_BY_CONFIG["k12"],
            self.pretrained_bridges_state,
        )

        # 3) 12 capas editadas (K12_mc: 12..23) + 3 puentes conectados para Math+Code
        tr_k12 = self.train_bridges_for_config("k12_mc", self.deltas["k12_mc"], PAIRS_BY_CONFIG["k12"])
        st_k12 = torch.load(tr_k12["ckpt_path"], map_location="cpu", weights_only=False)["state"]
        rep_k12_mc = self.evaluate_condition(
            "3_k12_mc_bridges_3",
            self.deltas["k12_mc"],
            PAIRS_BY_CONFIG["k12"],
            st_k12,
        )

        # 4) +4 capas editadas -> 16 capas (K16_mc: 8..23) + 4 puentes (8→12, 12→16, 16→20, 20→23)
        tr_k16 = self.train_bridges_for_config("k16_mc", self.deltas["k16_mc"], PAIRS_BY_CONFIG["k16"])
        st_k16 = torch.load(tr_k16["ckpt_path"], map_location="cpu", weights_only=False)["state"]
        rep_k16_mc = self.evaluate_condition(
            "4_k16_mc_bridges_4",
            self.deltas["k16_mc"],
            PAIRS_BY_CONFIG["k16"],
            st_k16,
        )

        # 5) +8 capas editadas -> 20 capas (K20_mc: 4..23) + 5 puentes (4→8, 8→12, 12→16, 16→20, 20→23)
        tr_k20 = self.train_bridges_for_config("k20_mc", self.deltas["k20_mc"], PAIRS_BY_CONFIG["k20"])
        st_k20 = torch.load(tr_k20["ckpt_path"], map_location="cpu", weights_only=False)["state"]
        rep_k20_mc = self.evaluate_condition(
            "5_k20_mc_bridges_5",
            self.deltas["k20_mc"],
            PAIRS_BY_CONFIG["k20"],
            st_k20,
        )

        # 6) +12 capas editadas -> 24 capas (K24_mc: 0..23) + 6 puentes (0→4, 4→8, 8→12, 12→16, 16→20, 20→23)
        tr_k24 = self.train_bridges_for_config("k24_mc", self.deltas["k24_mc"], PAIRS_BY_CONFIG["k24"])
        st_k24 = torch.load(tr_k24["ckpt_path"], map_location="cpu", weights_only=False)["state"]
        rep_k24_mc = self.evaluate_condition(
            "6_k24_mc_bridges_6",
            self.deltas["k24_mc"],
            PAIRS_BY_CONFIG["k24"],
            st_k24,
        )

        self.bank.remove_hooks()
        after_checksum = param_checksum(self.model)
        assert after_checksum == self.base_checksum, "Base model checksum mismatch!"

        all_reps = [
            rep_base,
            rep_k12_edit_only,
            rep_k12_mc,
            rep_k16_mc,
            rep_k20_mc,
            rep_k24_mc,
        ]
        best_rep = max(all_reps, key=lambda r: r["mean_acc"])

        def contrast(r_ref, r_new, seed=0):
            pf = paired_flags(r_ref["flat_ok"], r_new["flat_ok"])
            return {
                **pf,
                "bootstrap_ci95": bootstrap_delta_ci(r_ref["flat_ok"], r_new["flat_ok"], seed=seed),
                "mcnemar": mcnemar_exact(pf["n_other_better"], pf["n_other_worse"]),
                "math_delta": round(r_new["math_acc"] - r_ref["math_acc"], 4),
                "code_delta": round(r_new["code_acc"] - r_ref["code_acc"], 4),
            }

        summary = {
            "base_model": self.cfg.hf_id,
            "starting_point": "K12 (b8cf061d) + Puentes Entrenados (bridges_best.pt: 12->16, 16->20, 20->23)",
            "warm_extension_identity": warm_id,
            "base_checksum_ok": after_checksum == self.base_checksum,
            "split_sha256": self.split_man["prompt_sha256"],
            "n_hold_per_task": self.n_hold,
            "n_total_mc_holdout": len(rep_base["flat_ok"]),
            "winner": {
                "label": best_rep["label"],
                "n_layers_edited": best_rep["n_layers_edited"],
                "n_bridges": best_rep["n_bridges"],
                "mean_acc": best_rep["mean_acc"],
                "math_acc": best_rep["math_acc"],
                "code_acc": best_rep["code_acc"],
            },
            "progression_by_layers": [
                {
                    "label": r["label"],
                    "n_layers_edited": r["n_layers_edited"],
                    "layers_range": f"{min(r['layers_edited'])}..{max(r['layers_edited'])}",
                    "n_bridges": r["n_bridges"],
                    "bridge_pairs": r["bridge_pairs"],
                    "math_acc": r["math_acc"],
                    "code_acc": r["code_acc"],
                    "mean_acc": r["mean_acc"],
                    "legacy_arithmetic": r["legacy_bridges_k12_holdout"]["legacy_arithmetic"],
                    "legacy_code_lax": r["legacy_bridges_k12_holdout"]["legacy_code_lax"],
                    "legacy_code_strict_ast": r["legacy_bridges_k12_holdout"]["legacy_code_strict_ast"],
                    "ppl": r["ppl"],
                }
                for r in all_reps
            ],
            "per_task_table": {
                t["task"]: {r["label"]: next(x["acc"] for x in r["tasks"] if x["task"] == t["task"]) for r in all_reps}
                for t in rep_base["tasks"]
            },
            "paired_vs_base_k12_bridges": {
                "k12_mc_edit_only_vs_base": contrast(rep_base, rep_k12_edit_only, seed=1),
                "k12_mc_bridges_3_vs_base": contrast(rep_base, rep_k12_mc, seed=2),
                "k16_mc_bridges_4_vs_base": contrast(rep_base, rep_k16_mc, seed=3),
                "k20_mc_bridges_5_vs_base": contrast(rep_base, rep_k20_mc, seed=4),
                "k24_mc_bridges_6_vs_base": contrast(rep_base, rep_k24_mc, seed=5),
                "winner_vs_k12_mc_bridges_3": contrast(rep_k12_mc, best_rep, seed=6),
            },
            "training_logs": {
                "k12_mc": tr_k12,
                "k16_mc": tr_k16,
                "k20_mc": tr_k20,
                "k24_mc": tr_k24,
            },
            "total_wall_secs": round(time.time() - t_start, 1),
        }
        _jdump(os.path.join(self.out_dir, "summary.json"), summary)
        return summary
