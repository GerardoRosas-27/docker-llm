"""Entrenamiento (solo puentes) + eval base/K12/puentes-zero/puentes-entrenados."""
from __future__ import annotations
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from enjambre.adaptive.snapshot import SnapshotActivator
from enjambre.config import resolve
from enjambre.edits import EditDelta
from enjambre.experts import Item, all_expert_specs
from enjambre.fusion_confirm12.frozen import rebuild_k12_delta, delta_content_sha256
from enjambre.fusion_confirm12.stats import bootstrap_delta_ci, mcnemar_exact, paired_flags
from enjambre.generate import generate, perplexity

from .data import make_splits
from .modules import BridgeBank, BRIDGE_PAIRS

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "artifacts", "bridges-k12")
SEED = 0
K12_HASH_PREFIX = "b8cf061d"


def _jdump(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def param_checksum(model) -> str:
    h = hashlib.sha256()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "bridges" in name:  # skip if any
                continue
            h.update(name.encode())
            h.update(p.detach().cpu().float().contiguous().numpy().tobytes())
    return h.hexdigest()


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad_(False)


class BridgesHarness:
    def __init__(self, out_dir: Optional[str] = None, rank: int = 16,
                 n_train: int = 20, n_val: int = 10, n_hold: int = 50,
                 max_new: int = 64, lr: float = 1e-3, max_steps: int = 400,
                 pilot_steps: int = 20):
        self.cfg = resolve("qwen2.5-0.5b")
        self.out_dir = os.path.abspath(out_dir or DEFAULT_OUT)
        os.makedirs(self.out_dir, exist_ok=True)
        self.rank = rank
        self.n_train, self.n_val, self.n_hold = n_train, n_val, n_hold
        self.max_new = max_new
        self.lr = lr
        self.max_steps = max_steps
        self.pilot_steps = pilot_steps
        self.specs = all_expert_specs()
        torch.manual_seed(SEED)
        torch.set_num_threads(min(8, os.cpu_count() or 8))

        self.tok = AutoTokenizer.from_pretrained(self.cfg.hf_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.hf_id, dtype=torch.float32).eval()
        freeze_model(self.model)
        self.act = SnapshotActivator(self.model)
        self.k12, self.k12_meta = rebuild_k12_delta()
        assert self.k12_meta["delta_sha256"].startswith(K12_HASH_PREFIX)
        self.base_checksum = param_checksum(self.model)
        self.bank = BridgeBank(self.model.config.hidden_size, rank=rank)
        self.bank.attach(self.model)
        # trainables only bank
        for p in self.bank.parameters():
            p.requires_grad_(True)
        self.train_rows, self.val_rows, self.hold_items, self.split_man = make_splits(
            n_train, n_val, n_hold, seed=SEED
        )
        _jdump(os.path.join(self.out_dir, "split_manifest.json"), self.split_man)
        print(f"[bridges] k12={self.k12_meta['delta_sha256'][:16]}… rank={rank} "
              f"n_params={self.bank.n_params()} pairs={BRIDGE_PAIRS}", flush=True)

    def rss_mb(self) -> float:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024.0
        except OSError:
            return -1.0
        return -1.0

    def _chat(self, user: str) -> str:
        msgs = [{"role": "user", "content": user}]
        if hasattr(self.tok, "apply_chat_template"):
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return user

    def _loss_batch(self, prompts: List[str], targets: List[str]) -> torch.Tensor:
        """Teacher-forced loss on response tokens only."""
        self.bank.clear_buf()
        bos = [self._chat(p) for p in prompts]
        full = [b + t for b, t in zip(bos, targets)]
        enc_full = self.tok(full, return_tensors="pt", padding=True, truncation=True, max_length=384)
        enc_prompt = self.tok(bos, return_tensors="pt", padding=True, truncation=True, max_length=320)
        input_ids = enc_full["input_ids"]
        attn = enc_full["attention_mask"]
        # labels: mask prompt tokens (and pad) with -100
        labels = input_ids.clone()
        # left padding: prompt length = number of non-pad in enc_prompt
        prompt_lens = enc_prompt["attention_mask"].sum(1)
        full_lens = attn.sum(1)
        for i in range(input_ids.size(0)):
            # with left padding, content starts at T - full_len
            T = input_ids.size(1)
            start = T - int(full_lens[i])
            plen = int(prompt_lens[i])
            # mask everything before response
            labels[i, : start + plen] = -100
            labels[i, attn[i] == 0] = -100
        out = self.model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits = out.logits
        # shift
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        self.bank.clear_buf()
        return loss

    def flat_supervised(self, rows_by_task: Dict) -> List[Tuple[str, str]]:
        out = []
        for tid in [s.id for s in self.specs]:
            for p, tgt, _ in rows_by_task[tid]:
                out.append((p, tgt))
        return out

    def identity_check(self) -> Dict[str, Any]:
        """Con K12 activo y puentes en init: greedy debe coincidir con K12 solo."""
        prompt = "What is 3+4? Reply with ONLY the number."
        self.bank.enabled = True
        self.bank.clear_buf()
        # ensure zero-init identity of bank contribution
        with torch.no_grad():
            for b in self.bank.bridges.values():
                assert torch.allclose(b.down.weight, torch.zeros_like(b.down.weight))
        with self.act.session(self.k12):
            self.bank.clear_buf()
            a = generate(self.model, self.tok, [prompt], max_new_tokens=16, batch_size=1)[0]
        self.bank.enabled = False
        with self.act.session(self.k12):
            b = generate(self.model, self.tok, [prompt], max_new_tokens=16, batch_size=1)[0]
        self.bank.enabled = True
        ok = a == b
        # grad flow
        self.model.train()
        loss = self._loss_batch([prompt], ["7"])
        loss.backward()
        grads = {n: (p.grad is not None and float(p.grad.abs().sum()) > 0)
                 for n, p in self.bank.named_parameters()}
        self.bank.zero_grad(set_to_none=True)
        self.model.eval()
        self.bank.clear_buf()
        # base immutable
        after = param_checksum(self.model)
        return {
            "identity_greedy": ok, "out_a": a[:80], "out_b": b[:80],
            "grad_nonzero_any": any(grads.values()),
            "grad_details": grads,
            "base_checksum_match": after == self.base_checksum,
        }

    def pilot(self) -> Dict[str, Any]:
        opt = torch.optim.AdamW([p for p in self.bank.parameters() if p.requires_grad], lr=self.lr)
        data = self.flat_supervised(self.train_rows)
        t0 = time.time()
        self.model.train()
        losses = []
        for step in range(self.pilot_steps):
            batch = data[step % len(data):(step % len(data)) + 2]
            if len(batch) < 2:
                batch = data[:2]
            ps, ts = zip(*batch)
            with self.act.session(self.k12):
                loss = self._loss_batch(list(ps), list(ts))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            self.bank.clear_buf()
        secs = time.time() - t0
        per = secs / max(1, self.pilot_steps)
        est_full = per * self.max_steps
        self.model.eval()
        # restore zero-ish? no — pilot mutates weights; re-init bank for real train
        info = {
            "pilot_steps": self.pilot_steps, "secs": round(secs, 2),
            "sec_per_step": round(per, 3), "est_full_secs": round(est_full, 1),
            "loss_first": losses[0], "loss_last": losses[-1],
            "rss_mb": round(self.rss_mb(), 1),
        }
        _jdump(os.path.join(self.out_dir, "pilot.json"), info)
        print(f"[bridges] pilot {info}", flush=True)
        # reinit bank weights for clean train
        self.bank.remove_hooks()
        self.bank = BridgeBank(self.model.config.hidden_size, rank=self.rank)
        self.bank.attach(self.model)
        for p in self.bank.parameters():
            p.requires_grad_(True)
        freeze_model(self.model)
        for p in self.bank.parameters():
            p.requires_grad_(True)
        return info

    @torch.no_grad()
    def eval_split(self, items_by_task: Dict, label: str, delta: Optional[EditDelta],
                   bridges_on: bool, measure_ppl: bool = True) -> Dict[str, Any]:
        path = os.path.join(self.out_dir, f"results_{label}.json")
        self.bank.enabled = bridges_on
        self.bank.clear_buf()
        t0 = time.time()
        tasks = []
        all_ok = []
        with self.act.session(delta):
            ppl = perplexity(self.model, self.tok) if measure_ppl else None
            for spec in self.specs:
                items = items_by_task[spec.id]
                outs = generate(self.model, self.tok, [it.prompt for it in items],
                                max_new_tokens=self.max_new, batch_size=4)
                oks = []
                rows = []
                for it, o in zip(items, outs):
                    ok = bool(it.check(o))
                    oks.append(ok)
                    rows.append({"prompt": it.prompt, "output": o, "ok": ok})
                all_ok.extend(oks)
                tasks.append({"task": spec.id, "n": len(items), "n_ok": int(sum(oks)),
                              "acc": round(sum(oks) / len(oks), 4), "examples": rows[:2]})
                print(f"[bridges] {label} {spec.id} {tasks[-1]['acc']}", flush=True)
                self.bank.clear_buf()
        mean = round(sum(t["acc"] for t in tasks) / len(tasks), 4)
        rep = {
            "label": label, "mean_acc": mean, "global_acc": round(sum(all_ok)/len(all_ok), 4),
            "n_total": len(all_ok), "ppl": None if ppl is None else round(ppl, 3),
            "secs": round(time.time() - t0, 2), "rss_mb": round(self.rss_mb(), 1),
            "bridges_on": bridges_on, "tasks": tasks, "flat_ok": all_ok,
        }
        _jdump(path, rep)
        return rep

    def eval_val_mean(self) -> float:
        # quick val: convert val rows to items
        items = {
            tid: [Item(prompt=p, check=chk, meta={"target": tgt}) for p, tgt, chk in rows]
            for tid, rows in self.val_rows.items()
        }
        # temporary eval without saving huge
        self.bank.enabled = True
        oks = []
        with self.act.session(self.k12):
            for spec in self.specs:
                its = items[spec.id]
                outs = generate(self.model, self.tok, [it.prompt for it in its],
                                max_new_tokens=self.max_new, batch_size=4)
                for it, o in zip(its, outs):
                    oks.append(bool(it.check(o)))
                self.bank.clear_buf()
        return sum(oks) / max(1, len(oks))

    def train(self) -> Dict[str, Any]:
        opt = torch.optim.AdamW([p for p in self.bank.parameters() if p.requires_grad], lr=self.lr)
        data = self.flat_supervised(self.train_rows)
        # shuffle each epoch
        import random
        rng = random.Random(SEED)
        t0 = time.time()
        best_val = -1.0
        best_path = os.path.join(self.out_dir, "bridges_best.pt")
        history = []
        self.model.train()
        step = 0
        deadline = t0 + 28 * 60  # hard wall ~28 min train
        while step < self.max_steps and time.time() < deadline:
            rng.shuffle(data)
            for i in range(0, len(data), 2):
                if step >= self.max_steps or time.time() >= deadline:
                    break
                batch = data[i:i + 2]
                if len(batch) < 1:
                    continue
                ps, ts = zip(*batch)
                with self.act.session(self.k12):
                    loss = self._loss_batch(list(ps), list(ts))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.bank.parameters(), 1.0)
                opt.step()
                self.bank.clear_buf()
                if step % 40 == 0:
                    self.model.eval()
                    # checksum base still frozen
                    assert param_checksum(self.model) == self.base_checksum
                    v = self.eval_val_mean()
                    self.model.train()
                    row = {"step": step, "loss": float(loss.detach()), "val": round(v, 4),
                           "secs": round(time.time() - t0, 1), "rss_mb": round(self.rss_mb(), 1)}
                    history.append(row)
                    print(f"[bridges] step={step} loss={row['loss']:.3f} val={v:.3f}", flush=True)
                    if v > best_val:
                        best_val = v
                        torch.save({"state": self.bank.state_dict(), "step": step, "val": v,
                                    "rank": self.rank, "pairs": BRIDGE_PAIRS}, best_path)
                step += 1
        self.model.eval()
        # load best
        if os.path.isfile(best_path):
            ck = torch.load(best_path, map_location="cpu", weights_only=False)
            self.bank.load_state_dict(ck["state"])
            print(f"[bridges] loaded best step={ck['step']} val={ck['val']}", flush=True)
        info = {
            "steps": step, "best_val": best_val, "history": history,
            "secs": round(time.time() - t0, 1), "rss_mb": round(self.rss_mb(), 1),
            "n_params": self.bank.n_params(), "base_checksum": self.base_checksum,
            "k12_sha": self.k12_meta["delta_sha256"],
        }
        _jdump(os.path.join(self.out_dir, "train_log.json"), info)
        # also save final
        torch.save({"state": self.bank.state_dict(), "rank": self.rank, "pairs": BRIDGE_PAIRS},
                   os.path.join(self.out_dir, "bridges_final.pt"))
        return info

    def run(self) -> Dict[str, Any]:
        idn = self.identity_check()
        _jdump(os.path.join(self.out_dir, "identity.json"), idn)
        print(f"[bridges] identity={idn}", flush=True)
        if not idn["identity_greedy"] or not idn["grad_nonzero_any"] or not idn["base_checksum_match"]:
            print("[bridges] WARNING identity/grad/checksum issue — continuing with caution", flush=True)
        pilot = self.pilot()
        # adjust max_steps if estimate too large
        if pilot["est_full_secs"] > 26 * 60:
            new_steps = max(80, int(26 * 60 / max(0.1, pilot["sec_per_step"])))
            print(f"[bridges] reducing max_steps {self.max_steps}→{new_steps} by pilot budget", flush=True)
            self.max_steps = new_steps
        train_info = self.train()
        # holdout evals
        hold = self.hold_items
        # zero bridges: reinit clone for zero condition
        # save trained state
        trained = {k: v.detach().cpu().clone() for k, v in self.bank.state_dict().items()}
        # zero bank
        self.bank.remove_hooks()
        zero_bank = BridgeBank(self.model.config.hidden_size, rank=self.rank)
        zero_bank.attach(self.model)
        self.bank = zero_bank

        base = self.eval_split(hold, "base", None, bridges_on=False)
        k12 = self.eval_split(hold, "k12", self.k12, bridges_on=False)
        k12_zero = self.eval_split(hold, "k12_bridges_zero", self.k12, bridges_on=True)
        # restore trained
        self.bank.remove_hooks()
        self.bank = BridgeBank(self.model.config.hidden_size, rank=self.rank)
        self.bank.load_state_dict(trained)
        self.bank.attach(self.model)
        k12_tr = self.eval_split(hold, "k12_bridges_trained", self.k12, bridges_on=True)

        def flat(r):
            return r["flat_ok"]

        paired = {
            "trained_vs_k12": {**paired_flags(flat(k12), flat(k12_tr)),
                               "bootstrap_ci95": bootstrap_delta_ci(flat(k12), flat(k12_tr), seed=0),
                               "mcnemar": mcnemar_exact(*[paired_flags(flat(k12), flat(k12_tr))[k]
                                                          for k in ("n_other_better", "n_other_worse")])},
            "trained_vs_base": {**paired_flags(flat(base), flat(k12_tr)),
                                "bootstrap_ci95": bootstrap_delta_ci(flat(base), flat(k12_tr), seed=1)},
            "zero_vs_k12": {**paired_flags(flat(k12), flat(k12_zero)),
                            "bootstrap_ci95": bootstrap_delta_ci(flat(k12), flat(k12_zero), seed=2)},
            "k12_vs_base": {**paired_flags(flat(base), flat(k12)),
                            "bootstrap_ci95": bootstrap_delta_ci(flat(base), flat(k12), seed=3)},
        }
        # after train checksum
        after = param_checksum(self.model)
        summary = {
            "means": {
                "base": base["mean_acc"], "k12": k12["mean_acc"],
                "k12_bridges_zero": k12_zero["mean_acc"],
                "k12_bridges_trained": k12_tr["mean_acc"],
            },
            "ppl": {k: v.get("ppl") for k, v in
                    [("base", base), ("k12", k12), ("zero", k12_zero), ("trained", k12_tr)]},
            "paired": paired,
            "identity": idn,
            "pilot": pilot,
            "train": {k: train_info[k] for k in ("steps", "best_val", "secs", "rss_mb", "n_params")},
            "n_hold_per_task": self.n_hold,
            "n_total": base["n_total"],
            "rank": self.rank,
            "pairs": BRIDGE_PAIRS,
            "k12_sha": self.k12_meta["delta_sha256"],
            "base_checksum_ok": after == self.base_checksum,
            "split_sha": self.split_man["prompt_sha256"],
            "limits": [
                "10 tareas sintéticas; holdout n=50/tarea si viable",
                "Puentes entrenados ≠ inteligencia general",
                "Selección por val; holdout no usado para tuning",
            ],
        }
        _jdump(os.path.join(self.out_dir, "summary.json"), summary)
        print(json.dumps({"means": summary["means"], "paired_trained_k12": paired["trained_vs_k12"]["bootstrap_ci95"],
                          "secs_train": train_info["secs"]}, indent=2), flush=True)
        self.bank.remove_hooks()
        return summary
