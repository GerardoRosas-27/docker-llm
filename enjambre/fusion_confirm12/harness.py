"""Confirmación K12 vs base vs prior en holdout nuevo (sin reoptimizar)."""
from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from enjambre.adaptive.snapshot import SnapshotActivator
from enjambre.build_experts import load_delta
from enjambre.config import resolve
from enjambre.edits import EditDelta
from enjambre.experts import all_expert_specs
from enjambre.generate import generate, perplexity

from .confirm_holdout import assert_no_overlap, build_confirm_holdout, holdout_manifest
from .frozen import DEFAULT_DIRS, DEFAULT_PRIOR, FROZEN_K12, export_k12, rebuild_k12_delta
from .stats import bootstrap_delta_ci, mcnemar_exact, paired_flags

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "artifacts", "fusion-confirm12")
SEED = 0


def _jdump(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


class Confirm12Harness:
    def __init__(
        self,
        model_key: str = "qwen2.5-0.5b",
        out_dir: Optional[str] = None,
        holdout_n: int = 50,
        max_new: int = 64,
        load_model: bool = True,
    ):
        self.cfg = resolve(model_key)
        self.out_dir = os.path.abspath(out_dir or DEFAULT_OUT)
        os.makedirs(self.out_dir, exist_ok=True)
        self.holdout_n = holdout_n
        self.max_new = max_new
        self.specs = all_expert_specs()
        self.model = None
        self.tok = None
        self.act = None
        self.base_ppl = None
        self.k12: Optional[EditDelta] = None
        self.k12_meta: Optional[Dict] = None
        self.prior: Optional[EditDelta] = None
        self.holdout = None
        if load_model:
            self._load()

    def rss_mb(self) -> float:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024.0
        except OSError:
            return -1.0
        return -1.0

    def _load(self):
        torch.manual_seed(SEED)
        torch.set_num_threads(min(8, os.cpu_count() or 8))
        self.tok = AutoTokenizer.from_pretrained(self.cfg.hf_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.hf_id, dtype=torch.float32).eval()
        self.act = SnapshotActivator(self.model)
        self.base_ppl = perplexity(self.model, self.tok)
        self.prior = load_delta(DEFAULT_PRIOR)
        self.k12, self.k12_meta = rebuild_k12_delta(DEFAULT_DIRS, DEFAULT_PRIOR)
        export_k12(self.out_dir, self.k12, self.k12_meta)
        print(f"[confirm12] base_ppl={self.base_ppl:.3f} k12_sha={self.k12_meta['delta_sha256'][:16]}… layers={FROZEN_K12['layers']}", flush=True)

    def ensure_holdout(self):
        if self.holdout is None:
            self.holdout = build_confirm_holdout(self.holdout_n)
            assert_no_overlap(self.holdout)
            man = holdout_manifest(self.holdout)
            _jdump(os.path.join(self.out_dir, "confirm_holdout_manifest.json"), man)
            print(f"[confirm12] holdout n/task={self.holdout_n} sha={man['prompt_sha256'][:16]}…", flush=True)
        return self.holdout

    def _eval_condition(self, label: str, delta: Optional[EditDelta], resume: bool = True) -> Dict[str, Any]:
        path = os.path.join(self.out_dir, f"results_{label}.json")
        if resume and os.path.isfile(path):
            print(f"[confirm12] resume {label}", flush=True)
            return json.load(open(path, encoding="utf-8"))
        hold = self.ensure_holdout()
        t0 = time.time()
        tasks = []
        all_ok = []
        with self.act.session(delta):
            ppl = perplexity(self.model, self.tok)
            for spec in self.specs:
                items = hold[spec.id]
                prompts = [it.prompt for it in items]
                outs = generate(self.model, self.tok, prompts, max_new_tokens=self.max_new, batch_size=4)
                rows = []
                oks = []
                for it, o in zip(items, outs):
                    ok = bool(it.check(o))
                    oks.append(ok)
                    rows.append({
                        "prompt": it.prompt,
                        "output": o,
                        "ok": ok,
                        "meta": it.meta,
                    })
                all_ok.extend(oks)
                tasks.append({
                    "task": spec.id,
                    "n": len(items),
                    "n_ok": int(sum(oks)),
                    "acc": round(sum(oks) / len(oks), 4),
                    "examples": rows,
                })
                print(f"[confirm12] {label} {spec.id} acc={tasks[-1]['acc']} ({tasks[-1]['n_ok']}/{tasks[-1]['n']})", flush=True)
        mean = round(sum(t["acc"] for t in tasks) / len(tasks), 4)
        rep = {
            "label": label,
            "mean_acc": mean,
            "n_total": len(all_ok),
            "n_ok_total": int(sum(all_ok)),
            "global_acc": round(sum(all_ok) / len(all_ok), 4),
            "ppl": round(ppl, 3),
            "secs": round(time.time() - t0, 2),
            "rss_mb": round(self.rss_mb(), 1),
            "max_new": self.max_new,
            "delta_name": None if delta is None else delta.name,
            "tasks": tasks,
            "flat_ok": all_ok,
        }
        _jdump(path, rep)
        return rep

    def run(self, resume: bool = True) -> Dict[str, Any]:
        T0 = time.time()
        self.ensure_holdout()
        # freeze meta before any generation of new labels beyond export
        _jdump(os.path.join(self.out_dir, "frozen_config.json"), {
            **FROZEN_K12,
            "k12_meta": self.k12_meta,
            "prior_layers": [b.layer for b in self.prior.bias_edits],
            "prior_scales": [b.scale for b in self.prior.bias_edits],
            "base_ppl": self.base_ppl,
            "holdout_n": self.holdout_n,
            "max_new": self.max_new,
            "selection_note": "NO selection on confirm data; K12 frozen from sweep val",
        })
        base = self._eval_condition("base", None, resume=resume)
        prior = self._eval_condition("prior", self.prior, resume=resume)
        k12 = self._eval_condition("k12", self.k12, resume=resume)

        # paired stats flatten same order (specs × items)
        def flat(rep):
            return [ex["ok"] for t in rep["tasks"] for ex in t["examples"]]

        fb, fp, fk = flat(base), flat(prior), flat(k12)
        paired = {
            "k12_vs_base": {
                **paired_flags(fb, fk),
                "bootstrap_ci95": bootstrap_delta_ci(fb, fk, seed=0),
                "mcnemar": mcnemar_exact(
                    paired_flags(fb, fk)["n_other_better"],
                    paired_flags(fb, fk)["n_other_worse"],
                ),
            },
            "k12_vs_prior": {
                **paired_flags(fp, fk),
                "bootstrap_ci95": bootstrap_delta_ci(fp, fk, seed=1),
                "mcnemar": mcnemar_exact(
                    paired_flags(fp, fk)["n_other_better"],
                    paired_flags(fp, fk)["n_other_worse"],
                ),
            },
            "prior_vs_base": {
                **paired_flags(fb, fp),
                "bootstrap_ci95": bootstrap_delta_ci(fb, fp, seed=2),
                "mcnemar": mcnemar_exact(
                    paired_flags(fb, fp)["n_other_better"],
                    paired_flags(fb, fp)["n_other_worse"],
                ),
            },
        }
        # per-task paired gains
        per_task = []
        for tb, tk in zip(base["tasks"], k12["tasks"]):
            assert tb["task"] == tk["task"]
            b_ok = [ex["ok"] for ex in tb["examples"]]
            k_ok = [ex["ok"] for ex in tk["examples"]]
            per_task.append({
                "task": tb["task"],
                "base_acc": tb["acc"],
                "k12_acc": tk["acc"],
                "prior_acc": next(t["acc"] for t in prior["tasks"] if t["task"] == tb["task"]),
                "delta_k12_base": round(tk["acc"] - tb["acc"], 4),
                **{k: paired_flags(b_ok, k_ok)[k] for k in ("n_other_better", "n_other_worse")},
            })

        summary = {
            "protocol": "Confirmación K12 congelada; holdout nuevo independiente; sin tuning",
            "frozen": FROZEN_K12,
            "k12_delta_sha256": self.k12_meta["delta_sha256"],
            "holdout_n": self.holdout_n,
            "n_total": base["n_total"],
            "means": {
                "base": base["mean_acc"],
                "prior": prior["mean_acc"],
                "k12": k12["mean_acc"],
            },
            "global_acc": {
                "base": base["global_acc"],
                "prior": prior["global_acc"],
                "k12": k12["global_acc"],
            },
            "ppl": {"base": base["ppl"], "prior": prior["ppl"], "k12": k12["ppl"]},
            "paired": paired,
            "per_task": per_task,
            "secs_total": round(time.time() - T0, 1),
            "rss_mb": round(self.rss_mb(), 1),
            "limits": [
                "10 tareas sintéticas; no razonamiento general",
                "No comparar 0.77 del holdout n=20 previo con base de este holdout",
                "K12 no se re-seleccionó; params del barrido congelados",
            ],
        }
        _jdump(os.path.join(self.out_dir, "confirm_summary.json"), summary)
        print(json.dumps({"means": summary["means"], "paired_k12_base": paired["k12_vs_base"]["bootstrap_ci95"], "secs": summary["secs_total"]}, indent=2), flush=True)
        return summary
