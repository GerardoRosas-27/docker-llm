"""Evalúa base / K12 / K20-A / K20-B en holdout nuevo (diseños congelados)."""
from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from enjambre.adaptive.snapshot import SnapshotActivator
from enjambre.config import resolve
from enjambre.edits import EditDelta
from enjambre.experts import all_expert_specs
from enjambre.fusion_confirm12.stats import bootstrap_delta_ci, mcnemar_exact, paired_flags
from enjambre.generate import generate, perplexity

from .build import build_all_deltas, export_deltas
from .holdout import assert_no_overlap, build_k20_holdout, holdout_manifest

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "artifacts", "fusion-20capas")
SEED = 0
CONDS = ("base", "k12", "k20_A", "k20_B")


def _jdump(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


class Fusion20Harness:
    def __init__(self, model_key: str = "qwen2.5-0.5b", out_dir: Optional[str] = None,
                 holdout_n: int = 50, max_new: int = 64, load_model: bool = True):
        self.cfg = resolve(model_key)
        self.out_dir = os.path.abspath(out_dir or DEFAULT_OUT)
        os.makedirs(self.out_dir, exist_ok=True)
        self.holdout_n = holdout_n
        self.max_new = max_new
        self.specs = all_expert_specs()
        self.model = self.tok = self.act = None
        self.base_ppl = None
        self.bundle = None
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
        # freeze designs + export BEFORE holdout eval
        self.bundle = build_all_deltas()
        export_deltas(self.out_dir, self.bundle)
        print(
            f"[k20] base_ppl={self.base_ppl:.3f} "
            f"l2_k12={self.bundle['l2_k12']:.4f} l2_A={self.bundle['k20_A']['meta']['total_l2']:.4f} "
            f"l2_B={self.bundle['k20_B']['meta']['total_l2']:.4f}",
            flush=True,
        )

    def ensure_holdout(self):
        if self.holdout is None:
            self.holdout = build_k20_holdout(self.holdout_n)
            assert_no_overlap(self.holdout)
            man = holdout_manifest(self.holdout)
            _jdump(os.path.join(self.out_dir, "holdout_manifest.json"), man)
            print(f"[k20] holdout n={self.holdout_n} sha={man['prompt_sha256'][:16]}…", flush=True)
        return self.holdout

    def _delta(self, label: str) -> Optional[EditDelta]:
        if label == "base":
            return None
        if label == "k12":
            return self.bundle["k12"]["delta"]
        if label == "k20_A":
            return self.bundle["k20_A"]["delta"]
        if label == "k20_B":
            return self.bundle["k20_B"]["delta"]
        raise ValueError(label)

    def _eval_condition(self, label: str, resume: bool = True) -> Dict[str, Any]:
        path = os.path.join(self.out_dir, f"results_{label}.json")
        if resume and os.path.isfile(path):
            print(f"[k20] resume {label}", flush=True)
            return json.load(open(path, encoding="utf-8"))
        hold = self.ensure_holdout()
        delta = self._delta(label)
        t0 = time.time()
        tasks = []
        all_ok = []
        # ALWAYS fresh base via session restore — never stack edits
        with self.act.session(delta):
            ppl = perplexity(self.model, self.tok)
            for spec in self.specs:
                items = hold[spec.id]
                outs = generate(self.model, self.tok, [it.prompt for it in items],
                                max_new_tokens=self.max_new, batch_size=4)
                rows, oks = [], []
                for it, o in zip(items, outs):
                    ok = bool(it.check(o))
                    oks.append(ok)
                    rows.append({"prompt": it.prompt, "output": o, "ok": ok, "meta": it.meta})
                all_ok.extend(oks)
                tasks.append({
                    "task": spec.id, "n": len(items), "n_ok": int(sum(oks)),
                    "acc": round(sum(oks) / len(oks), 4), "examples": rows,
                })
                print(f"[k20] {label} {spec.id} {tasks[-1]['acc']} ({tasks[-1]['n_ok']}/{tasks[-1]['n']})", flush=True)
        mean = round(sum(t["acc"] for t in tasks) / len(tasks), 4)
        rep = {
            "label": label, "mean_acc": mean, "n_total": len(all_ok),
            "n_ok_total": int(sum(all_ok)),
            "global_acc": round(sum(all_ok) / len(all_ok), 4),
            "ppl": round(ppl, 3), "secs": round(time.time() - t0, 2),
            "rss_mb": round(self.rss_mb(), 1), "max_new": self.max_new,
            "delta_name": None if delta is None else delta.name,
            "tasks": tasks, "flat_ok": all_ok,
        }
        _jdump(path, rep)
        return rep

    def run(self, resume: bool = True) -> Dict[str, Any]:
        T0 = time.time()
        self.ensure_holdout()
        _jdump(os.path.join(self.out_dir, "frozen_before_eval.json"), {
            "note": "Designs/hashes frozen before generating confirm outputs",
            "hashes": {k: self.bundle[k]["meta"]["delta_sha256"] for k in ("k12", "k20_A", "k20_B")},
            "l2": {
                "k12": self.bundle["l2_k12"],
                "k20_A": self.bundle["k20_A"]["meta"]["total_l2"],
                "k20_B": self.bundle["k20_B"]["meta"]["total_l2"],
            },
            "layers": {
                "k12": self.bundle["k12"]["meta"]["layers"],
                "k20_A": self.bundle["k20_A"]["meta"]["layers"],
                "k20_B": self.bundle["k20_B"]["meta"]["layers"],
            },
            "scales": {
                "k12": self.bundle["scale_k12"],
                "k20_A_per_layer": self.bundle["k20_A"]["meta"]["scale_per_layer"],
                "k20_B_per_layer": self.bundle["k20_B"]["meta"]["scale_per_layer"],
            },
            "holdout_n": self.holdout_n,
            "seed": SEED,
        })
        results = {c: self._eval_condition(c, resume=resume) for c in CONDS}

        def flat(rep):
            return [ex["ok"] for t in rep["tasks"] for ex in t["examples"]]

        fb, f12, fa, fb20 = flat(results["base"]), flat(results["k12"]), flat(results["k20_A"]), flat(results["k20_B"])

        def contrast(name, a, b, seed):
            pf = paired_flags(a, b)
            return {
                **pf,
                "bootstrap_ci95": bootstrap_delta_ci(a, b, seed=seed),
                "mcnemar": mcnemar_exact(pf["n_other_better"], pf["n_other_worse"]),
                "name": name,
            }

        paired = {
            "k20A_vs_k12": contrast("k20A_vs_k12", f12, fa, 10),
            "k20B_vs_k12": contrast("k20B_vs_k12", f12, fb20, 11),
            "k20A_vs_base": contrast("k20A_vs_base", fb, fa, 12),
            "k20B_vs_base": contrast("k20B_vs_base", fb, fb20, 13),
            "k12_vs_base": contrast("k12_vs_base", fb, f12, 14),
            "k20A_vs_k20B": contrast("k20A_vs_k20B", fb20, fa, 15),
        }

        per_task = []
        for i, spec in enumerate(self.specs):
            row = {"task": spec.id}
            for lab in CONDS:
                t = results[lab]["tasks"][i]
                row[f"{lab}_acc"] = t["acc"]
            b_ok = [ex["ok"] for ex in results["base"]["tasks"][i]["examples"]]
            a_ok = [ex["ok"] for ex in results["k20_A"]["tasks"][i]["examples"]]
            bb_ok = [ex["ok"] for ex in results["k20_B"]["tasks"][i]["examples"]]
            k_ok = [ex["ok"] for ex in results["k12"]["tasks"][i]["examples"]]
            row["A_vs_k12"] = paired_flags(k_ok, a_ok)
            row["B_vs_k12"] = paired_flags(k_ok, bb_ok)
            row["A_vs_base"] = paired_flags(b_ok, a_ok)
            per_task.append(row)

        summary = {
            "protocol": "K20 extend K12: A literal more L2; B same total L2; no tuning",
            "means": {c: results[c]["mean_acc"] for c in CONDS},
            "global_acc": {c: results[c]["global_acc"] for c in CONDS},
            "ppl": {c: results[c]["ppl"] for c in CONDS},
            "n_total": results["base"]["n_total"],
            "holdout_n": self.holdout_n,
            "hashes": {k: self.bundle[k]["meta"]["delta_sha256"] for k in ("k12", "k20_A", "k20_B")},
            "l2": {
                "k12": self.bundle["l2_k12"],
                "k20_A": self.bundle["k20_A"]["meta"]["total_l2"],
                "k20_B": self.bundle["k20_B"]["meta"]["total_l2"],
            },
            "paired": paired,
            "per_task": per_task,
            "secs_total": round(time.time() - T0, 1),
            "rss_mb": round(self.rss_mb(), 1),
            "limits": [
                "10 tareas sintéticas; no inteligencia general",
                "No comparar 0.738 del confirm K12 previo con este holdout",
                "A tiene más presupuesto L2; B iguala L2(K12)",
                "Sesgos mlp.down_proj; no ΔW ni bloques nuevos",
            ],
        }
        _jdump(os.path.join(self.out_dir, "summary.json"), summary)
        print(json.dumps({"means": summary["means"], "l2": summary["l2"],
                          "A_vs_k12": paired["k20A_vs_k12"]["bootstrap_ci95"],
                          "B_vs_k12": paired["k20B_vs_k12"]["bootstrap_ci95"],
                          "secs": summary["secs_total"]}, indent=2), flush=True)
        return summary
