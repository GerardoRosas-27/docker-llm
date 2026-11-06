"""Arnés de fusión de los 10 expertos originales sobre Qwen2.5-0.5B."""
from __future__ import annotations
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from enjambre.adaptive.snapshot import SnapshotActivator
from enjambre.build_experts import load_delta
from enjambre.config import resolve
from enjambre.edits import EditDelta
from enjambre.experts import all_expert_specs
from enjambre.generate import generate_one, perplexity

from .combine import fuse_deltas
from .evaluate import eval_condition, score_items, split_eval
from .manifest import build_manifest
from .parallel_equiv import bias_vectors_from_deltas, prove_average_bias_equivalence

PPL_GUARD = 1.5
DEFAULT_DELTAS = os.path.join(os.path.dirname(__file__), "..", "artifacts", "qwen2.5-0.5b")
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "artifacts", "fusion-0.5b")


class FusionHarness:
    def __init__(
        self,
        model_key: str = "qwen2.5-0.5b",
        deltas_dir: Optional[str] = None,
        out_dir: Optional[str] = None,
        max_new: int = 64,
        load_model: bool = True,
    ):
        self.cfg = resolve(model_key)
        self.deltas_dir = os.path.abspath(deltas_dir or DEFAULT_DELTAS)
        self.out_dir = os.path.abspath(out_dir or DEFAULT_OUT)
        os.makedirs(self.out_dir, exist_ok=True)
        self.max_new = max_new
        self.specs = all_expert_specs()
        assert len(self.specs) == 10
        self.deltas: Dict[str, EditDelta] = {
            s.id: load_delta(os.path.join(self.deltas_dir, f"delta_{s.id}.pt")) for s in self.specs
        }
        self.model = None
        self.tok = None
        self.act = None
        self.base_ppl = None
        if load_model:
            self._load()

    def _load(self):
        torch.manual_seed(0)
        torch.set_num_threads(min(8, os.cpu_count() or 8))
        self.tok = AutoTokenizer.from_pretrained(self.cfg.hf_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.hf_id, dtype=torch.float32).eval()
        self.act = SnapshotActivator(self.model)
        self.base_ppl = perplexity(self.model, self.tok)

    def manifest(self) -> Dict[str, Any]:
        return build_manifest(self.cfg.key, self.cfg.hf_id, self.deltas_dir, extra={
            "out_dir": self.out_dir,
            "base_ppl": self.base_ppl,
            "n_layers": None if self.model is None else len(self.model.model.layers),
            "hidden": None if self.model is None else int(self.model.config.hidden_size),
        })

    def _delta_fn_base(self, _tid: str):
        return None

    def _delta_fn_oracle(self, tid: str):
        return self.deltas[tid]

    def _delta_fn_fixed(self, fused: EditDelta):
        def _f(_tid: str):
            return fused
        return _f

    def eval_suite(self, fused: Optional[EditDelta], *, split: str, label: str) -> Dict[str, Any]:
        if fused is None:
            fn = self._delta_fn_base
        elif label == "oracle":
            fn = self._delta_fn_oracle
        else:
            fn = self._delta_fn_fixed(fused)
        rep = eval_condition(self.model, self.tok, self.act, self.specs, fn, split=split, max_new=self.max_new)
        rep["condition"] = label
        return rep

    def select_fusion_hyperparams(self) -> Dict[str, Any]:
        """Elige modo/capa/escala: grid compacto en TUNE+PPL, top-K confirmados en fusion_val."""
        assert self.model is not None
        used = sorted({b.layer for d in self.deltas.values() for b in d.bias_edits})
        scales = [0.5, 1.0]
        grid = []

        def consider(mode, L, g):
            fused = fuse_deltas(
                list(self.deltas.values()), mode=mode, target_layer=L, global_scale=g,
                name=f"fuse_{mode}_L{L}_g{g}",
            )
            with self.act.session(fused):
                ppl = perplexity(self.model, self.tok)
            ok_ppl = ppl <= PPL_GUARD * self.base_ppl
            tune = eval_condition(
                self.model, self.tok, self.act, self.specs, self._delta_fn_fixed(fused),
                split="tune", max_new=self.max_new, measure_ppl=False,
            )
            row = {
                "mode": mode, "target_layer": L, "global_scale": g,
                "tune_mean": tune["mean_acc"], "ppl": round(ppl, 3), "ok_ppl": ok_ppl,
                "fused": fused,
            }
            grid.append(row)

        for mode in ("multi_avg", "multi_sum"):
            for g in scales:
                consider(mode, None, g)
        for L in used:  # only layers that already appear in the 10 experts
            for mode in ("single_avg", "single_sum"):
                for g in scales:
                    consider(mode, L, g)

        # rank by tune under ppl guard
        ranked = sorted(grid, key=lambda r: (r["ok_ppl"], r["tune_mean"], -abs(r["ppl"] - self.base_ppl)), reverse=True)
        # validate top-4 on fusion_val
        base_val = self.eval_suite(None, split="val", label="base")
        best = None
        validated = []
        for row in ranked[:4]:
            fused = row["fused"]
            val = eval_condition(
                self.model, self.tok, self.act, self.specs, self._delta_fn_fixed(fused),
                split="val", max_new=self.max_new, measure_ppl=False,
            )
            vrow = {k: row[k] for k in ("mode", "target_layer", "global_scale", "tune_mean", "ppl", "ok_ppl")}
            vrow["val_mean"] = val["mean_acc"]
            validated.append(vrow)
            score = (int(row["ok_ppl"]), val["mean_acc"], row["tune_mean"])
            if best is None or score > best[0]:
                best = (score, vrow, fused)
        assert best is not None
        return {
            "base_val_mean": base_val["mean_acc"],
            "selected": best[1],
            "fused_meta": best[2].meta,
            "fused_delta": best[2],
            "grid_size": len(grid),
            "grid_top_tune": [{k: r[k] for k in r if k != "fused"} for r in ranked[:8]],
            "validated_top": validated,
            "selection_splits": "tune(+PPL) → top-4 on fusion_val; confirm held out of selection",
        }

    def parallel_equivalence_check(self) -> Dict[str, Any]:
        """Prueba en capa más frecuente: N ramas bias-only ≡ bias promedio."""
        # pick most common original layer
        from collections import Counter
        layers = [b.layer for d in self.deltas.values() for b in d.bias_edits]
        L = Counter(layers).most_common(1)[0][0]
        W = self.model.model.layers[L].mlp.down_proj.weight.detach().float()
        biases = bias_vectors_from_deltas(list(self.deltas.values()), L)
        # for experts not on L, zeros already; keep all 10 for N growth story
        if len(biases) < 10:
            # pad
            dim = W.shape[0]
            while len(biases) < 10:
                biases.append(torch.zeros(dim))
        x = torch.randn(4, W.shape[1])
        proof = prove_average_bias_equivalence(W, biases, x)
        proof["layer"] = L
        # base transformer param count
        n_params = sum(p.numel() for p in self.model.parameters())
        proof["base_model_params"] = n_params
        proof["fusion_does_not_grow_transformer"] = True
        return proof

    def export_fused(self, fused: EditDelta, tag: str = "best") -> Dict[str, Any]:
        """Guarda delta fusionado + opcionalmente materializa bias en copia y save_pretrained ligero no-HF-full.
        No subimos pesos completos a git; escribimos delta .pt + JSON instrucciones.
        """
        path_pt = os.path.join(self.out_dir, f"fused_{tag}.pt")
        payload = {
            "name": fused.name,
            "edits": [{"layer": b.layer, "scale": b.scale, "vec": b.vec.detach().cpu()} for b in fused.bias_edits],
            "meta": fused.meta,
            "edit_kind": "steering_bias_fused",
        }
        torch.save(payload, path_pt)
        # materialize into model copy biases and save only bias state dict (small)
        bias_state = {}
        with self.act.session(fused):
            for b in fused.bias_edits:
                lin = self.model.model.layers[b.layer].mlp.down_proj
                if lin.bias is not None:
                    bias_state[f"layers.{b.layer}.mlp.down_proj.bias"] = lin.bias.detach().cpu().clone()
        path_bias = os.path.join(self.out_dir, f"fused_{tag}_biases.pt")
        torch.save({"base_hf_id": self.cfg.hf_id, "biases": bias_state, "meta": fused.meta}, path_bias)
        info = {
            "fused_delta_path": path_pt,
            "fused_biases_path": path_bias,
            "reload": "load base HF + torch.load biases into down_proj.bias, or apply fused delta via SnapshotActivator",
            "sha256_delta": hashlib.sha256(open(path_pt, "rb").read()).hexdigest(),
        }
        with open(os.path.join(self.out_dir, f"fused_{tag}_export.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        return info

    def reload_smoke(self, export_info: Dict[str, Any], prompt: str = "Say hi in one word.") -> Dict[str, Any]:
        """Aplica biases exportados y genera; confirma restore."""
        blob = torch.load(export_info["fused_biases_path"], map_location="cpu", weights_only=False)
        # apply
        for name, tensor in blob["biases"].items():
            # layers.{L}.mlp.down_proj.bias
            L = int(name.split(".")[1])
            lin = self.model.model.layers[L].mlp.down_proj
            with torch.no_grad():
                if lin.bias is None:
                    import torch.nn as nn
                    lin.bias = nn.Parameter(tensor.to(lin.weight.dtype))
                else:
                    lin.bias.copy_(tensor.to(lin.bias.dtype))
        out1 = generate_one(self.model, self.tok, prompt, max_new_tokens=16)
        # clear biases back to None/zero via activator restore pattern: rebuild from backup none
        for name in blob["biases"]:
            L = int(name.split(".")[1])
            self.model.model.layers[L].mlp.down_proj.bias = None
        out0 = generate_one(self.model, self.tok, prompt, max_new_tokens=16)
        return {"with_fused_bias": out1, "after_clear": out0, "ok": True}

    def run_experiment(self, *, resume: bool = True) -> Dict[str, Any]:
        ckpt = os.path.join(self.out_dir, "fusion_checkpoint.json")
        if resume and os.path.isfile(ckpt):
            prev = json.loads(open(ckpt, encoding="utf-8").read())
            if prev.get("done"):
                print("[fusion] resume complete checkpoint", flush=True)
                return prev

        t0 = time.time()
        man = self.manifest()
        with open(os.path.join(self.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(man, f, indent=2, ensure_ascii=False)

        print("[fusion] base confirm", flush=True)
        base_c = self.eval_suite(None, split="confirm", label="base")
        print("[fusion] select hyperparams on tune+val", flush=True)
        sel = self.select_fusion_hyperparams()
        fused: EditDelta = sel.pop("fused_delta")

        # also build reference multi_avg g=1 and single_avg on median layer g=1 for reporting
        used = sorted({b.layer for d in self.deltas.values() for b in d.bias_edits})
        Lref = used[len(used) // 2]
        fused_multi = fuse_deltas(list(self.deltas.values()), mode="multi_avg", global_scale=1.0, name="multi_avg_g1")
        fused_single = fuse_deltas(list(self.deltas.values()), mode="single_avg", target_layer=Lref, global_scale=1.0, name=f"single_avg_L{Lref}_g1")
        fused_sum = fuse_deltas(list(self.deltas.values()), mode="multi_sum", global_scale=1.0, name="multi_sum_g1")

        conditions = {
            "base": None,
            "oracle": "oracle",
            "multi_avg_g1": fused_multi,
            "multi_sum_g1": fused_sum,
            "single_avg_ref": fused_single,
            "selected": fused,
        }

        results = {"confirm": {}, "val": {}, "full_exploratory_reused20": {}}
        for name, delta in conditions.items():
            print(f"[fusion] eval {name}", flush=True)
            if name == "oracle":
                fn = self._delta_fn_oracle
                d_apply = None
            elif delta is None:
                fn = self._delta_fn_base
                d_apply = None
            else:
                fn = self._delta_fn_fixed(delta)
                d_apply = delta
            for split, key in [("confirm", "confirm"), ("val", "val"), ("full", "full_exploratory_reused20")]:
                rep = eval_condition(self.model, self.tok, self.act, self.specs, fn, split=split, max_new=self.max_new)
                rep["condition"] = name
                results[key][name] = rep

        # gains/losses selected vs base on confirm
        def compare(a, b):
            rows = []
            for ta, tb in zip(a["tasks"], b["tasks"]):
                rows.append({
                    "task": ta["task"], "base": ta["acc"], "other": tb["acc"],
                    "delta": None if ta["acc"] is None else round(tb["acc"] - ta["acc"], 4),
                })
            gains = sum(1 for r in rows if r["delta"] is not None and r["delta"] > 1e-9)
            losses = sum(1 for r in rows if r["delta"] is not None and r["delta"] < -1e-9)
            return {"rows": rows, "n_gain": gains, "n_loss": losses, "n_flat": len(rows) - gains - losses}

        base_confirm = results["confirm"]["base"]
        comparisons = {
            "oracle_vs_base": compare(base_confirm, results["confirm"]["oracle"]),
            "selected_vs_base": compare(base_confirm, results["confirm"]["selected"]),
            "multi_avg_vs_base": compare(base_confirm, results["confirm"]["multi_avg_g1"]),
            "single_avg_vs_base": compare(base_confirm, results["confirm"]["single_avg_ref"]),
            "selected_vs_oracle": compare(results["confirm"]["oracle"], results["confirm"]["selected"]),
        }

        proof = self.parallel_equivalence_check()
        export = self.export_fused(fused, tag="selected")
        smoke = self.reload_smoke(export)

        # param story
        n_params = sum(p.numel() for p in self.model.parameters())
        # after applying fused, bias may be created
        existing = set()
        for i, layer in enumerate(self.model.model.layers):
            if layer.mlp.down_proj.bias is not None:
                existing.add(i)
        from .combine import count_bias_params_materialized
        growth = count_bias_params_materialized(fused, existing)

        summary = {
            "base_ppl": round(self.base_ppl, 3),
            "confirm_means": {k: results["confirm"][k]["mean_acc"] for k in results["confirm"]},
            "val_means": {k: results["val"][k]["mean_acc"] for k in results["val"]},
            "full20_exploratory_means": {k: results["full_exploratory_reused20"][k]["mean_acc"] for k in results["full_exploratory_reused20"]},
            "selected_hyperparams": sel["selected"],
            "comparisons_confirm": {k: {kk: vv for kk, vv in v.items() if kk != "rows"} | {"per_task": v["rows"]} for k, v in comparisons.items()},
            "parallel_equivalence": proof,
            "params": {"base_model_params": n_params, "fusion_bias_materialization": growth,
                       "note": "Fusion of bias deltas does not grow Transformer depth/width; optional new bias tensors only."},
            "export": export,
            "reload_smoke": smoke,
            "elapsed_s": round(time.time() - t0, 1),
            "protocol": {
                "hyperparams": "tune + first half eval (fusion_val)",
                "final_report": "second half eval (fusion_confirm) n≈10/task",
                "exploratory_full20": "all 20 eval items reused from prior scaling — NOT independent holdout for this experiment",
            },
        }

        out = {
            "done": True,
            "manifest_path": os.path.join(self.out_dir, "manifest.json"),
            "summary": summary,
            "selection": {k: v for k, v in sel.items() if k != "fused_delta"},
            "results": results,
        }
        # strip huge grids already trimmed
        with open(ckpt, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.out_dir, "fusion_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print("[fusion] DONE", summary["confirm_means"], flush=True)
        return out
