"""Barrido K/layout/escala sobre Qwen2.5-0.5B con presupuesto L2 comparable.

Selección SOLO en tune + fusion_val (mitad histórica de eval). Confirmación FINAL
en holdout nuevo (>=20/tarea). No se reutiliza el confirm histórico como holdout
independiente.
"""
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
from enjambre.edits import BiasEdit, EditDelta
from enjambre.experts import all_expert_specs
from enjambre.fusion.evaluate import eval_condition, split_eval
from enjambre.generate import perplexity
from enjambre.scaling import save_delta

from .build_delta import (
    budget_B_from_prior_fusion,
    fuse_directions_on_layers,
    redistribute_prior_biases,
    total_l2,
)
from .directions import compute_expert_directions, load_directions, save_directions
from .evaluate_holdout import eval_holdout, paired_vs_base
from .holdout import build_new_holdout
from .layouts import all_configs, default_Ks, layout_layers

PPL_GUARD = 1.5
SEED = 0
DEFAULT_PRIOR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "fusion-0.5b", "fused_selected.pt")
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "artifacts", "fusion-sweep-0.5b")
SCALE_MULTS = [0.5, 1.0, 1.5]


def _jdump(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_ser)


def _ser(o):
    if isinstance(o, EditDelta):
        return {"name": o.name, "layers": [b.layer for b in o.bias_edits], "meta": o.meta}
    if isinstance(o, torch.Tensor):
        return f"Tensor{tuple(o.shape)}"
    raise TypeError(type(o))


def _cfg_key(cfg: Dict) -> str:
    layers = "-".join(str(x) for x in cfg["layers"])
    return f"K{cfg['K']}_{cfg['layout']}_L{layers}"


def select_key(row: Dict) -> Tuple:
    """Macro: ok_ppl, mayor mean, menos K, menos L2, layout estable."""
    return (
        0 if row.get("ok_ppl") else 1,
        -float(row.get("score", row.get("val_mean", row.get("tune_mean", 0.0))) or 0.0),
        int(row.get("K", 99)),
        float(row.get("total_l2", 99.0) or 99.0),
        str(row.get("layout", "z")),
    )


class SweepHarness:
    def __init__(
        self,
        model_key: str = "qwen2.5-0.5b",
        out_dir: Optional[str] = None,
        prior_path: Optional[str] = None,
        max_new: int = 64,
        holdout_n: int = 20,
        load_model: bool = True,
    ):
        self.cfg = resolve(model_key)
        self.out_dir = os.path.abspath(out_dir or DEFAULT_OUT)
        os.makedirs(self.out_dir, exist_ok=True)
        self.prior_path = os.path.abspath(prior_path or DEFAULT_PRIOR)
        self.max_new = max_new
        self.holdout_n = holdout_n
        self.specs = all_expert_specs()
        self.model = None
        self.tok = None
        self.act = None
        self.base_ppl = None
        self.nL = 24
        self.hidden = 896
        self.B = None
        self.prior: Optional[EditDelta] = None
        self.directions = None
        self.holdout = None
        if load_model:
            self._load()

    def _load(self):
        torch.manual_seed(SEED)
        torch.set_num_threads(min(8, os.cpu_count() or 8))
        self.tok = AutoTokenizer.from_pretrained(self.cfg.hf_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.hf_id, dtype=torch.float32).eval()
        self.act = SnapshotActivator(self.model)
        self.nL = len(self.model.model.layers)
        self.hidden = int(self.model.config.hidden_size)
        assert self.nL == 24 and self.hidden == 896, (self.nL, self.hidden)
        self.base_ppl = perplexity(self.model, self.tok)
        self.prior = load_delta(self.prior_path)
        self.B = budget_B_from_prior_fusion(self.prior)
        print(f"[sweep] nL={self.nL} hidden={self.hidden} base_ppl={self.base_ppl:.3f} B={self.B:.4f}", flush=True)

    def _ckpt_path(self, name: str) -> str:
        return os.path.join(self.out_dir, name)

    def _load_json(self, name: str) -> Optional[Any]:
        p = self._ckpt_path(name)
        if not os.path.isfile(p):
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def ensure_directions(self, force: bool = False) -> Dict[str, torch.Tensor]:
        path = self._ckpt_path("directions.pt")
        if (not force) and os.path.isfile(path):
            self.directions = load_directions(path)
            print(f"[dir] loaded cache {path} n={len(self.directions)}", flush=True)
            return self.directions
        assert self.model is not None
        self.directions = compute_expert_directions(self.model, self.tok, batch_size=4)
        save_directions(path, self.directions)
        return self.directions

    def ensure_holdout(self) -> Dict:
        if self.holdout is None:
            self.holdout = build_new_holdout(self.holdout_n)
            # persist prompts (not check fns) for audit
            audit = {
                tid: [{"prompt": it.prompt, "meta": it.meta} for it in items]
                for tid, items in self.holdout.items()
            }
            _jdump(self._ckpt_path("holdout_manifest.json"), {
                "n_per_task": self.holdout_n,
                "seed_note": "deterministic item lists in holdout.py",
                "tasks": {k: len(v) for k, v in self.holdout.items()},
                "prompt_sha256": hashlib.sha256(
                    json.dumps(audit, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest(),
                "items": audit,
            })
        return self.holdout

    def build_cfg_delta(self, cfg: Dict, *, scale_mult: float = 1.0, kind: str = "dir") -> EditDelta:
        layers = cfg["layers"]
        B_eff = float(self.B) * float(scale_mult)
        name = f"{kind}_{_cfg_key(cfg)}_s{scale_mult}"
        if kind == "dir":
            return fuse_directions_on_layers(self.directions, layers, B=B_eff, name=name)
        if kind == "redistribute":
            return redistribute_prior_biases([self.prior], layers, B=B_eff, name=name)
        raise ValueError(kind)

    def _eval_fixed(self, delta: Optional[EditDelta], split: str, measure_ppl: bool = False) -> Dict[str, Any]:
        def fn(_tid):
            return delta
        return eval_condition(
            self.model, self.tok, self.act, self.specs, fn,
            split=split, max_new=self.max_new, measure_ppl=measure_ppl,
        )

    def _ppl_ok(self, delta: EditDelta) -> Tuple[float, bool]:
        with self.act.session(delta):
            ppl = perplexity(self.model, self.tok)
        return float(ppl), ppl <= PPL_GUARD * self.base_ppl

    def rss_mb(self) -> float:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024.0
        except OSError:
            return -1.0
        return -1.0

    def phase_coarse(self, resume: bool = True) -> List[Dict]:
        """PPL + tune para todos los layouts (direcciones por capa)."""
        path = self._ckpt_path("coarse_grid.json")
        if resume and os.path.isfile(path):
            rows = json.load(open(path, encoding="utf-8"))
            done = {r["key"] for r in rows}
            print(f"[coarse] resume {len(done)} done", flush=True)
        else:
            rows, done = [], set()

        self.ensure_directions()
        configs = all_configs(self.nL)
        for cfg in configs:
            key = _cfg_key(cfg)
            if key in done:
                continue
            t0 = time.time()
            delta = self.build_cfg_delta(cfg, scale_mult=1.0, kind="dir")
            ppl, ok = self._ppl_ok(delta)
            if not ok:
                row = {
                    "key": key, **cfg, "scale_mult": 1.0, "kind": "dir",
                    "ppl": round(ppl, 3), "ok_ppl": False, "tune_mean": None,
                    "total_l2": round(total_l2(delta), 4),
                    "scale_per_layer": delta.meta.get("scale_per_layer"),
                    "rejected": "ppl_guard", "secs": round(time.time() - t0, 2),
                    "rss_mb": round(self.rss_mb(), 1),
                }
                rows.append(row)
                done.add(key)
                _jdump(path, rows)
                print(f"[coarse] REJECT {key} ppl={ppl:.2f}", flush=True)
                continue
            tune = self._eval_fixed(delta, "tune", measure_ppl=False)
            row = {
                "key": key, **cfg, "scale_mult": 1.0, "kind": "dir",
                "ppl": round(ppl, 3), "ok_ppl": True,
                "tune_mean": tune["mean_acc"], "tune_tasks": tune["tasks"],
                "total_l2": round(total_l2(delta), 4),
                "scale_per_layer": delta.meta.get("scale_per_layer"),
                "rejected": None, "secs": round(time.time() - t0, 2),
                "rss_mb": round(self.rss_mb(), 1),
            }
            rows.append(row)
            done.add(key)
            _jdump(path, rows)
            print(f"[coarse] {key} tune={tune['mean_acc']} ppl={ppl:.2f} l2={row['total_l2']}", flush=True)
        return rows

    def phase_scale_val(self, coarse: List[Dict], resume: bool = True) -> List[Dict]:
        """Por cada K: mejor layout en tune → probar SCALE_MULTS en fusion_val."""
        path = self._ckpt_path("val_grid.json")
        if resume and os.path.isfile(path):
            rows = json.load(open(path, encoding="utf-8"))
            done = {r["val_key"] for r in rows}
            print(f"[val] resume {len(done)}", flush=True)
        else:
            rows, done = [], set()

        self.ensure_directions()
        # best layout per K by tune among ok_ppl
        by_k: Dict[int, Dict] = {}
        for r in coarse:
            if not r.get("ok_ppl"):
                continue
            K = r["K"]
            cand = {**r, "score": r["tune_mean"]}
            if K not in by_k or select_key(cand) < select_key({**by_k[K], "score": by_k[K]["tune_mean"]}):
                by_k[K] = r

        for K, base_row in sorted(by_k.items()):
            cfg = {"K": K, "layout": base_row["layout"], "layers": base_row["layers"]}
            for sm in SCALE_MULTS:
                val_key = f"{_cfg_key(cfg)}_s{sm}"
                if val_key in done:
                    continue
                t0 = time.time()
                delta = self.build_cfg_delta(cfg, scale_mult=sm, kind="dir")
                ppl, ok = self._ppl_ok(delta)
                if not ok:
                    row = {
                        "val_key": val_key, **cfg, "scale_mult": sm, "kind": "dir",
                        "ppl": round(ppl, 3), "ok_ppl": False, "val_mean": None,
                        "total_l2": round(total_l2(delta), 4),
                        "rejected": "ppl_guard", "secs": round(time.time() - t0, 2),
                    }
                    rows.append(row)
                    done.add(val_key)
                    _jdump(path, rows)
                    continue
                val = self._eval_fixed(delta, "val", measure_ppl=False)
                row = {
                    "val_key": val_key, **cfg, "scale_mult": sm, "kind": "dir",
                    "ppl": round(ppl, 3), "ok_ppl": True,
                    "val_mean": val["mean_acc"], "val_tasks": val["tasks"],
                    "tune_mean_ref": base_row.get("tune_mean"),
                    "total_l2": round(total_l2(delta), 4),
                    "scale_per_layer": delta.meta.get("scale_per_layer"),
                    "rejected": None, "secs": round(time.time() - t0, 2),
                    "rss_mb": round(self.rss_mb(), 1),
                }
                rows.append(row)
                done.add(val_key)
                _jdump(path, rows)
                print(f"[val] {val_key} val={val['mean_acc']} ppl={ppl:.2f}", flush=True)
        return rows

    def pick_winners(self, val_rows: List[Dict]) -> Dict[str, Any]:
        """Mejor por K (val) y ganador global (val); no mira final."""
        ok = [r for r in val_rows if r.get("ok_ppl") and r.get("val_mean") is not None]
        by_k: Dict[int, Dict] = {}
        for r in ok:
            K = r["K"]
            cand = {**r, "score": r["val_mean"]}
            if K not in by_k or select_key(cand) < select_key({**by_k[K], "score": by_k[K]["val_mean"]}):
                by_k[K] = r
        best_per_k = {str(k): by_k[k] for k in sorted(by_k)}
        if not by_k:
            return {"best_per_k": {}, "selected": None}
        selected = min(by_k.values(), key=lambda r: select_key({**r, "score": r["val_mean"]}))
        return {"best_per_k": best_per_k, "selected": selected}

    def phase_baselines_val(self, resume: bool = True) -> Dict[str, Any]:
        path = self._ckpt_path("baselines_val.json")
        if resume and os.path.isfile(path):
            return json.load(open(path, encoding="utf-8"))
        out = {}
        for label, delta in [("base", None), ("prior_fusion", self.prior)]:
            t0 = time.time()
            rep = self._eval_fixed(delta, "val", measure_ppl=True)
            out[label] = {**rep, "secs": round(time.time() - t0, 2), "total_l2": 0.0 if delta is None else round(total_l2(delta), 4)}
            print(f"[base-val] {label} mean={rep['mean_acc']}", flush=True)
        # single-layer mid K=1 as reference from directions
        self.ensure_directions()
        cfg1 = {"K": 1, "layout": "mid", "layers": layout_layers(self.nL, 1, "mid")}
        d1 = self.build_cfg_delta(cfg1, scale_mult=1.0, kind="dir")
        rep = self._eval_fixed(d1, "val", measure_ppl=True)
        out["single_mid_dir"] = {**rep, "cfg": cfg1, "total_l2": round(total_l2(d1), 4)}
        _jdump(path, out)
        return out

    def phase_redistribute_ablation(self, resume: bool = True) -> List[Dict]:
        """Ablación: redistribuir bias previo (NO dirección por capa). Fuera de selección."""
        path = self._ckpt_path("redistribute_ablation.json")
        if resume and os.path.isfile(path):
            return json.load(open(path, encoding="utf-8"))
        rows = []
        ablations = [
            {"K": 3, "layout": "mid", "layers": layout_layers(self.nL, 3, "mid")},
            {"K": 6, "layout": "uniform", "layers": layout_layers(self.nL, 6, "uniform")},
            {"K": 3, "layout": "uniform", "layers": layout_layers(self.nL, 3, "uniform")},
        ]
        for cfg in ablations:
            delta = self.build_cfg_delta(cfg, scale_mult=1.0, kind="redistribute")
            ppl, ok = self._ppl_ok(delta)
            if not ok:
                rows.append({**cfg, "kind": "redistribute", "ok_ppl": False, "ppl": round(ppl, 3), "rejected": "ppl_guard"})
                continue
            val = self._eval_fixed(delta, "val", measure_ppl=False)
            rows.append({
                **cfg, "kind": "redistribute", "ok_ppl": True, "ppl": round(ppl, 3),
                "val_mean": val["mean_acc"], "total_l2": round(total_l2(delta), 4),
                "note": "NOT used for selection; relocation of prior bias direction",
            })
            print(f"[redist] K={cfg['K']} {cfg['layout']} val={val['mean_acc']}", flush=True)
        _jdump(path, rows)
        return rows

    def phase_final(self, winners: Dict[str, Any], resume: bool = True) -> Dict[str, Any]:
        path = self._ckpt_path("final_holdout.json")
        if resume and os.path.isfile(path):
            return json.load(open(path, encoding="utf-8"))
        self.ensure_directions()
        self.ensure_holdout()
        results = {}

        def run(label, delta):
            t0 = time.time()
            rep = eval_holdout(
                self.model, self.tok, self.act, self.specs, self.holdout, delta,
                max_new=self.max_new, measure_ppl=True, label=label,
            )
            rep["secs"] = round(time.time() - t0, 2)
            rep["total_l2"] = 0.0 if delta is None else round(total_l2(delta), 4)
            if delta is not None:
                rep["layers"] = [b.layer for b in delta.bias_edits]
                rep["scale_mult_meta"] = delta.meta
            results[label] = rep
            print(f"[final] {label} mean={rep['mean_acc']} n/task={rep['n_per_task']}", flush=True)

        run("base", None)
        run("prior_fusion", self.prior)

        # best per K (preselected on val) — for curve; winner not chosen here
        for Ks, row in winners["best_per_k"].items():
            cfg = {"K": row["K"], "layout": row["layout"], "layers": row["layers"]}
            delta = self.build_cfg_delta(cfg, scale_mult=row["scale_mult"], kind="dir")
            run(f"bestK{Ks}", delta)

        sel = winners.get("selected")
        if sel is not None:
            cfg = {"K": sel["K"], "layout": sel["layout"], "layers": sel["layers"]}
            delta = self.build_cfg_delta(cfg, scale_mult=sel["scale_mult"], kind="dir")
            run("selected", delta)
            # export
            save_delta(self._ckpt_path("best_delta.pt"), delta)
            export = {
                "name": delta.name,
                "K": sel["K"],
                "layout": sel["layout"],
                "layers": sel["layers"],
                "scale_mult": sel["scale_mult"],
                "B": self.B,
                "total_l2": total_l2(delta),
                "scale_per_layer": delta.meta.get("scale_per_layer"),
                "kind": "per_layer_direction_mean",
                "selection": "val_macro_predefined",
                "edits": [{"layer": b.layer, "scale": b.scale} for b in delta.bias_edits],
            }
            _jdump(self._ckpt_path("best_delta_export.json"), export)

        base_tasks = results["base"]["tasks"]
        comps = {}
        for lab, rep in results.items():
            if lab == "base":
                continue
            comps[lab] = paired_vs_base(base_tasks, rep["tasks"])
        results["_paired_vs_base"] = comps
        _jdump(path, results)
        return results

    def curve_by_K(self, winners: Dict, final: Dict, val_rows: List[Dict]) -> Dict[str, Any]:
        curve = {}
        for Ks, row in winners["best_per_k"].items():
            flab = f"bestK{Ks}"
            curve[Ks] = {
                "K": row["K"],
                "layout": row["layout"],
                "layers": row["layers"],
                "scale_mult": row["scale_mult"],
                "val_mean": row["val_mean"],
                "val_ppl": row["ppl"],
                "total_l2": row["total_l2"],
                "final_mean": None if flab not in final else final[flab]["mean_acc"],
                "final_n_per_task": None if flab not in final else final[flab]["n_per_task"],
                "final_tasks": None if flab not in final else [
                    {"task": t["task"], "acc": t["acc"], "n": t["n"], "n_ok": t["n_ok"]}
                    for t in final[flab]["tasks"]
                ],
            }
        _jdump(self._ckpt_path("curve_by_K.json"), curve)
        return curve

    def run(self, resume: bool = True) -> Dict[str, Any]:
        T0 = time.time()
        self.ensure_directions()
        self.ensure_holdout()
        coarse = self.phase_coarse(resume=resume)
        baselines_val = self.phase_baselines_val(resume=resume)
        val_rows = self.phase_scale_val(coarse, resume=resume)
        winners = self.pick_winners(val_rows)
        _jdump(self._ckpt_path("winners.json"), winners)
        redist = self.phase_redistribute_ablation(resume=resume)
        final = self.phase_final(winners, resume=resume)
        curve = self.curve_by_K(winners, final, val_rows)

        sel = winners.get("selected")
        summary = {
            "model": self.cfg.key,
            "hf_id": self.cfg.hf_id,
            "nL": self.nL,
            "hidden": self.hidden,
            "B_from_prior": self.B,
            "base_ppl": self.base_ppl,
            "ppl_guard": PPL_GUARD,
            "max_new": self.max_new,
            "holdout_n": self.holdout_n,
            "seed": SEED,
            "scope": "Distribuye CAMBIOS en bloques EXISTENTES; no añade bloques entrenados.",
            "direction_kind": "per_layer_direction_mean (contraste tune system vs no-system)",
            "selection_split": "tune (coarse) + fusion_val=primera mitad eval histórica; final=holdout NUEVO",
            "selected_by_val": None if sel is None else {
                "K": sel["K"], "layout": sel["layout"], "layers": sel["layers"],
                "scale_mult": sel["scale_mult"], "val_mean": sel["val_mean"],
                "total_l2": sel["total_l2"], "ppl": sel["ppl"],
            },
            "baselines_val": {k: v.get("mean_acc") for k, v in baselines_val.items()},
            "final_means": {k: v["mean_acc"] for k, v in final.items() if not k.startswith("_") and isinstance(v, dict) and "mean_acc" in v},
            "curve_by_K": curve,
            "redistribute_ablation_val": redist,
            "n_coarse": len(coarse),
            "n_rejected_ppl": sum(1 for r in coarse if r.get("rejected") == "ppl_guard"),
            "secs_total": round(time.time() - T0, 1),
            "rss_mb": round(self.rss_mb(), 1),
            "limits": [
                "Cobertura acotada a 10 tareas sintéticas; no inteligencia general.",
                "Templates nuevos en holdout pero dominio similar → no fingir independencia total.",
                "Más capas no implica mejor; presupuesto L2 comparable entre K.",
                "Confirm histórico de fusión previa NO se reutiliza como holdout independiente.",
            ],
        }
        _jdump(self._ckpt_path("sweep_summary.json"), summary)
        _jdump(self._ckpt_path("manifest.json"), {
            "model": self.cfg.key,
            "hf_id": self.cfg.hf_id,
            "B": self.B,
            "prior": self.prior_path,
            "out_dir": self.out_dir,
            "holdout_prompt_sha": (self._load_json("holdout_manifest.json") or {}).get("prompt_sha256"),
            "selected": summary["selected_by_val"],
            "git_branch_target": "exp/fusion-barrido-capas-0.5b",
        })
        return summary
