"""Evaluación y adaptación de arquitecturas de conectores en la Suite Compleja de Matemáticas y Programación.

Compara:
1. `baseline_k24_residual`: Conectores estándar `ResidualBridge` de `K24_mc (24L+6B)`.
2. `constelacion_v4`: Conectores `ConstelacionV4Bridge` (Koopman 2x2 + precisión normalizada).
3. `bq10_dastr`: Conectores `BQ10DaStrBridge` (ATLAS-FW + M7 DA-STR da=10 + innovación gemma_campo).
4. `soft_bellman_simplicial`: Conectores `SoftBellmanSimplicialBridge` (métrica Riemanniana SPD + tetraedro baricéntrico).
5. `neuro_adapted_best`: Conectores neuro-geométricos adaptados para razonamiento multi-paso.

Evalúa en 10 tareas complejas (15 ítems holdout cada una = 150 ítems de prueba):
- Matemáticas: MCD/MCM, Fibonacci/recurrencias, exp modular, ecuaciones lineales, combinatoria.
- Programación: Bucles/filtros, algoritmos de cadenas, recursión, diccionarios/frecuencias, matrices anidadas.
"""
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
from enjambre.bridges.modules import ResidualBridge
from enjambre.bridges.train_eval import freeze_model, param_checksum
from enjambre.config import resolve
from enjambre.edits import BiasEdit, EditDelta
from enjambre.experts import Item
from enjambre.generate import generate

from .complex_tasks import (
    ALL_COMPLEX_TASKS,
    COMPLEX_CODE_TASKS,
    COMPLEX_MATH_TASKS,
    COMPLEX_SPECS,
    make_complex_mc_splits,
)
from .neuro_connectors import (
    ARCHITECTURES,
    PAIRS_K24,
    NeuroBridgeBank,
    build_warm_neuro_bridge_bank,
)

DEFAULT_COMPLEX_OUT = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "complex-bench-k24"
)
K24_MC_ARTIFACTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "math-code-k12plus"
)
NEURO_ARTIFACTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "neuro-connectors-k24"
)
SEED = 42


def _jdump(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def _load_saved_delta(path: str) -> EditDelta:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, EditDelta):
        return raw
    edits = [
        BiasEdit(layer=int(e["layer"]), vec=e["vec"].float().clone(), scale=float(e["scale"]))
        for e in raw["edits"]
    ]
    return EditDelta(name=str(raw.get("name", "k24_mc")), bias_edits=edits, meta=dict(raw.get("meta", {})))


class ComplexBenchmarkHarness:
    def __init__(
        self,
        out_dir: Optional[str] = None,
        rank: int = 16,
        n_train: int = 12,
        n_val: int = 6,
        n_hold: int = 15,
        max_new: int = 48,  # Permitir más tokens para funciones con bucles y recursión
    ):
        self.cfg = resolve("qwen2.5-0.5b")
        self.out_dir = os.path.abspath(out_dir or DEFAULT_COMPLEX_OUT)
        os.makedirs(self.out_dir, exist_ok=True)
        self.rank = rank
        self.n_train, self.n_val, self.n_hold = n_train, n_val, n_hold
        self.max_new = max_new

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

        # Splits Complejos (10 tareas disjuntas)
        self.train_rows, self.val_rows, self.hold_items, self.split_man = make_complex_mc_splits(
            n_train=n_train, n_val=n_val, n_hold=n_hold, seed=242
        )
        _jdump(os.path.join(self.out_dir, "split_manifest_complex.json"), self.split_man)

        # Delta de 24 capas editadas de K24_mc
        k24_delta_path = os.path.join(K24_MC_ARTIFACTS_DIR, "deltas", "k24_mc.pt")
        k24_bridges_path = os.path.join(K24_MC_ARTIFACTS_DIR, "bridges_k24_mc_best.pt")
        self.k24_delta: EditDelta = _load_saved_delta(k24_delta_path)

        ck = torch.load(k24_bridges_path, map_location="cpu", weights_only=False)
        self.k24_bridge_state: Dict[str, torch.Tensor] = {
            k: v.detach().cpu().float().clone()
            for k, v in (ck["state"] if isinstance(ck, dict) and "state" in ck else ck).items()
        }

        # Checkpoints de los conectores neuro-geométricos
        self.neuro_states: Dict[str, Dict[str, torch.Tensor]] = {}
        for arch in ARCHITECTURES:
            p = os.path.join(NEURO_ARTIFACTS_DIR, f"connectors_{arch}_best.pt")
            if os.path.isfile(p):
                c = torch.load(p, map_location="cpu", weights_only=False)
                self.neuro_states[arch] = {
                    k: v.detach().cpu().float().clone()
                    for k, v in (c["state"] if isinstance(c, dict) and "state" in c else c).items()
                }

        self.bank: Optional[NeuroBridgeBank] = None

    def _chat(self, user: str) -> str:
        msgs = [{"role": "user", "content": user}]
        if hasattr(self.tok, "apply_chat_template"):
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return user

    def _loss_batch(self, prompts: List[str], targets: List[str]) -> torch.Tensor:
        assert self.bank is not None
        self.bank.clear_buf()
        bos = [self._chat(p) for p in prompts]
        full = [b + t + self.eos for b, t in zip(bos, targets)]
        enc_full = self.tok(full, return_tensors="pt", padding=True, truncation=True, max_length=160)
        enc_prompt = self.tok(bos, return_tensors="pt", padding=True, truncation=True, max_length=128)
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
            [1.25 if len(t.strip()) <= 10 else 1.0 for t in targets],
            dtype=seq_losses.dtype,
            device=seq_losses.device,
        )
        loss = (seq_losses * weights).sum() / weights.sum()
        self.bank.clear_buf()
        return loss

    @torch.no_grad()
    def evaluate_condition(
        self,
        label: str,
        arch: str,
        bridge_state: Dict[str, torch.Tensor],
        per_pair_arch: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Evalúa una configuración de conectores en los 150 ítems de la suite compleja."""
        out_path = os.path.join(self.out_dir, f"results_{label}.json")
        if os.path.isfile(out_path):
            cached = json.load(open(out_path, encoding="utf-8"))
            if len(cached.get("flat_ok", [])) == self.n_hold * len(COMPLEX_SPECS):
                print(f"[eval:{label}] Reutilizando caché mean_acc={cached['mean_acc']:.4f}", flush=True)
                return cached

        if self.bank is not None:
            self.bank.remove_hooks()
        self.bank = build_warm_neuro_bridge_bank(
            self.model.config.hidden_size,
            arch=arch,
            rank=self.rank,
            pairs=PAIRS_K24,
            k24_pretrained_state=bridge_state,
            per_pair_arch=per_pair_arch,
        )
        self.bank.attach(self.model)
        self.bank.enabled = True
        self.bank.clear_buf()

        t0 = time.time()
        reports = []
        math_oks: List[bool] = []
        code_oks: List[bool] = []
        all_oks: List[bool] = []

        with self.act.session(self.k24_delta):
            for spec in COMPLEX_SPECS:
                items = self.hold_items[spec.id]
                # Para tareas de código usamos max_new=36 (funciones multi-línea) y para math max_new=18
                toks_limit = 36 if spec.domain == "code" else 18
                outs = generate(
                    self.model, self.tok, [it.prompt for it in items],
                    max_new_tokens=toks_limit, batch_size=8,
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
                reports.append({
                    "task": spec.id,
                    "domain": spec.domain,
                    "n": len(items),
                    "n_ok": int(sum(oks)),
                    "acc": acc,
                    "examples": rows[:2],
                })
                print(f"[eval:{label}] {spec.id} ({spec.domain}) acc={acc:.4f}", flush=True)
                self.bank.clear_buf()

        math_acc = round(sum(math_oks) / len(math_oks), 4)
        code_acc = round(sum(code_oks) / len(code_oks), 4)
        mean_acc = round(sum(all_oks) / len(all_oks), 4)

        rep = {
            "label": label,
            "arch": arch,
            "per_pair_arch": per_pair_arch,
            "delta_name": self.k24_delta.name,
            "n_layers_edited": len(self.k24_delta.bias_edits),
            "bridge_pairs": PAIRS_K24,
            "n_bridges": len(PAIRS_K24),
            "n_bridge_params": self.bank.n_params(),
            "n_neuro_extra_params": self.bank.n_trainable_neuro_params(),
            "math_acc": math_acc,
            "code_acc": code_acc,
            "mean_acc": mean_acc,
            "secs": round(time.time() - t0, 2),
            "tasks": reports,
            "flat_ok": all_oks,
            "math_flat_ok": math_oks,
            "code_flat_ok": code_oks,
        }
        _jdump(out_path, rep)
        return rep

    def adapt_connectors_on_complex(
        self,
        tag: str,
        arch: str,
        init_state: Dict[str, torch.Tensor],
        *,
        per_pair_arch: Optional[Dict[str, str]] = None,
        steps: int = 32,
        lr_subspace: float = 2.0e-3,
        lr_direct: float = 4.0e-4,
    ) -> Dict[str, Any]:
        """Adapta únicamente los conectores inter-capa con el conjunto de entrenamiento complejo (140 ejemplos)."""
        ckpt_path = os.path.join(self.out_dir, f"connectors_complex_{tag}_best.pt")
        log_path = os.path.join(self.out_dir, f"train_log_complex_{tag}.json")
        if os.path.isfile(ckpt_path) and os.path.isfile(log_path):
            print(f"[adapt:{tag}] Reutilizando checkpoint {ckpt_path}", flush=True)
            return json.load(open(log_path, encoding="utf-8"))

        if self.bank is not None:
            self.bank.remove_hooks()

        self.bank = build_warm_neuro_bridge_bank(
            self.model.config.hidden_size,
            arch=arch,
            rank=self.rank,
            pairs=PAIRS_K24,
            k24_pretrained_state=init_state,
            per_pair_arch=per_pair_arch,
        )
        self.bank.attach(self.model)
        freeze_model(self.model)

        base_suffixes = (".up.weight", ".down.weight", ".gate.weight", ".gate.bias")
        params_base, params_subspace, params_direct = [], [], []
        for pname, p in self.bank.named_parameters():
            if arch == "residual_base":
                p.requires_grad_(True)
                params_base.append(p)
            elif pname.endswith(base_suffixes):
                p.requires_grad_(False)
            elif pname.endswith(".neuro_down.weight"):
                p.requires_grad_(True)
                params_direct.append(p)
            else:
                p.requires_grad_(True)
                params_subspace.append(p)

        if arch == "residual_base":
            param_groups = [{"params": params_base, "lr": 4e-4, "weight_decay": 1e-4}]
        else:
            param_groups = [
                {"params": params_subspace, "lr": lr_subspace, "weight_decay": 1e-3},
                {"params": params_direct, "lr": lr_direct, "weight_decay": 2e-2},
            ]
        opt = torch.optim.AdamW(param_groups)

        # Construir lista plana de entrenamiento (10 tareas * n_train = 120-140 ejemplos)
        data = []
        for tid in ALL_COMPLEX_TASKS:
            for p, tgt, _ in self.train_rows[tid]:
                data.append((p, tgt))

        rng = random.Random(SEED + 99)
        t0 = time.time()
        print(f"[adapt:{tag}] Entrenando conectores en la suite compleja: steps={steps} items={len(data)}", flush=True)

        self.model.train()
        step = 0
        bs = 4
        while step < steps:
            rng.shuffle(data)
            for i in range(0, len(data), bs):
                if step >= steps:
                    break
                batch = data[i : i + bs]
                if not batch:
                    continue
                ps, ts = zip(*batch)
                with self.act.session(self.k24_delta):
                    loss = self._loss_batch(list(ps), list(ts))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.bank.parameters(), 1.0)
                opt.step()
                self.bank.clear_buf()
                step += 1
                if step % 8 == 0 or step == steps:
                    print(f"[adapt:{tag}] step={step}/{steps} loss={float(loss):.4f} ({time.time()-t0:.1f}s)", flush=True)

        self.model.eval()
        best_state = {k: v.detach().cpu().clone() for k, v in self.bank.state_dict().items()}
        torch.save({"state": best_state, "arch": arch, "pairs": PAIRS_K24, "tag": tag}, ckpt_path)
        info = {"tag": tag, "arch": arch, "steps": steps, "ckpt_path": ckpt_path, "secs": round(time.time() - t0, 1)}
        _jdump(log_path, info)
        return info

    def run_full_complex_experiment(self, only_stage: Optional[str] = None) -> Dict[str, Any]:
        """Ejecuta el experimento completo en la suite compleja:
        1. Base K24_mc Residual (zero-shot transfer a tareas complejas).
        2. Base K24_mc Residual adaptada en tareas complejas (límite de la arquitectura residual).
        3. CONSTELACIÓN v4 (Koopman 2x2 en recurrencias y ecuaciones).
        4. BQ10 DA-STR (M7 DA-STR en algoritmos, bucles y lógica compleja).
        5. Soft-Bellman Simplicial (métrica SPD en matrices y estructuras).
        6. Ensamble / Conectores adaptados de Neuro-Geométrica.
        """
        t_start = time.time()

        # 1. Zero-shot transfer de la base K24_mc Residual
        if only_stage in (None, "1"):
            rep_base_zs = self.evaluate_condition(
                "1_base_k24_residual_zeroshot",
                "residual_base",
                self.k24_bridge_state,
            )
            if only_stage == "1":
                return rep_base_zs

        # 2. Conectores K24_mc Residual adaptados en la suite compleja (línea base entrenada)
        if only_stage in (None, "2"):
            tr_base_adapt = self.adapt_connectors_on_complex(
                "residual_adapted",
                "residual_base",
                self.k24_bridge_state,
                steps=30,
            )
            st_base_adapt = torch.load(tr_base_adapt["ckpt_path"], map_location="cpu", weights_only=False)["state"]
            rep_base_adapt = self.evaluate_condition(
                "2_base_k24_residual_adapted",
                "residual_base",
                st_base_adapt,
            )
            if only_stage == "2":
                return rep_base_adapt

        # 3. CONSTELACIÓN v4 adaptada en suite compleja (especializada en dinámicas/recurrencias)
        if only_stage in (None, "3"):
            init_cst = self.neuro_states.get("constelacion_v4", self.k24_bridge_state)
            tr_cst = self.adapt_connectors_on_complex(
                "constelacion_v4_adapted",
                "constelacion_v4",
                init_cst,
                steps=30,
            )
            st_cst = torch.load(tr_cst["ckpt_path"], map_location="cpu", weights_only=False)["state"]
            rep_cst = self.evaluate_condition(
                "3_constelacion_v4_adapted",
                "constelacion_v4",
                st_cst,
            )
            if only_stage == "3":
                return rep_cst

        # 4. BQ10 DA-STR adaptada en suite compleja (regularización por grado en bucles y lógica)
        if only_stage in (None, "4"):
            init_bq10 = self.neuro_states.get("bq10_dastr", self.k24_bridge_state)
            tr_bq10 = self.adapt_connectors_on_complex(
                "bq10_dastr_adapted",
                "bq10_dastr",
                init_bq10,
                steps=30,
            )
            st_bq10 = torch.load(tr_bq10["ckpt_path"], map_location="cpu", weights_only=False)["state"]
            rep_bq10 = self.evaluate_condition(
                "4_bq10_dastr_adapted",
                "bq10_dastr",
                st_bq10,
            )
            if only_stage == "4":
                return rep_bq10

        # 5. Soft-Bellman Simplicial 3D adaptada en suite compleja
        if only_stage in (None, "5"):
            init_sbs = self.neuro_states.get("soft_bellman_simplicial", self.k24_bridge_state)
            tr_sbs = self.adapt_connectors_on_complex(
                "soft_bellman_adapted",
                "soft_bellman_simplicial",
                init_sbs,
                steps=30,
            )
            st_sbs = torch.load(tr_sbs["ckpt_path"], map_location="cpu", weights_only=False)["state"]
            rep_sbs = self.evaluate_condition(
                "5_soft_bellman_adapted",
                "soft_bellman_simplicial",
                st_sbs,
            )
            if only_stage == "5":
                return rep_sbs

        # 6. Neuro-Geométrica Compuesta Jerárquica:
        # Capas tempranas 0->4, 4->8: Constelación v4 (Koopman para secuencias/recurrencias)
        # Capas medias 8->12: Soft-Bellman (métrica baricéntrica para matrices y grafos)
        # Capas tardías 12->16, 16->20, 20->23: BQ10 DA-STR (atenuación M7 para bucles/lógica)
        if only_stage in (None, "6"):
            p_cst = os.path.join(self.out_dir, "connectors_complex_constelacion_v4_adapted_best.pt")
            p_sbs = os.path.join(self.out_dir, "connectors_complex_soft_bellman_adapted_best.pt")
            p_bq10 = os.path.join(self.out_dir, "connectors_complex_bq10_dastr_adapted_best.pt")
            st_cst = torch.load(p_cst, map_location="cpu", weights_only=False)["state"]
            st_sbs = torch.load(p_sbs, map_location="cpu", weights_only=False)["state"]
            st_bq10 = torch.load(p_bq10, map_location="cpu", weights_only=False)["state"]

            hybrid_per_pair = {
                "0_4": "constelacion_v4",
                "4_8": "constelacion_v4",
                "8_12": "soft_bellman_simplicial",
                "12_16": "bq10_dastr",
                "16_20": "bq10_dastr",
                "20_23": "bq10_dastr",
            }
            hybrid_init_state: Dict[str, torch.Tensor] = {}
            source_states = {
                "constelacion_v4": st_cst,
                "soft_bellman_simplicial": st_sbs,
                "bq10_dastr": st_bq10,
            }
            base_suffixes = (".up.weight", ".down.weight", ".gate.weight", ".gate.bias")
            for s, t in PAIRS_K24:
                bkey = f"{s}_{t}"
                chosen = hybrid_per_pair[bkey]
                src_st = source_states[chosen]
                prefix = f"bridges.{bkey}."
                for k, v in src_st.items():
                    if k.startswith(prefix):
                        if k.endswith(base_suffixes) and k in self.k24_bridge_state:
                            hybrid_init_state[k] = self.k24_bridge_state[k].clone()
                        else:
                            hybrid_init_state[k] = v.clone()

            rep_hybrid = self.evaluate_condition(
                "6_k24_neuro_hybrid_complex",
                "bq10_dastr",
                hybrid_init_state,
                per_pair_arch=hybrid_per_pair,
            )
            if only_stage == "6":
                return rep_hybrid

        labels = [
            "1_base_k24_residual_zeroshot",
            "2_base_k24_residual_adapted",
            "3_constelacion_v4_adapted",
            "4_bq10_dastr_adapted",
            "5_soft_bellman_adapted",
            "6_k24_neuro_hybrid_complex",
        ]
        all_reps = [
            json.load(open(os.path.join(self.out_dir, f"results_{lbl}.json"), encoding="utf-8"))
            for lbl in labels
        ]
        rep_base_zs, rep_base_adapt, rep_cst, rep_bq10, rep_sbs, rep_hybrid = all_reps

        if self.bank is not None:
            self.bank.remove_hooks()
        after_checksum = param_checksum(self.model)
        assert after_checksum == self.base_checksum, "Base model weights must remain frozen!"

        all_reps = [rep_base_zs, rep_base_adapt, rep_cst, rep_bq10, rep_sbs, rep_hybrid]
        best_rep = max(all_reps, key=lambda r: r["mean_acc"])

        def diff_stats(ref: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
            r_ok = ref["flat_ok"]
            n_ok = new["flat_ok"]
            gains = sum(1 for a, b in zip(r_ok, n_ok) if not a and b)
            losses = sum(1 for a, b in zip(r_ok, n_ok) if a and not b)
            return {
                "delta_mean": round(new["mean_acc"] - ref["mean_acc"], 4),
                "delta_math": round(new["math_acc"] - ref["math_acc"], 4),
                "delta_code": round(new["code_acc"] - ref["code_acc"], 4),
                "gains": gains,
                "losses": losses,
                "net": gains - losses,
            }

        summary = {
            "title": "Experimento en Suite Compleja de Matemáticas y Programación (150 Ítems)",
            "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "layers_frozen": "24 capas editadas k24_mc.pt (100% congeladas, requires_grad=False)",
            "n_test_items": len(rep_base_zs["flat_ok"]),
            "winner": {
                "label": best_rep["label"],
                "arch": best_rep["arch"],
                "math_acc": best_rep["math_acc"],
                "code_acc": best_rep["code_acc"],
                "mean_acc": best_rep["mean_acc"],
                "gain_vs_base_zeroshot": round(best_rep["mean_acc"] - rep_base_zs["mean_acc"], 4),
                "gain_vs_base_adapted": round(best_rep["mean_acc"] - rep_base_adapt["mean_acc"], 4),
            },
            "comparison_table": [
                {
                    "label": r["label"],
                    "arch": r["arch"],
                    "math_acc": r["math_acc"],
                    "code_acc": r["code_acc"],
                    "mean_acc": r["mean_acc"],
                    "extra_params": r.get("n_neuro_extra_params", 0),
                    "secs": r["secs"],
                }
                for r in all_reps
            ],
            "per_task_table": {
                t["task"]: {
                    r["label"]: next(x["acc"] for x in r["tasks"] if x["task"] == t["task"])
                    for r in all_reps
                }
                for t in rep_base_zs["tasks"]
            },
            "contrasts_vs_base_adapted": {
                "constelacion_v4_vs_base_adapted": diff_stats(rep_base_adapt, rep_cst),
                "bq10_dastr_vs_base_adapted": diff_stats(rep_base_adapt, rep_bq10),
                "soft_bellman_vs_base_adapted": diff_stats(rep_base_adapt, rep_sbs),
                "hybrid_vs_base_adapted": diff_stats(rep_base_adapt, rep_hybrid),
            },
            "total_wall_secs": round(time.time() - t_start, 1),
        }
        _jdump(os.path.join(self.out_dir, "summary_complex_benchmark.json"), summary)
        return summary
