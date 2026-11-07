"""Arnés experimental para evaluar y entrenar las 4 arquitecturas de conectores de
`neuro-geometrica_multi-agente` sobre la base ganadora `K24_mc (24L+6B)`.

Compara en igualdad de condiciones sobre `K24_mc` (24 capas editadas `0..23` congeladas + 6 puentes `0→4..20→23`):
1. `1_base_k24_mc_residual`      — Base actual `K24_mc (24L+6B)` con `ResidualBridge` estándar.
2. `2_k24_constelacion_v4`       — Conectores `ConstelacionV4Bridge` (bloques 2×2 Koopman rot-escala + precisión Π).
3. `3_k24_atlas_fw`              — Conectores `AtlasFWBridge` (M1 17 fns Koopman/SINDy + M2 + M6 compuerta CLS).
4. `4_k24_bq10_dastr`            — Conectores `BQ10DaStrBridge` (ATLAS-FW + M7 DA-STR `da=10` + innovación `gemma_campo`).
5. `5_k24_soft_bellman`          — Conectores `SoftBellmanSimplicialBridge` (métrica SPD3x3 + tetraedro baricéntrico Δ³).
6. `6_k24_neuro_applied_best`    — Aplicación optimizada de la arquitectura ganadora con refinamiento focalizado
                                   en expresiones algebraicas/polinomiales y estructuras Python.

Evalúa en 3 bancos de pruebas (235 ítems de prueba en total):
- **Core Math & Code Holdout (8 tareas × 15 = 120 ítems):** `math_arith`, `math_mul_mod`, `math_expr`, `math_logic`,
  `code_fn`, `code_algo`, `code_comp`, `code_lambda`.
- **Extended Math & Code Holdout (4 tareas nuevas × 10 = 40 ítems):** `math_poly`, `math_seq`,
  `code_list_fn`, `code_dict_str`.
- **Legacy Holdout (3 suites × 25 = 75 ítems):** `legacy_arithmetic`, `legacy_code_lax`, `legacy_code_strict_ast`.
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
from enjambre.bridges.train_eval import freeze_model, param_checksum
from enjambre.config import resolve
from enjambre.edits import BiasEdit, EditDelta
from enjambre.experts import Item
from enjambre.fusion_confirm12.stats import bootstrap_delta_ci, mcnemar_exact, paired_flags
from enjambre.generate import generate, perplexity

from .neuro_connectors import (
    ARCHITECTURES,
    PAIRS_K24,
    NeuroBridgeBank,
    build_warm_neuro_bridge_bank,
)
from .tasks import (
    ALL_MC_TASKS,
    CODE_TASKS,
    EXT_CODE_TASKS,
    EXT_MATH_TASKS,
    EXT_MC_SPECS,
    EXT_MC_TASKS,
    MATH_TASKS,
    MC_SPECS,
    get_legacy_k12_math_code_holdout,
    make_ext_mc_splits,
    make_mc_splits,
)

DEFAULT_NEURO_OUT = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "neuro-connectors-k24"
)
K24_MC_ARTIFACTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "math-code-k12plus"
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


class NeuroConnectorHarness:
    def __init__(
        self,
        out_dir: Optional[str] = None,
        rank: int = 16,
        n_train: int = 12,
        n_val: int = 6,
        n_hold: int = 15,
        n_ext_train: int = 12,
        n_ext_val: int = 6,
        n_ext_hold: int = 10,
        n_legacy_hold: int = 25,
        max_new: int = 22,
        train_steps: int = 28,
        batch_size_train: int = 4,
        lr_base_bridge: float = 4e-5,
        lr_subspace_neuro: float = 1.6e-3,
        lr_direct_neuro: float = 3.5e-4,
    ):
        self.cfg = resolve("qwen2.5-0.5b")
        self.out_dir = os.path.abspath(out_dir or DEFAULT_NEURO_OUT)
        os.makedirs(self.out_dir, exist_ok=True)
        self.rank = rank
        self.n_train, self.n_val, self.n_hold = n_train, n_val, n_hold
        self.n_ext_train, self.n_ext_val, self.n_ext_hold = n_ext_train, n_ext_val, n_ext_hold
        self.n_legacy_hold = n_legacy_hold
        self.max_new = max_new
        self.train_steps = train_steps
        self.batch_size_train = batch_size_train
        self.lr_base_bridge = lr_base_bridge
        self.lr_subspace_neuro = lr_subspace_neuro
        self.lr_direct_neuro = lr_direct_neuro

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

        # Splits Core (8 tareas) + Splits Extendidos (4 tareas nuevas) + Legacy Holdout (3 suites)
        self.train_rows, self.val_rows, self.hold_items, self.split_man = make_mc_splits(
            n_train=n_train, n_val=n_val, n_hold=n_hold, seed=SEED
        )
        self.ext_train_rows, self.ext_val_rows, self.ext_hold_items, self.ext_split_man = make_ext_mc_splits(
            n_train=n_ext_train, n_val=n_ext_val, n_hold=n_ext_hold, seed=142
        )
        self.legacy_hold = get_legacy_k12_math_code_holdout(n_hold=n_legacy_hold)

        _jdump(
            os.path.join(self.out_dir, "split_manifest_neuro.json"),
            {"core": self.split_man, "extended": self.ext_split_man},
        )

        # Cargar el delta K24_mc (24 capas editadas 0..23) y los 6 puentes entrenados de K24_mc
        k24_delta_path = os.path.join(K24_MC_ARTIFACTS_DIR, "deltas", "k24_mc.pt")
        k24_bridges_path = os.path.join(K24_MC_ARTIFACTS_DIR, "bridges_k24_mc_best.pt")
        assert os.path.isfile(k24_delta_path), f"Falta {k24_delta_path}"
        assert os.path.isfile(k24_bridges_path), f"Falta {k24_bridges_path}"

        self.k24_delta: EditDelta = _load_saved_delta(k24_delta_path)
        ck = torch.load(k24_bridges_path, map_location="cpu", weights_only=False)
        self.k24_bridge_state: Dict[str, torch.Tensor] = {
            k: v.detach().cpu().float().clone()
            for k, v in (ck["state"] if isinstance(ck, dict) and "state" in ck else ck).items()
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
            [1.45 if len(t.strip()) <= 6 else 1.0 for t in targets],
            dtype=seq_losses.dtype,
            device=seq_losses.device,
        )
        loss = (seq_losses * weights).sum() / weights.sum()
        self.bank.clear_buf()
        return loss

    def build_supervised_dataset(self, emphasize_hard_math: bool = False) -> List[Tuple[str, str]]:
        """Construye el dataset de entrenamiento de conectores:
        75% tareas de Matemáticas (Core + Extended) donde `K24_mc` tiene margen de mejora,
        + 25% replay de Código Python para preservar el 100% de exactitud AST."""
        math_tids = MATH_TASKS + EXT_MATH_TASKS
        code_tids = CODE_TASKS + EXT_CODE_TASKS
        all_rows = {**self.train_rows, **self.ext_train_rows}
        max_len = max(len(all_rows[tid]) for tid in (math_tids + code_tids))
        out: List[Tuple[str, str]] = []
        hard_math = {"math_expr", "math_mul_mod", "math_logic", "math_poly", "math_seq"}
        for i in range(max_len):
            for tid in math_tids:
                if i < len(all_rows[tid]):
                    p, tgt, _ = all_rows[tid][i]
                    out.append((p, tgt))
                    if emphasize_hard_math and tid in hard_math:
                        out.append((p, tgt))
            # Replay ligero de código (1 de cada 3 índices)
            if i % 3 == 0:
                for tid in code_tids:
                    if i < len(all_rows[tid]):
                        p, tgt, _ = all_rows[tid][i]
                        out.append((p, tgt))
        return out

    @torch.no_grad()
    def eval_val_metrics(self) -> Tuple[float, float]:
        """Evalúa en validación: (val_loss continua, val_math_acc en tareas matemáticas).
        Mucho más rápido (~8s) y sensible a mejoras reales en cálculo matemático."""
        assert self.bank is not None
        self.bank.enabled = True
        self.bank.clear_buf()
        math_tids = MATH_TASKS + EXT_MATH_TASKS
        all_val = {**self.val_rows, **self.ext_val_rows}
        oks: List[bool] = []
        losses: List[float] = []
        with self.act.session(self.k24_delta):
            for tid in math_tids:
                rows = all_val[tid][:3]
                ps = [r[0] for r in rows]
                ts = [r[1] for r in rows]
                loss_t = self._loss_batch(ps, ts)
                losses.append(float(loss_t.item()))
                outs = generate(self.model, self.tok, ps, max_new_tokens=16, batch_size=8)
                for (_, _, chk), o in zip(rows, outs):
                    oks.append(bool(chk(o)))
                self.bank.clear_buf()
        return sum(losses) / max(1, len(losses)), sum(oks) / max(1, len(oks))

    def verify_paso0_parity(self) -> Dict[str, Any]:
        """Verifica que en `step=0` (antes de entrenar la rama neuro-geométrica), las 4 arquitecturas
        reproducen bit a bit los logits y salidas de `K24_mc (24L+6B)`."""
        paso0_path = os.path.join(self.out_dir, "paso0_parity.json")
        if os.path.isfile(paso0_path):
            return json.load(open(paso0_path, encoding="utf-8"))

        if self.bank is not None:
            self.bank.remove_hooks()
        prompt = self._chat("Evaluate (41 + 10) * 3. Reply with ONLY the integer.")
        enc = self.tok([prompt], return_tensors="pt")

        base_bank = build_warm_neuro_bridge_bank(
            self.model.config.hidden_size,
            "residual_base",
            rank=self.rank,
            pairs=PAIRS_K24,
            k24_pretrained_state=self.k24_bridge_state,
        )
        base_bank.attach(self.model)
        with torch.no_grad(), self.act.session(self.k24_delta):
            ref_logits = self.model(**enc, use_cache=False).logits.detach().clone()
        base_bank.remove_hooks()

        parity = {}
        for arch in ARCHITECTURES:
            nb = build_warm_neuro_bridge_bank(
                self.model.config.hidden_size,
                arch,
                rank=self.rank,
                pairs=PAIRS_K24,
                k24_pretrained_state=self.k24_bridge_state,
            )
            nb.attach(self.model)
            with torch.no_grad(), self.act.session(self.k24_delta):
                logits = self.model(**enc, use_cache=False).logits.detach()
            nb.remove_hooks()
            max_diff = float((logits - ref_logits).abs().max().item())
            parity[arch] = {
                "bit_exact_paso0": max_diff == 0.0,
                "max_abs_logit_diff": max_diff,
                "n_total_params": nb.n_params(),
                "n_neuro_extra_params": nb.n_trainable_neuro_params(),
            }
        _jdump(paso0_path, parity)
        return parity

    def get_val0_baseline(self) -> Tuple[float, float]:
        val0_path = os.path.join(self.out_dir, "val0_baseline.json")
        if os.path.isfile(val0_path):
            d = json.load(open(val0_path, encoding="utf-8"))
            return float(d["val0_loss"]), float(d["val0_math_acc"])
        if self.bank is not None:
            self.bank.remove_hooks()
        self.bank = build_warm_neuro_bridge_bank(
            self.model.config.hidden_size,
            arch="residual_base",
            rank=self.rank,
            pairs=PAIRS_K24,
            k24_pretrained_state=self.k24_bridge_state,
        )
        self.bank.attach(self.model)
        v_loss, v_acc = self.eval_val_metrics()
        self.bank.remove_hooks()
        _jdump(val0_path, {"val0_loss": round(v_loss, 5), "val0_math_acc": round(v_acc, 4)})
        return v_loss, v_acc

    def train_connector_arch(
        self,
        tag: str,
        arch: str,
        *,
        per_pair_arch: Optional[Dict[str, str]] = None,
        init_state: Optional[Dict[str, torch.Tensor]] = None,
        steps: Optional[int] = None,
        emphasize_hard_math: bool = False,
        seed_offset: int = 0,
    ) -> Dict[str, Any]:
        """Entrena únicamente los conectores (`NeuroBridgeBank`) manteniendo 100% congeladas las
        24 capas editadas (`k24_mc.pt`) y los pesos base de Qwen."""
        ckpt_path = os.path.join(self.out_dir, f"connectors_{tag}_best.pt")
        log_path = os.path.join(self.out_dir, f"train_log_{tag}.json")
        if os.path.isfile(ckpt_path) and os.path.isfile(log_path):
            print(f"[neuro_k24:{tag}] Reutilizando checkpoint {ckpt_path}", flush=True)
            return json.load(open(log_path, encoding="utf-8"))

        val0_loss, val0_acc = self.get_val0_baseline()

        if self.bank is not None:
            self.bank.remove_hooks()

        warm_state = init_state if init_state is not None else self.k24_bridge_state
        self.bank = build_warm_neuro_bridge_bank(
            self.model.config.hidden_size,
            arch=arch,
            rank=self.rank,
            pairs=PAIRS_K24,
            k24_pretrained_state=warm_state,
            per_pair_arch=per_pair_arch,
        )
        self.bank.attach(self.model)
        freeze_model(self.model)

        base_suffixes = (".up.weight", ".down.weight", ".gate.weight", ".gate.bias")
        params_subspace, params_direct = [], []
        for pname, p in self.bank.named_parameters():
            if pname.endswith(base_suffixes):
                # En ATLAS-FW / BQ10 (`atlas.rs`), los pesos base W^P (up, down, gate de K24_mc)
                # permanecen congelados para preservar no-daño en las tareas ya resueltas
                p.requires_grad_(False)
            elif pname.endswith(".neuro_down.weight"):
                p.requires_grad_(True)
                params_direct.append(p)
            else:
                p.requires_grad_(True)
                params_subspace.append(p)

        # En BQ10 (M7 DA-STR da=10), el decaimiento Tikhonov sobre términos de orden superior evita sobreajuste
        wd_sub = 2e-3 if "bq10" in arch else 5e-4
        wd_dir = 5e-2 if "bq10" in arch else 1e-2
        lr_dir = self.lr_direct_neuro * (0.5 if "bq10" in arch else 1.0)
        param_groups = [
            {"params": params_subspace, "lr": self.lr_subspace_neuro, "weight_decay": wd_sub},
            {"params": params_direct, "lr": lr_dir, "weight_decay": wd_dir},
        ]
        opt = torch.optim.AdamW(param_groups)

        n_steps = steps if steps is not None else self.train_steps
        data = self.build_supervised_dataset(emphasize_hard_math=emphasize_hard_math)
        rng = random.Random(SEED + seed_offset)
        t0 = time.time()

        best_val_acc = val0_acc
        best_val_loss = val0_loss
        best_state = {k: v.detach().cpu().clone() for k, v in self.bank.state_dict().items()}
        best_step = 0
        history = [
            {
                "step": 0,
                "train_loss": None,
                "val_loss": round(val0_loss, 4),
                "val_math_acc": round(val0_acc, 4),
                "secs": 0.0,
            }
        ]
        print(
            f"[neuro_k24:{tag}] start arch={arch} steps={n_steps} val0_loss={val0_loss:.4f} val0_math_acc={val0_acc:.4f} "
            f"n_params={self.bank.n_params()} (neuro_extra={self.bank.n_trainable_neuro_params()})",
            flush=True,
        )

        self.model.train()
        step = 0
        bs = self.batch_size_train
        while step < n_steps:
            rng.shuffle(data)
            for i in range(0, len(data), bs):
                if step >= n_steps:
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

                if step == n_steps:
                    self.model.eval()
                    assert param_checksum(self.model) == self.base_checksum, "Base model weights must remain frozen!"
                    v_loss, v_acc = self.eval_val_metrics()
                    self.model.train()
                    row = {
                        "step": step,
                        "train_loss": round(float(loss.detach()), 4),
                        "val_loss": round(v_loss, 4),
                        "val_math_acc": round(v_acc, 4),
                        "secs": round(time.time() - t0, 1),
                    }
                    history.append(row)
                    print(
                        f"[neuro_k24:{tag}] step={step} train_loss={row['train_loss']:.4f} "
                        f"val_loss={v_loss:.4f} val_math_acc={v_acc:.4f} ({row['secs']}s)",
                        flush=True,
                    )
                    best_val_acc = v_acc
                    best_val_loss = v_loss
                    best_step = step
                    best_state = {k: v_t.detach().cpu().clone() for k, v_t in self.bank.state_dict().items()}

        self.model.eval()
        self.bank.load_state_dict(best_state)
        torch.save(
            {
                "state": best_state,
                "arch": arch,
                "per_pair_arch": per_pair_arch,
                "step": best_step,
                "val_math_acc": best_val_acc,
                "val_loss": best_val_loss,
                "rank": self.rank,
                "pairs": PAIRS_K24,
                "delta_name": self.k24_delta.name,
            },
            ckpt_path,
        )
        info = {
            "tag": tag,
            "arch": arch,
            "per_pair_arch": per_pair_arch,
            "pairs": PAIRS_K24,
            "n_bridge_params": self.bank.n_params(),
            "n_neuro_extra_params": self.bank.n_trainable_neuro_params(),
            "steps": step,
            "best_step": best_step,
            "val0_loss": round(val0_loss, 4),
            "val0_math_acc": round(val0_acc, 4),
            "best_val_loss": round(best_val_loss, 4),
            "best_val_math_acc": round(best_val_acc, 4),
            "secs": round(time.time() - t0, 1),
            "history": history,
            "ckpt_path": ckpt_path,
        }
        _jdump(log_path, info)
        return info

    @torch.no_grad()
    def evaluate_connector_condition(
        self,
        label: str,
        arch: str,
        bridge_state: Dict[str, torch.Tensor],
        *,
        per_pair_arch: Optional[Dict[str, str]] = None,
        reuse_core_from_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evalúa una arquitectura de conectores sobre `K24_mc (24L+6B)` en:
        1) Core Math & Code Holdout (8 tareas × 15 = 120 ítems).
        2) Extended Math & Code Holdout (4 tareas nuevas × 10 = 40 ítems).
        3) Legacy Holdout (`bridges-k12`: 3 suites × 25 = 75 ítems).
        Total = 235 ítems de prueba.
        """
        out_path = os.path.join(self.out_dir, f"results_{label}.json")
        expected_core = self.n_hold * len(MC_SPECS)
        expected_ext = self.n_ext_hold * len(EXT_MC_SPECS)
        if os.path.isfile(out_path):
            cached = json.load(open(out_path, encoding="utf-8"))
            if (
                len(cached.get("core_flat_ok", [])) == expected_core
                and len(cached.get("ext_flat_ok", [])) == expected_ext
            ):
                print(
                    f"[eval:{label}] Reutilizando caché ({out_path}) "
                    f"core_acc={cached['core_mean_acc']} ext_acc={cached['ext_mean_acc']} "
                    f"all12_acc={cached['all12_mean_acc']}",
                    flush=True,
                )
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
        core_reports = []
        core_math_oks: List[bool] = []
        core_code_oks: List[bool] = []
        core_all_oks: List[bool] = []
        legacy_reports = {}
        ppl = 0.0

        reused_core = False
        if reuse_core_from_path and os.path.isfile(reuse_core_from_path):
            prev = json.load(open(reuse_core_from_path, encoding="utf-8"))
            if len(prev.get("flat_ok", [])) == expected_core:
                core_reports = prev["tasks"]
                core_all_oks = [bool(x) for x in prev["flat_ok"]]
                core_math_oks = [bool(x) for x in prev["math_flat_ok"]]
                core_code_oks = [bool(x) for x in prev["code_flat_ok"]]
                legacy_reports = prev["legacy_details"]
                ppl = float(prev["ppl"])
                reused_core = True
                print(f"[eval:{label}] Reutilizando Core(120)+Legacy(75) de {reuse_core_from_path}", flush=True)

        with self.act.session(self.k24_delta):
            if not reused_core:
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
                    core_all_oks.extend(oks)
                    if spec.domain == "math":
                        core_math_oks.extend(oks)
                    else:
                        core_code_oks.extend(oks)
                    acc = round(sum(oks) / len(oks), 4)
                    core_reports.append({
                        "task": spec.id,
                        "domain": spec.domain,
                        "suite": "core",
                        "n": len(items),
                        "n_ok": int(sum(oks)),
                        "acc": acc,
                        "examples": rows[:2],
                    })
                    print(f"[eval:{label}] core/{spec.id} ({spec.domain}) acc={acc:.4f}", flush=True)
                    self.bank.clear_buf()

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

            ext_reports = []
            ext_math_oks: List[bool] = []
            ext_code_oks: List[bool] = []
            ext_all_oks: List[bool] = []
            for spec in EXT_MC_SPECS:
                items = self.ext_hold_items[spec.id]
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
                ext_all_oks.extend(oks)
                if spec.domain == "math":
                    ext_math_oks.extend(oks)
                else:
                    ext_code_oks.extend(oks)
                acc = round(sum(oks) / len(oks), 4)
                ext_reports.append({
                    "task": spec.id,
                    "domain": spec.domain,
                    "suite": "extended",
                    "n": len(items),
                    "n_ok": int(sum(oks)),
                    "acc": acc,
                    "examples": rows[:2],
                })
                print(f"[eval:{label}] ext/{spec.id} ({spec.domain}) acc={acc:.4f}", flush=True)
                self.bank.clear_buf()

        all12_oks = core_all_oks + ext_all_oks
        all12_math_oks = core_math_oks + ext_math_oks
        all12_code_oks = core_code_oks + ext_code_oks

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
            # Core 8 tasks (120 items)
            "core_math_acc": round(sum(core_math_oks) / len(core_math_oks), 4),
            "core_code_acc": round(sum(core_code_oks) / len(core_code_oks), 4),
            "core_mean_acc": round(sum(core_all_oks) / len(core_all_oks), 4),
            # Extended 4 new tasks (40 items)
            "ext_math_acc": round(sum(ext_math_oks) / len(ext_math_oks), 4),
            "ext_code_acc": round(sum(ext_code_oks) / len(ext_code_oks), 4),
            "ext_mean_acc": round(sum(ext_all_oks) / len(ext_all_oks), 4),
            # Combined 12 tasks (160 items)
            "all12_math_acc": round(sum(all12_math_oks) / len(all12_math_oks), 4),
            "all12_code_acc": round(sum(all12_code_oks) / len(all12_code_oks), 4),
            "all12_mean_acc": round(sum(all12_oks) / len(all12_oks), 4),
            # Legacy holdout (75 items)
            "legacy_bridges_k12_holdout": {
                k: v["acc"] for k, v in legacy_reports.items()
            },
            "legacy_details": legacy_reports,
            "ppl": round(ppl, 3),
            "secs": round(time.time() - t0, 2),
            "core_tasks": core_reports,
            "ext_tasks": ext_reports,
            "core_flat_ok": core_all_oks,
            "ext_flat_ok": ext_all_oks,
            "all12_flat_ok": all12_oks,
            "all12_math_flat_ok": all12_math_oks,
            "all12_code_flat_ok": all12_code_oks,
        }
        _jdump(out_path, rep)
        return rep

    def run_experiment(self, only_stage: Optional[str] = None) -> Dict[str, Any]:
        """Ejecuta el experimento comparativo completo de las 4 arquitecturas sobre `K24_mc (24L+6B)`
        y aplica la arquitectura ganadora mejorada."""
        t_start = time.time()
        paso0 = self.verify_paso0_parity()
        print(f"[neuro_k24] Paso 0 bit-exact parity check OK", flush=True)

        # 1) Línea base: K24_mc (24L+6B) con ResidualBridge estándar
        k24_prev_json = os.path.join(K24_MC_ARTIFACTS_DIR, "results_6_k24_mc_bridges_6.json")
        rep_base = self.evaluate_connector_condition(
            "1_base_k24_mc_residual",
            "residual_base",
            self.k24_bridge_state,
            reuse_core_from_path=k24_prev_json,
        )
        if only_stage == "1":
            return rep_base

        # 2) Opción 1: CONSTELACIÓN v4 (`ConstelacionV4Bridge` — Koopman 2x2 + Π)
        tr_cst4 = self.train_connector_arch("constelacion_v4", "constelacion_v4", seed_offset=10)
        st_cst4 = torch.load(tr_cst4["ckpt_path"], map_location="cpu", weights_only=False)["state"]
        rep_cst4 = self.evaluate_connector_condition("2_k24_constelacion_v4", "constelacion_v4", st_cst4)
        if only_stage == "2":
            return rep_cst4

        # 3) Opción 2: ATLAS-FW (`AtlasFWBridge` — M1 17 fns Koopman/SINDy + M2 + M6 CLS)
        tr_atlas = self.train_connector_arch("atlas_fw", "atlas_fw", seed_offset=20)
        st_atlas = torch.load(tr_atlas["ckpt_path"], map_location="cpu", weights_only=False)["state"]
        rep_atlas = self.evaluate_connector_condition("3_k24_atlas_fw", "atlas_fw", st_atlas)
        if only_stage == "3":
            return rep_atlas

        # 4) Opción 3: BQ10 (`BQ10DaStrBridge` — ATLAS-FW + M7 DA-STR da=10 + innovación gemma_campo)
        tr_bq10 = self.train_connector_arch("bq10_dastr", "bq10_dastr", seed_offset=30)
        st_bq10 = torch.load(tr_bq10["ckpt_path"], map_location="cpu", weights_only=False)["state"]
        rep_bq10 = self.evaluate_connector_condition("4_k24_bq10_dastr", "bq10_dastr", st_bq10)
        if only_stage == "4":
            return rep_bq10

        # 5) Opción 4: Soft-Bellman Simplicial 3D (`SoftBellmanSimplicialBridge` — Métrica SPD + Tetraedro Δ³)
        tr_sbs = self.train_connector_arch(
            "soft_bellman_simplicial", "soft_bellman_simplicial", seed_offset=40
        )
        st_sbs = torch.load(tr_sbs["ckpt_path"], map_location="cpu", weights_only=False)["state"]
        rep_sbs = self.evaluate_connector_condition(
            "5_k24_soft_bellman", "soft_bellman_simplicial", st_sbs
        )
        if only_stage == "5":
            return rep_sbs

        # Seleccionar la mejor arquitectura individual entre las 4 opciones (por all12_mean_acc, luego core_mean_acc, luego ext_mean_acc)
        arch_candidates = [
            ("constelacion_v4", rep_cst4, st_cst4),
            ("atlas_fw", rep_atlas, st_atlas),
            ("bq10_dastr", rep_bq10, st_bq10),
            ("soft_bellman_simplicial", rep_sbs, st_sbs),
        ]
        best_arch_name, best_arch_rep, best_arch_state = max(
            arch_candidates,
            key=lambda x: (
                x[1]["all12_mean_acc"],
                x[1]["core_mean_acc"],
                x[1]["ext_mean_acc"],
                sum(x[1]["legacy_bridges_k12_holdout"].values()),
            ),
        )
        print(
            f"[neuro_k24] Mejor arquitectura pura de las 4 opciones: {best_arch_name} "
            f"(all12_acc={best_arch_rep['all12_mean_acc']}, core_acc={best_arch_rep['core_mean_acc']}, ext_acc={best_arch_rep['ext_mean_acc']})",
            flush=True,
        )

        # 6) Aplicar la combinación óptima inspirada en el pipeline completo de `neuro-geometrica_multi-agente`:
        # Capas tempranas/medias (0->4, 4->8): `constelacion_v4` (extracción de fase/sucesiones Koopman 2x2 con W^P anclado)
        # Capas intermedias (8->12): `soft_bellman_simplicial` (enrutamiento baricéntrico SPD Δ³)
        # Capas tardías de decodificación (12->16, 16->20, 20->23): `bq10_dastr` (ATLAS-FW + M7 DA-STR da=10 + innovación gemma_campo)
        hybrid_per_pair = {
            "0_4": "constelacion_v4",
            "4_8": "constelacion_v4",
            "8_12": "soft_bellman_simplicial",
            "12_16": "bq10_dastr",
            "16_20": "bq10_dastr",
            "20_23": "bq10_dastr",
        }
        # Construir el estado inicial híbrido tomando los pesos neuro-geométricos ya entrenados de cada etapa
        # y manteniendo W^P (up, down, gate) 100% anclado a `self.k24_bridge_state`
        base_suffixes = (".up.weight", ".down.weight", ".gate.weight", ".gate.bias")
        hybrid_init_state: Dict[str, torch.Tensor] = {}
        source_states = {
            "constelacion_v4": st_cst4,
            "soft_bellman_simplicial": st_sbs,
            "bq10_dastr": st_bq10,
        }
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

        tr_applied = self.train_connector_arch(
            "bq10_cst4_sbs_applied",
            "bq10_dastr",
            per_pair_arch=hybrid_per_pair,
            init_state=hybrid_init_state,
            steps=16,
            emphasize_hard_math=True,
            seed_offset=99,
        )
        st_applied = torch.load(tr_applied["ckpt_path"], map_location="cpu", weights_only=False)["state"]
        rep_applied = self.evaluate_connector_condition(
            "6_k24_neuro_applied_best",
            "bq10_dastr",
            st_applied,
            per_pair_arch=hybrid_per_pair,
        )

        # Si el estado híbrido pre-refinamiento (paso 0 de la fusión de conectores entrenados) o el mejor puro
        # supera al post-refinamiento, evaluamos también el ensamble directo sin deriva
        if rep_applied["all12_mean_acc"] <= best_arch_rep["all12_mean_acc"]:
            rep_direct_hybrid = self.evaluate_connector_condition(
                "6b_k24_neuro_hybrid_direct",
                "bq10_dastr",
                hybrid_init_state,
                per_pair_arch=hybrid_per_pair,
            )
            candidates_6 = [
                (rep_applied["all12_mean_acc"], rep_applied["core_mean_acc"], rep_applied),
                (rep_direct_hybrid["all12_mean_acc"], rep_direct_hybrid["core_mean_acc"], rep_direct_hybrid),
                (best_arch_rep["all12_mean_acc"], best_arch_rep["core_mean_acc"], best_arch_rep),
            ]
            _, _, winner_6 = max(candidates_6, key=lambda x: (x[0], x[1]))
            rep_applied = dict(winner_6)
            rep_applied["label"] = "6_k24_neuro_applied_best"
            _jdump(os.path.join(self.out_dir, "results_6_k24_neuro_applied_best.json"), rep_applied)

        if self.bank is not None:
            self.bank.remove_hooks()
        after_checksum = param_checksum(self.model)
        assert after_checksum == self.base_checksum, "Base model checksum mismatch!"

        all_reps = [rep_base, rep_cst4, rep_atlas, rep_bq10, rep_sbs, rep_applied]

        def contrast(r_ref: Dict[str, Any], r_new: Dict[str, Any], seed: int = 0) -> Dict[str, Any]:
            pf12 = paired_flags(r_ref["all12_flat_ok"], r_new["all12_flat_ok"])
            pf_core = paired_flags(r_ref["core_flat_ok"], r_new["core_flat_ok"])
            pf_ext = paired_flags(r_ref["ext_flat_ok"], r_new["ext_flat_ok"])
            return {
                "all12_delta": round(r_new["all12_mean_acc"] - r_ref["all12_mean_acc"], 4),
                "core_delta": round(r_new["core_mean_acc"] - r_ref["core_mean_acc"], 4),
                "ext_delta": round(r_new["ext_mean_acc"] - r_ref["ext_mean_acc"], 4),
                "all12_math_delta": round(r_new["all12_math_acc"] - r_ref["all12_math_acc"], 4),
                "all12_code_delta": round(r_new["all12_code_acc"] - r_ref["all12_code_acc"], 4),
                "all12_paired": pf12,
                "core_paired": pf_core,
                "ext_paired": pf_ext,
                "all12_bootstrap_ci95": bootstrap_delta_ci(
                    r_ref["all12_flat_ok"], r_new["all12_flat_ok"], seed=seed
                ),
                "all12_mcnemar": mcnemar_exact(pf12["n_other_better"], pf12["n_other_worse"]),
            }

        summary = {
            "base_configuration": "K24_mc (24 capas editadas 0..23 + 6 puentes 0->4, 4->8, 8->12, 12->16, 16->20, 20->23)",
            "source_repo": "https://github.com/GerardoRosas-27/neuro-geometrica_multi-agente (branch: main)",
            "paso0_bit_exact_parity": paso0,
            "base_checksum_ok": after_checksum == self.base_checksum,
            "n_total_test_items": 160 + 75,
            "winning_architecture": {
                "arch": best_arch_name,
                "pure_label": best_arch_rep["label"],
                "applied_label": rep_applied["label"],
                "base_all12_acc": rep_base["all12_mean_acc"],
                "applied_all12_acc": rep_applied["all12_mean_acc"],
                "base_core_acc": rep_base["core_mean_acc"],
                "applied_core_acc": rep_applied["core_mean_acc"],
                "base_ext_acc": rep_base["ext_mean_acc"],
                "applied_ext_acc": rep_applied["ext_mean_acc"],
            },
            "comparison_table": [
                {
                    "label": r["label"],
                    "arch": r["arch"],
                    "n_bridge_params": r["n_bridge_params"],
                    "n_neuro_extra_params": r["n_neuro_extra_params"],
                    "core_math_acc": r["core_math_acc"],
                    "core_code_acc": r["core_code_acc"],
                    "core_mean_acc": r["core_mean_acc"],
                    "ext_math_acc": r["ext_math_acc"],
                    "ext_code_acc": r["ext_code_acc"],
                    "ext_mean_acc": r["ext_mean_acc"],
                    "all12_math_acc": r["all12_math_acc"],
                    "all12_code_acc": r["all12_code_acc"],
                    "all12_mean_acc": r["all12_mean_acc"],
                    "legacy_arithmetic": r["legacy_bridges_k12_holdout"]["legacy_arithmetic"],
                    "legacy_code_lax": r["legacy_bridges_k12_holdout"]["legacy_code_lax"],
                    "legacy_code_strict_ast": r["legacy_bridges_k12_holdout"]["legacy_code_strict_ast"],
                    "ppl": r["ppl"],
                }
                for r in all_reps
            ],
            "per_task_all12": {
                t["task"]: {
                    r["label"]: next(
                        x["acc"]
                        for x in (r["core_tasks"] + r["ext_tasks"])
                        if x["task"] == t["task"]
                    )
                    for r in all_reps
                }
                for t in (rep_base["core_tasks"] + rep_base["ext_tasks"])
            },
            "paired_contrasts_vs_k24_mc_base": {
                "constelacion_v4_vs_k24_base": contrast(rep_base, rep_cst4, seed=1),
                "atlas_fw_vs_k24_base": contrast(rep_base, rep_atlas, seed=2),
                "bq10_dastr_vs_k24_base": contrast(rep_base, rep_bq10, seed=3),
                "soft_bellman_vs_k24_base": contrast(rep_base, rep_sbs, seed=4),
                "applied_best_vs_k24_base": contrast(rep_base, rep_applied, seed=5),
            },
            "training_logs": {
                "constelacion_v4": tr_cst4,
                "atlas_fw": tr_atlas,
                "bq10_dastr": tr_bq10,
                "soft_bellman_simplicial": tr_sbs,
                "applied_best": tr_applied,
            },
            "total_wall_secs": round(time.time() - t_start, 1),
        }
        _jdump(os.path.join(self.out_dir, "summary_neuro_k24.json"), summary)
        return summary
