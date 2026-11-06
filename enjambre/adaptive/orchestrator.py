"""Orquestador adaptativo: un modelo 0.5B + banco de deltas + memoria + auto-mejora."""
from __future__ import annotations
import json
import os
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from enjambre.config import artifacts_dir, resolve
from enjambre.generate import generate, generate_one, perplexity

from .bank import ACTIVE, EXPERIMENTAL, ExpertBank
from .catalog import TaskSpec, all_tasks, catalog_manifest, task_by_id
from .improve import AutoImprover, _acc
from .memory_net import MemoryNet
from .router_heuristic import route as heuristic_route
from .snapshot import SnapshotActivator

DEFAULT_MODEL = "qwen2.5-0.5b"


class AdaptiveHarness:
    def __init__(
        self,
        model_key: str = DEFAULT_MODEL,
        root: Optional[str] = None,
        dtype=torch.float32,
        max_new: int = 32,
        load_model: bool = True,
    ):
        self.cfg = resolve(model_key)
        if root is None:
            # artifacts/adaptive-0.5b por defecto (Qwen2.5-0.5B)
            self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "adaptive-0.5b"))
        else:
            self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.bank = ExpertBank(self.root)
        self.memory = MemoryNet(os.path.join(self.root, "memory_net.sqlite"))
        self.max_new = max_new
        self.model = None
        self.tok = None
        self.act = None
        self.improver = None
        self._decision_log: List[Dict[str, Any]] = []
        if load_model:
            self._load_model(dtype)

    def _load_model(self, dtype=torch.float32):
        torch.manual_seed(0)
        torch.set_num_threads(min(8, os.cpu_count() or 8))
        self.tok = AutoTokenizer.from_pretrained(self.cfg.hf_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.hf_id, dtype=dtype).eval()
        self.act = SnapshotActivator(self.model)
        self.improver = AutoImprover(self.model, self.tok, self.act, self.bank, self.memory, max_new=self.max_new)

    def status(self) -> Dict[str, Any]:
        import resource
        return {
            "model": self.cfg.hf_id,
            "key": self.cfg.key,
            "root": self.root,
            "bank": self.bank.summary(),
            "memory": self.memory.stats(),
            "catalog": {"n_tasks": len(all_tasks())},
            "rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
            "router": "heuristic_keywords (not LLM reasoning)",
            "active_expert": self.act.active() if self.act else None,
        }

    def list_experts(self) -> List[Dict[str, Any]]:
        out = []
        for t in all_tasks():
            active = self.bank.get_active(t.id)
            versions = self.bank.list_for_task(t.id)
            out.append({
                "task_id": t.id, "name": t.name, "family": t.family,
                "active_version": active.version_id if active else None,
                "active_status": active.status if active else None,
                "n_versions": len(versions),
                "statuses": {v.version_id: v.status for v in versions},
            })
        return out

    def catalog(self) -> Dict[str, Any]:
        return catalog_manifest()

    def route(self, prompt: str) -> Dict[str, Any]:
        tid, info = heuristic_route(prompt, all_tasks())
        return {"task_id": tid, **info}

    def solve(self, prompt: str, *, task_id: Optional[str] = None, use_expert: bool = True, max_new: Optional[int] = None) -> Dict[str, Any]:
        """Resuelve con delta activo de la tarea (o base). Restaura pesos al terminar."""
        assert self.model is not None
        t0 = time.time()
        route_info = None
        if task_id is None:
            task_id, route_info = heuristic_route(prompt, all_tasks())
        delta = None
        version_id = None
        if use_expert and task_id:
            active = self.bank.get_active(task_id)
            if active:
                version_id = active.version_id
                delta = self.bank.load_delta(version_id)
        with self.act.session(delta):
            out = generate_one(self.model, self.tok, prompt, max_new_tokens=max_new or self.max_new)
        # auto-check if we know the task and prompt matches an item — optional
        ok = None
        reason = ""
        if task_id:
            try:
                task = task_by_id(task_id)
                # try match against any known item prompt
                matched = None
                for it in task.all_non_heldout() + task.heldout_items:
                    if it.prompt.strip() == prompt.strip():
                        matched = it
                        break
                if matched is not None:
                    ok = bool(matched.check(out))
                    reason = "auto_check_matched_item"
                    self.memory.record_experience(task_id, prompt, out, ok, version_id=version_id, source="auto_check", reason=reason)
            except KeyError:
                pass
        decision = {
            "task_id": task_id, "version_id": version_id, "used_expert": delta is not None,
            "output": out, "ok": ok, "route": route_info, "seconds": round(time.time() - t0, 3),
        }
        self._decision_log.append({"ts": time.time(), "op": "solve", **{k: decision[k] for k in decision if k != "output"}})
        return decision

    def evaluate_task(self, task_id: str, split: str = "val") -> Dict[str, Any]:
        task = task_by_id(task_id)
        active = self.bank.get_active(task_id)
        vid = active.version_id if active else None
        base = self.improver.evaluate_version(task, None, split)
        exp = self.improver.evaluate_version(task, vid, split) if vid else None
        return {"task_id": task_id, "split": split, "base": base, "expert": exp, "active_version": vid}

    def feedback(self, task_id: str, prompt: str, output: str, ok: bool, *, reason: str = "user_feedback", version_id: Optional[str] = None) -> Dict[str, Any]:
        """Feedback marcado (usuario u orquestador). ok=True entra a retrieve_good."""
        active = self.bank.get_active(task_id)
        vid = version_id or (active.version_id if active else None)
        eid = self.memory.record_experience(task_id, prompt, output, ok, version_id=vid, source="user_feedback", reason=reason)
        return {"experience_id": eid, "ok": ok, "version_id": vid}

    def memory_query(self, task_id: str, *, good: bool = True, limit: int = 20) -> Dict[str, Any]:
        rows = self.memory.retrieve_good(task_id, limit) if good else self.memory.failures(task_id, limit)
        return {"task_id": task_id, "good": good, "n": len(rows), "rows": rows}

    def memory_reset(self, task_id: str, keep_audit: bool = True) -> Dict[str, Any]:
        return self.memory.reset_task(task_id, keep_audit=keep_audit)

    def create_expert(self, task_id: str) -> Dict[str, Any]:
        task = task_by_id(task_id)
        return self.improver.build_initial(task)

    def propose_replacement(self, task_id: str) -> Dict[str, Any]:
        return self.improver.propose_replacement(task_by_id(task_id))

    def evaluate_replacement(self, task_id: str, candidate_version_id: str) -> Dict[str, Any]:
        return self.improver.evaluate_and_maybe_promote(task_by_id(task_id), candidate_version_id)

    def auto_improve(self, *, task_ids: Optional[List[str]] = None, episodes: int = 1, max_tasks: Optional[int] = None) -> Dict[str, Any]:
        """Ciclo acotado: propose→val→promote/reject. No usa heldout. No garantiza mejora."""
        tasks = [task_by_id(t) for t in task_ids] if task_ids else all_tasks()
        result = self.improver.auto_improve(tasks, episodes=episodes, max_tasks=max_tasks)
        self._decision_log.append({"ts": time.time(), "op": "auto_improve", "accepted": result["accepted"], "rejected": result["rejected"]})
        return result

    def improve_history(self, task_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        return self.memory.improve_history(task_id, limit)

    def build_all(self, *, resume: bool = True, checkpoint_name: str = "build_checkpoint.json") -> Dict[str, Any]:
        """Construye expertos iniciales para las 50 tareas con checkpoint reanudable."""
        from .builder import build_all_experts
        return build_all_experts(self, resume=resume, checkpoint_name=checkpoint_name)
