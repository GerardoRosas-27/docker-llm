"""Banco de expertos: versiones de deltas persistidas (lazy load), estados y provenance."""
from __future__ import annotations
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch

from enjambre.edits import BiasEdit, EditDelta


ACTIVE = "active"
EXPERIMENTAL = "experimental"  # built but no heldout gain / not promoted
QUARANTINE = "quarantine"
CANDIDATE = "candidate"
RETIRED = "retired"


@dataclass
class ExpertVersion:
    task_id: str
    version_id: str
    status: str
    path: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    parent_version_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _save_delta(path: str, delta: EditDelta) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "name": delta.name,
        "edits": [{"layer": b.layer, "scale": b.scale, "vec": b.vec.detach().cpu()} for b in delta.bias_edits],
        "meta": dict(delta.meta),
    }
    torch.save(payload, path)


def _load_delta(path: str) -> EditDelta:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    edits = [BiasEdit(layer=int(e["layer"]), vec=e["vec"].float(), scale=float(e["scale"])) for e in raw["edits"]]
    return EditDelta(name=raw.get("name", "expert"), bias_edits=edits, meta=dict(raw.get("meta") or {}))


class ExpertBank:
    """Un delta por versión; el 'activo' por tarea es el promovido vigente."""

    def __init__(self, root: str):
        self.root = root
        self.meta_path = os.path.join(root, "bank_meta.json")
        self.deltas_dir = os.path.join(root, "deltas")
        os.makedirs(self.deltas_dir, exist_ok=True)
        self.versions: Dict[str, ExpertVersion] = {}  # version_id ->
        self.active_by_task: Dict[str, str] = {}  # task_id -> version_id
        self._cache: Dict[str, EditDelta] = {}
        self._load_meta()

    def _load_meta(self) -> None:
        if not os.path.isfile(self.meta_path):
            return
        data = json.loads(open(self.meta_path, encoding="utf-8").read())
        for vid, row in data.get("versions", {}).items():
            self.versions[vid] = ExpertVersion(**row)
        self.active_by_task = dict(data.get("active_by_task") or {})

    def save_meta(self) -> None:
        payload = {
            "versions": {vid: v.to_dict() for vid, v in self.versions.items()},
            "active_by_task": self.active_by_task,
            "updated_at": time.time(),
        }
        tmp = self.meta_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.meta_path)

    def list_for_task(self, task_id: str) -> List[ExpertVersion]:
        return sorted([v for v in self.versions.values() if v.task_id == task_id], key=lambda v: v.created_at)

    def get_active(self, task_id: str) -> Optional[ExpertVersion]:
        vid = self.active_by_task.get(task_id)
        return self.versions.get(vid) if vid else None

    def load_delta(self, version_id: str) -> EditDelta:
        if version_id in self._cache:
            return self._cache[version_id]
        v = self.versions[version_id]
        d = _load_delta(v.path)
        self._cache[version_id] = d
        return d

    def register(
        self,
        task_id: str,
        delta: EditDelta,
        *,
        status: str,
        metrics: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        parent_version_id: Optional[str] = None,
        make_active: bool = False,
        version_id: Optional[str] = None,
    ) -> ExpertVersion:
        vid = version_id or f"{task_id}__{uuid.uuid4().hex[:10]}"
        path = os.path.join(self.deltas_dir, f"{vid}.pt")
        delta.name = task_id
        delta.meta = {**delta.meta, "version_id": vid, "task_id": task_id, "status": status}
        _save_delta(path, delta)
        ev = ExpertVersion(
            task_id=task_id,
            version_id=vid,
            status=status,
            path=path,
            metrics=dict(metrics or {}),
            provenance=dict(provenance or {}),
            parent_version_id=parent_version_id,
        )
        self.versions[vid] = ev
        self._cache[vid] = delta
        if make_active:
            self.active_by_task[task_id] = vid
            ev.status = ACTIVE
        self.save_meta()
        return ev

    def set_status(self, version_id: str, status: str) -> ExpertVersion:
        v = self.versions[version_id]
        v.status = status
        self.save_meta()
        return v

    def promote_atomic(self, task_id: str, new_version_id: str, *, retire_previous: bool = True) -> Dict[str, Any]:
        """Promueve new_version a activo; marca el anterior retired (conserva archivo)."""
        if new_version_id not in self.versions:
            raise KeyError(new_version_id)
        prev = self.active_by_task.get(task_id)
        self.active_by_task[task_id] = new_version_id
        self.versions[new_version_id].status = ACTIVE
        rolled = None
        if prev and prev != new_version_id and retire_previous:
            self.versions[prev].status = RETIRED
            rolled = prev
        self.save_meta()
        return {"task_id": task_id, "active": new_version_id, "previous": rolled}

    def rollback(self, task_id: str, previous_version_id: str) -> Dict[str, Any]:
        if previous_version_id not in self.versions:
            raise KeyError(previous_version_id)
        cur = self.active_by_task.get(task_id)
        self.active_by_task[task_id] = previous_version_id
        self.versions[previous_version_id].status = ACTIVE
        if cur and cur != previous_version_id:
            self.versions[cur].status = QUARANTINE
        self.save_meta()
        return {"task_id": task_id, "active": previous_version_id, "quarantined": cur}

    def summary(self) -> Dict[str, Any]:
        by_status: Dict[str, int] = {}
        for v in self.versions.values():
            by_status[v.status] = by_status.get(v.status, 0) + 1
        return {
            "n_versions": len(self.versions),
            "n_active_tasks": len(self.active_by_task),
            "by_status": by_status,
            "active_by_task": dict(self.active_by_task),
        }
