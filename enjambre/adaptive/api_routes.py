"""Rutas FastAPI del arnés adaptativo (montadas bajo /adaptive)."""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .orchestrator import AdaptiveHarness

router = APIRouter(prefix="/adaptive", tags=["adaptive"])
_h: Optional[AdaptiveHarness] = None


def get_harness() -> AdaptiveHarness:
    global _h
    if _h is None:
        model = os.environ.get("ENJAMBRE_ADAPTIVE_MODEL", "qwen2.5-0.5b")
        root = os.environ.get("ENJAMBRE_ADAPTIVE_ROOT")
        _h = AdaptiveHarness(model_key=model, root=root, load_model=True)
    return _h


class SolveIn(BaseModel):
    prompt: str
    task_id: Optional[str] = None
    use_expert: bool = True
    max_new_tokens: int = 32


class FeedbackIn(BaseModel):
    task_id: str
    prompt: str
    output: str
    ok: bool
    reason: str = "user_feedback"


class AutoImproveIn(BaseModel):
    task_ids: Optional[List[str]] = None
    episodes: int = Field(1, ge=1, le=5)
    max_tasks: Optional[int] = Field(None, ge=1, le=50)


class ReplaceIn(BaseModel):
    task_id: str


class EvalReplaceIn(BaseModel):
    task_id: str
    candidate_version_id: str


@router.get("/status")
def status():
    return get_harness().status()


@router.get("/catalog")
def catalog():
    return get_harness().catalog()


@router.get("/experts")
def experts():
    return {"experts": get_harness().list_experts()}


@router.post("/route")
def route(body: SolveIn):
    return get_harness().route(body.prompt)


@router.post("/solve")
def solve(body: SolveIn):
    return get_harness().solve(body.prompt, task_id=body.task_id, use_expert=body.use_expert, max_new=body.max_new_tokens)


@router.post("/feedback")
def feedback(body: FeedbackIn):
    return get_harness().feedback(body.task_id, body.prompt, body.output, body.ok, reason=body.reason)


@router.get("/memory/query")
def memory_query(task_id: str, good: bool = True, limit: int = 20):
    return get_harness().memory_query(task_id, good=good, limit=limit)


@router.post("/memory/reset")
def memory_reset(task_id: str, keep_audit: bool = True):
    return get_harness().memory_reset(task_id, keep_audit=keep_audit)


@router.post("/experts/create")
def create_expert(task_id: str):
    try:
        return get_harness().create_expert(task_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


@router.post("/experts/propose_replacement")
def propose(body: ReplaceIn):
    return get_harness().propose_replacement(body.task_id)


@router.post("/experts/evaluate_replacement")
def eval_replace(body: EvalReplaceIn):
    return get_harness().evaluate_replacement(body.task_id, body.candidate_version_id)


@router.post("/auto_improve")
def auto_improve(body: AutoImproveIn):
    """Presupuesto acotado de episodios. No garantiza mejora. Heldout intacto."""
    return get_harness().auto_improve(task_ids=body.task_ids, episodes=body.episodes, max_tasks=body.max_tasks)


@router.get("/improve/history")
def improve_history(task_id: Optional[str] = None, limit: int = 50):
    return {"history": get_harness().improve_history(task_id, limit)}


@router.post("/evaluate")
def evaluate(task_id: str, split: str = "val"):
    if split not in ("val", "tune", "heldout"):
        raise HTTPException(400, "split must be val|tune|heldout")
    return get_harness().evaluate_task(task_id, split)
