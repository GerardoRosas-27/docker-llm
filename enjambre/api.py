"""API HTTP (FastAPI) tool-callable para que un LLM en el contenedor orqueste el enjambre."""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .orchestrator import Swarm

DELTAS = os.environ.get("ENJAMBRE_DELTAS", os.path.join(os.path.dirname(__file__), "artifacts"))
MODEL = os.environ.get("ENJAMBRE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")

app = FastAPI(title="Enjambre de micro-expertos", version="0.1.0")
_swarm: Optional[Swarm] = None


def get_swarm() -> Swarm:
    global _swarm
    if _swarm is None:
        _swarm = Swarm(model_name=MODEL, deltas_dir=DELTAS if os.path.isdir(DELTAS) else None)
        if os.path.isdir(DELTAS):
            _swarm.load_deltas(DELTAS)
    return _swarm


class GenerateIn(BaseModel):
    prompt: str
    expert_id: Optional[str] = None
    max_new_tokens: int = 80


class RouteIn(BaseModel):
    prompt: str
    strategy: str = "hybrid"
    max_new_tokens: int = 80


class ExpertEditIn(BaseModel):
    expert_id: str
    contrast_prompts: List[Dict[str, Any]] = Field(
        ..., description="[{prompt, positive_system?}] o [{pos_text, neg_text}]"
    )
    layer: int = 10
    scale: float = 1.0
    system_prompt: Optional[str] = None


class MemoryWriteIn(BaseModel):
    subject: str
    object: str
    fact_id: Optional[str] = None
    layer: Optional[int] = None


class MemoryReadIn(BaseModel):
    subject: str
    max_new_tokens: int = 40


class ApplyIn(BaseModel):
    expert_id: str
    keep_memory: bool = True


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/status")
def status():
    return get_swarm().status()


@app.get("/experts")
def list_experts():
    return {"experts": get_swarm().list_experts()}


@app.post("/experts/apply")
def apply_expert(body: ApplyIn):
    try:
        return get_swarm().apply_expert(body.expert_id, keep_memory=body.keep_memory)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.post("/experts/revert")
def revert():
    get_swarm().revert()
    return {"active": []}


@app.post("/experts/create_or_modify")
def create_or_modify(body: ExpertEditIn):
    return get_swarm().create_or_modify_expert(
        body.expert_id, body.contrast_prompts, body.layer, body.scale, body.system_prompt
    )


@app.post("/generate")
def gen(body: GenerateIn):
    out = get_swarm().generate(body.prompt, expert_id=body.expert_id, max_new_tokens=body.max_new_tokens)
    return {"output": out}


@app.post("/route")
def route(body: RouteIn):
    return get_swarm().route_and_run(body.prompt, strategy=body.strategy, max_new_tokens=body.max_new_tokens)


@app.post("/memory/write")
def mem_write(body: MemoryWriteIn):
    return get_swarm().memory_write(body.subject, body.object, body.fact_id, body.layer)


@app.post("/memory/read")
def mem_read(body: MemoryReadIn):
    return get_swarm().memory_read(body.subject, body.max_new_tokens)


@app.get("/memory/list")
def mem_list():
    return {"facts": get_swarm().memory_list()}


@app.get("/ppl")
def ppl():
    return {"ppl": get_swarm().ppl()}


@app.post("/bench/quick")
def bench_quick():
    """Eval corta: 1 prompt por experto + recall de hechos actuales."""
    s = get_swarm()
    from .experts import all_expert_specs
    rows = []
    for e in all_expert_specs():
        if not e.eval_items:
            continue
        it = e.eval_items[0]
        r = s.route_and_run(it.prompt)
        rows.append({
            "task": e.id,
            "routed": r["expert"],
            "ok": bool(it.check(r["output"])),
            "output": r["output"][:200],
        })
    return {"rows": rows, "memory": s.memory_list()}


def main():
    import uvicorn
    host = os.environ.get("ENJAMBRE_HOST", "0.0.0.0")
    port = int(os.environ.get("ENJAMBRE_PORT", "8090"))
    uvicorn.run("enjambre.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
