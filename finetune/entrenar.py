"""Fine-tuning LoRA (o QLoRA en GPU) + fusión del adaptador con los pesos base.

Mide tiempo, RAM pico (RSS) y loss, y evalúa el holdout antes y después.
Escribe métricas en finetune/out/metricas.json.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import resource
import threading
import time
from pathlib import Path

import psutil
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluar import cargar_holdout, evaluar

HERE = Path(__file__).parent


class RamPico:
    """Muestrea la RSS del proceso cada 0.5 s."""

    def __init__(self) -> None:
        self.pico = 0
        self._stop = threading.Event()
        self._p = psutil.Process()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.pico = max(self.pico, self._p.memory_info().rss)
            time.sleep(0.5)

    def mb(self) -> float:
        return round(max(self.pico, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024) / 2**20, 1)


def tokenizar(tok, ejemplo: dict, max_len: int) -> dict:
    """Solo la respuesta del asistente cuenta en el loss (labels=-100 en el prompt)."""
    msgs = ejemplo["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=False)
    full = tok.apply_chat_template(msgs, tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return {"input_ids": f_ids, "labels": labels[:len(f_ids)]}


def lotes(data: list[dict], bs: int, pad: int):
    for i in range(0, len(data), bs):
        b = data[i:i + bs]
        L = max(len(x["input_ids"]) for x in b)
        ids = torch.full((len(b), L), pad)
        lab = torch.full((len(b), L), -100)
        att = torch.zeros((len(b), L), dtype=torch.long)
        for j, x in enumerate(b):
            n = len(x["input_ids"])
            ids[j, :n] = torch.tensor(x["input_ids"])
            lab[j, :n] = torch.tensor(x["labels"])
            att[j, :n] = 1
        yield ids, att, lab


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--qlora", action="store_true", help="4-bit con bitsandbytes (solo GPU CUDA)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--skip-eval-base", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ram = RamPico()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m: dict = {"base": a.base, "device": dev, "threads": torch.get_num_threads(), "args": vars(a)}

    t = time.time()
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw: dict = {"dtype": torch.float32 if dev == "cpu" else torch.bfloat16}
    if a.qlora:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
        kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(a.base, **kw)
    if not a.qlora:
        model.to(dev)
    m["carga_base_s"] = round(time.time() - t, 1)
    m["parametros_base"] = sum(p.numel() for p in model.parameters())

    holdout = cargar_holdout(str(Path(a.data) / "holdout.jsonl"))
    if not a.skip_eval_base:
        m["eval_antes"] = evaluar(model.cpu() if dev == "cpu" else model, tok, holdout)
        print("eval antes:", {k: v for k, v in m["eval_antes"].items() if k != "ejemplos"}, flush=True)

    if a.qlora:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    cfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    m["parametros_entrenables"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"entrenables: {m['parametros_entrenables']:,} de {m['parametros_base']:,}", flush=True)

    rows = [json.loads(l) for l in (Path(a.data) / "train.jsonl").read_text(encoding="utf-8").splitlines() if l]
    data = [tokenizar(tok, r, a.max_len) for r in rows]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    pasos = a.epochs * math.ceil(len(data) / a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / pasos))
    model.train()
    hist, paso, t = [], 0, time.time()
    for ep in range(a.epochs):
        random.shuffle(data)
        acc = []
        for ids, att, lab in lotes(data, a.bs, tok.pad_token_id):
            loss = model(input_ids=ids.to(dev), attention_mask=att.to(dev), labels=lab.to(dev)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            acc.append(loss.item()); paso += 1
        hist.append({"epoch": ep + 1, "loss": round(sum(acc) / len(acc), 4), "s": round(time.time() - t, 1)})
        print(hist[-1], flush=True)
    m["entreno_s"] = round(time.time() - t, 1)
    m["pasos"] = paso
    m["ejemplos"] = len(data)
    m["loss_por_epoch"] = hist

    model.save_pretrained(out / "adaptador")
    t = time.time()
    fused = model.merge_and_unload()
    fused.save_pretrained(out / "fusionado", safe_serialization=True)
    tok.save_pretrained(out / "fusionado")
    m["fusion_s"] = round(time.time() - t, 1)

    m["eval_despues"] = evaluar(fused.cpu() if dev == "cpu" else fused, tok, holdout)
    print("eval después:", {k: v for k, v in m["eval_despues"].items() if k != "ejemplos"}, flush=True)
    m["ram_pico_mb"] = ram.mb()
    (out / "metricas.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RAM pico (MB):", m["ram_pico_mb"])


if __name__ == "__main__":
    main()
