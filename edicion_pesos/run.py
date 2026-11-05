"""Edicion directa de pesos (sin gradientes) de SmolLM2-135M-Instruct + evaluacion."""
import torch, json, time, math, sys, copy, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench import build, CTYPES, PPL_TEXT
torch.manual_seed(0); torch.set_num_threads(8)
BASE="HuggingFaceTB/SmolLM2-135M"; INST="HuggingFaceTB/SmolLM2-135M-Instruct"
OUT=os.path.dirname(os.path.abspath(__file__))
tok=AutoTokenizer.from_pretrained(INST); tok.padding_side="left"
def load(n): return AutoModelForCausalLM.from_pretrained(n, torch_dtype=torch.float32).eval()
SYS_STRICT="You follow the user's instructions EXACTLY and literally. Obey every format, length, language and word constraint."
def chat(p, sys_=None):
    m=([{"role":"system","content":sys_}] if sys_ else [])+[{"role":"user","content":p}]
    return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def generate(model, prompts, bs=18, mnt=100):
    outs=[]
    for i in range(0,len(prompts),bs):
        enc=tok([chat(p) for p in prompts[i:i+bs]], return_tensors="pt", padding=True)
        g=model.generate(**enc, max_new_tokens=mnt, do_sample=False, pad_token_id=tok.pad_token_id, repetition_penalty=1.0)
        outs+= [tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True).strip() for x in g]
    return outs

@torch.no_grad()
def ppl(model):
    ids=tok(PPL_TEXT, return_tensors="pt").input_ids
    return math.exp(model(ids, labels=ids).loss.item())

def evaluate(model, items, name):
    t=time.time(); outs=generate(model,[x["prompt"] for x in items])
    per={c:[0,0] for c in CTYPES}
    for x,o in zip(items,outs): per[x["type"]][1]+=1; per[x["type"]][0]+=int(x["check"](o))
    acc=sum(v[0] for v in per.values())/len(items)
    r={"name":name,"strict_acc":round(acc,4),"per_type":{k:round(v[0]/v[1],3) for k,v in per.items()},
       "ppl":round(ppl(model),3),"eval_seconds":round(time.time()-t,1),
       "samples":[{"type":x["type"],"prompt":x["prompt"],"out":o} for x,o in zip(items,outs)]}
    print(f"{name:28s} acc={acc:.3f} ppl={r['ppl']} ({r['eval_seconds']}s)", flush=True); return r

# ---- metodo 1: aritmetica de tareas ----
def task_arith(base_sd, inst, alpha):
    m=copy.deepcopy(inst); sd=m.state_dict()
    for k in sd: sd[k].copy_(base_sd[k]+alpha*(sd[k]-base_sd[k]))
    return m

# ---- metodo 2: direcciones de activacion ----
@torch.no_grad()
def last_hidden(model, texts):
    hs=[]
    for i in range(0,len(texts),16):
        enc=tok(texts[i:i+16], return_tensors="pt", padding=True)
        out=model(**enc, output_hidden_states=True)
        hs.append(torch.stack(out.hidden_states)[:, :, -1, :])  # [L+1,B,d]
    return torch.cat(hs,1)

def add_bias(model, layer, vec):
    """Edicion rank-1 en forma cerrada: agrega un sesgo a down_proj de la capa `layer`
    => residual += vec en todas las posiciones a partir de esa capa."""
    m=copy.deepcopy(model); lin=m.model.layers[layer].mlp.down_proj
    lin.bias=torch.nn.Parameter(vec.clone()); return m

def orthogonalize(model, r, layers=None):
    """Elimina la direccion r (normalizada) de todo lo que escribe al residual: W <- (I - r r^T) W."""
    m=copy.deepcopy(model); r=r/r.norm()
    P=torch.eye(r.numel())-torch.outer(r,r)
    with torch.no_grad():
        for i,L in enumerate(m.model.layers):
            if layers is not None and i not in layers: continue
            L.self_attn.o_proj.weight.copy_(P@L.self_attn.o_proj.weight)
            L.mlp.down_proj.weight.copy_(P@L.mlp.down_proj.weight)
    return m

if __name__=="__main__":
    ev=build("eval"); dv=build("dir"); print(len(ev),"eval /",len(dv),"dir", flush=True)
    T0=time.time(); res=[]; times={}
    inst=load(INST); base=load(BASE); base_sd={k:v.clone() for k,v in base.state_dict().items()}; del base
    res.append(evaluate(inst,ev,"baseline_instruct"))
    for a in [0.5,1.25,1.5,2.0]:
        t=time.time(); m=task_arith(base_sd,inst,a); times[f"task_arith_a{a}"]=round(time.time()-t,2)
        res.append(evaluate(m,ev,f"task_arith_a{a}")); del m
    # direccion A: con vs sin system prompt estricto (solo set 'dir')
    t=time.time()
    h_s=last_hidden(inst,[chat(x["prompt"],SYS_STRICT) for x in dv]); h_n=last_hidden(inst,[chat(x["prompt"]) for x in dv])
    dirA=(h_s-h_n).mean(1)  # [L+1,d]
    # direccion B: cumple vs viola (salidas del baseline en el set 'dir', verificadas)
    outs=generate(inst,[x["prompt"] for x in dv]); ok=torch.tensor([x["check"](o) for x,o in zip(dv,outs)])
    h=last_hidden(inst,[chat(x["prompt"]) for x in dv]); dirB=h[:,ok].mean(1)-h[:,~ok].mean(1)
    times["direcciones"]=round(time.time()-t,1); print("dir set: cumple",int(ok.sum()),"/",len(dv), flush=True)
    nL=len(inst.model.layers)
    for L in [nL//3, nL//2]:
        for c in [1.0, 2.0]:
            # hidden_states[L+1] = salida de la capa L
            res.append(evaluate(add_bias(inst,L,c*dirA[L+1]),ev,f"actadd_strict_L{L}_c{c}"))
    for L in [nL//3, nL//2]:
        res.append(evaluate(add_bias(inst,L,1.0*dirB[L+1]),ev,f"actadd_comply_L{L}_c1.0"))
    res.append(evaluate(orthogonalize(inst,-dirB[nL//2+1]),ev,f"orth_noncomply_L{nL//2}"))
    best=max(res,key=lambda r:r["strict_acc"])
    json.dump({"model":INST,"n_eval":len(ev),"n_dir":len(dv),"dir_set_baseline_comply":int(ok.sum()),
               "edit_times_s":times,"total_s":round(time.time()-T0,1),"best":best["name"],"results":res},
              open(f"{OUT}/metrics.json","w"),ensure_ascii=False,indent=1)
    torch.save({"dirA":dirA,"dirB":dirB},f"{OUT}/direcciones.pt")
    print("BEST",best["name"],"total",round(time.time()-T0,1))
