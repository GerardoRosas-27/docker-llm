"""Guarda el mejor modelo editado (actadd_strict_L10_c2.0) como HF con mlp_bias=True (sesgos cero salvo capa 10) para convertir a GGUF."""
import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
INST="HuggingFaceTB/SmolLM2-135M-Instruct"; L=10; C=2.0; out=sys.argv[1]
d=torch.load("direcciones.pt")["dirA"]
m=AutoModelForCausalLM.from_pretrained(INST, dtype=torch.float32)
cfg=m.config; cfg.mlp_bias=True
m2=AutoModelForCausalLM.from_config(cfg); m2.load_state_dict(m.state_dict(), strict=False)
with torch.no_grad():
    for i,ly in enumerate(m2.model.layers):
        for p in (ly.mlp.gate_proj, ly.mlp.up_proj, ly.mlp.down_proj): p.bias.zero_()
    m2.model.layers[L].mlp.down_proj.bias.copy_(C*d[L+1])
m2.save_pretrained(out); AutoTokenizer.from_pretrained(INST).save_pretrained(out); print("ok", out)
