# extract_prosst_perres_1280.py
import os, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from Bio import SeqIO
from tqdm import tqdm

fasta_path = "/workspace/EITLEM-Kinetics/Data/Feature/seq_str.fasta"
out_dir    = "/workspace/EITLEM-Kinetics/Data/Feature/prosst_perres_1280"
os.makedirs(out_dir, exist_ok=True)

model_name = "AI4Protein/ProSST-2048"   # 该仓库 hidden_size=768（2048是结构codebook大小）
tokenizer  = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model      = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).eval()
device     = "cuda" if torch.cuda.is_available() else "cpu"
model      = model.to(device)

# 关结构分支（若实现有这些开关）
if hasattr(model, "config"):
    for k in ["use_ss", "enable_ss_stream", "structure_conditioning", "use_structure"]:
        if hasattr(model.config, k):
            setattr(model.config, k, False)

# 线性投影：768 -> 1280
proj = torch.nn.Linear(768, 1280, bias=False)
torch.nn.init.orthogonal_(proj.weight)
proj = proj.to(device).eval()

@torch.inference_mode()
def prosst_perres_1280(seq: str) -> torch.Tensor:
    toks = tokenizer(seq, return_tensors="pt",add_special_tokens=False)
    toks = {k: v.to(device) for k, v in toks.items()}
    ss_ids = torch.zeros_like(toks["input_ids"])       # 哑结构流，无需 PDB
    out = model(**toks, ss_input_ids=ss_ids, output_hidden_states=True)
    if not out.hidden_states: raise RuntimeError("hidden_states 为空")
    h = out.hidden_states[-1].squeeze(0)               # [L, 768]
    h1280 = proj(h)                                    # [L, 1280]
    return h1280.detach().cpu()

records = list(SeqIO.parse(fasta_path, "fasta"))
for rec in tqdm(records, desc="ProSST per-residue 1280D"):
    sid = rec.id  # 0,1,2,...
    emb = prosst_perres_1280(str(rec.seq))             # [L, 1280]
    torch.save(emb, os.path.join(out_dir, f"{sid}.pt"))

print(f"[DONE] saved per-residue embeddings to {out_dir}")
