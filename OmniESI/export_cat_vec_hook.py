#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export OmniESI CCFM joint embedding (cat_vec) for EITLEM-style index files,
with safe resume (checkpointing) and atomic writes.

Inputs
------
--ckpt         : OmniESI checkpoint (.pth)
--seq_fasta    : FASTA of all protein sequences (e.g., seq_str.fasta)
--index_seq    : mapping file:  seq_key -> fasta_header
--index_smiles : mapping file:  smi_key -> SMILES
--pairs        : pairs file (one per line), formats supported:
                 1) "id seq_key smi_key"
                 2) "seq_key smi_key"        (id auto = "seq_key__smi_key")
                 3) "id,seq_key,smi_key"     (CSV)
--out          : output directory (saves <id>.npy)
--resume       : (default) skip existing outputs; track state in _export_state.json
--no-resume    : ignore state; still skips existing unless --overwrite
--overwrite    : re-export even if <id>.npy exists
--device       : cuda / cpu
--max_errors   : stop after this many failures (default: inf)

Usage examples
--------------
# pairing (ESP) ckpt
python export_cat_vec_hook.py \
  --ckpt  /workspace/OmniESI/datasets/OmniESI_additional_data/additional_data/results/esp/OmniESI/best_model_epoch.pth \
  --seq_fasta   /workspace/EITLEM-Kinetics/Data/Feature/seq_str.fasta \
  --index_seq   /workspace/EITLEM-Kinetics/Data/Feature/index_seq \
  --index_smiles /workspace/EITLEM-Kinetics/Data/Feature/index_smiles \
  --pairs       /workspace/EITLEM-Kinetics/Data/Feature/pairs.txt \
  --out         /workspace/OmniESI/exports/cat_vec_pair

# active-site (mut_classify) ckpt
python export_cat_vec_hook.py \
  --ckpt  /workspace/OmniESI/datasets/OmniESI_additional_data/additional_data/results/mut_classify/OmniESI/best_model_epoch.pth \
  --seq_fasta   /workspace/EITLEM-Kinetics/Data/Feature/seq_str.fasta \
  --index_seq   /workspace/EITLEM-Kinetics/Data/Feature/index_seq \
  --index_smiles /workspace/EITLEM-Kinetics/Data/Feature/index_smiles \
  --pairs       /workspace/EITLEM-Kinetics/Data/Feature/pairs.txt \
  --out         /workspace/OmniESI/exports/cat_vec_activesite
"""


import os, json, argparse, signal
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Tuple

# --- Use OmniESI official modules ---
import dataloader as dl          # build_protein_input_from_seq / build_mol_input_from_smiles
import models as M               # build_model_from_cfg
import configs as C              # load_yaml

# ------------------- IO helpers -------------------

def _device(pref="cuda"):
    return torch.device(pref if torch.cuda.is_available() else "cpu")

def _read_fasta(path: str) -> Dict[str, str]:
    seqs, hdr, buf = {}, None, []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: 
                continue
            if line.startswith(">"):
                if hdr is not None:
                    seqs[hdr]= "".join(buf).replace(" ", "").replace("\t","")
                hdr = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if hdr is not None:
            seqs[hdr]= "".join(buf).replace(" ", "").replace("\t","")
    if not seqs:
        raise RuntimeError(f"FASTA empty or unreadable: {path}")
    return seqs

def _read_kv(path: str):
    """
    Load key->value mapping.
    Supports:
      - Torch serialized dict (.pt/.pth)
      - 2-column text (tab/comma/space separated)
    """
    # 先尝试 torch.load
    try:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
    except Exception:
        pass

    # 否则按文本解析
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if "," in ln and "\t" not in ln:
                parts = [p.strip() for p in ln.split(",")]
            elif "\t" in ln:
                parts = [p.strip() for p in ln.split("\t")]
            else:
                parts = ln.split()
            if len(parts) < 2:
                continue
            k, v = parts[0], parts[1]
            out[k] = v
    if not out:
        raise RuntimeError(f"No mappings found in {path}")
    return out


def _read_pairs(path: str) -> List[Tuple[str, str, str]]:
    """
    Return list of (sample_id, seq_key, smi_key)
    Supports:
      - 'id,seq_key,smi_key'
      - 'id\tseq_key\tsmi_key'
      - 'id seq_key smi_key'
      - 'seq_key smi_key'   (auto id = 'seq_key__smi_key')
    """
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if "," in ln and "\t" not in ln:
                parts = [p.strip() for p in ln.split(",")]
            elif "\t" in ln:
                parts = [p.strip() for p in ln.split("\t")]
            else:
                parts = ln.split()
            if len(parts) == 2:
                seq_key, smi_key = parts
                sid = f"{seq_key}__{smi_key}"
            elif len(parts) >= 3:
                sid, seq_key, smi_key = parts[0], parts[1], parts[2]
            else:
                raise RuntimeError(f"Unrecognized pair line: {ln}")
            pairs.append((sid, seq_key, smi_key))
    if not pairs:
        raise RuntimeError(f"No pairs found in {path}")
    return pairs

# -------------- OmniESI wrappers ----------------

def _inputs_from_strings(seq_str: str, smiles: str, device):
    prot_in = dl.build_protein_input_from_seq(seq_str, device=device)
    mol_in  = dl.build_mol_input_from_smiles(smiles, device=device)
    return prot_in, mol_in

def _find_ccfm(model: torch.nn.Module):
    if hasattr(model, "ccfm") and isinstance(getattr(model, "ccfm"), torch.nn.Module):
        return getattr(model, "ccfm")
    for name, mod in model.named_modules():
        nm = name.lower()
        cls = mod.__class__.__name__.lower()
        if "ccfm" in nm or "ccfm" in cls or "catalysis" in cls:
            return mod
    raise RuntimeError("CCFM module not found in model; check models.py.")

# -------------- Resume state ----------------

def _load_state(state_path: str):
    if os.path.exists(state_path):
        try:
            return json.load(open(state_path, "r", encoding="utf-8"))
        except Exception:
            pass
    return {"done": [], "failed": []}

def _save_state(state_path: str, state: dict):
    tmp = state_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, state_path)

# -------------- Main export ----------------

@torch.no_grad()
def export_cat_vec(ckpt, seq_fasta, index_seq, index_smiles, pairs_file, out_dir,
                   device="cuda", resume=True, overwrite=False, max_errors=None):
    os.makedirs(out_dir, exist_ok=True)
    err_log_path = os.path.join(out_dir, "_errors.log")
    state_path   = os.path.join(out_dir, "_export_state.json")

    dev = _device(device)
    fasta = _read_fasta(seq_fasta)
    seq_map = _read_kv(index_seq)
    smi_map = _read_kv(index_smiles)
    pairs   = _read_pairs(pairs_file)

    # build model
    cfg = C.load_yaml("configs/model/OmniESI.yaml")
    model = M.build_model_from_cfg(cfg)
    state_dict = torch.load(ckpt, map_location=dev)
    model.load_state_dict(state_dict, strict=True)
    model.to(dev).eval()

    # hook
    holder = {"joint": None}
    def _hook_ccfm(_mod, _inp, out):
        try:
            _, _, joint = out
        except Exception:
            if isinstance(out, dict) and "joint" in out:
                joint = out["joint"]
            else:
                raise
        holder["joint"] = joint.detach()

    ccfm = _find_ccfm(model)
    h = ccfm.register_forward_hook(_hook_ccfm)

    # resume state
    state = _load_state(state_path) if resume else {"done": [], "failed": []}
    done_set  = set(state.get("done", []))
    fail_set  = set(state.get("failed", []))
    n_errors = 0

    # graceful shutdown: persist state on SIGINT/TERM
    def _flush_and_exit(signum, frame):
        _save_state(state_path, {"done": sorted(list(done_set)), "failed": sorted(list(fail_set))})
        print(f"\n[Signal {signum}] state saved to {state_path}.")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _flush_and_exit)
    signal.signal(signal.SIGTERM, _flush_and_exit)

    with open(err_log_path, "a", encoding="utf-8") as errlog:
        for sid, seq_key, smi_key in tqdm(pairs, desc=f"Exporting from {os.path.dirname(ckpt)}"):
            out_file = os.path.join(out_dir, f"{sid}.npy")
            if not overwrite:
                if resume and sid in done_set:
                    continue
                if os.path.exists(out_file):
                    done_set.add(sid)
                    continue

            try:
                if seq_key not in seq_map:
                    raise KeyError(f"seq_key '{seq_key}' not in index_seq.")
                fasta_hdr = seq_map[seq_key]
                if fasta_hdr not in fasta:
                    raise KeyError(f"fasta header '{fasta_hdr}' not in FASTA.")
                if smi_key not in smi_map:
                    raise KeyError(f"smi_key '{smi_key}' not in index_smiles.")

                seq_str = fasta[fasta_hdr]
                smiles  = smi_map[smi_key]

                prot_in, mol_in = _inputs_from_strings(seq_str, smiles, dev)
                _ = model.forward_inference(prot_in, mol_in)  # run forward; hook captures joint

                vec = holder["joint"]
                if vec is None:
                    raise RuntimeError("CCFM joint not captured.")
                vec = vec.squeeze(0).cpu().numpy().astype("float32")

                tmp_file = out_file + ".tmp"
                np.save(tmp_file, vec)
                os.replace(tmp_file, out_file)  # atomic
                done_set.add(sid)

            except Exception as e:
                n_errors += 1
                fail_set.add(sid)
                errlog.write(f"{sid}\t{seq_key}\t{smi_key}\t{repr(e)}\n")
                errlog.flush()
                if max_errors is not None and n_errors >= max_errors:
                    break

    h.remove()
    _save_state(state_path, {"done": sorted(list(done_set)), "failed": sorted(list(fail_set))})
    print(f"Finished. OK={len(done_set)}  FAIL={len(fail_set)}")
    print(f"State: {state_path}")
    print(f"Errors: {err_log_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq_fasta", required=True)
    ap.add_argument("--index_seq", required=True)
    ap.add_argument("--index_smiles", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", dest="resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--overwrite", action="store_true", default=False)
    ap.add_argument("--max_errors", type=int, default=None)
    args = ap.parse_args()

    export_cat_vec(
        ckpt=args.ckpt,
        seq_fasta=args.seq_fasta,
        index_seq=args.index_seq,
        index_smiles=args.index_smiles,
        pairs_file=args.pairs,
        out_dir=args.out,
        device=args.device,
        resume=args.resume,
        overwrite=args.overwrite,
        max_errors=args.max_errors
    )

if __name__ == "__main__":
    main()