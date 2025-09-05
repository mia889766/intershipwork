#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 EITLEM 的一一对应关系，查表生成 OmniESI 可用 CSV：id,seq,smiles
- 输入：
  seq_str.fasta              : 全部蛋白序列（当 index_seq 存 header 时才会用到）
  index_seq                  : torch.save 的索引；可能是:
                               1) dict[int|str -> AA 序列]（直接用，优先级最高）
                               2) dict[int|str -> fasta_header]（再去 FASTA 查）
                               3) list/tuple[AA 序列] 或 list/tuple[fasta_header]
  index_smiles               : torch.save 的索引；可能是:
                               1) dict[int|str -> SMILES]
                               2) list/tuple[SMILES]
- 配对策略：一一对应（按相同索引位次 i 对齐），长度取两者最小
- 断点续写：若 out_csv 已存在，跳过已存在的 id 行（按第一列匹配）
- id 规则：S000000, S000001, ...

用法示例：
python make_eitlem_omniesi_csv.py \
  --seq_fasta   /workspace/EITLEM-Kinetics/Data/Feature/seq_str.fasta \
  --index_seq   /workspace/EITLEM-Kinetics/Data/Feature/index_seq \
  --index_smiles /workspace/EITLEM-Kinetics/Data/Feature/index_smiles \
  --out_csv     /workspace/OmniESI/paths/eitlem_pairs.csv
"""

import os
import csv
import argparse
import torch
from Bio import SeqIO

AA_CHARS = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")  # 宽松集合，容错非常规字母

def load_fasta(path):
    """FASTA: header -> sequence"""
    hdr2seq = {}
    for rec in SeqIO.parse(path, "fasta"):
        hdr2seq[rec.id] = str(rec.seq)
    if not hdr2seq:
        raise SystemExit(f"[ERR] empty FASTA: {path}")
    return hdr2seq

def load_index_any(path):
    """优先 torch.load（EITLEM 索引通常是 .pt）；失败则报错"""
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception as e:
        raise SystemExit(f"[ERR] torch.load({path}) failed: {e}")
    return obj

def get_len_and_getter(obj, name):
    """
    统一提供长度 N 和 get(i)->value
    - list/tuple: 直接按 i 取
    - dict:
        * 若存在 int 键：N = max(int_keys)+1，按 i 取（缺则报错）
        * 若存在 "0","1"... 字符串键：同上逻辑
        * 否则：按排序后的键顺序取值（兜底，不保证原顺序）
    """
    if isinstance(obj, (list, tuple)):
        N = len(obj)
        def get_i(i): return obj[i]
        return N, get_i

    if isinstance(obj, dict):
        int_keys = [k for k in obj.keys() if isinstance(k, int)]
        if int_keys:
            N = max(int_keys) + 1
            def get_i(i):
                if i in obj: return obj[i]
                raise KeyError(f"{name}: missing int key {i}")
            return N, get_i

        strint_ok, strint_keys = True, []
        for k in obj.keys():
            if isinstance(k, str) and k.isdigit():
                strint_keys.append(int(k))
            else:
                strint_ok = False
                break
        if strint_ok and strint_keys:
            N = max(strint_keys) + 1
            def get_i(i):
                k = str(i)
                if k in obj: return obj[k]
                raise KeyError(f"{name}: missing str-int key '{i}'")
            return N, get_i

        keys = sorted(list(obj.keys()), key=lambda x: str(x))
        N = len(keys)
        def get_i(i):
            return obj[keys[i]]
        return N, get_i

    raise SystemExit(f"[ERR] unsupported index type for {name}: {type(obj)}")

def looks_like_aa_sequence(val: str) -> bool:
    """判断字符串是否像氨基酸序列（宽松规则）"""
    if not isinstance(val, str): return False
    s = val.strip().upper()
    if len(s) < 10: return False
    if any(ch.isspace() for ch in s): return False
    good = sum(1 for c in s if c in AA_CHARS)
    return good / max(1, len(s)) >= 0.85

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_fasta", required=True)
    ap.add_argument("--index_seq", required=True)
    ap.add_argument("--index_smiles", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    # 读取索引
    idx_seq = load_index_any(args.index_seq)
    idx_smi = load_index_any(args.index_smiles)

    # 为了兼容“index_seq 值是 header”的情形，读 FASTA 作为后备查表
    # 如果 index_seq 值本身是 AA 序列，则不会用到 FASTA
    hdr2seq = load_fasta(args.seq_fasta)

    N_seq, get_seq_val = get_len_and_getter(idx_seq, "index_seq")
    N_smi, get_smi_val = get_len_and_getter(idx_smi, "index_smiles")
    N = min(N_seq, N_smi)
    if N == 0:
        raise SystemExit("[ERR] zero length after alignment (index_seq / index_smiles)")

    # 断点续写：收集已有 id
    existing_ids = set()
    if os.path.exists(args.out_csv) and os.path.getsize(args.out_csv) > 0:
        with open(args.out_csv, "r", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r, None)  # 跳过表头（若无也不影响）
            for row in r:
                if not row: continue
                existing_ids.add(row[0])

    need_header = not os.path.exists(args.out_csv) or os.path.getsize(args.out_csv) == 0
    f = open(args.out_csv, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if need_header:
        w.writerow(["id", "seq", "smiles"])

    n_write, n_skip = 0, 0
    for i in range(N):
        sid = f"S{i:06d}"
        if sid in existing_ids:
            n_skip += 1
            continue

        seq_val = get_seq_val(i)
        smi_val = get_smi_val(i)

        # —— 关键修正：优先把 index_seq 的值当作 AA 序列；否则当作 header 查 FASTA ——
        if isinstance(seq_val, str) and looks_like_aa_sequence(seq_val):
            seq = seq_val.strip().upper()
        else:
            header = str(seq_val).split()[0]
            if header not in hdr2seq:
                raise KeyError(f"[index_seq] header '{header}' not in FASTA")
            seq = hdr2seq[header]

        smiles = str(smi_val).strip()
        if not smiles:
            n_skip += 1
            continue

        w.writerow([sid, seq, smiles])
        n_write += 1

    f.flush()
    f.close()
    print(f"[OK] wrote {n_write} rows to {args.out_csv}; skipped {n_skip} rows.")
    print(f"[INFO] aligned length: {N} (seq:{N_seq}, smiles:{N_smi})")

if __name__ == "__main__":
    main()
