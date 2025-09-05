# save as esmfold_generate_pdb_with_smiles_eitlem.py
import os
import csv
import json
import torch
import gc
from esm.pretrained import esmfold_v1
from Bio import SeqIO
from tqdm import tqdm

# ==== 路径 ====
feature_dir = "EITLEM-Kinetics/Data/Feature"
input_fasta = os.path.join(feature_dir, "seq_str.fasta")
index_seq_path = os.path.join(feature_dir, "index_seq")
index_smiles_path = os.path.join(feature_dir, "index_smiles")
output_dir = os.path.join(feature_dir, "pdb_structures")
easifa_csv = os.path.join(feature_dir, "easifa_input.csv")
easifa_json = os.path.join(feature_dir, "easifa_input.json")

# ==== 加载索引 ====
print("Loading index files...")
index_seqs = torch.load(index_seq_path)      # 蛋白质序列索引
index_smiles = torch.load(index_smiles_path) # 底物SMILES字符串列表

# ==== 读取FASTA，建立索引到序列ID映射 ====
seq_records = list(SeqIO.parse(input_fasta, "fasta"))
seq_idx_to_id = {i: rec.id for i, rec in enumerate(seq_records)}
seq_idx_to_seq = {i: str(rec.seq) for i, rec in enumerate(seq_records)}

# ==== 初始化ESMFold ====
print("Loading ESMFold model...")
model = esmfold_v1()
model = model.eval().cuda()
print(type(index_seqs), len(index_seqs))
print(type(index_seqs[0]), index_seqs[0][:50])  # 预计是 str，且是氨基酸序列
print(type(index_smiles), len(index_smiles), type(index_smiles[0]))

# ==== 创建输出文件夹 ====
os.makedirs(output_dir, exist_ok=True)

# ==== 已完成的样本（断点续传） ====
done_ids = set()
if os.path.exists(easifa_csv):
    with open(easifa_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # 跳过表头
        for row in reader:
            pdb_file = os.path.basename(row[0])
            done_ids.add(os.path.splitext(pdb_file)[0])

# ==== 打开输出文件 ====
csv_file = open(easifa_csv, "a", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
if not os.path.exists(easifa_csv) or os.path.getsize(easifa_csv) == 0:
    csv_writer.writerow(["pdb_path", "substrate_smiles"])

json_list = []
if os.path.exists(easifa_json):
    with open(easifa_json, "r", encoding="utf-8") as f:
        try:
            json_list = json.load(f)
        except:
            json_list = []

# ==== 遍历所有样本 ====
with torch.no_grad():
    for i in tqdm(range(len(index_seqs)), desc="Generating PDB"):
        # 1) 用 i 取 ID/序列（你的两个映射的键都是整数 i）
        seq_id = seq_idx_to_id[i]             # ✔ 正确：键=整数
        seq    = index_seqs[i]                # ✔ 正确：index_seqs 是 dict[int->str]

        # 2) 去重（断点续传）
        if seq_id in done_ids:
            continue

        # 3) 取 SMILES（index_smiles 是 dict，但键未知；做三种情形的兜底）
        smiles = None
        if isinstance(index_smiles, dict):
            if i in index_smiles:
                smiles = index_smiles[i]            # 情形 A：键是整数 i
            elif seq_id in index_smiles:
                smiles = index_smiles[seq_id]       # 情形 B：键是 fasta 的 id
        elif isinstance(index_smiles, (list, tuple)) and i < len(index_smiles):
            smiles = index_smiles[i]                # 情形 C：顺序列表
        if smiles is None:
            # 若缺失，给个空串或直接 continue，看你的业务要求
            smiles = ""

        # 4) 文件名安全
        safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in seq_id)
        pdb_path = os.path.join(output_dir, f"{safe_id}.pdb")

        # 5) 生成并落盘
        if not os.path.exists(pdb_path):
            try:
                output_pdb = model.infer_pdb(seq)
                with open(pdb_path, "w") as f:
                    f.write(output_pdb)
            except Exception as e:
                print(f"[Error] {seq_id} failed: {e}")
                torch.cuda.empty_cache()
                continue

        # 6) 记录 CSV/JSON
        csv_writer.writerow([pdb_path, smiles])
        csv_file.flush()
        json_list.append({"pdb_path": pdb_path, "substrate_smiles": smiles})
        with open(easifa_json, "w", encoding="utf-8") as jf:
            json.dump(json_list, jf, indent=2)
        
        torch.cuda.empty_cache()
        gc.collect()


csv_file.close()
print("Done! EasIFA input CSV/JSON saved.")
