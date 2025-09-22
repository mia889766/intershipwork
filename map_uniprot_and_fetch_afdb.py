import os, re, time, csv, requests, torch
from datetime import datetime
from tqdm import tqdm

# ========= 路径（按你的工程改） =========
feature_dir = "EITLEM-Kineticsold/Data/Feature"      # 改为你的真实路径
index_seq_path = os.path.join(feature_dir, "index_seq")  # torch.save 的二进制
output_dir     = os.path.join(feature_dir, "pdb_structures")  # 输出目录（与 ESMFold 一样）
manifest_path  = os.path.join(output_dir, "_manifest.csv")    # 断点续传/状态记录
os.makedirs(output_dir, exist_ok=True)

# ========= 超参 =========
SLEEP       = 0.35       # API 限速
TIMEOUT     = 45
RETRY       = 3
MAX_TEST    = None       # 冒烟测试：设 200 只跑前 200 条；生产设 None
VERBOSE_N   = 50         # 每 N 条打印一次统计
DEBUG_FIRST = 5          # 前 N 条打印详细调试
FILE_FORMAT = "pdb"      # "pdb" 或 "cif"
UA_HEADERS  = {"User-Agent": "eitlem-afdb/1.0 (mailto:you@example.com)"}

# 只保留 20 个标准氨基酸做精确匹配；其余（B/Z/J/U/O/X/* 以及空白）会被清洗掉
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

sess = requests.Session()

def clean_seq(seq: str) -> str:
    s = "".join(ch for ch in str(seq).strip().upper() if ch in VALID_AA)
    return s

def uniprot_exact(seq: str):
    """
    UniProt 精确序列搜索（POST；避免 URL 过长导致 400）
    返回: (uid or None, status_code or str)
    """
    url = "https://rest.uniprot.org/uniprotkb/search"
    data = {"query": f'sequence:"{seq}"', "fields":"accession", "format":"json", "size":1}
    for attempt in range(3):
        try:
            r = sess.post(url, data=data, headers=UA_HEADERS, timeout=TIMEOUT)
            if r.status_code == 400:
                return None, 400
            if not r.ok:
                time.sleep(1.5*(attempt+1)); continue
            js = r.json().get("results", [])
            return (js[0]["primaryAccession"], 200) if js else (None, 200)
        except requests.exceptions.RequestException:
            time.sleep(1.5*(attempt+1))
    return None, "timeout"

def afdb_download(uid: str, out_path: str, fmt="pdb") -> bool:
    """
    调 AlphaFold DB `/api/prediction/{uid}` 下载 PDB/CIF。
    采用 .part 临时文件，完成后原子重命名，支持断点续传。
    """
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True

    tmp = out_path + ".part"
    for _ in range(RETRY):
        try:
            meta = sess.get(f"https://alphafold.ebi.ac.uk/api/prediction/{uid}",
                            headers=UA_HEADERS, timeout=TIMEOUT)
            if not meta.ok:
                time.sleep(SLEEP); continue
            info = meta.json()
            if not info:
                return False
            e0 = info[0]
            url = e0.get("pdbUrl") if fmt=="pdb" else (e0.get("cifUrl") or e0.get("mmcifUrl"))
            if not url:
                return False
            binr = sess.get(url, headers=UA_HEADERS, timeout=TIMEOUT)
            if not binr.ok:
                time.sleep(SLEEP); continue
            with open(tmp, "wb") as f:
                f.write(binr.content)
            os.replace(tmp, out_path)   # 原子移动
            time.sleep(SLEEP)
            return True
        except requests.exceptions.RequestException:
            time.sleep(SLEEP)
    # 清理临时文件
    if os.path.exists(tmp):
        try: os.remove(tmp)
        except: pass
    return False

def load_index_seq(path):
    x = torch.load(path)
    if isinstance(x, dict):
        items = list(x.items())      # [(key, seq), ...]
    elif isinstance(x, (list, tuple)):
        items = list(enumerate(x))   # 索引作 key
    else:
        raise TypeError(f"Unsupported index_seq type: {type(x)}")
    return items

def load_done_keys_from_manifest(manifest_csv):
    done = set()
    if os.path.exists(manifest_csv):
        with open(manifest_csv, "r", newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r, None)
            for row in r:
                # 行：key, pdb_path, source, uniprot, status, ts
                if len(row) >= 5 and row[4] == "done":
                    done.add(row[0])
    return done

def append_manifest(manifest_csv, key, pdb_path, source, uniprot, status):
    new_file = not os.path.exists(manifest_csv)
    with open(manifest_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["key","pdb_path","source","uniprot","status","timestamp"])
        w.writerow([str(key), pdb_path, source, (uniprot or ""), status,
                    datetime.now().isoformat(timespec="seconds")])

def main():
    items = load_index_seq(index_seq_path)
    if MAX_TEST: items = items[:MAX_TEST]

    # 断点续传：已完成（manifest 标记 done 或同名 PDB 存在）直接跳过
    done_keys = load_done_keys_from_manifest(manifest_path)
    seen=mapped=downloaded=skipped=no_uid=0

    for key, raw_seq in tqdm(items, desc="map(index_seq)+download"):
        seen += 1
        key_str = str(key)
        out_pdb = os.path.join(output_dir, f"{key_str}.pdb")

        # 若已有同名 PDB（包含你之前 ESMFold 生成的），直接跳过
        if key_str in done_keys or (os.path.exists(out_pdb) and os.path.getsize(out_pdb)>0):
            skipped += 1
            if seen <= DEBUG_FIRST:
                print(f"[DEBUG] key={key_str} -> skip(already exists)")
            continue

        seq = clean_seq(raw_seq)
        if len(seq) < 20:  # 清洗后过短，放弃映射，直接记 no_uid
            no_uid += 1
            append_manifest(manifest_path, key_str, out_pdb, "NA", None, "no_uid_short")
            continue

        uid, code = uniprot_exact(seq)
        if uid:
            mapped += 1
            ok = afdb_download(uid, out_pdb, FILE_FORMAT)
            if ok:
                downloaded += 1
                append_manifest(manifest_path, key_str, out_pdb, "AFDB", uid, "done")
                if seen <= DEBUG_FIRST:
                    print(f"[DEBUG] key={key_str} uid={uid} -> OK")
            else:
                append_manifest(manifest_path, key_str, out_pdb, "AFDB", uid, "download_fail")
                if seen <= DEBUG_FIRST:
                    print(f"[DEBUG] key={key_str} uid={uid} -> FAIL")
        else:
            no_uid += 1
            append_manifest(manifest_path, key_str, out_pdb, "NA", None, f"no_uid(search={code})")
            if seen <= DEBUG_FIRST:
                print(f"[DEBUG] key={key_str} no_uid (search={code})")

        if seen % VERBOSE_N == 0:
            print(f"[STAT] seen={seen} mapped={mapped} downloaded={downloaded} "
                  f"skipped={skipped} no_uid={no_uid}")

    print(f"[DONE] seen={seen} mapped={mapped} downloaded={downloaded} "
          f"skipped={skipped} no_uid={no_uid}")
    print(f"Manifest: {manifest_path}\nOutput dir: {output_dir}")

if __name__ == "__main__":
    main()
