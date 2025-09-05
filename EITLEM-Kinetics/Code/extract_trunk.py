import argparse, os
from pathlib import Path
import torch
from tqdm import tqdm
from esm import pretrained

# ---------- 原子保存，支持断点续跑 ----------
def safe_torch_save(tensor: torch.Tensor, out_path: Path, legacy: bool = False):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if out_path.exists() and out_path.stat().st_size > 0:
        return "skipped"
    if legacy:
        torch.save(tensor, tmp, _use_new_zipfile_serialization=False)
    else:
        torch.save(tensor, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, out_path)
    return "saved"

def read_fasta(path: Path):
    with open(path, "r") as f:
        label, seq = None, []
        for line in f:
            line = line.strip()
            if not line: 
                continue
            if line.startswith(">"):
                if label is not None:
                    yield label, "".join(seq)
                label = line[1:].strip()
                seq = []
            else:
                seq.append(line)
        if label is not None:
            yield label, "".join(seq)

def parse_args():
    p = argparse.ArgumentParser("ESMFold trunk extractor: GPU-first with per-seq CPU fallback (auto return to GPU)")
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument("--out",   type=Path, required=True)
    p.add_argument("--chunk_size", type=int, default=128)
    p.add_argument("--recycles",   type=int, default=4)
    p.add_argument("--concat_pair", action="store_true")
    p.add_argument("--compat_1280", action="store_true")
    p.add_argument("--legacy_save", action="store_true")
    p.add_argument("--cpu_only",    action="store_true")
    p.add_argument("--min_chunk_size", type=int, default=32)
    p.add_argument("--min_recycles",   type=int, default=2)
    p.add_argument("--gpu_fp16", type=lambda s: s.lower()!="false", default=True,
                   help="on GPU, use fp16 for ESM sub-net to save memory (default: True; pass --gpu_fp16 false to disable)")
    return p.parse_args()

@torch.no_grad()
def run_once(model, seq, recycles, chunk_size, concat_pair, compat_1280):
    if chunk_size is not None:
        model.set_chunk_size(int(chunk_size))
    out = model.infer([seq], num_recycles=int(recycles))  # dict
    s_s = out["s_s"][0].to("cpu").float()                 # [L,c_s]
    feat = s_s
    if concat_pair:
        s_z = out["s_z"][0]                               # [L,L,c_z] (device of model)
        s_z = s_z.mean(dim=1).to("cpu").float()           # [L,c_z]
        feat = torch.cat([s_s, s_z], dim=-1)              # [L,c_s+c_z]
    if compat_1280:
        target = 1280
        C = feat.shape[-1]
        if C < target:
            pad = torch.zeros(feat.shape[0], target - C, dtype=feat.dtype)
            feat = torch.cat([feat, pad], dim=-1)
        elif C > target:
            feat = feat[:, :target]
    return feat.contiguous()

def to_gpu(model, gpu_fp16: bool):
    model = model.to("cuda")
    # ESMFold 官方实现里在 GPU 路径默认把 ESM 子网 half；我们按开关处理
    if hasattr(model, "esm"):
        model.esm = model.esm.half() if gpu_fp16 else model.esm.float()
    return model

def to_cpu_float(model):
    model = model.to("cpu").float()
    if hasattr(model, "esm"):
        model.esm = model.esm.float()
    return model

@torch.no_grad()
def main():
    # 降低碎片化 OOM 概率
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    args = parse_args()

    use_gpu = (torch.cuda.is_available() and not args.cpu_only)
    device  = torch.device("cuda:0" if use_gpu else "cpu")

    model = pretrained.esmfold_v1()  # 包含 ESM2 + Trunk
    model.eval()
    if device.type == "cuda":
        model = to_gpu(model, args.gpu_fp16)
        print("[info] model on GPU (esm fp16: %s)" % str(args.gpu_fp16))
    else:
        model = to_cpu_float(model)
        print("[info] model on CPU (float32)")

    outdir = args.out
    outdir.mkdir(parents=True, exist_ok=True)

    entries = list(read_fasta(args.fasta))

    for label, seq in tqdm(entries, desc="[extract trunk]"):
        out_path = outdir / f"{label}.pt"
        if out_path.exists() and out_path.stat().st_size > 0:
            continue

        # 每条序列的尝试表
        chunk_try = [args.chunk_size, 64, 32, None]
        recy_try  = [args.recycles, 3, 2]
        chunk_try = [c for c in chunk_try if (c is None) or (c >= args.min_chunk_size)]
        recy_try  = [r for r in recy_try  if r >= args.min_recycles]
        tried_disable_pair = False

        # 首先尝试在 GPU（若可用）
        while True:
            try:
                feat = run_once(
                    model=model,
                    seq=seq,
                    recycles=recy_try[0],
                    chunk_size=chunk_try[0],
                    concat_pair=(args.concat_pair and (not tried_disable_pair)),
                    compat_1280=args.compat_1280,
                )
                safe_torch_save(feat, out_path, legacy=args.legacy_save)
                break  # 该条完成
            except RuntimeError as e:
                msg = str(e).lower()
                oom_like = ("out of memory" in msg) or ("cuda oom" in msg)
                if oom_like and device.type == "cuda":
                    # 1) 降 chunk
                    if len(chunk_try) > 1:
                        chunk_try.pop(0)
                        torch.cuda.empty_cache()
                        continue
                    # 2) 降 recycles
                    if len(recy_try) > 1:
                        recy_try.pop(0)
                        torch.cuda.empty_cache()
                        continue
                    # 3) 关 concat_pair
                    if args.concat_pair and not tried_disable_pair:
                        tried_disable_pair = True
                        torch.cuda.empty_cache()
                        continue
                    # 4) 本条切 CPU 完成
                    print(f"[warn] OOM on GPU for {label}, falling back to CPU for this sequence ...")
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    model = to_cpu_float(model)
                    device = torch.device("cpu")
                    # 在 CPU 上跑（用当前的最小化设定；可选也能直接用 recycles=2/chunk=None）
                    try:
                        feat = run_once(
                            model=model,
                            seq=seq,
                            recycles=recy_try[-1],
                            chunk_size=chunk_try[-1],
                            concat_pair=(args.concat_pair and tried_disable_pair is False),  # 可保守也关
                            compat_1280=args.compat_1280,
                        )
                        safe_torch_save(feat, out_path, legacy=args.legacy_save)
                    finally:
                        # ★ 完成该条后，若 GPU 可用，切回 GPU 继续下一条
                        if torch.cuda.is_available() and not args.cpu_only:
                            model = to_gpu(model, args.gpu_fp16)
                            device = torch.device("cuda")
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                    break  # 该条完成（CPU 路径）
                else:
                    # 非 OOM 或已在 CPU 的其他错误：带上 label 抛出
                    raise RuntimeError(f"[{label}] failed: {e}")

if __name__ == "__main__":
    main()
