# import os
# import torch
# from tqdm import tqdm
# cmd = 'python ./extract.py esm2_t33_650M_UR50D ../Data/Feature/seq_str.fasta ../Data/Feature/esm2_t33_650M_UR50D/ --repr_layers 33 --include per_tok'
# os.system(cmd)
# base = "../Data/Feature/esm2_t33_650M_UR50D/"
# def change(index, layer):
#     data = torch.load(base+f'{index}.pt')
#     data = data['representations'][layer]
#     torch.save(data, base+f'{index}.pt')
# file_list = os.listdir(base)
# length = len(file_list)
# for index in tqdm(range(length)):
#     change(index, 33)
import argparse
import pathlib
import torch
from tqdm import tqdm
from esm import pretrained, FastaBatchedDataset, MSATransformer

# ========== 新增：安全保存 ==========
import os
from pathlib import Path

def safe_torch_save(tensor, out_path: Path, use_legacy=False, max_retries=3):
    """
    安全保存：临时文件 + fsync + 原子重命名；存在即跳过；失败自动重试。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    # 已存在且非空 -> 跳过（支持断点续跑）
    if out_path.exists() and out_path.stat().st_size > 0:
        return "skipped"

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            if use_legacy:
                # 旧序列化容器，避免 zip 容器在网络盘上的偶发写失败
                torch.save(tensor, tmp_path, _use_new_zipfile_serialization=False)
            else:
                torch.save(tensor, tmp_path)

            # 刷盘确保数据真正落盘
            with open(tmp_path, "rb") as f:
                os.fsync(f.fileno())

            # 原子替换，避免半截文件
            os.replace(tmp_path, out_path)
            return "saved"
        except Exception as e:
            last_err = e
            # 清理临时文件，准备重试
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            if attempt == max_retries:
                raise last_err
    # 理论到不了这
    return "error"


def parse_args():
    p = argparse.ArgumentParser("EITLEM-friendly ESM2 extractor (layer fusion + optional pair prior)")
    p.add_argument("--model", type=str, default="esm2_t33_650M_UR50D")
    p.add_argument("--fasta", type=pathlib.Path, default="../Data/Feature/seq_str.fasta")
    p.add_argument("--out",   type=pathlib.Path, default="../Data/Feature/esm1v_t33_650M_UR90S_1_embeding_1280/")
    p.add_argument("--toks_per_batch", type=int, default=4096)

    p.add_argument("--fuse", type=str, choices=["mean", "last"], default="mean",
                   help="layer fusion strategy: mean over all layers, or last layer only")
    p.add_argument("--use_pair", action="store_true", help="inject pair prior from attentions")
    p.add_argument("--no_pair",  action="store_true", help="disable pair prior even if --use_pair given")
    p.add_argument("--alpha", type=float, default=0.7, help="residual factor for pair injection")
    p.add_argument("--nogpu", action="store_true", help="force CPU")

    # ===== 新增：保存相关参数 =====
    p.add_argument("--legacy_save", action="store_true",
                   help="use legacy serialization (more robust on network/mounted disks)")
    p.add_argument("--max_retries", type=int, default=3,
                   help="max retries for safe save")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    use_pair = (args.use_pair and not args.no_pair)

    model, alphabet = pretrained.load_model_and_alphabet(args.model)
    if isinstance(model, MSATransformer):
        raise ValueError("This script does not support MSA Transformer.")

    device = "cuda" if (torch.cuda.is_available() and not args.nogpu) else "cpu"
    if device == "cuda":
        model = model.cuda()
        print("[info] model on GPU")

    model.eval()

    dataset = FastaBatchedDataset.from_file(args.fasta)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
    )
    print(f"[info] read {args.fasta} with {len(dataset)} sequences")

    args.out.mkdir(parents=True, exist_ok=True)

    repr_layers = list(range(model.num_layers + 1))
    need_heads = use_pair

    for bidx, (labels, strs, toks) in enumerate(loader):
        print(f"[info] batch {bidx+1}/{len(batches)} ({toks.size(0)} seqs)")
        toks = toks.to(device=device, non_blocking=True)

        out = model(
            toks,
            repr_layers=repr_layers if args.fuse == "mean" else [max(repr_layers)],
            need_head_weights=need_heads,
            return_contacts=False,
        )

        reps = {l: t.to("cpu") for l, t in out["representations"].items()}  # [B,T,C]
        attn = out.get("attentions", None)
        if attn is not None:
            attn = attn.to("cpu")  # memory friendly

        B = toks.size(0)

        for i, label in enumerate(labels):
            L = len(strs[i])

            # -------- layer fusion --------
            if args.fuse == "mean":
                xs = [reps[l][i, 1:L+1] for l in reps.keys()]  # strip BOS/EOS
                X = torch.stack(xs, dim=0).mean(dim=0)          # [L,1280]
            else:
                last = max(reps.keys())
                X = reps[last][i, 1:L+1]                        # [L,1280]

            # -------- optional pair prior: Y = X + alpha * (A @ X) --------
            if use_pair and attn is not None:
                dims = attn.shape
                if len(dims) == 5:
                    # 常见两种： [num_layers, B, num_heads, T, T] 或 [B, num_layers, num_heads, T, T]
                    if dims[1] == B:
                        A_raw = attn[:, i]        # [num_layers, H, T, T]
                        A = A_raw.mean(dim=(0,1)) # [T, T]
                    elif dims[0] == B:
                        A_raw = attn[i]           # [num_layers, H, T, T]
                        A = A_raw.mean(dim=(0,1))
                    else:
                        A = attn.mean(dim=(0,1,2))
                else:
                    # 兜底：平均除最后两个T,T之外的所有维
                    reduce_dims = tuple(range(attn.dim() - 2))
                    A = attn.mean(dim=reduce_dims)  # [T, T]

                # strip BOS/EOS
                A = A[1:L+1, 1:L+1]                 # [L,L]
                A = torch.nan_to_num(A, nan=0.0)
                row_sum = A.sum(dim=-1, keepdim=True) + 1e-8
                A = A / row_sum

                AX = A @ X
                Y = X + float(args.alpha) * AX
            else:
                Y = X

            # -------- 安全保存（断点续跑 + 原子写入） --------
            out_path = (args.out / f"{label}.pt")
            status = safe_torch_save(
                Y.clone(),
                out_path,
                use_legacy=args.legacy_save,
                max_retries=args.max_retries
            )
            # 可选打印每条状态（大量样本时可注释）
            print(f"[save] {label}.pt -> {status}")

if __name__ == "__main__":
    main()
