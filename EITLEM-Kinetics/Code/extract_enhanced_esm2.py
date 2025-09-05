import argparse
import pathlib
import torch
from tqdm import tqdm
from esm import pretrained, FastaBatchedDataset, MSATransformer

# ---------------------------
# 核心思路：
# 1) 层融合：对 0..num_layers 的 per-residue 表示做均值或自定义权重融合 => [B, L, 1280]
# 2) pair 先验：取多头注意力，平均 layer×head => A[B, T, T]，去掉BOS/EOS => A[B, L, L]
#    对每个序列做 Y = X + alpha * (A @ X)，保持 [B, L, 1280]
# ---------------------------

def create_parser():
    p = argparse.ArgumentParser("Enhanced ESM2 per-residue extractor (layer fusion + pair prior)")
    p.add_argument("model_location", type=str, help="e.g., esm2_t33_650M_UR50D")
    p.add_argument("fasta_file", type=pathlib.Path, help="input FASTA")
    p.add_argument("output_dir", type=pathlib.Path, help="dir for .pt outputs")
    p.add_argument("--toks_per_batch", type=int, default=4096)
    p.add_argument("--fuse", type=str, choices=["mean", "last"], default="mean",
                   help="layer fusion: mean over all layers, or take last only")
    p.add_argument("--use_pair", action="store_true",
                   help="inject pair prior from attentions: Y = X + alpha * (A @ X)")
    p.add_argument("--alpha", type=float, default=0.7, help="residual factor for pair injection")
    p.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return p

@torch.no_grad()
def main(args):
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    model.eval()
    if isinstance(model, MSATransformer):
        raise ValueError("This script does not support MSA Transformer models.")

    device = "cuda" if (torch.cuda.is_available() and not args.nogpu) else "cpu"
    if device == "cuda":
        model = model.cuda()
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(args.fasta_file)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
    )
    print(f"Read {args.fasta_file} with {len(dataset)} sequences")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # repr layers: 0..num_layers (inclusive). ESM-2 t33 has 34 layers with indices [0..33].
    repr_layers = list(range(model.num_layers + 1))
    need_heads = args.use_pair

    for batch_idx, (labels, strs, toks) in enumerate(data_loader):
        print(f"Processing {batch_idx + 1}/{len(batches)} batches ({toks.size(0)} sequences)")
        toks = toks.to(device=device, non_blocking=True)

        # forward: get per-layer reps; if use_pair, also get attentions
        out = model(
            toks,
            repr_layers=repr_layers,
            need_head_weights=need_heads,   # 必须为 True 才会返回注意力
            return_contacts=False,
        )

        # per-layer token reps: dict[layer] -> [B, T, C]
        reps = {l: t.to("cpu") for l, t in out["representations"].items()}
        # 注意：reps 的 T 含 BOS/EOS；我们需要切掉
        # attentions: [num_layers, num_heads, B, T, T]  (ESM实现里通常是这样)
        attn = out.get("attentions", None)
        if attn is not None:
            attn = attn.to("cpu")

        B = toks.size(0)

        for i, label in enumerate(labels):
            # 该序列实际长度（不含特殊符号）
            L = len(strs[i])

            # 1) 取每层 per-residue 表示，并裁掉 BOS/EOS => [L, C]
            #    层融合策略：
            if args.fuse == "mean":
                # 对所有层做简单均值
                xs = [reps[l][i, 1: L + 1] for l in repr_layers]  # 每个: [L, C]
                X = torch.stack(xs, dim=0).mean(dim=0)            # [L, C]
            else:  # "last"
                last = max(repr_layers)
                X = reps[last][i, 1: L + 1]                       # [L, C]

            # 2) 如果需要注入 pair 先验：A @ X，再做残差
            if args.use_pair and attn is not None:
                # 平均 layer×head，得到 A_raw: [T, T]
                # 这里的 T = L + 2（含 BOS/EOS），我们要取中间 L×L 区域
                A_raw = attn[:, :, i]            # [num_layers, num_heads, T, T]
                A = A_raw.mean(dim=(0, 1))       # [T, T]
                A = A[1: L + 1, 1: L + 1]        # 去掉BOS/EOS => [L, L]

                # 行归一化（保证每行和为1），避免数值偏置
                A = torch.nan_to_num(A, nan=0.0)
                row_sum = A.sum(dim=-1, keepdim=True) + 1e-8
                A = A / row_sum                   # [L, L]

                # A @ X  => [L, C]，并做残差融合
                AX = torch.matmul(A, X)           # [L, C]
                Y = X + args.alpha * AX           # [L, C]
            else:
                Y = X

            # 最终 per-residue 1280-d（与原 EITLEM 对接）
            out_path = (args.output_dir / f"{label}.pt")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "label": label,
                    "per_residue_1280": Y.clone(),  # [L, 1280]（对 t33_650M 来说 C=1280）
                    "meta": {
                        "fuse": args.fuse,
                        "use_pair": bool(args.use_pair),
                        "alpha": float(args.alpha),
                        "model": args.model_location,
                        "length": int(L),
                    },
                },
                out_path,
            )

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
