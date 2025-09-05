# /workspace/EITLEM-Kinetics/Code/omniesi/extractor.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 只保留 BCFM/CCFM（不依赖 DGL）
from omniesi.CN import BCFM, CCFM

# 使用 PyG 版分子编码器（需要 omniesi/pyg_encoder.py 存在）
from omniesi.pyg_encoder import EncoderDrugPyG

def _pad_by_batch(x: torch.Tensor, batch: torch.Tensor, pad_value: float = 0.0):
    """
    x: [N_total, D], batch: [N_total]  ->  out: [B, Lmax, D], mask: [B, Lmax]  (True=pad)
    """
    B = int(batch.max().item()) + 1
    D = x.size(-1)
    lengths = torch.bincount(batch, minlength=B)
    Lmax = int(lengths.max().item())
    device = x.device

    out = x.new_full((B, Lmax, D), pad_value)
    mask = torch.ones((B, Lmax), dtype=torch.bool, device=device)  # True 表示 padding

    idx = 0
    for b in range(B):
        l = int(lengths[b].item())
        if l == 0:
            continue
        out[b, :l] = x[idx: idx + l]
        mask[b, :l] = False
        idx += l
    return out, mask  # mask: True=pad


class Encoder_protein(nn.Module):
    """
    轻量蛋白编码器：逐残基线性投影到 dim_model；不依赖 DGL/dgllife。
    输入:  per-residue embedding [N_res, protein_dim]
    输出:  [N_res, dim_model]
    """
    def __init__(self, embedding_dim: int, target_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, target_dim),
            nn.LayerNorm(target_dim),
        )

    def forward(self, per_residue: torch.Tensor) -> torch.Tensor:
        return self.proj(per_residue)


class OmniESIExtractor(nn.Module):
    """
    适配器：将 EITLEM 的 batch 输入转为 BCFM+CCFM 所需的 [B,L,D]+mask，
    输出固定维度的 catalysis-aware 表征，供原 decoder 使用。
    - 期望 data 字段：
        data.x              : [N_atom, in_feats]
        data.edge_index     : [2, E]
        data.batch          : [N_atom]
        data.pro_emb        : [N_res, protein_dim]
        data.pro_emb_batch  : [N_res]
    - 输出：
        [B, d_out]
    """
    def __init__(
        self,
        drug_in_feats: int,
        protein_dim: int = 1280,
        dim_model: int = 128,
        d_out: int = 512,
        gnn_hidden=None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if gnn_hidden is None:
            gnn_hidden = [dim_model, dim_model, dim_model]

        # 分子：PyG GCNConv 堆叠 -> 原子 token 序列 [B, Ld, D]
        self.enc_drug = EncoderDrugPyG(
            in_feats=drug_in_feats,
            dim_embedding=dim_model,
            num_layers=len(gnn_hidden),
            dropout=dropout,
        )

        # 蛋白：逐残基线性投影 -> [N_res, D]，后续 pad
        self.enc_prot = Encoder_protein(
            embedding_dim=protein_dim,
            target_dim=dim_model,
        )

        # 红框：BCFM + CCFM
        self.bcfm = BCFM(dim_model=dim_model)
        self.ccfm = CCFM(dim_model=dim_model, num_head=4)

        # 融合到固定维度，接回原 decoder
        self.fuse = nn.Sequential(
            nn.LayerNorm(dim_model),
            nn.Linear(dim_model, d_out), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, data):
        # --- 分子端（PyG） ---
        # 需要 data.x / data.edge_index / data.batch
        drug_tokens, d_mask = self.enc_drug(data)  # [B, Ld, D], [B, Ld]

        # --- 蛋白端：逐残基 -> pad ---
        prot_seq = self.enc_prot(data.pro_emb)     # [N_res, D]
        prot_tokens, p_mask = _pad_by_batch(prot_seq, data.pro_emb_batch)  # [B, Lp, D], [B, Lp]

        # --- BCFM + CCFM ---
        y_p, y_d = self.bcfm(prot_tokens, drug_tokens, p_mask, d_mask)
        rep = self.ccfm(y_p, y_d, p_mask, d_mask)  # [B, D]

        # --- 融合到固定维度 ---
        return self.fuse(rep)                      # [B, d_out]
