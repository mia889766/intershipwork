# /workspace/EITLEM-Kinetics/Code/omniesi/pyg_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

def _is_valid_edge_index(edge_index: torch.Tensor) -> bool:
    return (
        isinstance(edge_index, torch.Tensor)
        and edge_index.dtype in (torch.long, torch.int64)
        and edge_index.dim() == 2
        and edge_index.size(0) == 2
        and edge_index.numel() > 0
    )

class EncoderDrugPyG(nn.Module):
    """
    PyG 分子编码器（自适应 图 / 向量）：
      - 图模式：data.x [N_atom, in_feats], data.edge_index [2, E], data.batch [N_atom]
                -> GCNConv 堆叠 -> 原子 token -> [B, Ld, D] + mask
      - 向量模式：data.x [B, in_feats]（无有效 edge_index / 无 batch）
                -> 线性投影 -> 每样本 1 个 token -> [B, 1, D] + 全 False mask
    """
    def __init__(self, in_feats: int, dim_embedding: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout
        self.in_feats = in_feats
        self.dim_embedding = dim_embedding
        self.num_layers = num_layers

        # 图模式用的 GCN 堆叠
        hs = [in_feats] + [dim_embedding] * num_layers
        self.convs = nn.ModuleList(GCNConv(hs[i], hs[i+1]) for i in range(num_layers))
        self.ln_graph = nn.LayerNorm(dim_embedding)

        # 向量模式用的线性投影
        self.proj_vec = nn.Sequential(
            nn.LayerNorm(in_feats),
            nn.Linear(in_feats, dim_embedding),
            nn.LayerNorm(dim_embedding),
        )

    @staticmethod
    def pad_by_batch(x: torch.Tensor, batch: torch.Tensor, pad_value: float = 0.0):
        B = int(batch.max().item()) + 1
        D = x.size(-1)
        lengths = torch.bincount(batch, minlength=B)
        Lmax = int(lengths.max().item())
        device = x.device
        out = x.new_full((B, Lmax, D), pad_value)
        mask = torch.ones((B, Lmax), dtype=torch.bool, device=device)  # True=pad
        idx = 0
        for b in range(B):
            l = int(lengths[b].item())
            if l == 0:
                continue
            out[b, :l] = x[idx: idx + l]
            mask[b, :l] = False
            idx += l
        return out, mask

    def forward(self, data):
        """
        兼容两种输入：
          图模式需要: data.x, data.edge_index, data.batch
          向量模式仅需: data.x（形状 [B, in_feats]）
        返回:
          tokens: [B, Ld, D], mask: [B, Ld] (True=pad)
        """
        # 判定是否为“图模式”
        edge_index = getattr(data, "edge_index", None)
        batch = getattr(data, "batch", None)
        graph_mode = edge_index is not None and _is_valid_edge_index(edge_index) and batch is not None

        if graph_mode:
            # --- 图模式：GCN over atoms ---
            h = data.x
            for conv in self.convs:
                h = conv(h, edge_index)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.ln_graph(h)  # [N_atom, D]
            tokens, mask = self.pad_by_batch(h, batch)  # [B, Ld, D], [B, Ld]
            return tokens, mask

        else:
            # --- 向量模式（MACCS/ECFP 等）：每样本一个 token ---
            # data.x: [B, in_feats]
            x = data.x
            if x.dim() != 2 or x.size(1) != self.in_feats:
                raise ValueError(f"[EncoderDrugPyG] Expect vector-mode data.x shape [B, {self.in_feats}], got {tuple(x.shape)}")
            h = self.proj_vec(x)                 # [B, D]
            tokens = h.unsqueeze(1)              # [B, 1, D]
            mask = torch.zeros((h.size(0), 1), dtype=torch.bool, device=h.device)  # 无 pad
            return tokens, mask
