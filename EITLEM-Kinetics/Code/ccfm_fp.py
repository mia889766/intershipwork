import torch
import torch.nn as nn
import torch.nn.functional as F

class CCFMFromFPPatch(nn.Module):
    """
    把指纹分块成 N 个“伪 token”，与蛋白序列一起被全局 query 聚合。
    参数:
      F: 指纹维度，如 167/1024
      n_patch: 切成多少块（167 可设 8；1024 可设 16/32）
      d_model: 模型维度
    """
    def __init__(self, fp_dim: int, n_patch: int, d_model: int):
        super().__init__()
        self.fp_dim = fp_dim
        self.np = n_patch
        self.d = d_model
        # 映射到伪 token：每块 -> d
        self.proj = nn.Linear(fp_dim, n_patch * d_model)    # 简单起见做一次性映射
        self.ln_d = nn.LayerNorm(d_model)
        self.k_d  = nn.Linear(d_model, d_model)

        # 全局 token + 查询
        self.gd = nn.Sequential(nn.LayerNorm(fp_dim), nn.Linear(fp_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.gp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.q  = nn.Linear(2*d_model, d_model)

        # 蛋白侧
        self.ln_p = nn.LayerNorm(d_model)
        self.k_p  = nn.Linear(d_model, d_model)

    def attention_pool(self, q, K, V, mask=None):
        q = q.unsqueeze(1)
        att = torch.matmul(q, K.transpose(1,2)) / (K.size(-1)**0.5)
        if mask is not None:
            att = att.masked_fill(~mask.unsqueeze(1), float("-inf"))
        w = torch.softmax(att, dim=-1)
        out = torch.matmul(w, V)
        return out.squeeze(1)

    def forward(self, fp, pro_tok, pro_mask=None):
        B = fp.size(0)
        # 指纹 → 伪序列
        d_tok = self.proj(fp).view(B, self.np, self.d)         # (B, Np, d)
        d_tok = self.ln_d(d_tok)
        Kd = self.k_d(d_tok)

        # 全局查询
        g_d = self.gd(fp)                                      # (B,d)
        g_p = self.gp(pro_tok.mean(dim=1))                     # (B,d)
        q_all = self.q(torch.cat([g_d, g_p], -1))              # (B,d)

        # 蛋白侧
        P = self.ln_p(pro_tok); Kp = self.k_p(P)

        vp = self.attention_pool(q_all, Kp, P, pro_mask)       # (B,d)
        vd = self.attention_pool(q_all, Kd, d_tok)             # (B,d)
        return vp + vd                                         # (B,d)
