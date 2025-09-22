import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax
from torch_geometric.utils import degree
import numpy as np
import math

class Resnet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.lin3 = nn.Linear(out_dim, out_dim)
        self.lin4 = nn.Linear(out_dim, out_dim)
    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = x + F.relu(self.lin2(x))
        x = x + F.relu(self.lin3(x))
        x = x + F.relu(self.lin4(x))
        return x
    
class ProMolAtt(nn.Module):
    def __init__(self, hidden_dim):
        super(ProMolAtt, self).__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.merge = nn.Linear(2*hidden_dim, 1, bias=False) # 计算相似性函数
        self.k = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, mol, prot, batch):
        q = F.relu(self.q(mol)) # 分子映射
        r = q.repeat_interleave(degree(batch,  dtype=batch.dtype), dim=0) # 分子扩增
        k = F.relu(self.k(prot))
        score = self.merge(torch.cat([k, r], dim=-1)) # 计算相似性分数
        score = softmax(score, batch, dim=0) # 权重加权
        o = global_add_pool(k * score, batch) # 聚合全局向量
        return o, q
    
class AttentionAgg(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionAgg, self).__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
    def forward(self, x, y):
        """
        x -> y ==> y^
        """
        q = F.relu(self.q(x.mean(dim=1)))
        k = F.relu(self.k(y))
        score = F.softmax(torch.matmul(q.unsqueeze(1), k.transpose(-1, -2)), dim=-1)
        out = torch.matmul(score, y).squeeze(1)
        return out
    
class MultiHeadAttenAgg(nn.Module):
    def __init__(self, hidden_dim, att_layer, dropout):
        super().__init__()
        self.seq_m = nn.ModuleList(AttentionAgg(hidden_dim) for _ in range(att_layer))
        self.seq_o = nn.Sequential(
        nn.Linear(hidden_dim*att_layer, 4*hidden_dim*att_layer),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(4*hidden_dim*att_layer, hidden_dim)
        )
    def forward(self, x, y):
        return self.seq_o(torch.cat([m(x, y) for m in self.seq_m], dim=-1))

class CCFMFromFPPatch(nn.Module):
    """
    指纹 F 维 -> N 个伪 token; 用全局 query 统一聚合 蛋白序列 & 指纹伪序列，返回 (vd, vp)
    """
    def __init__(self, fp_dim: int, n_patch: int, d_model: int):
        super().__init__()
        self.fp_dim = fp_dim; self.np = n_patch; self.d = d_model
        self.fp_proj = nn.Sequential(
            nn.LayerNorm(fp_dim),
            nn.Linear(fp_dim, n_patch * d_model)  # -> (B, Np*d)
        )
        # 生成全局 query 所需的全局 token
        self.gd = nn.Sequential(nn.LayerNorm(fp_dim), nn.Linear(fp_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.gp = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.q  = nn.Linear(2*d_model, d_model)

        # K 投影
        self.ln_d = nn.LayerNorm(d_model); self.k_d = nn.Linear(d_model, d_model)
        self.ln_p = nn.LayerNorm(d_model); self.k_p = nn.Linear(d_model, d_model)

    def _attn_pool(self, q, K, V, mask=None):
        # q: (B,d)->(B,1,d); K,V: (B,L,d)
        q = q.unsqueeze(1)
        att = torch.matmul(q, K.transpose(1,2)) / (K.size(-1) ** 0.5)  # (B,1,L)
        if mask is not None: att = att.masked_fill(~mask.unsqueeze(1), float("-inf"))
        w = torch.softmax(att, dim=-1)
        out = torch.matmul(w, V).squeeze(1)                            # (B,d)
        return out

    def forward(self, fp, pro_tok, pro_mask=None):
        B = fp.size(0)
        # 指纹 -> 伪序列
        d_tok = self.fp_proj(fp).view(B, self.np, self.d)  # (B, Np, d)
        d_tok = self.ln_d(d_tok); Kd = self.k_d(d_tok)

        # 全局 query
        g_d = self.gd(fp)                                  # (B,d)
        g_p = self.gp(pro_tok.mean(dim=1))                 # (B,d)
        q_all = self.q(torch.cat([g_d, g_p], dim=-1))      # (B,d)

        # 蛋白侧
        P = self.ln_p(pro_tok); Kp = self.k_p(P)
        vp = self._attn_pool(q_all, Kp, P, pro_mask)       # (B,d)  ← 聚合蛋白
        vd = self._attn_pool(q_all, Kd, d_tok)             # (B,d)  ← 聚合指纹伪序列
        return vd, vp









class EitlemKKmPredictor(nn.Module):
    def __init__(self, 
                 mol_in_dim, 
                 hidden_dim=128, 
                 protein_dim=1280, 
                 layer=10, 
                 dropout=0.2, 
                 att_layer=10
                ):
        super(EitlemKKmPredictor, self).__init__()
        self.prej1 = Resnet(mol_in_dim, hidden_dim)
        self.prej2 = nn.Linear(protein_dim, hidden_dim, bias=False)
        self.pro_extrac = nn.ModuleList([ProMolAtt(hidden_dim) for _ in range(layer)])
        #self.att1 = MultiHeadAttenAgg(hidden_dim,  att_layer, dropout)
        self.ccfm = CCFMFromFPPatch(
            fp_dim=mol_in_dim,                # 指纹维度(MACCS=167 / ECFP/RDKIT=1024)
            n_patch=8 if mol_in_dim<=256 else 16,
            d_model=hidden_dim
        )
        #self.att2 = MultiHeadAttenAgg(hidden_dim,  att_layer, dropout)
        self.out = nn.Sequential(
            nn.Linear(2*hidden_dim, 4*hidden_dim),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4*hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def final_stage(self, mol, pro, raw_fp=None, pro_mask=None):
        #pro_out = self.att1(mol, pro)
        #mol_out = self.att2(pro, mol)
        vd, vp = self.ccfm(raw_fp, pro, pro_mask)
        return self.out(torch.cat([vd, vp], dim=-1)).squeeze(dim=-1)
    
    def forward(self, data):
        mol = F.relu(self.prej1(data.x))
        prot = F.relu(self.prej2(data.pro_emb))
        att_pro = []
        att_mol = []
        for m in self.pro_extrac:
            o, q = m(mol, prot, data.pro_emb_batch)
            att_pro.append(o)
            att_mol.append(q)
        att_mol = torch.stack(att_mol, dim=1)
        att_pro = torch.stack(att_pro, dim=1)
        raw_fp = data.x 
        return self.final_stage(att_mol, att_pro, raw_fp=raw_fp, pro_mask=None)