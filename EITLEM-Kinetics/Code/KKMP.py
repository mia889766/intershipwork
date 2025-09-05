# /workspace/EITLEM-Kinetics/Code/KKMP.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from omniesi.extractor import OmniESIExtractor

class OmniBackbone(nn.Module):
    def __init__(self, mol_in_dim, protein_dim=1280, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.omni = OmniESIExtractor(
            drug_in_feats=mol_in_dim,
            protein_dim=protein_dim,
            dim_model=hidden_dim,
            d_out=512,
            gnn_hidden=[hidden_dim, hidden_dim, hidden_dim],
            dropout=dropout
        )
    def forward(self, data):
        return self.omni(data)  # [B,512]

class KKMHead(nn.Module):
    """ KKM 头：concat([512,512])=1024 -> 1 """
    def __init__(self, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(1024, 4*hidden_dim),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4*hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, cat_kcat, cat_km):
        x = torch.cat([cat_kcat, cat_km], dim=-1)  # [B,1024]
        return self.out(x).squeeze(-1)

class EitlemKKmPredictor(nn.Module):
    """ KKM = 两个 backbone（各 512） + 1024 头；只迁移骨干参数 """
    def __init__(self, mol_in_dim, hidden_dim=128, protein_dim=1280, dropout=0.2):
        super().__init__()
        self.kcat_backbone = OmniBackbone(mol_in_dim, protein_dim, hidden_dim, dropout)
        self.km_backbone   = OmniBackbone(mol_in_dim, protein_dim, hidden_dim, dropout)
        self.head = KKMHead(hidden_dim, dropout)

    # 兼容旧脚本写法
    @property
    def kcat(self):
        return self.kcat_backbone

    @property
    def km(self):
        return self.km_backbone

    @property
    def o(self):
        return self.head

    def forward(self, data):
        f_kcat = self.kcat_backbone(data)  # [B,512]
        f_km   = self.km_backbone(data)    # [B,512]
        return self.head(f_kcat, f_km)

    # -------- 只加载骨干（从 KCAT/KM 的 ckpt 里抽取 omni.*） ----------
    @staticmethod
    def _extract_backbone_sd(full_sd, target_backbone_state):
        """从完整 state_dict 里抽取 'omni.' 前缀参数，去前缀后对齐到 backbone"""
        filtered = {}
        for k, v in full_sd.items():
            if not k.startswith("backbone.omni.") and not k.startswith("omni."):
                continue
            k2 = k.split("omni.", 1)[1]  # 去掉前缀 '...omni.'
            if k2 in target_backbone_state and target_backbone_state[k2].shape == v.shape:
                filtered[k2] = v
        return filtered

    def load_backbones(self, kcat_sd, km_sd, verbose=True):
        kcat_tgt = self.kcat_backbone.state_dict()
        km_tgt   = self.km_backbone.state_dict()

        kcat_f = self._extract_backbone_sd(kcat_sd, kcat_tgt)
        km_f   = self._extract_backbone_sd(km_sd,   km_tgt)

        self.kcat_backbone.load_state_dict(kcat_f, strict=False)
        self.km_backbone.load_state_dict(km_f,     strict=False)

        if verbose:
            print(f"[KKM] kcat_backbone loaded: {len(kcat_f)} keys; km_backbone loaded: {len(km_f)} keys")











# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import global_add_pool
# from torch_geometric.utils import softmax
# from torch_geometric.utils import degree
# import numpy as np
# import math
# from omniesi.extractor import OmniESIExtractor
# class Resnet(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super().__init__()
#         self.lin1 = nn.Linear(in_dim, out_dim)
#         self.lin2 = nn.Linear(out_dim, out_dim)
#         self.lin3 = nn.Linear(out_dim, out_dim)
#         self.lin4 = nn.Linear(out_dim, out_dim)
#     def forward(self, x):
#         x = F.relu(self.lin1(x))
#         x = x + F.relu(self.lin2(x))
#         x = x + F.relu(self.lin3(x))
#         x = x + F.relu(self.lin4(x))
#         return x
    
# class ProMolAtt(nn.Module):
#     def __init__(self, hidden_dim):
#         super(ProMolAtt, self).__init__()
#         self.q = nn.Linear(hidden_dim, hidden_dim)
#         self.merge = nn.Linear(2*hidden_dim, 1, bias=False) # 计算相似性函数
#         self.k = nn.Linear(hidden_dim, hidden_dim)
#     def forward(self, mol, prot, batch):
#         q = F.relu(self.q(mol)) # 分子映射
#         r = q.repeat_interleave(degree(batch,  dtype=batch.dtype), dim=0) # 分子扩增
#         k = F.relu(self.k(prot))
#         score = self.merge(torch.cat([k, r], dim=-1)) # 计算相似性分数
#         score = softmax(score, batch, dim=0) # 权重加权
#         o = global_add_pool(k * score, batch) # 聚合全局向量
#         return o, q
    
# class AttentionAgg(nn.Module):
#     def __init__(self, hidden_dim):
#         super(AttentionAgg, self).__init__()
#         self.q = nn.Linear(hidden_dim, hidden_dim)
#         self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
#     def forward(self, x, y):
#         """
#         x -> y ==> y^
#         """
#         q = F.relu(self.q(x.mean(dim=1)))
#         k = F.relu(self.k(y))
#         score = F.softmax(torch.matmul(q.unsqueeze(1), k.transpose(-1, -2)), dim=-1)
#         out = torch.matmul(score, y).squeeze(1)
#         return out
    
# class MultiHeadAttenAgg(nn.Module):
#     def __init__(self, hidden_dim, att_layer, dropout):
#         super().__init__()
#         self.seq_m = nn.ModuleList(AttentionAgg(hidden_dim) for _ in range(att_layer))
#         self.seq_o = nn.Sequential(
#         nn.Linear(hidden_dim*att_layer, 4*hidden_dim*att_layer),
#         nn.ReLU(),
#         nn.Dropout(p=dropout),
#         nn.Linear(4*hidden_dim*att_layer, hidden_dim)
#         )
#     def forward(self, x, y):
#         return self.seq_o(torch.cat([m(x, y) for m in self.seq_m], dim=-1))

# # class EitlemKKmPredictor(nn.Module):
# #     def __init__(self, 
# #                  mol_in_dim, 
# #                  hidden_dim=128, 
# #                  protein_dim=1280, 
# #                  layer=10, 
# #                  dropout=0.2, 
# #                  att_layer=10
# #                 ):
# #         super(EitlemKKmPredictor, self).__init__()
# #         self.prej1 = Resnet(mol_in_dim, hidden_dim)
# #         self.prej2 = nn.Linear(protein_dim, hidden_dim, bias=False)
# #         self.pro_extrac = nn.ModuleList([ProMolAtt(hidden_dim) for _ in range(layer)])
# #         self.att1 = MultiHeadAttenAgg(hidden_dim,  att_layer, dropout)
# #         self.att2 = MultiHeadAttenAgg(hidden_dim,  att_layer, dropout)
# #         self.out = nn.Sequential(
# #             nn.Linear(2*hidden_dim, 4*hidden_dim),
# #             nn.PReLU(),
# #             nn.Dropout(p=dropout),
# #             nn.Linear(4*hidden_dim, hidden_dim),
# #             nn.PReLU(),
# #             nn.Dropout(p=dropout),
# #             nn.Linear(hidden_dim, 1)
# #         )
        
# #     def final_stage(self, mol, pro):
# #         pro_out = self.att1(mol, pro)
# #         mol_out = self.att2(pro, mol)
# #         return self.out(torch.cat([mol_out, pro_out], dim=-1)).squeeze(dim=-1)
    
# #     def forward(self, data):
# #         mol = F.relu(self.prej1(data.x))
# #         prot = F.relu(self.prej2(data.pro_emb))
# #         att_pro = []
# #         att_mol = []
# #         for m in self.pro_extrac:
# #             o, q = m(mol, prot, data.pro_emb_batch)
# #             att_pro.append(o)
# #             att_mol.append(q)
# #         att_mol = torch.stack(att_mol, dim=1)
# #         att_pro = torch.stack(att_pro, dim=1)
# #         return self.final_stage(att_mol, att_pro)
        


# class EitlemKKmPredictor(nn.Module):
#     def __init__(self,
#                  mol_in_dim,
#                  hidden_dim=128,
#                  protein_dim=1280,
#                  layer=10,
#                  dropout=0.2,
#                  att_layer=10):
#         super().__init__()
#         # 其余保持——如果你还需要 mol/pro 的 MLP，可保留；但我们的适配器已经内含编码器
#         self.omni = OmniESIExtractor(
#             drug_in_feats=mol_in_dim,
#             protein_dim=protein_dim,
#             dim_model=hidden_dim,   # 与原 hidden_dim 对齐
#             d_out=512,              # 设成你原 decoder 期望的维度（常见 512）
#             gnn_hidden=[hidden_dim, hidden_dim, hidden_dim],
#             dropout=dropout
#         )

#         # 下游解码头沿用你原来的 (输入维 = d_out)
#         self.out = nn.Sequential(
#             nn.Linear(512, 4*hidden_dim),
#             nn.PReLU(),
#             nn.Dropout(p=dropout),
#             nn.Linear(4*hidden_dim, hidden_dim),
#             nn.PReLU(),
#             nn.Dropout(p=dropout),
#             nn.Linear(hidden_dim, 1)
#         )

#     def forward(self, data):
#         # 直接用适配器得到红框融合表示
#         fused = self.omni(data)        # [B, 512]
#         return self.out(fused).squeeze(-1)