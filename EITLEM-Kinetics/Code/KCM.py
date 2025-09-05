# /workspace/EITLEM-Kinetics/Code/KCM.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from omniesi.extractor import OmniESIExtractor  # 产出 512 cat_vec

class OmniBackbone(nn.Module):
    """ 产出 512 的 catalysis-aware 表征；训练/迁移都复用这一块 """
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
        return self.omni(data)  # [B, 512]

class RegressionHead(nn.Module):
    """ 通用回归头:512 -> 1 """
    def __init__(self, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(512, 4*hidden_dim),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4*hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):  # x: [B,512]
        return self.out(x).squeeze(-1)

class EitlemKcatPredictor(nn.Module):
    """ Kcat 全模型 = Backbone + Head """
    def __init__(self, mol_in_dim, hidden_dim=128, protein_dim=1280, dropout=0.2):
        super().__init__()
        self.backbone = OmniBackbone(mol_in_dim, protein_dim, hidden_dim, dropout)
        self.head = RegressionHead(hidden_dim, dropout)
    def forward(self, data):
        x = self.backbone(data)   # [B,512]
        return self.head(x)

    # 便于 KKM 阶段只迁移骨干参数
    def backbone_state_dict(self):
        return self.backbone.state_dict()

















# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import global_add_pool
# from torch_geometric.utils import softmax
# from torch_geometric.utils import degree
# # 在文件开头加入：
# from torch_geometric.nn import GCNConv, global_mean_pool
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

# # class EitlemKcatPredictor(nn.Module):
# #     def __init__(self, 
# #                  mol_in_dim, 
# #                  hidden_dim=128, 
# #                  protein_dim=1280, 
# #                  layer=10, 
# #                  dropout=0.2, 
# #                  att_layer=10
# #                 ):
# #         super(EitlemKcatPredictor, self).__init__()
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


# class EitlemKcatPredictorMM(nn.Module):
#     """
#     扩展的三模态模型：底物指纹 + 蛋白序列嵌入 + 蛋白 3D 图
#     graph_in_dim：蛋白节点特征维度（即 .pkl 中 graph_x[i].shape[1]）
#     """

#     def __init__(self,
#                  mol_in_dim,        # 与原模型一致：分子指纹维度，如 167(MACCS) 或 1024(ECFP/RDKIT)
#                  hidden_dim=128,
#                  protein_dim=1280,  # Sequence_Rep 的维度
#                  graph_in_dim=256,  # 图节点特征维度，根据 pkl 文件确定
#                  layer=10,
#                  dropout=0.2,
#                  att_layer=10):
#         super().__init__()
#         # 原有分子和蛋白序列部分
#         self.prej1 = Resnet(mol_in_dim, hidden_dim)
#         self.prej2 = nn.Linear(protein_dim, hidden_dim, bias=False)
#         self.pro_extrac = nn.ModuleList([ProMolAtt(hidden_dim) for _ in range(layer)])
#         self.att1 = MultiHeadAttenAgg(hidden_dim, att_layer, dropout)
#         self.att2 = MultiHeadAttenAgg(hidden_dim, att_layer, dropout)

#         # 新增：蛋白 3D 图编码器（两层 GCN）
#         self.gcn1 = GCNConv(graph_in_dim, hidden_dim)
#         self.gcn2 = GCNConv(hidden_dim, hidden_dim)
#         # 将图表示投影到 hidden_dim，便于融合
#         self.graph_proj = nn.Linear(hidden_dim, hidden_dim)

#         # 输出层维度由 2*hidden_dim 改为 3*hidden_dim
#         self.out = nn.Sequential(
#             nn.Linear(3 * hidden_dim, 4 * hidden_dim),
#             nn.PReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(4 * hidden_dim, hidden_dim),
#             nn.PReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, 1)
#         )

#     def final_stage(self, mol, pro, graph_feat):
#         """
#         返回融合后的标量预测值。
#         """
#         pro_out = self.att1(mol, pro)
#         mol_out = self.att2(pro, mol)
#         # graph_feat 已在 forward 中投影到 hidden_dim
#         # 拼接三个模态的向量
#         final_vec = torch.cat([mol_out, pro_out, graph_feat], dim=-1)
#         return self.out(final_vec).squeeze(dim=-1)

#     def forward(self, data):
#         # 1) 底物分子指纹
#         mol = F.relu(self.prej1(data.x))
#         # 2) 蛋白序列嵌入
#         prot = F.relu(self.prej2(data.pro_emb))
#         # 3) 蛋白 3D 图编码
#         #    data.graph_x: [total_nodes, graph_in_dim]
#         #    data.graph_edge_index: [2, total_edges]
#         #    data.graph_x_batch: [total_nodes]，表明每个节点属于哪个样本
#         x = F.relu(self.gcn1(data.graph_x, data.graph_edge_index))
#         x = F.relu(self.gcn2(x, data.graph_edge_index))
#         graph_raw = global_mean_pool(x, data.graph_x_batch)  # [batch_size, hidden_dim]
#         graph_feat = F.relu(self.graph_proj(graph_raw))       # [batch_size, hidden_dim]

#         # 与原模型相同的 cross-attention 部分
#         att_pro = []
#         att_mol = []
#         for m in self.pro_extrac:
#             o, q = m(mol, prot, data.pro_emb_batch)
#             att_pro.append(o)
#             att_mol.append(q)
#         att_mol = torch.stack(att_mol, dim=1)
#         att_pro = torch.stack(att_pro, dim=1)

#         return self.final_stage(att_mol, att_pro, graph_feat)




# class EitlemKcatPredictor(nn.Module):
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