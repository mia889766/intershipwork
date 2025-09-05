import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import numpy as np
import math
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit import Chem, RDLogger
# 在文件开头加入所需库
import json
import pickle

RDLogger.DisableLog('rdApp.*')
### if an error occurs here, please check the version of rdkit 
fpgen = AllChem.GetRDKitFPGenerator(fpSize=1024)
# 放在文件顶部的 import（若已有可忽略）
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors as rdMD
from rdkit.Chem import RDKFingerprint  # RDKit-topological

# def _fp_from_mol(mol, nbits, radius, Type):
#     """对单个 RDKit Mol 生成 bit 向量(np.uint8)。"""
#     if mol is None:
#         return None
#     if Type == "ECFP":
#         bv = rdMD.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
#         arr = np.zeros((nbits,), dtype=np.uint8)
#         from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
#         ConvertToNumpyArray(bv, arr)
#         return arr
#     elif Type == "MACCSKeys":
#         # MACCS 固定 167 位
#         fp = MACCSkeys.GenMACCSKeys(mol)
#         arr = np.fromiter((int(fp.GetBit(i)) for i in range(167)), dtype=np.uint8, count=167)
#         return arr
#     elif Type == "RDKIT":
#         bv = RDKFingerprint(mol, fpSize=nbits)  # nbits 默认为 2048/1024 等
#         arr = np.zeros((nbits,), dtype=np.uint8)
#         from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
#         ConvertToNumpyArray(bv, arr)
#         return arr
#     else:
#         raise ValueError(f"Unknown Type: {Type}")

# def _fp_or_from_smiles(smiles, nbits, radius, Type):
#     """
#     对单底物或多底物（字符串或列表）生成 OR 合并的指纹（np.float32）。
#     """
#     if isinstance(smiles, str):
#         smiles_list = [smiles]
#     else:
#         smiles_list = smiles

#     # 确定位数
#     if Type == "MACCSKeys":
#         total_bits = 167
#     else:
#         total_bits = nbits

#     acc = np.zeros((total_bits,), dtype=np.uint8)
#     for smi in smiles_list:
#         smi = (smi or "").strip()
#         if not smi:
#             continue
#         mol = Chem.MolFromSmiles(smi)
#         arr = _fp_from_mol(mol, nbits, radius, Type)
#         if arr is not None:
#             acc |= arr  # 关键：位或合并

#     return acc.astype(np.float32)

# def generateData(smiles_or_list, proteins, value, nbits, radius, Type):
#     """
#     兼容原接口：把 mol 参数替换为 smiles（字符串或列表），
#     返回一个 torch_geometric.data.Data，x 是合并后的单个指纹向量。
#     """
#     # 特例：同时输出 MACCS 与 RDKIT 两种（你原先的 MACCSKeys_RDKIT 分支）
#     if Type == 'MACCSKeys_RDKIT':
#         fp1 = _fp_or_from_smiles(smiles_or_list, nbits, radius, 'MACCSKeys')  # 167
#         fp2 = _fp_or_from_smiles(smiles_or_list, nbits, radius, 'RDKIT')      # nbits
#         return Data(
#             x=torch.from_numpy(fp1).unsqueeze(0),    # [1, 167]
#             y=torch.from_numpy(fp2).unsqueeze(0),    # [1, nbits]
#             pro_emb=proteins,
#             value=value
#         )

#     # 常规：单通道指纹
#     fp = _fp_or_from_smiles(smiles_or_list, nbits, radius, Type)  # [bits]
#     data = Data(
#         x=torch.from_numpy(fp).unsqueeze(0),  # [1, bits]
#         pro_emb=proteins,
#         value=value
#     )
#     return data



def generateData(mol, proteins, value, nbits, radius, Type):
    if Type == "ECFP":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits).ToList()
    elif Type == "MACCSKeys":
        fp = MACCSkeys.GenMACCSKeys(mol).ToList()
    elif Type == "RDKIT":
        fp = fpgen.GetFingerprint(mol).ToList()
    elif Type == 'MACCSKeys_RDKIT':
        fp1 = MACCSkeys.GenMACCSKeys(mol).ToList()
        fp2 = fpgen.GetFingerprint(mol).ToList()
        return Data(x = torch.FloatTensor(fp1).unsqueeze(0), y=torch.FloatTensor(fp2).unsqueeze(0), pro_emb=proteins, value=value)
    data = Data(x = torch.FloatTensor(fp).unsqueeze(0), pro_emb=proteins, value=value)
    return data
    
class EitlemDataSet(Dataset):
    def __init__(self, Pairinfo, ProteinsPath, Smiles, nbits, radius, log10=False, Type='ECFP'):
        super(EitlemDataSet, self).__init__()
        self.pairinfo = Pairinfo
        if isinstance(Smiles, str):
            self.smiles = torch.load(Smiles)
        elif isinstance(Smiles, dict):
            self.smiles = Smiles
        self.seq_path = os.path.join(ProteinsPath, '{}.pt')
        self.nbits = nbits
        self.radius = radius
        self.log10 = log10
        self.Type = Type
        print(f"log10:{self.log10} molType:{self.Type}")
    def __getitem__(self, idx):
        pro_id = self.pairinfo[idx][0]
        smi_id = self.pairinfo[idx][1]
        value = self.pairinfo[idx][2]
        protein_emb = torch.load(self.seq_path.format(pro_id))
        mol = AllChem.MolFromSmiles(self.smiles[smi_id].strip())
        if self.log10:
            value = math.log10(value)
        else:
            value = math.log2(value)
        data = generateData(mol,  protein_emb, value, self.nbits, self.radius, self.Type)
        return data
    def collate_fn(self, batch):
        return Batch.from_data_list(batch, follow_batch=['pro_emb'])
    def __len__(self):
        return len(self.pairinfo)
    
class EitlemDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)

def shuffle_dataset(dataset):
    np.random.shuffle(dataset)
    return dataset
def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2



class MMKcatDataset(Dataset):
    """
    使用 MMKcat 数据集的 Dataset。
    每条样本包含：
      - Substrate_Smiles: SMILES 字符串或字符串列表
      - Sequence_Rep: 长度为 1280 的蛋白序列嵌入（已通过 ESM2 预生成）
      - Value: log10(kcat) 或 kcat（根据 log10 参数决定是否取 log10）
      - graph_x: 节点特征矩阵（来自 .pkl）
      - graph_edge_index: 边索引（来自 .pkl）
    """

    def __init__(self,
                 json_path: str,
                 graph_x_pkl: str,
                 graph_ei_pkl: str,
                 nbits: int = 1024,
                 radius: int = 2,
                 Type: str = "ECFP",
                 log10: bool = True):
        super().__init__()
        # 读取 json （顶层是字典，每个样本是一个 value）
        with open(json_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        # 过滤掉没有底物 SMILES 或没有序列嵌入的条目
        self.items = [
            v for v in data_dict.values()
            if v.get("Substrate_Smiles") and v.get("Sequence_Rep") is not None
        ]
        # 载入蛋白 3D 图数据
        with open(graph_x_pkl, 'rb') as f:
            self.graph_x = pickle.load(f)
        with open(graph_ei_pkl, 'rb') as f:
            self.graph_edge_index = pickle.load(f)
        assert len(self.items) == len(self.graph_x) == len(self.graph_edge_index), \
            "json 与 pkl 文件长度不一致，请确保样本顺序一致"

        self.nbits = nbits
        self.radius = radius
        self.Type = Type
        self.log10 = log10

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sample = self.items[idx]
        # 取底物 SMILES（如果是列表则取第一条，也可以改为平均或拼接）
        #smi = sample['Substrate_Smiles']
        #smi_str = smi[0] if isinstance(smi, list) else smi
        # 利用 rdkit 生成分子指纹；下面示例沿用 EITLEM 的 generateData 方式
        #mol = AllChem.MolFromSmiles(smi_str.strip())
        # 读取蛋白序列嵌入
        smiles = sample['Substrate_Smiles']
        import numpy as np

        protein_dim = 1280

        # 1) 读取并强制成 2D [L, 1280]
        pro_arr = np.asarray(sample['Sequence_Rep'], dtype=np.float32)
        if pro_arr.ndim == 1:
    # 常见：就是一个 1280 向量
            assert pro_arr.size == protein_dim, f"Sequence_Rep length {pro_arr.size} != 1280"
            pro_arr = pro_arr.reshape(1, protein_dim)          # [1, 1280]
        elif pro_arr.ndim == 2:
    # 少见：若作者给了每位残基一行，则 pro_arr.shape[1] 必须是 1280
            assert pro_arr.shape[1] == protein_dim, f"Sequence_Rep shape {pro_arr.shape} incompatible"
        else:
    # 如果出现奇怪维度，尝试按 1280 分块 reshape；不行则报错
            if pro_arr.size % protein_dim == 0:
                pro_arr = pro_arr.reshape(-1, protein_dim)
            else:
                raise ValueError(f"Unexpected Sequence_Rep shape: {pro_arr.shape}")

        pro_emb = torch.from_numpy(pro_arr)  # [L, 1280]

        # 读取 kcat 数值，并按需取 log10 或 log2
        value = float(sample['Value'])
        

        data = generateData(smiles, pro_emb, value, self.nbits, self.radius, self.Type)
        # 加上 3D 结构
        data.graph_x = torch.as_tensor(self.graph_x[idx], dtype=torch.float32)
        data.graph_edge_index = torch.as_tensor(self.graph_edge_index[idx], dtype=torch.long)

        return data

class MMKcatDataLoader(DataLoader):
    """
    DataLoader, collate_fn 默认传入 follow_batch=['pro_emb','graph_x']
    这样 batch 中会自动生成 pro_emb_batch 和 graph_x_batch，方便后续模型处理。
    """
    def __init__(self, data, **kwargs):
        super().__init__(data,
                         collate_fn=lambda batch: Batch.from_data_list(
                             batch, follow_batch=['pro_emb', 'graph_x']
                         ),
                         **kwargs)
