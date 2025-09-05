# import torch
# sd_kcat = torch.load("/workspace/EITLEM-Kinetics/Resultsomniesi/KCAT/Transfer-iterativeTrain-KCAT-train-1/Weight/Eitlem_MACCSKeys_trainR2_0.7849_devR2_0.5936_RMSE_0.9802_MAE_0.6662", map_location="cuda:0")
# sd_km   = torch.load("/workspace/EITLEM-Kinetics/Resultsomniesi/KM/Transfer-iterativeTrain-KM-train-1/Weight/Eitlem_MACCSKeys_trainR2_0.8401_devR2_0.6260_RMSE_0.7992_MAE_0.5742",   map_location="cuda:0")
# print("KCAT has omni keys? ", any(k.startswith("backbone.omni.") or k.startswith("omni.") for k in sd_kcat.keys()))
# print("KM   has omni keys? ", any(k.startswith("backbone.omni.") or k.startswith("omni.") for k in sd_km.keys()))
import torch
from KKMP import EitlemKKmPredictor

# 1) 加载 ckpt
kcat_path = "/workspace/EITLEM-Kinetics/Resultsomniesi/KCAT/Transfer-iterativeTrain-KCAT-train-1/Weight/Eitlem_MACCSKeys_trainR2_0.7849_devR2_0.5936_RMSE_0.9802_MAE_0.6662"
km_path   = "/workspace/EITLEM-Kinetics/Resultsomniesi/KM/Transfer-iterativeTrain-KM-train-1/Weight/Eitlem_MACCSKeys_trainR2_0.8401_devR2_0.6260_RMSE_0.7992_MAE_0.5742"

sd_kcat = torch.load(kcat_path, map_location="cuda:0")
sd_km   = torch.load(km_path,   map_location="cuda:0")

print("==== KCAT ckpt keys sample ====")
print(list(sd_kcat.keys())[:20])   # 打印前 20 个键
print("==== KM ckpt keys sample ====")
print(list(sd_km.keys())[:20])

# 2) 初始化一个 KKM 模型，拿目标 state_dict 的键
model = EitlemKKmPredictor(mol_in_dim=167, hidden_dim=512, protein_dim=1280, layer=10, dropout=0.5, att_layer=10)
kcat_tgt = model.kcat_backbone.state_dict()
km_tgt   = model.km_backbone.state_dict()

print("==== KKM model target kcat_backbone keys sample ====")
print([k for k in kcat_tgt.keys() if k.startswith("omni.")][:20])
print("==== KKM model target km_backbone keys sample ====")
print([k for k in km_tgt.keys() if k.startswith("omni.")][:20])
