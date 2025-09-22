import torch, glob
p = sorted(glob.glob("/workspace/EITLEM-Kinetics/Data/Feature/esm1v_t33_650M_UR90S_1_embeding_1280/1.pt"))[0]
v = torch.load(p, map_location="cpu")
print(v.shape, v.numel())  # 期望输出: torch.Size([1280]) 1280
