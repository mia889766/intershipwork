# joint_model_triple.py（新建）
import torch.nn as nn
from KCM import EitlemKcatPredictor
from KMP import EitlemKmPredictor
from KKMP import EitlemKKmPredictor

class JointTripleModel(nn.Module):
    def __init__(self, mol_dim=167, hidden_dim=512, protein_dim=1280, layer=10, dropout=0.5, att_layer=10):
        super().__init__()
        self.kcat = EitlemKcatPredictor(mol_dim, hidden_dim, protein_dim, layer, dropout, att_layer)
        self.km   = EitlemKmPredictor  (mol_dim, hidden_dim, protein_dim, layer, dropout, att_layer)
        self.kkm  = EitlemKKmPredictor (mol_dim, hidden_dim, protein_dim, layer, dropout, att_layer)

    def forward(self, data):
        y_kcat = self.kcat(data)  # (B,)
        y_km   = self.km  (data)  # (B,)
        y_kkm  = self.kkm (data)  # (B,)
        return y_kcat, y_km, y_kkm
