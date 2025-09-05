from torch import nn
import sys
import re
import torch
from eitlem_utils import Tester, Trainer, get_pair_info
from KCM import EitlemKcatPredictor
from KMP import EitlemKmPredictor
#from ensemble import ensemble
from KKMP import EitlemKKmPredictor

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import EitlemDataSet, EitlemDataLoader
import os
import shutil
import argparse
# 新增：三模态模型（你按我之前给的类名放在 KCM_MM.py；如果你直接写在 KCM.py，就改成 from KCM import EitlemKcatPredictorMM）
#from KCM import EitlemKcatPredictorMM

# 新增：MMKcat 数据集/加载器（按你粘贴到 dataset.py 的类名）
from dataset import MMKcatDataset, MMKcatDataLoader


def kineticsTrainer(kkmPath, TrainType, Type, Iteration, log10, molType, device):
    train_info = f"Transfer-{TrainType}-{Type}-train-{Iteration}"

    if os.path.exists(f'../Resultsomniesi/{Type}/{train_info}'):
        return None
    
    if kkmPath is not None:
        Epoch = 40 // (Iteration // 2)
    else:
        Epoch = 100
    
    file_model = f'../Resultsomniesi/{Type}/{train_info}/Weight/'
    
    if kkmPath is not None:
        trained_weights = torch.load(kkmPath)
        if Type == 'KCAT':
            if molType == 'MACCSKeys':
                model = EitlemKcatPredictor(167, 512, 1280, 10, 0.5, 10)
            else:
                model = EitlemKcatPredictor(1024, 512, 1280, 10, 0.5, 10)
            weights = model.state_dict()
            pretrained_para = {k[5:]: v for k, v in trained_weights.items() if 'kcat' in k and k[5:] in weights}
            weights.update(pretrained_para)
            model.load_state_dict(weights)
        else:
            if molType == 'MACCSKeys':
                model = EitlemKmPredictor(167, 512, 1280, 10, 0.5, 10)
            else:
                model = EitlemKmPredictor(1024, 512, 1280, 10, 0.5, 10)
            weights = model.state_dict()
            pretrained_para = {k[3:]: v for k, v in trained_weights.items() if 'km' in k and k[3:] in weights}
            weights.update(pretrained_para)
            model.load_state_dict(weights)
    else:
        if Type == 'KCAT':
            if molType == 'MACCSKeys':
                #model = EitlemKcatPredictor(167, 512, 1280, 10, 0.5, 10)
                model = EitlemKcatPredictor(mol_in_dim=167, hidden_dim=128, protein_dim=1280, dropout=0.5)

            else:
                #model = EitlemKcatPredictor(1024, 512, 1280, 10, 0.5, 10)
                model = EitlemKcatPredictor(mol_in_dim=1024, hidden_dim=128, protein_dim=1280, dropout=0.5)

        else:
            if molType == 'MACCSKeys':
                #model = EitlemKmPredictor(167, 512, 1280, 10, 0.5, 10)
                model = EitlemKmPredictor(mol_in_dim=167, hidden_dim=128, protein_dim=1280, dropout=0.5)

            else:
                #model = EitlemKmPredictor(1024, 512, 1280, 10, 0.5, 10)
                model = EitlemKmPredictor(mol_in_dim=1024, hidden_dim=128, protein_dim=1280, dropout=0.5)

    
    if not os.path.exists(file_model):
        os.makedirs(file_model)
    file_model += 'Eitlem_'
    """Train setting."""
    train_pair_info, test_pair_info = get_pair_info("../Data/", Type, False)
    train_set = EitlemDataSet(train_pair_info, f'../Data/Feature/esm1v_t33_650M_UR90S_1_embeding_1280/', f'../Data/Feature/index_smiles', 1024, 4, log10, molType)
    test_set = EitlemDataSet(test_pair_info, f'../Data/Feature/esm1v_t33_650M_UR90S_1_embeding_1280/', f'../Data/Feature/index_smiles', 1024, 4, log10, molType)
    train_loader = EitlemDataLoader(data=train_set, batch_size=64, shuffle=True, drop_last=False, num_workers=4, prefetch_factor=2, persistent_workers=True, pin_memory=True)#改小batchsize,num_works,prefetch_factor
    valid_loader = EitlemDataLoader(data=test_set, batch_size=64, drop_last=False, num_workers=4, prefetch_factor=2, persistent_workers=True, pin_memory=True)#改小batchsize,num_works,prefetch_factor
    model = model.to(device)
    if kkmPath is not None:
        #out_param = list(map(id, model.out.parameters()))
        head_param_ids = list(map(id, model.head.parameters()))
        rest_params = filter(lambda p: id(p) not in head_param_ids, model.parameters())
        optimizer = torch.optim.AdamW([
            {'params': rest_params,        'lr': 1e-4},
            {'params': model.head.parameters(), 'lr': 1e-3},
        ])

        #rest_param = filter(lambda x:id(x) not in out_param, model.parameters())
        # optimizer = torch.optim.AdamW([
        #                                {'params': rest_param, 'lr':1e-4},
        #                                {'params':model.out.parameters(), 'lr':1e-3}, 
        #                               ])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.8)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.9)
    
    loss_fn = nn.MSELoss()
    tester = Tester(device, loss_fn, log10=log10)
    trainer = Trainer(device, loss_fn, log10=log10)
    
    print("start to train...")
    writer = SummaryWriter(f'../Resultsomniesi/{Type}/{train_info}/logs/')
    for epoch in range(1, Epoch + 1):
        train_MAE, train_rmse, train_r2, loss_train = trainer.run(model, train_loader, optimizer, len(train_pair_info), f"{Iteration}iter epoch {epoch} train:")
        MAE_dev, RMSE_dev, R2_dev, loss_dev = tester.test(model, valid_loader, len(test_pair_info), desc=f"{Iteration}iter epoch {epoch} valid:")
        scheduler.step()
        writer.add_scalars("loss",{'train_loss':loss_train, 'dev_loss':loss_dev}, epoch)
        writer.add_scalars("RMSE",{'train_RMSE':train_rmse, 'dev_RMSE':RMSE_dev}, epoch)
        writer.add_scalars("MAE",{'train_MAE':train_MAE, 'dev_MAE':MAE_dev}, epoch)
        writer.add_scalars("R2",{'train_R2':train_r2, 'dev_R2':R2_dev}, epoch)
        tester.save_model(model, file_model+f'{molType}_trainR2_{train_r2:.4f}_devR2_{R2_dev:.4f}_RMSE_{RMSE_dev:.4f}_MAE_{MAE_dev:.4f}') # 保存
    pass


def kineticsTrainer_MMKcat(mm_root, molType, device, log10=True, epochs=100, batch_size=32):
    """
    直接用 MMKcat 的三模态数据训练扩展后的 EITLEM kcat 模型（含 3D 结构分支）。
    不参与原来的 Transfer/KKM 流程。
    """
    Type = 'KCAT'
    train_info = f"MMKcat-{Type}-train"
    save_root = f'../Resultsmm1/{Type}/{train_info}'
    weight_dir = f'{save_root}/Weight/'
    log_dir = f'{save_root}/logs/'
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 1) 构造数据路径
    train_json = os.path.join(mm_root, 'concat_train_dataset_final_latest.json')
    train_x    = os.path.join(mm_root, 'concat_train_graph_x_latest.pkl')
    train_ei   = os.path.join(mm_root, 'concat_train_graph_edge_index_latest.pkl')

    test_json  = os.path.join(mm_root, 'concat_test_dataset_final_latest.json')
    test_x     = os.path.join(mm_root, 'concat_test_graph_x_latest.pkl')
    test_ei    = os.path.join(mm_root, 'concat_test_graph_edge_index_latest.pkl')

    # 2) 构造数据集与 DataLoader
    # 这里保持与原 EITLEM 指纹设置一致：MACCS=167; 其他=1024
    nbits = 167 if molType == 'MACCSKeys' else 1024
    # 半径与类型与原脚本一致（可按需改 ECFP/RDKIT）
    train_set = MMKcatDataset(train_json, train_x, train_ei, nbits=nbits, radius=4, Type=molType, log10=log10)
    test_set  = MMKcatDataset(test_json,  test_x,  test_ei,  nbits=nbits, radius=4, Type=molType, log10=log10)

    train_loader = MMKcatDataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False,
                                    num_workers=4, prefetch_factor=2, persistent_workers=True, pin_memory=True)
    valid_loader = MMKcatDataLoader(test_set,  batch_size=batch_size, shuffle=False, drop_last=False,
                                    num_workers=4, prefetch_factor=2, persistent_workers=True, pin_memory=True)

    # 3) 构造模型
    #   - mol_in_dim 由指纹类型决定
    #   - protein_dim 固定为 1280（Sequence_Rep 的维度）
    #   - graph_in_dim 从样本推断
    protein_dim = 1280
    graph_in_dim = train_set[0].graph_x.shape[1]
    if molType == 'MACCSKeys':
        model = EitlemKcatPredictorMM(167, 512, protein_dim, graph_in_dim, 10, 0.5, 10)
    else:
        model = EitlemKcatPredictorMM(1024, 512, protein_dim, graph_in_dim, 10, 0.5, 10)
    model = model.to(device)

    # 4) 优化器/调度与原脚本风格一致
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    epochs = args.epochs
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=5)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs-5), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
    GRAD_CLIP = 1.0


    loss_fn = nn.MSELoss()
    tester = Tester(device, loss_fn, log10=log10)
    trainer = Trainer(device, loss_fn, log10=log10)

    print("start to train on MMKcat three-modality dataset...")
    writer = SummaryWriter(log_dir)

    for epoch in range(1, epochs + 1):
        train_MAE, train_rmse, train_r2, loss_train = trainer.run(
            model, train_loader, optimizer, len(train_set), f"MMKcat epoch {epoch} train:"
        )
        MAE_dev, RMSE_dev, R2_dev, loss_dev = tester.test(
            model, valid_loader, len(test_set), desc=f"MMKcat epoch {epoch} valid:"
        )
        scheduler.step()

        writer.add_scalars("loss", {'train_loss': loss_train, 'dev_loss': loss_dev}, epoch)
        writer.add_scalars("RMSE", {'train_RMSE': train_rmse, 'dev_RMSE': RMSE_dev}, epoch)
        writer.add_scalars("MAE", {'train_MAE': train_MAE, 'dev_MAE': MAE_dev}, epoch)
        writer.add_scalars("R2",  {'train_R2':  train_r2,  'dev_R2':  R2_dev}, epoch)

        # 与原保存风格一致
        file_model = os.path.join(weight_dir, f"Eitlem_{molType}_trainR2_{train_r2:.4f}_devR2_{R2_dev:.4f}_RMSE_{RMSE_dev:.4f}_MAE_{MAE_dev:.4f}")
        tester.save_model(model, file_model)




def KKMTrainer(kcatPath, kmPath, TrainType, Iteration, log10, molType, device):
    train_info = f"Transfer-{TrainType}-KKM-train-{Iteration}"
    kkm_root  = f'../Resultsomniesi/KKM/{train_info}'
    kkm_weight_dir = f'{kkm_root}/Weight/'
    # 仅当存在且有权重文件才跳过
    if os.path.isdir(kkm_weight_dir):
        files = [f for f in os.listdir(kkm_weight_dir) if os.path.isfile(os.path.join(kkm_weight_dir, f))]
        if len(files) > 0:
            return None

    #if os.path.exists(f'../Resultsomniesi/KKM/{train_info}'):
        #return None
    
    Epoch = 40
    file_model = f'../Resultsomniesi/KKM/{train_info}/Weight/'
    if molType == 'MACCSKeys':
        #model = ensemble(167, 512, 1280, 10, 0.5, 10)
        model = EitlemKKmPredictor(mol_in_dim=167, hidden_dim=128, protein_dim=1280, dropout=0.5)
    else:
        #model = ensemble(1024, 512, 1280, 10, 0.5, 10)
        model = EitlemKKmPredictor(mol_in_dim=1024, hidden_dim=128, protein_dim=1280, dropout=0.5)
    # kcat_pretrained = torch.load(kcatPath)
    # km_pretrained = torch.load(kmPath)
    # kcat_parameters = model.kcat.state_dict()
    # km_parameters = model.km.state_dict()
    # pretrained_kcat_para = {k:v for k, v in kcat_pretrained.items() if k in kcat_parameters}
    # pretrained_km_para = {k:v for k, v in km_pretrained.items() if k in km_parameters}
    # kcat_parameters.update(pretrained_kcat_para)
    # km_parameters.update(pretrained_km_para)
    #model.kcat.load_state_dict(kcat_parameters)
    #model.km.load_state_dict(km_parameters)
    kcat_sd = torch.load(kcatPath, map_location=device)
    km_sd   = torch.load(kmPath,   map_location=device)
    model = model.to(device)
    model.load_backbones(kcat_sd, km_sd, verbose=True)


    if not os.path.exists(file_model):
        os.makedirs(file_model)

    file_model += 'Eitlem_'
    """Train setting."""
    train_pair_info, test_pair_info = get_pair_info("../Data/", 'KKM',False)
    train_set = EitlemDataSet(train_pair_info, f'../Data/Feature/esm1v_t33_650M_UR90S_1_embeding_1280/', f'../Data/Feature/index_smiles', 1024, 4, log10, molType)
    test_set = EitlemDataSet(test_pair_info, f'../Data/Feature/esm1v_t33_650M_UR90S_1_embeding_1280/', f'../Data/Feature/index_smiles', 1024, 4, log10, molType)
    train_loader = EitlemDataLoader(data=train_set, batch_size=64, shuffle=True, drop_last=False, num_workers=4, prefetch_factor=2, persistent_workers=True, pin_memory=True)
    valid_loader = EitlemDataLoader(data=test_set, batch_size=64, drop_last=False, num_workers=4, prefetch_factor=2, persistent_workers=True, pin_memory=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW([
                                   {'params': model.kcat.parameters(), 'lr':1e-4},
                                   {'params':model.km.parameters(), 'lr':1e-4},
                                   {'params':model.o.parameters(), 'lr':1e-3},
                                  ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.9)
    loss_fn = nn.MSELoss()
    tester = Tester(device, loss_fn, log10=log10)
    trainer = Trainer(device, loss_fn, log10=log10)
    print("start to train...")
    writer = SummaryWriter(f'../Resultsomniesi/KKM/{train_info}/logs/')
    for epoch in range(1, Epoch + 1):
        train_MAE, train_rmse, train_r2, loss_train = trainer.run(model, train_loader, optimizer, len(train_set), f"{Iteration}iter epoch {epoch} train:")
        MAE_dev, RMSE_dev, R2_dev, loss_dev = tester.test(model, valid_loader, len(test_set), desc=f"{Iteration}iter epoch {epoch} valid:")
        scheduler.step()
        writer.add_scalars("loss",{'train_loss':loss_train, 'dev_loss':loss_dev}, epoch)
        writer.add_scalars("RMSE",{'train_RMSE':train_rmse, 'dev_RMSE':RMSE_dev}, epoch)
        writer.add_scalars("MAE",{'train_MAE':train_MAE, 'dev_MAE':MAE_dev}, epoch)
        writer.add_scalars("R2",{'train_R2':train_r2, 'dev_R2':R2_dev}, epoch)
        tester.save_model(model, file_model+f'{molType}_trainR2:{train_r2:.4f}_devR2_{R2_dev:.4f}_RMSE_{RMSE_dev:.4f}_MAE_{MAE_dev:.4f}')


def getPath(Type, TrainType, Iteration):
    train_info = f"Transfer-{TrainType}-{Type}-train-{Iteration}"
    file_model = f'../Resultsomniesi/{Type}/{train_info}/Weight/'
    fileList = os.listdir(file_model)
    return os.path.join(file_model, fileList[0])

def TransferLearing(Iterations, TrainType, log10=False, molType='MACCSKeys', device=None):
    for iteration in range(1, Iterations + 1):
        if iteration == 1:
            kineticsTrainer(None, TrainType, 'KCAT', iteration, log10, molType, device)
            kineticsTrainer(None, TrainType, 'KM', iteration, log10, molType, device)
        else:
            kkmPath = getPath('KKM', TrainType, iteration-1)
            kineticsTrainer(kkmPath, TrainType, 'KCAT', iteration, log10, molType, device)
            kineticsTrainer(kkmPath, TrainType, 'KM', iteration, log10, molType, device)
        
        kcatPath = getPath('KCAT', TrainType, iteration)
        kmPath = getPath('KM', TrainType, iteration)
        KKMTrainer(kcatPath, kmPath, TrainType, iteration, log10, molType, device)




# def str2bool(v):
#     if isinstance(v, bool):
#         return v
#     s = str(v).lower()
#     if s in ('yes', 'true', 't', '1', 'y'):
#         return True
#     if s in ('no', 'false', 'f', '0', 'n'):
#         return False
#     raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--Iteration', type=int, required=True)
    parser.add_argument('-t', '--TrainType', type=str, required=True)
    parser.add_argument('-l', '--log10', type=bool, required=False, default=True)
    parser.add_argument('-m', '--molType', type=str, required=False, default='MACCSKeys')
    parser.add_argument('-d', '--device', type=int, required=True)
    # parser.add_argument('--mm_root', type=str, required=False, default='../Data')#新增mmkcat

    # # === 新增参数 ===
    # parser.add_argument('--epochs', type=int, default=100)
    # parser.add_argument('--batch_size', type=int, default=32)

    return parser.parse_args()

if __name__ == '__main__':
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')
    print(f"use device {device}")

    # if args.TrainType.upper() == 'MMKCAT':
    # # 只跑 MMKcat 三模态的 KCAT 训练（不做 transfer，不做 KKM）
    #     kineticsTrainer_MMKcat(args.mm_root, args.molType, device,
    #                    log10=args.log10, epochs=args.epochs, batch_size=args.batch_size)
    # else:
    # # 走原来的 transfer 学习 + KKM 流程（兼容老数据）
    #     TransferLearing(args.Iteration, args.TrainType, args.log10, args.molType, device)

    TransferLearing(args.Iteration, args.TrainType, args.log10, args.molType, device)
