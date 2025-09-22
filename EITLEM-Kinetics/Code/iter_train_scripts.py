from torch import nn
import sys
import re
import torch
from eitlem_utils import Tester, Trainer, get_pair_info
from KCM import EitlemKcatPredictor
from KMP import EitlemKmPredictor
from ensemble import ensemble
from torch.utils.tensorboard import SummaryWriter
from dataset import EitlemDataSet, EitlemDataLoader
import os
import argparse

# --------- 新增：更安全的 DataLoader 构造 ---------
def make_loader(pairs, batch_size, num_workers, prefetch_factor, log10, molType, shuffle):
    ds = EitlemDataSet(
        pairs,
        '../Data/Feature/prosst_perres_1280/',
        '../Data/Feature/index_smiles',
        1024, 4, log10, molType
    )
    # 当 num_workers=0 时，prefetch_factor 必须为 None
    pf = prefetch_factor if num_workers > 0 else None
    return EitlemDataLoader(
        data=ds, batch_size=batch_size, shuffle=shuffle, drop_last=False,
        num_workers=num_workers, prefetch_factor=pf,
        persistent_workers=(num_workers > 0), pin_memory=True
    ), ds

# --------- 单任务（KCAT/KM）训练 ---------
def kineticsTrainer(kkmPath, TrainType, Type, Iteration, log10, molType, device,
                    batch_size=200, num_workers=4, prefetch_factor=2):
    train_info = f"Transfer-{TrainType}-{Type}-train-{Iteration}"
    out_dir = f'../Results/{Type}/{train_info}'
    if os.path.exists(out_dir):
        return None

    # 轮次策略保持原有
    #Epoch = 40 // (Iteration // 2) if kkmPath is not None else 100
    Epoch = 40
    # ---- 构建模型 ----
    mol_dim = 167 if molType == 'MACCSKeys' else 1024
    if Type == 'KCAT':
        model = EitlemKcatPredictor(mol_dim, 512, 1280, 10, 0.5, 10)
    else:
        model = EitlemKmPredictor(mol_dim, 512, 1280, 10, 0.5, 10)

    # 迁移加载（从上一轮 KKM ensemble 中抽取对应子网）
    if kkmPath is not None:
        trained_weights = torch.load(kkmPath, map_location='cpu')  # ✅ 显式 map
        weights = model.state_dict()
        if Type == 'KCAT':
            # 取 ensemble 中以 'kcat.' 开头且与当前权重同名的参数
            pretrained = {k[5:]: v for k, v in trained_weights.items()
                          if k.startswith('kcat.') and (k[5:] in weights) and (weights[k[5:]].shape == v.shape)}
        else:
            pretrained = {k[3:]: v for k, v in trained_weights.items()
                          if k.startswith('km.') and (k[3:] in weights) and (weights[k[3:]].shape == v.shape)}
        weights.update(pretrained)
        model.load_state_dict(weights, strict=False)

    # ---- 数据 ----
    train_pairs, test_pairs = get_pair_info("../Data/", Type, False)
    train_loader, train_ds = make_loader(train_pairs, batch_size, num_workers, prefetch_factor, log10, molType, True)
    valid_loader, valid_ds = make_loader(test_pairs,  batch_size, num_workers, prefetch_factor, log10, molType, False)

    # ---- 优化器 / 调度 ----
    model = model.to(device)
    if kkmPath is not None:
        out_param = list(map(id, model.out.parameters()))
        rest_param = filter(lambda p: id(p) not in out_param, model.parameters())
        optimizer = torch.optim.AdamW([
            {'params': rest_param,            'lr': 1e-4},
            {'params': model.out.parameters(),'lr': 1e-3},
        ])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.9)

    # ---- 训练循环 ----
    loss_fn = nn.MSELoss()
    tester = Tester(device, loss_fn, log10=log10)
    trainer = Trainer(device, loss_fn, log10=log10)

    os.makedirs(os.path.join(out_dir, "Weight"), exist_ok=True)
    writer = SummaryWriter(os.path.join(out_dir, "logs"))
    file_prefix = os.path.join(out_dir, "Weight", f"Eitlem_{molType}_")

    print(f"start to train {Type} ...")
    for epoch in range(1, Epoch + 1):
        train_MAE, train_rmse, train_r2, loss_train = trainer.run(
            model, train_loader, optimizer, len(train_pairs), f"{Iteration}iter epoch {epoch} train:"
        )
        MAE_dev, RMSE_dev, R2_dev, loss_dev = tester.test(
            model, valid_loader, len(test_pairs), desc=f"{Iteration}iter epoch {epoch} valid:"
        )
        scheduler.step()
        writer.add_scalars("loss", {'train_loss': loss_train, 'dev_loss': loss_dev}, epoch)
        writer.add_scalars("RMSE", {'train_RMSE': train_rmse, 'dev_RMSE': RMSE_dev}, epoch)
        writer.add_scalars("MAE", {'train_MAE': train_MAE, 'dev_MAE': MAE_dev}, epoch)
        writer.add_scalars("R2",  {'train_R2': train_r2, 'dev_R2': R2_dev}, epoch)
        # 保存
        torch.save(model.state_dict(),
                   file_prefix + f"trainR2_{train_r2:.4f}_devR2_{R2_dev:.4f}_RMSE_{RMSE_dev:.4f}_MAE_{MAE_dev:.4f}.pt")

# --------- KKM（ensemble）训练 ---------
def KKMTrainer(kcatPath, kmPath, TrainType, Iteration, log10, molType, device,
               batch_size=200, num_workers=4, prefetch_factor=2):
    train_info = f"Transfer-{TrainType}-KKM-train-{Iteration}"
    out_dir = f'../Results/KKM/{train_info}'
    if os.path.exists(out_dir):
        return None

    Epoch = 40
    mol_dim = 167 if molType == 'MACCSKeys' else 1024
    model = ensemble(mol_dim, 512, 1280, 10, 0.5, 10).to(device)

    # 从单任务权重初始化子网（形状不匹配的层自动跳过）
    kcat_pretrained = torch.load(kcatPath, map_location='cpu')
    km_pretrained   = torch.load(kmPath,   map_location='cpu')
    kcat_parameters = model.kcat.state_dict()
    km_parameters   = model.km.state_dict()

    pretrained_kcat_para = {k: v for k, v in kcat_pretrained.items()
                            if k in kcat_parameters and (kcat_parameters[k].shape == v.shape)}
    pretrained_km_para   = {k: v for k, v in km_pretrained.items()
                            if k in km_parameters and (km_parameters[k].shape == v.shape)}
    kcat_parameters.update(pretrained_kcat_para)
    km_parameters.update(pretrained_km_para)
    model.kcat.load_state_dict(kcat_parameters, strict=False)
    model.km.load_state_dict(km_parameters,   strict=False)

    # 数据
    train_pairs, test_pairs = get_pair_info("../Data/", 'KKM')
    train_loader, train_ds = make_loader(train_pairs, batch_size, num_workers, prefetch_factor, log10, molType, True)
    valid_loader, valid_ds = make_loader(test_pairs,  batch_size, num_workers, prefetch_factor, log10, molType, False)

    # 优化器（子网小 lr，head 大 lr）
    optimizer = torch.optim.AdamW([
        {'params': model.kcat.parameters(), 'lr': 1e-4},
        {'params': model.km.parameters(),   'lr': 1e-4},
        {'params': model.o.parameters(),    'lr': 1e-3},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    loss_fn = nn.MSELoss()
    tester = Tester(device, loss_fn, log10=log10)
    trainer = Trainer(device, loss_fn, log10=log10)

    os.makedirs(os.path.join(out_dir, "Weight"), exist_ok=True)
    writer = SummaryWriter(os.path.join(out_dir, "logs"))
    file_prefix = os.path.join(out_dir, "Weight", f"Eitlem_{molType}_")

    print("start to train KKM (ensemble)...")
    for epoch in range(1, Epoch + 1):
        train_MAE, train_rmse, train_r2, loss_train = trainer.run(
            model, train_loader, optimizer, len(train_ds), f"{Iteration}iter epoch {epoch} train:"
        )
        MAE_dev, RMSE_dev, R2_dev, loss_dev = tester.test(
            model, valid_loader, len(valid_ds), desc=f"{Iteration}iter epoch {epoch} valid:"
        )
        scheduler.step()
        writer.add_scalars("loss", {'train_loss': loss_train, 'dev_loss': loss_dev}, epoch)
        writer.add_scalars("RMSE", {'train_RMSE': train_rmse, 'dev_RMSE': RMSE_dev}, epoch)
        writer.add_scalars("MAE",  {'train_MAE': train_MAE,   'dev_MAE': MAE_dev}, epoch)
        writer.add_scalars("R2",   {'train_R2': train_r2,     'dev_R2': R2_dev}, epoch)
        # 保存
        torch.save(model.state_dict(),
                   file_prefix + f"trainR2_{train_r2:.4f}_devR2_{R2_dev:.4f}_RMSE_{RMSE_dev:.4f}_MAE_{MAE_dev:.4f}.pt")

# --------- 取路径 ---------
def getPath(Type, TrainType, Iteration):
    train_info = f"Transfer-{TrainType}-{Type}-train-{Iteration}"
    file_model = f'../Results/{Type}/{train_info}/Weight/'
    fileList = sorted(os.listdir(file_model))  # 稳妥些：排序取最新
    return os.path.join(file_model, fileList[-1])

# --------- 主流程（与原来一致） ---------
def TransferLearing(Iterations, TrainType, log10=False, molType='MACCSKeys', device=None,
                    batch_size=200, num_workers=4, prefetch_factor=2):
    for iteration in range(1, Iterations + 1):
        if iteration == 1:
            kineticsTrainer(None, TrainType, 'KCAT', iteration, log10, molType, device,
                            batch_size, num_workers, prefetch_factor)
            kineticsTrainer(None, TrainType, 'KM',   iteration, log10, molType, device,
                            batch_size, num_workers, prefetch_factor)
        else:
            kkmPath = getPath('KKM', TrainType, iteration-1)
            kineticsTrainer(kkmPath, TrainType, 'KCAT', iteration, log10, molType, device,
                            batch_size, num_workers, prefetch_factor)
            kineticsTrainer(kkmPath, TrainType, 'KM',   iteration, log10, molType, device,
                            batch_size, num_workers, prefetch_factor)
        # 训练 KKM（ensemble）
        kcatPath = getPath('KCAT', TrainType, iteration)
        kmPath   = getPath('KM',   TrainType, iteration)
        KKMTrainer(kcatPath, kmPath, TrainType, iteration, log10, molType, device,
                   batch_size, num_workers, prefetch_factor)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--Iteration', type=int, required=True)
    parser.add_argument('-t', '--TrainType', type=str, required=True)
    parser.add_argument('-l', '--log10', type=bool, default=True)
    parser.add_argument('-m', '--molType', type=str, default='MACCSKeys')
    parser.add_argument('-d', '--device', type=int, required=True)
    # 新增可控参数（与 DataLoader 相关）
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    return parser.parse_args()

if __name__ == '__main__':
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    args = parse_args()
    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')
    print(f"use device {device}")
    TransferLearing(
        args.Iteration, args.TrainType, args.log10, args.molType, device,
        batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor
    )
