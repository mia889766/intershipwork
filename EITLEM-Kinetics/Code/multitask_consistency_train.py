# -*- coding: utf-8 -*-
"""
Multi-task joint training with physical consistency for EITLEM-Kinetics.

- 数据：三种 split（KCAT/KM/KKM），标签均在 log10 空间
- 模型：并列复用现有三类单任务模型（EitlemKcatPredictor/EitlemKmPredictor/EitlemKKmPredictor）
- 损失：
    * 监督：按 batch 所属 split 分别用 L_kcat / L_km / L_kkm（MSE）
    * 一致性：L_cons = | yhat_kcat - yhat_km - yhat_kkm |_1  （对所有 batch 都加）
- 评估：分别在三种 split 验证集上计算 (MAE/RMSE/R2/PCC)
- 选模：以 KKM 验证 R2 最优保存（可按需改为综合指标）
- 优化：AdamW, lr=1e-4, weight_decay=1e-5, batch_size=128, prefetch_factor=2, num_workers=4
"""
import os
import argparse
import multiprocessing as mp
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from dataset import EitlemDataSet, EitlemDataLoader  # Batch fields: .x, .pro_emb, .value
from eitlem_utils import get_pair_info
from KCM import EitlemKcatPredictor                  # returns scalar (B,)
from KMP import EitlemKmPredictor                    # returns scalar (B,)
from KKMP import EitlemKKmPredictor                  # returns scalar (B,)


# ---------------- utils ----------------
def seed_all(seed: int = 2024):
    import random, numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(pairs, seq_dir, smi_idx, batch_size, num_workers, prefetch_factor,
                pin_memory, log10, molType, shuffle):
    ds = EitlemDataSet(pairs, seq_dir, smi_idx, 1024, 4, log10, molType)
    dl = EitlemDataLoader(
        data=ds, batch_size=batch_size, shuffle=shuffle, drop_last=False,
        num_workers=num_workers,
        prefetch_factor=(prefetch_factor if num_workers > 0 else None),
        persistent_workers=(num_workers > 0),
        pin_memory=pin_memory,
    )
    return ds, dl


# ---------------- joint model (wrap three existing predictors) ----------------
class JointTripleModel(nn.Module):
    """
    外壳模型：内部并列复用三类单任务预测器，实现一次前向拿到三头标量。
    """
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


# ---------------- losses / metrics ----------------
def consistency_loss(y_kcat, y_km, y_kkm):
    # all values are in log-space
    return (y_kcat - y_km - y_kkm).abs().mean()


@torch.no_grad()
def evaluate_head(model, loader, device, which: str):
    """
    对指定头（KCAT/KM/KKM）在对应 split 验证集上评估 MAE/RMSE/R2/PCC。
    """
    model.eval()
    ys, yh = [], []
    for batch in loader:
        batch = batch.to(device)
        #y = batch.value.float().squeeze(-1)  # (B,)
        y = batch.value.float().reshape(-1)
        y_kcat, y_km, y_kkm = model(batch)
        if which == "KCAT":
            #y_hat = y_kcat
            y_hat = y_kcat.reshape(-1)
        elif which == "KM":
            y_hat = y_km.reshape(-1)
        elif which == "KKM":
            y_hat = y_kkm.reshape(-1)
        else:
            raise ValueError(which)
        ys.append(y.detach().cpu())
        yh.append(y_hat.detach().cpu())

    if not ys:
        return float("nan"), float("nan"), float("nan"), float("nan")

    y_true = torch.cat(ys)
    y_pred = torch.cat(yh)

    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()
    var_y = torch.var(y_true, unbiased=False).item()
    r2 = 1.0 - (torch.mean((y_pred - y_true) ** 2).item() / (var_y + 1e-12))

    # PCC
    x = (y_pred - torch.mean(y_pred)) / (torch.std(y_pred) + 1e-12)
    t = (y_true - torch.mean(y_true)) / (torch.std(y_true) + 1e-12)
    pcc = torch.mean(x * t).item()
    return mae, rmse, r2, pcc


def train_epoch(model, opt, device,
                dl_kcat, dl_km, dl_kkm,
                w_kcat=1.0, w_km=1.0, w_kkm=1.0, lam_cons=0.3,
                loss_sup=nn.MSELoss(),scaler: GradScaler = None):
    """
    一个 epoch：轮转三种 DataLoader。
    - 当前 batch 属于哪个任务，就用对应监督项 + 一致性项；
    - 一致性项对所有 batch 都施加（无需额外标签）。
    """
    print("[DBG] enter train loop", flush=True)
    model.train()
    it_kcat, it_km, it_kkm = iter(dl_kcat), iter(dl_km), iter(dl_kkm)
    sched = [("KCAT", it_kcat, w_kcat), ("KM", it_km, w_km), ("KKM", it_kkm, w_kkm)]

    total_loss, total_n = 0.0, 0
    steps = max(len(dl_kcat), len(dl_km), len(dl_kkm))
    for _ in range(steps):
        for tag, it_, w in sched:
            try:
                batch = next(it_)
            except StopIteration:
                continue
            batch = batch.to(device)
            #y_true = batch.value.float().squeeze(-1)
            y_true = batch.value.float().reshape(-1)

            # ---- AMP 前向与损失 ----
            use_amp = (scaler is not None) and (device.type == "cuda")
            with autocast(enabled=use_amp):
                y_kcat, y_km, y_kkm = model(batch)
                y_kcat = y_kcat.reshape(-1)
                y_km   = y_km.reshape(-1)
                y_kkm  = y_kkm.reshape(-1)
                l_cons = consistency_loss(y_kcat, y_km, y_kkm)
                if tag == "KCAT":
                    l_sup = loss_sup(y_kcat, y_true)
                elif tag == "KM":
                    l_sup = loss_sup(y_km, y_true)
                else:
                    l_sup = loss_sup(y_kkm, y_true)

                loss = w * l_sup + lam_cons * l_cons

            # ---- AMP 后向 ----
            opt.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            

            bs = y_true.numel()
            total_loss += loss.item() * bs
            total_n += bs

    return total_loss / max(1, total_n)


# ---------------- args & main ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--molType", type=str, default="MACCSKeys",
                    choices=["MACCSKeys", "ECFP", "RDKIT", "UniMol"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_cons", type=float, default=0.3)
    ap.add_argument("--w_kcat", type=float, default=1.0)
    ap.add_argument("--w_km", type=float, default=1.0)
    ap.add_argument("--w_kkm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--outdir", type=str, default="../Results/JointMTL")
    # log10 flag（避免 type=bool 坑）
    ap.add_argument("--log10", dest="log10", action="store_true")
    ap.add_argument("--no-log10", dest="log10", action="store_false")
    ap.set_defaults(log10=True)
    # dataloader workers
    ap.add_argument("--num_workers", type=int, default=30)
    ap.add_argument("--prefetch_factor", type=int, default=5)
    return ap.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device} molType={args.molType} log10={args.log10}")

    # ---------- paths ----------
    data_root = "../Data/"
    seq_dir   = "../Data/Feature/esm1v_t33_650M_UR90S_1_embeding_1280/"
    smi_idx   = "../Data/Feature/index_smiles"

    # ---------- split pairs ----------
    kcat_train, kcat_test = get_pair_info(data_root, "KCAT",False)  # log10(kcat)
    km_train,   km_test   = get_pair_info(data_root, "KM",False)    # log10(Km)
    kkm_train,  kkm_test  = get_pair_info(data_root, "KKM")   # log10(kcat/Km)

    # ---------- dataloaders ----------
    cpu_n = max(1, (mp.cpu_count() or 1) - 1)
    nworkers = min(args.num_workers, cpu_n)
    pin = (device.type == "cuda")

    _, dl_kcat_tr = make_loader(kcat_train, seq_dir, smi_idx, args.batch_size, nworkers, args.prefetch_factor, pin, args.log10, args.molType, shuffle=True)
    _, dl_km_tr   = make_loader(km_train,   seq_dir, smi_idx, args.batch_size, nworkers, args.prefetch_factor, pin, args.log10, args.molType, shuffle=True)
    _, dl_kkm_tr  = make_loader(kkm_train,  seq_dir, smi_idx, args.batch_size, nworkers, args.prefetch_factor, pin, args.log10, args.molType, shuffle=True)

    _, dl_kcat_te = make_loader(kcat_test, seq_dir, smi_idx, args.batch_size, nworkers, args.prefetch_factor, pin, args.log10, args.molType, shuffle=False)
    _, dl_km_te   = make_loader(km_test,   seq_dir, smi_idx, args.batch_size, nworkers, args.prefetch_factor, pin, args.log10, args.molType, shuffle=False)
    _, dl_kkm_te  = make_loader(kkm_test,  seq_dir, smi_idx, args.batch_size, nworkers, args.prefetch_factor, pin, args.log10, args.molType, shuffle=False)


    # ---------- model ----------
    
    mol_dim = 167 if args.molType == "MACCSKeys" else 1024  # 按你的特征维度定义
    model = JointTripleModel(mol_dim=mol_dim, hidden_dim=512, protein_dim=1280, layer=10, dropout=0.5, att_layer=10).to(device)

    # ---------- optim / sched / log ----------
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)
    scaler = GradScaler(enabled=(device.type == "cuda"))



    run_name = f"MTL-{args.molType}-lam{args.lambda_cons}"
    log_dir = os.path.join(args.outdir, run_name, "logs")
    ckpt_dir = os.path.join(args.outdir, run_name, "Weight")
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    print("[INFO] start multi-task training with consistency ...")
    best_r2_kkm = -1e9
    for ep in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, opt, device,
            dl_kcat_tr, dl_km_tr, dl_kkm_tr,
            w_kcat=args.w_kcat, w_km=args.w_km, w_kkm=args.w_kkm,
            lam_cons=args.lambda_cons,
            loss_sup=nn.MSELoss(),
            scaler=scaler
        )

        mae_kcat, rmse_kcat, r2_kcat, pcc_kcat = evaluate_head(model, dl_kcat_te, device, "KCAT")
        mae_km,   rmse_km,   r2_km,   pcc_km   = evaluate_head(model, dl_km_te,   device, "KM")
        mae_kkm,  rmse_kkm,  r2_kkm,  pcc_kkm  = evaluate_head(model, dl_kkm_te,  device, "KKM")
        scheduler.step()

        # tensorboard logs
        writer.add_scalar("loss/train", train_loss, ep)
        writer.add_scalars("dev/KCAT", {"MAE": mae_kcat, "RMSE": rmse_kcat, "R2": r2_kcat, "PCC": pcc_kcat}, ep)
        writer.add_scalars("dev/KM",   {"MAE": mae_km,   "RMSE": rmse_km,   "R2": r2_km,   "PCC": pcc_km},   ep)
        writer.add_scalars("dev/KKM",  {"MAE": mae_kkm,  "RMSE": rmse_kkm,  "R2": r2_kkm,  "PCC": pcc_kkm},  ep)

        # save by best KKM-R2（可改为综合指标）
        if r2_kkm > best_r2_kkm:
            best_r2_kkm = r2_kkm
            tag = f"best_e{ep}_kkmR2_{r2_kkm:.4f}_kcatR2_{r2_kcat:.4f}_kmR2_{r2_km:.4f}"
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"EITLEM_JointMTL_{tag}.pt"))

        print(f"[Epoch {ep:03d}] train_loss={train_loss:.4f} | "
              f"KCAT R2={r2_kcat:.4f} KM R2={r2_km:.4f} KKM R2={r2_kkm:.4f}")

    print(f"[INFO] finished. best KKM R2 = {best_r2_kkm:.4f}")
    writer.flush(); writer.close()


if __name__ == "__main__":
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
