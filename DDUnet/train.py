import argparse
import csv
import glob
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import SimpleITK as sitk
import torch
import cv2
from monai.data import DataLoader
from monai.transforms import Compose, EnsureTyped, Lambdad, RandAffined, RandFlipd, RandGaussianNoised, RandScaleIntensityd, RandShiftIntensityd, Resized
from torch.utils.data import Dataset
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import scipy.signal


def parse_args(default_runs_root: str):
    p = argparse.ArgumentParser(description='2D segmentation training with 5-fold CV.')
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--runs_root', type=str, default=default_runs_root)
    p.add_argument('--num_folds', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_epochs', type=int, default=100)
    p.add_argument('--early_stop_patience', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--eta_min', type=float, default=1e-6)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--grad_clip', type=float, default=12.0)
    p.add_argument('--input_h', type=int, default=512)
    p.add_argument('--input_w', type=int, default=512)
    p.add_argument('--a_min', type=float, default=-160.0)
    p.add_argument('--a_max', type=float, default=240.0)
    p.add_argument('--only_fold', type=int, default=None)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--save_period', type=int, default=10)
    return p.parse_args()


class LossHistory:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.losses = []
        self.val_losses = []

    def append_loss(self, loss: float, val_loss: float):
        self.losses.append(loss)
        self.val_losses.append(val_loss)
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), "a", encoding="utf-8") as f:
            f.write(f"{loss}\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), "a", encoding="utf-8") as f:
            f.write(f"{val_loss}\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))
        plt.figure()
        plt.plot(iters, self.losses, "red", linewidth=2, label="train loss")
        plt.plot(iters, self.val_losses, "coral", linewidth=2, label="val loss")
        try:
            num = 5 if len(self.losses) < 25 else 15
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), "green", linestyle="--", linewidth=2, label="smooth train loss")
            plt.plot(iters, scipy.signal.savgol_filter(self.val_losses, num, 3), "#8B4513", linestyle="--", linewidth=2, label="smooth val loss")
        except Exception:
            pass
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def strip_nii_ext(filename: str) -> str:
    if filename.endswith('.nii.gz'): return filename[:-7]
    if filename.endswith('.nii'): return filename[:-4]
    return os.path.splitext(filename)[0]


def image_case_id(path: str) -> str:
    n = strip_nii_ext(os.path.basename(path)); return n[:-5] if n.endswith('_0000') else n


def collect_data_list(data_root: str) -> List[Dict[str, str]]:
    img_dir = os.path.join(data_root, 'imagesTr'); lbl_dir = os.path.join(data_root, 'labelsTr')
    imgs = sorted(glob.glob(os.path.join(img_dir, '*_0000.nii.gz')) + glob.glob(os.path.join(img_dir, '*_0000.nii')))
    data = []
    for ip in imgs:
        cid = image_case_id(ip)
        p1, p2 = os.path.join(lbl_dir, f'{cid}.nii.gz'), os.path.join(lbl_dir, f'{cid}.nii')
        lp = p1 if os.path.exists(p1) else p2
        if not os.path.exists(lp):
            continue
        data.append({'case_id': cid, 'image': ip, 'label': lp})
    if not data:
        raise RuntimeError('No paired train cases found.')
    return data


def make_kfold_splits(data_list: List[Dict[str, str]], num_folds: int, seed: int):
    idx = list(range(len(data_list))); rng = random.Random(seed); rng.shuffle(idx)
    sizes = [len(idx)//num_folds]*num_folds
    for i in range(len(idx)%num_folds): sizes[i] += 1
    folds = []; s = 0
    for fs in sizes: folds.append(idx[s:s+fs]); s += fs
    out = []
    for f in range(num_folds):
        val = folds[f]; tr = [j for i, fold in enumerate(folds) if i != f for j in fold]
        out.append(([data_list[i] for i in tr], [data_list[i] for i in val]))
    return out


class NiftiSliceDataset(Dataset):
    def __init__(self, case_files: List[Dict[str, str]], transform=None):
        self.transform = transform
        self.case_files = case_files
        self.cache = {}
        self.records = []
        for item in case_files:
            img = sitk.GetArrayFromImage(sitk.ReadImage(item['image']))
            lbl = sitk.GetArrayFromImage(sitk.ReadImage(item['label']))
            if img.shape != lbl.shape:
                raise ValueError(f"shape mismatch: {item['case_id']}")
            self.cache[item['case_id']] = (img.astype(np.float32), (lbl > 0).astype(np.float32))
            for z in range(img.shape[0]):
                self.records.append((item['case_id'], z))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        cid, z = self.records[idx]
        img, lbl = self.cache[cid]
        sample = {
            'image': img[z][None, ...],
            'label': lbl[z][None, ...],
            'case_id': cid,
            'slice_idx': z,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def get_transforms(args):
    def norm(x):
        x = np.clip(x, args.a_min, args.a_max)
        return (x - args.a_min) / max(args.a_max - args.a_min, 1e-6)

    train_tf = Compose([
        Lambdad(keys=['image'], func=norm),
        Resized(keys=['image', 'label'], spatial_size=(args.input_h, args.input_w), mode=('bilinear', 'nearest')),
        RandFlipd(keys=['image', 'label'], spatial_axis=0, prob=0.5),
        RandFlipd(keys=['image', 'label'], spatial_axis=1, prob=0.5),
        RandAffined(keys=['image', 'label'], prob=0.2, rotate_range=(0.1,), scale_range=(0.1, 0.1), mode=('bilinear', 'nearest'), padding_mode='border'),
        RandShiftIntensityd(keys=['image'], offsets=0.1, prob=0.3),
        RandScaleIntensityd(keys=['image'], factors=0.1, prob=0.3),
        RandGaussianNoised(keys=['image'], prob=0.15, mean=0.0, std=0.01),
        EnsureTyped(keys=['image', 'label'], dtype=torch.float32),
    ])

    val_tf = Compose([
        Lambdad(keys=['image'], func=norm),
        Resized(keys=['image', 'label'], spatial_size=(args.input_h, args.input_w), mode=('bilinear', 'nearest')),
        EnsureTyped(keys=['image', 'label'], dtype=torch.float32),
    ])
    return train_tf, val_tf


def dice_from_case_stats(case_stats: Dict[str, Tuple[float, float, float]]) -> float:
    dices = []
    for inter, pred_sum, gt_sum in case_stats.values():
        dice = (2.0 * inter + 1e-6) / (pred_sum + gt_sum + 1e-6)
        dices.append(float(dice))
    return float(np.mean(dices)) if dices else 0.0

from ddunet import DDUNet2D, DiceBCELoss


def patient_3d_dice_ddunet(
    model: torch.nn.Module,
    case_items: List[Dict[str, str]],
    args,
    device: torch.device,
    threshold: float = 0.5,
) -> float:
    model.eval()
    dices = []

    def norm(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, args.a_min, args.a_max)
        return (x - args.a_min) / max(args.a_max - args.a_min, 1e-6)

    with torch.no_grad():
        for item in case_items:
            vol = sitk.GetArrayFromImage(sitk.ReadImage(item["image"])).astype(np.float32)
            lab = (sitk.GetArrayFromImage(sitk.ReadImage(item["label"])) > 0).astype(np.float32)
            preds = []
            for z in range(vol.shape[0]):
                h0, w0 = vol[z].shape
                img = norm(vol[z])
                img = cv2.resize(img, (args.input_w, args.input_h), interpolation=cv2.INTER_LINEAR)
                img = img[None, None, ...]
                x = torch.from_numpy(img).float().to(device)
                p = model(x)[0, 0].detach().cpu().numpy()
                p_bin = (p > threshold).astype(np.float32)
                p_back = cv2.resize(p_bin, (w0, h0), interpolation=cv2.INTER_NEAREST)
                preds.append(p_back)
            pred_vol = np.stack(preds, axis=0)
            inter = float((pred_vol * lab).sum())
            den = float(pred_vol.sum() + lab.sum())
            dices.append((2.0 * inter + 1e-6) / (den + 1e-6))
    return float(np.mean(dices)) if dices else 0.0


def main():
    args = parse_args('./runs/DDUNet')
    os.makedirs(args.runs_root, exist_ok=True)
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    splits = make_kfold_splits(collect_data_list(args.data_root), args.num_folds, args.seed)
    all_results = []

    for fold_idx, (train_cases, val_cases) in enumerate(splits):
        if args.only_fold is not None and fold_idx != args.only_fold:
            continue
        fold_dir = os.path.join(args.runs_root, f'fold_{fold_idx}'); os.makedirs(fold_dir, exist_ok=True)

        train_tf, val_tf = get_transforms(args)
        train_loader = DataLoader(NiftiSliceDataset(train_cases, train_tf), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type=='cuda')
        val_loader = DataLoader(NiftiSliceDataset(val_cases, val_tf), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type=='cuda')

        model = DDUNet2D(in_channels=1, out_channels=1).to(device)
        criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.eta_min)

        log_path = os.path.join(fold_dir, 'epoch_metrics.csv')
        best_path = os.path.join(fold_dir, f"best_model_fold{fold_idx}.pth")
        latest_path = os.path.join(fold_dir, f"latest_model_fold{fold_idx}.pth")
        latest_state_path = os.path.join(fold_dir, f'latest_state_fold{fold_idx}.pt')
        loss_history = LossHistory(fold_dir)

        best_dice, best_epoch, no_improve, start_epoch = -1.0, -1, 0, 1
        if args.resume and os.path.exists(latest_state_path):
            st = torch.load(latest_state_path, map_location=device)
            model.load_state_dict(st['model']); optimizer.load_state_dict(st['optimizer']); scheduler.load_state_dict(st['scheduler'])
            best_dice = float(st.get('best_dice', -1.0)); best_epoch = int(st.get('best_epoch', -1)); no_improve = int(st.get('epochs_no_improve', 0)); start_epoch = int(st.get('epoch', 0)) + 1

        with open(log_path, 'a' if (args.resume and os.path.exists(log_path) and start_epoch > 1) else 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if f.tell() == 0: w.writerow(['epoch','train_loss','val_loss','val_dice_fg','lr','is_best'])
            print(f"[Fold {fold_idx}] start training: train_cases={len(train_cases)}, val_cases={len(val_cases)}, start_epoch={start_epoch}, max_epochs={args.max_epochs}")
            for epoch in range(start_epoch, args.max_epochs + 1):
                model.train(); tl = 0.0; ts = 0
                for b in train_loader:
                    ts += 1
                    x, y = b['image'].to(device), b['label'].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    p = model(x)
                    loss = criterion(p, y)
                    loss.backward()
                    if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step(); tl += loss.item()
                tl /= max(ts, 1)

                model.eval(); vl = 0.0; vs = 0
                with torch.no_grad():
                    for b in val_loader:
                        vs += 1
                        x, y = b['image'].to(device), b['label'].to(device)
                        p = model(x)
                        loss = criterion(p, y)
                        vl += loss.item()
                vl /= max(vs, 1)
                vd = patient_3d_dice_ddunet(model, val_cases, args, device, threshold=0.5)
                lr_now = optimizer.param_groups[0]['lr']; scheduler.step()

                is_best = int(vd > best_dice)
                w.writerow([epoch, f'{tl:.6f}', f'{vl:.6f}', f'{vd:.6f}', f'{lr_now:.8f}', is_best]); f.flush()
                loss_history.append_loss(tl, vl)

                if is_best:
                    best_dice, best_epoch, no_improve = vd, epoch, 0
                    torch.save(model.state_dict(), best_path)
                else:
                    no_improve += 1
                print(
                    f"[Fold {fold_idx}] Epoch {epoch:03d}/{args.max_epochs} | "
                    f"train_loss={tl:.6f} | val_loss={vl:.6f} | val_dice={vd:.6f} | "
                    f"lr={lr_now:.8f} | best_dice={best_dice:.6f}(ep{best_epoch}) | "
                    f"no_improve={no_improve}/{args.early_stop_patience}"
                )
                torch.save(model.state_dict(), latest_path)
                torch.save({'fold': fold_idx, 'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'best_dice': best_dice, 'best_epoch': best_epoch, 'epochs_no_improve': no_improve, 'args': vars(args)}, latest_state_path)
                if no_improve >= args.early_stop_patience:
                    print(f"[Fold {fold_idx}] Early stopping triggered at epoch {epoch}.")
                    break

            print(f"[Fold {fold_idx}] finished. best_epoch={best_epoch}, best_val_dice={best_dice:.6f}")

        all_results.append({'fold': fold_idx, 'best_epoch': best_epoch, 'best_dice_fg': best_dice, 'num_train': len(train_cases), 'num_val': len(val_cases), 'checkpoint': best_path})

    with open(os.path.join(args.runs_root, 'cv_results.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['fold','best_epoch','best_dice_fg','num_train','num_val','checkpoint'])
        w.writeheader()
        for r in sorted(all_results, key=lambda x: x['fold']):
            w.writerow({'fold': r['fold'], 'best_epoch': r['best_epoch'], 'best_dice_fg': f"{r['best_dice_fg']:.6f}", 'num_train': r['num_train'], 'num_val': r['num_val'], 'checkpoint': r['checkpoint']})


if __name__ == '__main__':
    main()
