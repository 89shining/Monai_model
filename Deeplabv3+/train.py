import argparse
import csv
import glob
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

from nets.deeplabv3_plus import DeepLab


WINDOW_WIDTH = 400.0
WINDOW_LEVEL = 40.0
WINDOW_MIN = WINDOW_LEVEL - WINDOW_WIDTH / 2.0
WINDOW_MAX = WINDOW_LEVEL + WINDOW_WIDTH / 2.0


@dataclass
class CaseData:
    case_id: str
    image_path: str
    label_path: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def strip_nii_ext(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return os.path.splitext(name)[0]


def image_case_id(path: str) -> str:
    name = strip_nii_ext(os.path.basename(path))
    return name[:-5] if name.endswith("_0000") else name


def collect_cases(data_root: str) -> List[CaseData]:
    images = sorted(glob.glob(os.path.join(data_root, "imagesTr", "*.nii*")))
    labels = sorted(glob.glob(os.path.join(data_root, "labelsTr", "*.nii*")))
    if not images or not labels:
        raise FileNotFoundError("imagesTr/labelsTr not found or empty.")

    label_map = {strip_nii_ext(os.path.basename(p)): p for p in labels}
    cases: List[CaseData] = []
    for ip in images:
        cid = image_case_id(ip)
        if cid not in label_map:
            raise FileNotFoundError(f"Missing label for case: {cid}")
        cases.append(CaseData(case_id=cid, image_path=ip, label_path=label_map[cid]))
    return sorted(cases, key=lambda x: x.case_id)


def window_and_norm(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, WINDOW_MIN, WINDOW_MAX)
    arr = (arr - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN)
    return arr.astype(np.float32)


class SliceDataset(Dataset):
    def __init__(self, cases: List[CaseData], out_size: Tuple[int, int] = (512, 512), augment: bool = False):
        self.out_size = out_size
        self.augment = augment
        self.slices = []

        for case in cases:
            img = sitk.GetArrayFromImage(sitk.ReadImage(case.image_path)).astype(np.float32)
            lab = sitk.GetArrayFromImage(sitk.ReadImage(case.label_path)).astype(np.float32)
            if img.shape != lab.shape:
                raise ValueError(f"Shape mismatch: {case.case_id}")
            img = window_and_norm(img)
            lab = (lab > 0).astype(np.float32)
            for z in range(img.shape[0]):
                self.slices.append((img[z], lab[z]))

    def __len__(self):
        return len(self.slices)

    @staticmethod
    def rand(a=0.0, b=1.0) -> float:
        return np.random.rand() * (b - a) + a

    def augment_image_mask(self, img: np.ndarray, lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = self.out_size
        ih, iw = img.shape

        jitter = 0.3
        new_ar = (iw / max(ih, 1)) * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2.0)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        nw = max(nw, 1)
        nh = max(nh, 1)

        img_rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        lab_rs = cv2.resize(lab, (nw, nh), interpolation=cv2.INTER_NEAREST)

        canvas_img = np.zeros((h, w), dtype=np.float32)
        canvas_lab = np.zeros((h, w), dtype=np.float32)
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        sx0 = max(0, -dx)
        sy0 = max(0, -dy)
        sx1 = min(nw, w - dx)
        sy1 = min(nh, h - dy)
        tx0 = max(0, dx)
        ty0 = max(0, dy)
        tx1 = tx0 + (sx1 - sx0)
        ty1 = ty0 + (sy1 - sy0)
        if sx1 > sx0 and sy1 > sy0:
            canvas_img[ty0:ty1, tx0:tx1] = img_rs[sy0:sy1, sx0:sx1]
            canvas_lab[ty0:ty1, tx0:tx1] = lab_rs[sy0:sy1, sx0:sx1]
        img, lab = canvas_img, canvas_lab

        if self.rand() < 0.5:
            img = np.fliplr(img).copy()
            lab = np.fliplr(lab).copy()

        if self.rand() < 0.25:
            img = cv2.GaussianBlur(img, (5, 5), 0)

        if self.rand() < 0.25:
            center = (w // 2, h // 2)
            rotation = np.random.randint(-10, 11)
            m = cv2.getRotationMatrix2D(center, -rotation, 1.0)
            img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0)
            lab = cv2.warpAffine(lab, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0.0)

        return img.astype(np.float32), (lab > 0.5).astype(np.float32)

    def __getitem__(self, idx):
        img, lab = self.slices[idx]
        if self.augment:
            img, lab = self.augment_image_mask(img, lab)
            x = torch.from_numpy(img)[None, ...]
            y = torch.from_numpy(lab)[None, ...]
        else:
            x = torch.from_numpy(img)[None, ...]
            y = torch.from_numpy(lab)[None, ...]
            x = F.interpolate(x[None, ...], size=self.out_size, mode="bilinear", align_corners=False)[0]
            y = F.interpolate(y[None, ...], size=self.out_size, mode="nearest")[0]
            y = (y > 0.5).float()

        x = x.repeat(3, 1, 1)
        return x, y


def dice_score_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (den + eps)
    return dice.mean().item()


def dice_bce_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    target = target.float()
    probs = torch.sigmoid(logits)
    p = probs.contiguous().view(probs.shape[0], -1)
    t = target.contiguous().view(target.shape[0], -1)
    inter = (p * t).sum(dim=1)
    den = p.sum(dim=1) + t.sum(dim=1)
    dice_loss = 1.0 - ((2.0 * inter + eps) / (den + eps))
    dice_loss = dice_loss.mean()

    bce_raw = torch.nn.functional.binary_cross_entropy(probs, target)
    bce_norm = bce_raw / (1.0 + bce_raw)
    return 0.5 * dice_loss + 0.5 * bce_norm


def save_loss_plot(csv_path: str, out_png: str) -> None:
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return
        epochs = [int(r["epoch"]) for r in rows]
        train_loss = [float(r["train_loss"]) for r in rows]
        val_loss = [float(r["val_loss"]) for r in rows]
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, train_loss, label="train_loss")
        plt.plot(epochs, val_loss, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Fold Loss Curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
    except Exception as e:
        print(f"[Warn] save_loss_plot failed: {e}", flush=True)


def build_model(backbone: str) -> nn.Module:
    model_dir = os.path.join(os.path.dirname(__file__), "model_data")
    pretrain_file = {
        "xception": "xception_pytorch_imagenet.pth",
        "mobilenet": "mobilenet_v2.pth.tar",
    }[backbone]
    has_local_pretrain = os.path.exists(os.path.join(model_dir, pretrain_file))
    model = DeepLab(num_classes=1, backbone=backbone, pretrained=has_local_pretrain, downsample_factor=16)
    if not has_local_pretrain:
        print(f"[Info] Local pretrained not found ({pretrain_file}), skip pretrained init.")
    return model


def run_fold(args, fold: int, train_cases: List[CaseData], val_cases: List[CaseData], device: torch.device) -> Tuple[int, float]:
    fold_dir = os.path.join(args.runs_root, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    latest_path = os.path.join(fold_dir, "latest.pth")
    best_path = os.path.join(fold_dir, f"best_model_fold{fold}.pth")
    metrics_csv = os.path.join(fold_dir, "epoch_metrics.csv")

    train_ds = SliceDataset(train_cases, out_size=(512, 512), augment=True)
    val_ds = SliceDataset(val_cases, out_size=(512, 512), augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = build_model(args.backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_dice = -1.0
    best_epoch = -1
    bad_epochs = 0

    if args.resume and os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_dice = float(ckpt["best_dice"])
        best_epoch = int(ckpt["best_epoch"])
        bad_epochs = int(ckpt["bad_epochs"])
        print(f"[Resume] fold={fold}, start_epoch={start_epoch}")

    if start_epoch == 0:
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_loss", "val_dice_fg"])

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = dice_bce_loss_from_logits(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                val_loss += dice_bce_loss_from_logits(logits, y).item()
                val_dice += dice_score_from_logits(logits, y)
        val_loss /= max(len(val_loader), 1)
        val_dice /= max(len(val_loader), 1)
        scheduler.step()

        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_dice:.6f}"])

        improved = val_dice > best_dice
        if improved:
            best_dice = val_dice
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_dice": best_dice}, best_path)
        else:
            bad_epochs += 1

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_dice": best_dice,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
            },
            latest_path,
        )

        print(
            f"Fold {fold} | Epoch {epoch + 1}/{args.epochs} | "
            f"TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | ValDice {val_dice:.4f} | Best {best_dice:.4f}",
            flush=True,
        )

        if bad_epochs >= args.early_stop:
            print(f"[EarlyStop] fold={fold}, epoch={epoch}")
            break
    save_loss_plot(metrics_csv, os.path.join(fold_dir, "loss_curve.png"))

    return best_epoch, best_dice


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--runs_root", type=str, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--early_stop", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone", type=str, default="xception", choices=["xception", "mobilenet"])
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = collect_cases(args.data_root)

    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    split = list(kf.split(cases))[args.fold]
    train_idx, val_idx = split
    train_cases = [cases[i] for i in train_idx]
    val_cases = [cases[i] for i in val_idx]

    os.makedirs(args.runs_root, exist_ok=True)
    fold_dir = os.path.join(args.runs_root, f"fold_{args.fold}")
    os.makedirs(fold_dir, exist_ok=True)

    with open(os.path.join(fold_dir, "train_cases.txt"), "w", encoding="utf-8") as f:
        for c in train_cases:
            f.write(c.case_id + "\n")
    with open(os.path.join(fold_dir, "val_cases.txt"), "w", encoding="utf-8") as f:
        for c in val_cases:
            f.write(c.case_id + "\n")

    best_epoch, best_dice = run_fold(args, args.fold, train_cases, val_cases, device)
    with open(os.path.join(fold_dir, "fold_done.flag"), "w", encoding="utf-8") as f:
        f.write(f"best_epoch={best_epoch},best_dice={best_dice:.6f}\n")


if __name__ == "__main__":
    main()
