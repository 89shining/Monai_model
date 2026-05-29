import argparse
import csv
import glob
import os
import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.data import DataLoader, Dataset, decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
)
from sklearn.model_selection import KFold

from vnet import VNet


def parse_args():
    parser = argparse.ArgumentParser(description="3D VNet training with K-fold CV.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--runs_root", type=str, default="./runs/VNet")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=12.0)

    parser.add_argument("--roi_x", type=int, default=256)
    parser.add_argument("--roi_y", type=int, default=256)
    parser.add_argument("--roi_z", type=int, default=128)
    parser.add_argument("--strong_aug", action="store_true")

    parser.add_argument("--a_min", type=float, default=-160.0)
    parser.add_argument("--a_max", type=float, default=240.0)

    parser.add_argument("--pixdim_x", type=float, default=1.0)
    parser.add_argument("--pixdim_y", type=float, default=1.0)
    parser.add_argument("--pixdim_z", type=float, default=1.0)
    parser.add_argument("--axcodes", type=str, default="RAS")

    parser.add_argument("--only_fold", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def binarize_label(x):
    if isinstance(x, torch.Tensor):
        return (x > 0).float()
    return (x > 0).astype(np.float32)


class VNet3DSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = VNet(elu=True, nll=False, logits=True)

    def forward(self, x):
        n, _, d, h, w = x.shape
        out = self.net(x)
        out = out.view(n, d, h, w, 2).permute(0, 4, 1, 2, 3).contiguous()
        return out[:, 1:2]


def strip_nii_ext(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return os.path.splitext(filename)[0]


def image_case_id(path: str) -> str:
    name = strip_nii_ext(os.path.basename(path))
    return name[:-5] if name.endswith("_0000") else name


def collect_data_list(data_root: str) -> List[Dict[str, str]]:
    image_dir = os.path.join(data_root, "imagesTr")
    label_dir = os.path.join(data_root, "labelsTr")
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")) + glob.glob(os.path.join(image_dir, "*.nii")))
    if not image_paths:
        raise RuntimeError(f"No NIfTI files found in: {image_dir}")

    data_list = []
    for image_path in image_paths:
        case_id = image_case_id(image_path)
        p1 = os.path.join(label_dir, f"{case_id}.nii.gz")
        p2 = os.path.join(label_dir, f"{case_id}.nii")
        if os.path.exists(p1):
            label_path = p1
        elif os.path.exists(p2):
            label_path = p2
        else:
            raise FileNotFoundError(f"Label not found for case: {case_id}")
        data_list.append({"case_id": case_id, "image": image_path, "label": label_path})
    return data_list


def make_kfold_splits(data_list: List[Dict[str, str]], num_folds: int, seed: int):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits = []
    for train_idx, val_idx in kf.split(data_list):
        splits.append(([data_list[i] for i in train_idx], [data_list[i] for i in val_idx]))
    return splits


def dice_bce_loss_from_probs(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    target = target.float()
    probs = torch.clamp(probs, min=1e-6, max=1.0 - 1e-6)
    p = probs.contiguous().view(probs.shape[0], -1)
    t = target.contiguous().view(target.shape[0], -1)
    inter = (p * t).sum(dim=1)
    den = p.sum(dim=1) + t.sum(dim=1)
    dice_loss = 1.0 - ((2.0 * inter + eps) / (den + eps))
    dice_loss = dice_loss.mean()

    bce_raw = F.binary_cross_entropy(probs, target)
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


def get_transforms(args):
    roi_size = (args.roi_x, args.roi_y, args.roi_z)
    pixdim = (args.pixdim_x, args.pixdim_y, args.pixdim_z)

    train_list = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes=args.axcodes),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=0.0, b_max=1.0, clip=True),
        Lambdad(keys=["label"], func=binarize_label),
        Resized(keys=["image", "label"], spatial_size=roi_size, mode=("trilinear", "nearest")),
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ]

    if args.strong_aug:
        train_list.extend([
            RandAffined(
                keys=["image", "label"], prob=0.2,
                rotate_range=(0.1, 0.1, 0.05),
                scale_range=(0.1, 0.1, 0.05),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
            RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
        ])
    else:
        train_list.extend([
            RandAffined(
                keys=["image", "label"], prob=0.1,
                rotate_range=(0.05, 0.05, 0.02),
                scale_range=(0.05, 0.05, 0.02),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
            RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.1),
            RandScaleIntensityd(keys=["image"], factors=0.05, prob=0.1),
            RandGaussianNoised(keys=["image"], prob=0.05, mean=0.0, std=0.005),
        ])

    train_transforms = Compose(train_list)

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes=args.axcodes),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=0.0, b_max=1.0, clip=True),
        Lambdad(keys=["label"], func=binarize_label),
        Resized(keys=["image", "label"], spatial_size=roi_size, mode=("trilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ])
    return train_transforms, val_transforms


def train_one_epoch(model, loader, optimizer, loss_func, device, args, fold_idx: int, epoch: int, max_epochs: int):
    model.train()
    epoch_loss = 0.0
    steps = 0
    total_steps = len(loader)
    for batch_data in loader:
        steps += 1
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        probs = torch.sigmoid(logits)
        loss = loss_func(probs, labels)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        epoch_loss += loss.item()
        print(
            f"Fold {fold_idx} | Epoch {epoch}/{max_epochs} | "
            f"Step {steps}/{total_steps} | TrainLoss {loss.item():.4f}",
            flush=True,
        )
    return epoch_loss / max(steps, 1)


@torch.no_grad()
def validate(model, loader, loss_func, dice_metric, post_pred, post_label, device):
    model.eval()
    dice_metric.reset()
    val_loss = 0.0
    steps = 0

    for batch_data in loader:
        steps += 1
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        logits = model(images)
        probs = torch.sigmoid(logits)
        val_loss += loss_func(probs, labels).item()
        preds_list = [post_pred(x) for x in decollate_batch(logits)]
        labels_list = [post_label(x) for x in decollate_batch(labels)]
        dice_metric(y_pred=preds_list, y=labels_list)

    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return val_loss / max(steps, 1), mean_dice


def save_case_list(file_list, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        for item in file_list:
            f.write(item["case_id"] + "\n")


def train_one_fold(fold_idx: int, train_files, val_files, args, device):
    fold_dir = os.path.join(args.runs_root, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    print("=" * 80, flush=True)
    print(f"Start Fold {fold_idx}", flush=True)
    print(f"Train cases: {len(train_files)} | Val cases: {len(val_files)}", flush=True)
    print("=" * 80, flush=True)
    save_case_list(train_files, os.path.join(fold_dir, "train_cases.txt"))
    save_case_list(val_files, os.path.join(fold_dir, "val_cases.txt"))

    train_tf, val_tf = get_transforms(args)
    train_ds = Dataset(data=train_files, transform=train_tf)
    val_ds = Dataset(data=val_files, transform=val_tf)

    g = torch.Generator()
    g.manual_seed(args.seed + fold_idx)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
    )

    model = VNet3DSeg().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.eta_min)
    loss_func = dice_bce_loss_from_probs

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete(threshold=0.5)])

    log_path = os.path.join(fold_dir, "epoch_metrics.csv")
    best_path = os.path.join(fold_dir, f"best_model_fold{fold_idx}.pth")
    last_state_path = os.path.join(fold_dir, f"last_state_fold{fold_idx}.pt")

    best_dice = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    start_epoch = 1

    if args.resume and os.path.exists(last_state_path):
        state = torch.load(last_state_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best_dice = float(state.get("best_dice", -1.0))
        best_epoch = int(state.get("best_epoch", -1))
        epochs_no_improve = int(state.get("epochs_no_improve", 0))
        start_epoch = int(state.get("epoch", 0)) + 1
        print(f"Resume fold {fold_idx} from epoch {start_epoch}")

    log_mode = "a" if (args.resume and os.path.exists(log_path) and start_epoch > 1) else "w"
    with open(log_path, log_mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if log_mode == "w":
            writer.writerow(["epoch", "train_loss", "val_loss", "val_dice_fg", "lr", "is_best"])

        for epoch in range(start_epoch, args.max_epochs + 1):
            print(f"[Fold {fold_idx}] Epoch {epoch}/{args.max_epochs} start...", flush=True)
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_func, device, args, fold_idx, epoch, args.max_epochs)
            print(f"[Fold {fold_idx}] Epoch {epoch}/{args.max_epochs} validation...", flush=True)
            val_loss, val_dice = validate(model, val_loader, loss_func, dice_metric, post_pred, post_label, device)

            lr_now = optimizer.param_groups[0]["lr"]
            scheduler.step()

            is_best = int(val_dice > best_dice)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_dice:.6f}", f"{lr_now:.8f}", is_best])
            f.flush()

            print(
                f"Fold {fold_idx} | Epoch {epoch}/{args.max_epochs} | "
                f"Train {train_loss:.4f} | ValLoss {val_loss:.4f} | ValDice {val_dice:.4f} | LR {lr_now:.2e}"
            )

            if is_best:
                best_dice = val_dice
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(
                    {
                        "fold": fold_idx,
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_dice": best_dice,
                        "best_epoch": best_epoch,
                        "args": vars(args),
                    },
                    best_path,
                )
            else:
                epochs_no_improve += 1

            torch.save(
                {
                    "fold": fold_idx,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_dice": best_dice,
                    "best_epoch": best_epoch,
                    "epochs_no_improve": epochs_no_improve,
                    "args": vars(args),
                },
                last_state_path,
            )

            if epochs_no_improve >= args.early_stop_patience:
                print(f"Early stopping on fold {fold_idx} at epoch {epoch}")
                break

    save_loss_plot(log_path, os.path.join(fold_dir, "loss_curve.png"))

    return {
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "best_dice_fg": best_dice,
        "num_train": len(train_files),
        "num_val": len(val_files),
        "checkpoint": best_path,
    }


def main():
    args = parse_args()
    os.makedirs(args.runs_root, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_list = collect_data_list(args.data_root)
    if len(data_list) < args.num_folds:
        raise RuntimeError(f"Number of cases {len(data_list)} is smaller than num_folds {args.num_folds}.")

    splits = make_kfold_splits(data_list, args.num_folds, args.seed)
    all_results = []

    for fold_idx, (train_files, val_files) in enumerate(splits):
        if args.only_fold is not None and fold_idx != args.only_fold:
            continue
        result = train_one_fold(fold_idx, train_files, val_files, args, device)
        all_results.append(result)

    if not all_results:
        raise ValueError("No fold was trained. Check --only_fold setting.")

    summary_path = os.path.join(args.runs_root, "cv_results.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "best_epoch", "best_dice_fg", "num_train", "num_val", "checkpoint"])
        writer.writeheader()
        for r in sorted(all_results, key=lambda x: x["fold"]):
            writer.writerow(
                {
                    "fold": r["fold"],
                    "best_epoch": r["best_epoch"],
                    "best_dice_fg": f"{r['best_dice_fg']:.6f}",
                    "num_train": r["num_train"],
                    "num_val": r["num_val"],
                    "checkpoint": r["checkpoint"],
                }
            )

    scores = np.array([r["best_dice_fg"] for r in all_results], dtype=np.float32)
    print("\n===== Cross-Validation Summary =====")
    print(f"Saved summary: {summary_path}")
    print(f"Dice(FG) mean={scores.mean():.4f}, std={scores.std(ddof=0):.4f}")


if __name__ == "__main__":
    main()
