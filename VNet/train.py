import argparse
import csv
import glob
import math
import os
import random

import numpy as np
import torch
from sklearn.model_selection import KFold

from monai.data import DataLoader, Dataset, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import VNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityRanged,
    SpatialPadd,
)
from monai.utils import set_determinism


def parse_args():
    parser = argparse.ArgumentParser(description="VNet 3D segmentation training (KFold).")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--runs_root", type=str, default="./runs/VNet")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--early_stop_patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_workers", type=int, default=4)
    parser.add_argument("--val_workers", type=int, default=2)
    parser.add_argument("--roi_x", type=int, default=128)
    parser.add_argument("--roi_y", type=int, default=128)
    parser.add_argument("--roi_z", type=int, default=96)
    parser.add_argument("--only_fold", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def set_all_seeds(seed):
    set_determinism(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_data_dicts(root):
    images = sorted(glob.glob(os.path.join(root, "imagesTr", "*_0000.nii.gz")))
    labels = sorted(glob.glob(os.path.join(root, "labelsTr", "*.nii.gz")))

    image_map = {os.path.basename(p)[:-12]: p for p in images}
    label_map = {os.path.basename(p)[:-7]: p for p in labels}

    ids = sorted(image_map.keys())
    if not ids:
        raise FileNotFoundError(f"No training images found under: {os.path.join(root, 'imagesTr')}")
    missing = [i for i in ids if i not in label_map]
    if missing:
        raise ValueError(f"Missing labels for {len(missing)} cases, e.g. {missing[:5]}")
    return [{"image": image_map[i], "label": label_map[i], "case_id": i} for i in ids]


def _bin(x):
    return (x > 0).astype(x.dtype)


def build_transforms(args):
    train = Compose([
        LoadImaged(["image", "label"]),
        EnsureChannelFirstd(["image", "label"]),
        ScaleIntensityRanged("image", -1000, 1000, 0, 1, True),
        Lambdad("label", _bin),
        SpatialPadd(["image", "label"], (args.roi_x, args.roi_y, args.roi_z)),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            pos=3,
            neg=1,
            num_samples=4,
        ),
        RandFlipd(["image", "label"], 0.5, 0),
        RandFlipd(["image", "label"], 0.5, 1),
        RandFlipd(["image", "label"], 0.5, 2),
        RandRotate90d(["image", "label"], 0.5, 3),
        EnsureTyped(["image", "label"]),
    ])

    val = Compose([
        LoadImaged(["image", "label"]),
        EnsureChannelFirstd(["image", "label"]),
        ScaleIntensityRanged("image", -1000, 1000, 0, 1, True),
        Lambdad("label", _bin),
        EnsureTyped(["image", "label"]),
    ])

    return train, val


def build_model(device: torch.device):
    return VNet(spatial_dims=3, in_channels=1, out_channels=1).to(device)


def train_one_fold(fold, train_files, val_files, args, device):
    fold_dir = os.path.join(args.runs_root, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    best_ckpt = os.path.join(fold_dir, f"best_model_fold{fold}.pth")
    last_state_path = os.path.join(fold_dir, f"last_state_fold{fold}.pt")
    epoch_log_path = os.path.join(fold_dir, "epoch_metrics.csv")

    train_tf, val_tf = build_transforms(args)

    train_loader = DataLoader(
        Dataset(train_files, train_tf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.train_workers,
        collate_fn=list_data_collate,
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        Dataset(val_files, val_tf),
        batch_size=1,
        num_workers=args.val_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    model = build_model(device)

    loss_fn = DiceCELoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(3, args.early_stop_patience // 3),
        min_lr=1e-6,
    )

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    roi = (args.roi_x, args.roi_y, args.roi_z)
    best_metric = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    start_epoch = 0

    if args.resume and os.path.exists(last_state_path):
        state = torch.load(last_state_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_metric = float(state["best_metric"])
        best_epoch = int(state["best_epoch"])
        epochs_no_improve = int(state["epochs_no_improve"])
        print(f"Resume fold {fold} from epoch {start_epoch + 1} | best_dice_fg={best_metric:.4f}")

    log_mode = "a" if (args.resume and os.path.exists(epoch_log_path) and start_epoch > 0) else "w"
    with open(epoch_log_path, log_mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if log_mode == "w":
            writer.writerow(["epoch", "train_loss", "val_dice_fg", "lr", "is_best"])

        for epoch in range(start_epoch, args.max_epochs):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                x = batch["image"].to(device)
                y = batch["label"].to(device)

                pred = model(x)
                loss = loss_fn(pred, y)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= max(len(train_loader), 1)

            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["image"].to(device)
                    y = batch["label"].to(device)

                    pred = sliding_window_inference(
                        x, roi, 1, model,
                        sw_device=device,
                        device=torch.device("cpu")
                    )

                    pred = torch.sigmoid(pred)
                    pred = (pred > 0.5).float()
                    dice_metric(pred, y)

            metric = float(dice_metric.aggregate().item())
            if math.isnan(metric):
                metric = 0.0
            dice_metric.reset()
            scheduler.step(metric)
            lr_now = optimizer.param_groups[0]["lr"]

            is_best = int(metric > best_metric)
            writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{metric:.6f}", f"{lr_now:.8f}", is_best])
            f.flush()

            print(f"Fold {fold} | Epoch {epoch + 1}/{args.max_epochs} | Loss {train_loss:.4f} | Val Dice(FG) {metric:.4f}")

            if is_best:
                best_metric = metric
                best_epoch = epoch + 1
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_ckpt)
            else:
                epochs_no_improve += 1

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                    "epochs_no_improve": epochs_no_improve,
                },
                last_state_path,
            )

            if epochs_no_improve >= args.early_stop_patience:
                print(f"Early stopping on fold {fold} at epoch {epoch + 1}")
                break

    return {
        "fold": fold,
        "best_epoch": best_epoch,
        "best_dice_fg": best_metric,
        "num_train": len(train_files),
        "num_val": len(val_files),
        "checkpoint": best_ckpt,
    }


def main():
    args = parse_args()
    set_all_seeds(args.seed)
    os.makedirs(args.runs_root, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = build_data_dicts(args.data_root)

    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    fold_results = []
    for fold, (tr, va) in enumerate(kf.split(data)):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        train_files = [data[i] for i in tr]
        val_files = [data[i] for i in va]
        result = train_one_fold(fold, train_files, val_files, args, device)
        fold_results.append(result)

    if not fold_results:
        raise ValueError("No fold was trained. Check --only_fold setting.")

    summary_path = os.path.join(args.runs_root, "cv_results.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "best_epoch", "best_dice_fg", "num_train", "num_val", "checkpoint"])
        for r in fold_results:
            writer.writerow(
                [
                    r["fold"],
                    r["best_epoch"],
                    f"{r['best_dice_fg']:.6f}",
                    r["num_train"],
                    r["num_val"],
                    r["checkpoint"],
                ]
            )

    scores = np.array([r["best_dice_fg"] for r in fold_results], dtype=np.float32)
    print(f"Saved summary: {summary_path}")
    print(f"Dice(FG) mean={scores.mean():.4f}, std={scores.std(ddof=0):.4f}")


if __name__ == "__main__":
    main()
