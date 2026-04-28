import argparse
import csv
import glob
import math
import os
import random

import numpy as np
import torch
from sklearn.model_selection import KFold

from monai.data import DataLoader, Dataset, pad_list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import AttentionUnet
from monai.transforms import (
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityd,
)
from monai.utils import set_determinism


def parse_args():
    parser = argparse.ArgumentParser(description="AttentionUNet 3D segmentation training (KFold).")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root directory.")
    parser.add_argument("--runs_root", type=str, default="./runs/AttentionUNet", help="Output root.")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of KFold splits.")
    parser.add_argument("--max_epochs", type=int, default=200, help="Maximum epochs.")
    parser.add_argument("--early_stop_patience", type=int, default=30, help="Early-stop patience.")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--train_workers", type=int, default=4, help="Num workers for training loader.")
    parser.add_argument("--val_workers", type=int, default=2, help="Num workers for validation loader.")
    parser.add_argument("--roi_x", type=int, default=96, help="Sliding-window ROI x.")
    parser.add_argument("--roi_y", type=int, default=96, help="Sliding-window ROI y.")
    parser.add_argument("--roi_z", type=int, default=96, help="Sliding-window ROI z.")
    parser.add_argument(
        "--only_fold",
        type=int,
        default=None,
        help="Train only one fold index (e.g. 0~4).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last_state checkpoint of a fold if available.",
    )
    return parser.parse_args()


def set_all_seeds(seed: int):
    set_determinism(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def strip_nii_ext(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return os.path.splitext(filename)[0]


def image_case_id(path: str) -> str:
    name = strip_nii_ext(os.path.basename(path))
    if name.endswith("_0000"):
        return name[:-5]
    return name


def label_case_id(path: str) -> str:
    return strip_nii_ext(os.path.basename(path))


def build_data_dicts(data_root: str):
    image_patterns = [
        os.path.join(data_root, "imagesTr", "*_0000.nii.gz"),
        os.path.join(data_root, "imagesTr", "*_0000.nii"),
    ]
    label_patterns = [
        os.path.join(data_root, "labelsTr", "*.nii.gz"),
        os.path.join(data_root, "labelsTr", "*.nii"),
    ]

    images = []
    labels = []
    for p in image_patterns:
        images.extend(glob.glob(p))
    for p in label_patterns:
        labels.extend(glob.glob(p))

    images = sorted(set(images))
    labels = sorted(set(labels))

    if not images:
        raise FileNotFoundError(f"No training images found under: {os.path.join(data_root, 'imagesTr')}")
    if not labels:
        raise FileNotFoundError(f"No training labels found under: {os.path.join(data_root, 'labelsTr')}")

    image_map = {image_case_id(p): p for p in images}
    label_map = {label_case_id(p): p for p in labels}

    image_ids = set(image_map.keys())
    label_ids = set(label_map.keys())

    missing_labels = sorted(image_ids - label_ids)
    missing_images = sorted(label_ids - image_ids)
    if missing_labels or missing_images:
        msg = [
            "Image/label case IDs are inconsistent.",
            f"Missing labels for {len(missing_labels)} image IDs.",
            f"Missing images for {len(missing_images)} label IDs.",
        ]
        if missing_labels:
            msg.append(f"Example missing labels: {missing_labels[:5]}")
        if missing_images:
            msg.append(f"Example missing images: {missing_images[:5]}")
        raise ValueError(" ".join(msg))

    case_ids = sorted(image_ids)
    return [{"image": image_map[cid], "label": label_map[cid], "case_id": cid} for cid in case_ids]


def build_transforms():
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys="image"),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            DivisiblePadd(keys=["image", "label"], k=16),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys="image"),
            DivisiblePadd(keys=["image", "label"], k=16),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    return train_transforms, val_transforms


def build_model(device: torch.device):
    model = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    ).to(device)
    return model


def train_one_fold(fold: int, train_files, val_files, args, device: torch.device):
    fold_dir = os.path.join(args.runs_root, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    checkpoint_path = os.path.join(fold_dir, f"best_model_fold{fold}.pth")
    epoch_log_path = os.path.join(fold_dir, "epoch_metrics.csv")
    last_state_path = os.path.join(fold_dir, f"last_state_fold{fold}.pt")

    train_transforms, val_transforms = build_transforms()
    train_ds = Dataset(train_files, train_transforms)
    val_ds = Dataset(val_files, val_transforms)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.train_workers,
        pin_memory=pin_memory,
        collate_fn=pad_list_data_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.val_workers,
        pin_memory=pin_memory,
    )

    model = build_model(device)
    loss_function = DiceCELoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(3, args.early_stop_patience // 3),
        min_lr=1e-6,
    )

    dice_metric = DiceMetric(include_background=False, reduction="mean")

    best_metric = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    start_epoch = 0
    roi_size = (args.roi_x, args.roi_y, args.roi_z)

    if args.resume and os.path.exists(last_state_path):
        state = torch.load(last_state_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_metric = float(state["best_metric"])
        best_epoch = int(state["best_epoch"])
        epochs_no_improve = int(state["epochs_no_improve"])
        print(
            f"Resume fold {fold} from epoch {start_epoch + 1} | "
            f"best_dice_fg={best_metric:.4f}"
        )

    log_mode = "a" if (args.resume and os.path.exists(epoch_log_path) and start_epoch > 0) else "w"
    with open(epoch_log_path, log_mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if log_mode == "w":
            writer.writerow(["epoch", "train_loss", "val_dice_fg", "lr", "is_best"])

        for epoch in range(start_epoch, args.max_epochs):
            model.train()
            train_loss = 0.0

            for batch in train_loader:
                inputs = batch["image"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss = train_loss / max(len(train_loader), 1)

            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device)

                    val_outputs = sliding_window_inference(
                        val_inputs, roi_size=roi_size, sw_batch_size=1, predictor=model
                    )
                    val_outputs = torch.sigmoid(val_outputs)
                    val_outputs = (val_outputs > 0.5).float()
                    dice_metric(y_pred=val_outputs, y=val_labels)

            metric = float(dice_metric.aggregate().item())
            if math.isnan(metric):
                metric = 0.0
            dice_metric.reset()
            scheduler.step(metric)

            lr_now = optimizer.param_groups[0]["lr"]
            is_best = int(metric > best_metric)
            writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{metric:.6f}", f"{lr_now:.8f}", is_best])
            f.flush()

            print(
                f"Fold {fold} | Epoch {epoch + 1}/{args.max_epochs} | "
                f"Loss {train_loss:.4f} | Val Dice(FG) {metric:.4f} | LR {lr_now:.2e}"
            )

            if is_best:
                best_metric = metric
                best_epoch = epoch + 1
                epochs_no_improve = 0
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved best model to: {checkpoint_path}")
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
        "checkpoint": checkpoint_path,
    }


def main():
    args = parse_args()
    os.makedirs(args.runs_root, exist_ok=True)

    set_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dicts = build_data_dicts(args.data_root)

    if len(data_dicts) < args.num_folds:
        raise ValueError(
            f"Not enough samples ({len(data_dicts)}) for num_folds={args.num_folds}. "
            "Please reduce num_folds or add more training cases."
        )

    print("=" * 88)
    print("AttentionUNet Training")
    print(f"Device: {device}")
    print(f"Cases: {len(data_dicts)}")
    print(f"Folds: {args.num_folds}")
    print(f"Runs root: {args.runs_root}")
    print("=" * 88)

    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(data_dicts)):
        if args.only_fold is not None and fold != args.only_fold:
            continue

        print(f"\n===== Fold {fold} =====")
        train_files = [data_dicts[i] for i in train_idx]
        val_files = [data_dicts[i] for i in val_idx]

        result = train_one_fold(fold, train_files, val_files, args, device)
        fold_results.append(result)
        print(
            f"Fold {fold} done | best_epoch={result['best_epoch']} | "
            f"best_dice_fg={result['best_dice_fg']:.4f}"
        )

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

    best_scores = np.array([r["best_dice_fg"] for r in fold_results], dtype=np.float32)
    print("\n===== Cross-Validation Summary =====")
    print(f"Saved summary: {summary_path}")
    print(f"Dice(FG) mean={best_scores.mean():.4f}, std={best_scores.std(ddof=0):.4f}")


if __name__ == "__main__":
    main()
