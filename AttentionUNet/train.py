import argparse
import csv
import glob
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from monai.data import DataLoader, Dataset, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
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
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)


def parse_args():
    parser = argparse.ArgumentParser(description="3D AttentionUNet training with K-fold CV.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--runs_root", type=str, default="./runs/AttentionUNet")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--early_stop_patience", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=12.0)

    parser.add_argument("--roi_x", type=int, default=96)
    parser.add_argument("--roi_y", type=int, default=96)
    parser.add_argument("--roi_z", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--sw_batch_size", type=int, default=2)
    parser.add_argument("--infer_overlap", type=float, default=0.5)

    parser.add_argument("--a_min", type=float, default=-160.0)
    parser.add_argument("--a_max", type=float, default=240.0)

    parser.add_argument("--pixdim_x", type=float, default=1.0)
    parser.add_argument("--pixdim_y", type=float, default=1.0)
    parser.add_argument("--pixdim_z", type=float, default=1.0)
    parser.add_argument("--axcodes", type=str, default="RAS")

    parser.add_argument("--f1", type=int, default=32)
    parser.add_argument("--f2", type=int, default=64)
    parser.add_argument("--f3", type=int, default=128)
    parser.add_argument("--f4", type=int, default=256)
    parser.add_argument("--f5", type=int, default=512)

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


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    def __init__(self, g_channels: int, x_channels: int, inter_channels: int):
        super().__init__()
        self.wg = nn.Sequential(
            nn.Conv3d(g_channels, inter_channels, kernel_size=1, bias=True),
            nn.InstanceNorm3d(inter_channels),
        )
        self.wx = nn.Sequential(
            nn.Conv3d(x_channels, inter_channels, kernel_size=1, bias=True),
            nn.InstanceNorm3d(inter_channels),
        )
        self.psi = nn.Sequential(nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True), nn.Sigmoid())
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.wg(g) + self.wx(x))
        psi = self.psi(psi)
        return x * psi


class AttentionUNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, features: Tuple[int, int, int, int, int] = (32, 64, 128, 256, 512)):
        super().__init__()
        f1, f2, f3, f4, f5 = features
        self.enc1 = ConvBlock(in_channels, f1)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.enc2 = ConvBlock(f1, f2)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.enc3 = ConvBlock(f2, f3)
        self.pool3 = nn.MaxPool3d(2, 2)
        self.enc4 = ConvBlock(f3, f4)
        self.pool4 = nn.MaxPool3d(2, 2)
        self.bottleneck = ConvBlock(f4, f5)

        self.up4 = nn.ConvTranspose3d(f5, f4, kernel_size=2, stride=2)
        self.att4 = AttentionGate(f4, f4, f4 // 2)
        self.dec4 = ConvBlock(f4 + f4, f4)

        self.up3 = nn.ConvTranspose3d(f4, f3, kernel_size=2, stride=2)
        self.att3 = AttentionGate(f3, f3, f3 // 2)
        self.dec3 = ConvBlock(f3 + f3, f3)

        self.up2 = nn.ConvTranspose3d(f3, f2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(f2, f2, f2 // 2)
        self.dec2 = ConvBlock(f2 + f2, f2)

        self.up1 = nn.ConvTranspose3d(f2, f1, kernel_size=2, stride=2)
        self.att1 = AttentionGate(f1, f1, f1 // 2)
        self.dec1 = ConvBlock(f1 + f1, f1)

        self.out_conv = nn.Conv3d(f1, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.att4(self.up4(b), e4), self.up4(b)], dim=1))
        d3 = self.dec3(torch.cat([self.att3(self.up3(d4), e3), self.up3(d4)], dim=1))
        d2 = self.dec2(torch.cat([self.att2(self.up2(d3), e2), self.up2(d3)], dim=1))
        d1 = self.dec1(torch.cat([self.att1(self.up1(d2), e1), self.up1(d2)], dim=1))
        return self.out_conv(d1)


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


def collect_data_list(data_root: str) -> List[Dict[str, str]]:
    image_dir = os.path.join(data_root, "imagesTr")
    label_dir = os.path.join(data_root, "labelsTr")
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")) + glob.glob(os.path.join(image_dir, "*.nii")))
    if len(image_paths) == 0:
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
    indices = list(range(len(data_list)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    fold_sizes = [len(indices) // num_folds] * num_folds
    for i in range(len(indices) % num_folds):
        fold_sizes[i] += 1

    folds = []
    start = 0
    for fs in fold_sizes:
        end = start + fs
        folds.append(indices[start:end])
        start = end

    splits = []
    for fold_idx in range(num_folds):
        val_indices = folds[fold_idx]
        train_indices = []
        for i in range(num_folds):
            if i != fold_idx:
                train_indices.extend(folds[i])
        splits.append(([data_list[i] for i in train_indices], [data_list[i] for i in val_indices]))
    return splits


def get_transforms(args):
    roi_size = (args.roi_x, args.roi_y, args.roi_z)
    pixdim = (args.pixdim_x, args.pixdim_y, args.pixdim_z)

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes=args.axcodes),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=0.0, b_max=1.0, clip=True),
        Lambdad(keys=["label"], func=binarize_label),
        SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_size,
            pos=1,
            neg=1,
            num_samples=args.num_samples,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
        RandAffined(
            keys=["image", "label"],
            prob=0.2,
            rotate_range=(0.1, 0.1, 0.05),
            scale_range=(0.1, 0.1, 0.05),
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes=args.axcodes),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=0.0, b_max=1.0, clip=True),
        Lambdad(keys=["label"], func=binarize_label),
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
        loss = loss_func(logits, labels)
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
def validate(model, loader, loss_func, dice_metric, post_pred, post_label, device, args):
    model.eval()
    dice_metric.reset()
    val_loss = 0.0
    steps = 0

    for batch_data in loader:
        steps += 1
        images = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        logits = sliding_window_inference(
            inputs=images,
            roi_size=(args.roi_x, args.roi_y, args.roi_z),
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=args.infer_overlap,
        )
        val_loss += loss_func(logits, labels).item()
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
        collate_fn=list_data_collate,
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

    model = AttentionUNet3D(
        in_channels=1,
        out_channels=1,
        features=(args.f1, args.f2, args.f3, args.f4, args.f5),
    ).to(device)

    loss_func = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean", lambda_dice=1.0, lambda_ce=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.eta_min)

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
            val_loss, val_dice = validate(model, val_loader, loss_func, dice_metric, post_pred, post_label, device, args)

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
