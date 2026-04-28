# ================= train.py =================
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
    parser = argparse.ArgumentParser()
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
    return parser.parse_args()


def set_all_seeds(seed):
    set_determinism(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_data_dicts(root):
    images = sorted(glob.glob(os.path.join(root, "imagesTr", "*_0000.nii.gz")))
    labels = sorted(glob.glob(os.path.join(root, "labelsTr", "*.nii.gz")))

    image_map = {os.path.basename(p)[:-12]: p for p in images}
    label_map = {os.path.basename(p)[:-7]: p for p in labels}

    ids = sorted(image_map.keys())
    return [{"image": image_map[i], "label": label_map[i]} for i in ids]


def _bin(x):
    return (x > 0).astype(x.dtype)


def build_transforms(args):
    train = Compose([
        LoadImaged(["image", "label"]),
        EnsureChannelFirstd(["image", "label"]),
        ScaleIntensityRanged("image", -1000, 1000, 0, 1, True),
        Lambdad("label", _bin),

        SpatialPadd(["image", "label"], (args.roi_x, args.roi_y, args.roi_z)),

        # ⭐关键：加强前景采样
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


def train_one_fold(fold, train_files, val_files, args, device):
    train_tf, val_tf = build_transforms(args)

    train_loader = DataLoader(
        Dataset(train_files, train_tf),
        batch_size=1,
        shuffle=True,
        num_workers=args.train_workers,
        collate_fn=list_data_collate,
    )

    val_loader = DataLoader(
        Dataset(val_files, val_tf),
        batch_size=1,
        num_workers=args.val_workers,
    )

    model = VNet(3, 1, 1).to(device)

    loss_fn = DiceCELoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    dice_metric = DiceMetric(include_background=True)

    roi = (args.roi_x, args.roi_y, args.roi_z)

    for epoch in range(args.max_epochs):
        model.train()
        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        dice_metric.reset()

        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = batch["label"]

                pred = sliding_window_inference(
                    x, roi, 1, model,
                    sw_device=device,
                    device=torch.device("cpu")
                )

                pred = torch.sigmoid(pred)
                dice_metric(pred, y)

        metric = dice_metric.aggregate().item()
        dice_metric.reset()

        print(f"Fold {fold} Epoch {epoch+1} Dice {metric:.4f}")


def main():
    args = parse_args()
    set_all_seeds(args.seed)
    device = torch.device("cuda")

    data = build_data_dicts(args.data_root)

    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=42)

    for fold, (tr, va) in enumerate(kf.split(data)):
        train_files = [data[i] for i in tr]
        val_files = [data[i] for i in va]
        train_one_fold(fold, train_files, val_files, args, device)


if __name__ == "__main__":
    main()