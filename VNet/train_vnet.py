import csv
import os
import random
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import torch
from monai.data import DataLoader, Dataset, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import VNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    SaveImage,
    ScaleIntensityd,
)
from monai.utils import set_determinism


# =========================
# Hard-coded configuration
# =========================
DATA_ROOT = r"D:\project\Monai_model\data"
IMAGES_TR_DIR = os.path.join(DATA_ROOT, "imagesTr")
LABELS_TR_DIR = os.path.join(DATA_ROOT, "labelsTr")
IMAGES_TS_DIR = os.path.join(DATA_ROOT, "imagesTs")
LABELS_TS_DIR = os.path.join(DATA_ROOT, "labelsTs")

RUNS_ROOT = os.path.join("..", "runs", "VNet")

SEED = 42
NUM_FOLDS = 5
EPOCHS = 200
BATCH_SIZE = 1
NUM_WORKERS = 4
LEARNING_RATE = 1e-4

PATCH_SIZE = (96, 96, 96)
SW_BATCH_SIZE = 1

SAVE_PREDICTIONS = True


# =========================
# Utilities
# =========================
def strip_nii_gz(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return os.path.splitext(filename)[0]


def case_key_from_image_path(image_path: str) -> str:
    name = strip_nii_gz(os.path.basename(image_path))
    return name.replace("_0000", "")


def dice_from_probs(probs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5, eps: float = 1e-8) -> torch.Tensor:
    preds = (probs >= threshold).float()
    labels = (labels > 0.5).float()

    dims = tuple(range(2, preds.ndim))
    intersection = torch.sum(preds * labels, dim=dims)
    denominator = torch.sum(preds, dim=dims) + torch.sum(labels, dim=dims)

    # If both prediction and label are empty, define Dice as 1 for that sample.
    dice = (2.0 * intersection + eps) / (denominator + eps)
    both_empty = denominator == 0
    dice = torch.where(both_empty, torch.ones_like(dice), dice)
    return dice.squeeze(1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


def collect_train_files(images_dir: str, labels_dir: str) -> List[Dict[str, str]]:
    image_paths = sorted(glob(os.path.join(images_dir, "*.nii*")))
    if not image_paths:
        raise FileNotFoundError(f"No training images found in: {images_dir}")

    files = []
    missing_labels = []
    for image_path in image_paths:
        case_key = case_key_from_image_path(image_path)
        label_path = os.path.join(labels_dir, f"{case_key}.nii.gz")
        if not os.path.exists(label_path):
            alt_label_path = os.path.join(labels_dir, f"{case_key}.nii")
            if os.path.exists(alt_label_path):
                label_path = alt_label_path
            else:
                missing_labels.append((image_path, label_path))
                continue
        files.append({"image": image_path, "label": label_path, "case_name": case_key})

    if missing_labels:
        sample = "\n".join([f"image={i}, expected_label={l}" for i, l in missing_labels[:5]])
        raise FileNotFoundError(
            "Some training labels are missing after applying the '_0000' matching rule. "
            f"Examples:\n{sample}"
        )

    return files


def collect_test_files(images_dir: str, labels_dir: str) -> List[Dict[str, str]]:
    image_paths = sorted(glob(os.path.join(images_dir, "*.nii*")))
    if not image_paths:
        raise FileNotFoundError(f"No test images found in: {images_dir}")

    files = []
    missing_labels = []
    for image_path in image_paths:
        case_key = case_key_from_image_path(image_path)
        label_path = os.path.join(labels_dir, f"{case_key}.nii.gz")
        if not os.path.exists(label_path):
            alt_label_path = os.path.join(labels_dir, f"{case_key}.nii")
            if os.path.exists(alt_label_path):
                label_path = alt_label_path
            else:
                missing_labels.append((image_path, label_path))
                continue
        files.append({"image": image_path, "label": label_path, "case_name": case_key})

    if missing_labels:
        sample = "\n".join([f"image={i}, expected_label={l}" for i, l in missing_labels[:5]])
        raise FileNotFoundError(
            "Some test labels are missing after applying the '_0000' matching rule. "
            f"Examples:\n{sample}"
        )

    return files


def build_kfold_indices(num_samples: int, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if num_samples < n_splits:
        raise ValueError(f"num_samples ({num_samples}) must be >= n_splits ({n_splits}).")

    rng = np.random.RandomState(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_splits)

    split_pairs = []
    for fold_idx in range(n_splits):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(n_splits) if i != fold_idx])
        split_pairs.append((train_idx, val_idx))
    return split_pairs


def get_train_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"]),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=PATCH_SIZE,
                pos=1,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_eval_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"]),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def build_model(device: torch.device) -> torch.nn.Module:
    model = VNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
    )
    return model.to(device)


def validate_one_epoch(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    dices: List[float] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = sliding_window_inference(
                inputs=images,
                roi_size=PATCH_SIZE,
                sw_batch_size=SW_BATCH_SIZE,
                predictor=model,
                overlap=0.25,
            )
            probs = torch.sigmoid(logits)
            batch_dice = dice_from_probs(probs, labels, threshold=0.5)
            dices.extend(batch_dice.detach().cpu().numpy().tolist())

    if not dices:
        return 0.0
    return float(np.mean(dices))


def run_test_for_fold(
    fold_idx: int,
    best_ckpt: str,
    output_dir: str,
    model: torch.nn.Module,
    test_files: List[Dict[str, str]],
    device: torch.device,
) -> None:
    print(f"\n[Best Fold {fold_idx}] Testing with selected best fold model...")

    if not os.path.exists(best_ckpt):
        raise FileNotFoundError(f"Best model not found: {best_ckpt}")

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()

    test_ds = Dataset(data=test_files, transform=get_eval_transforms())
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "test_metrics.csv")
    case_dices: List[float] = []
    saver = SaveImage(
        output_dir=pred_dir,
        output_postfix="pred",
        output_ext=".nii.gz",
        separate_folder=False,
        print_log=False,
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_name", "dice"])

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                case_name = batch["case_name"][0]

                logits = sliding_window_inference(
                    inputs=images,
                    roi_size=PATCH_SIZE,
                    sw_batch_size=SW_BATCH_SIZE,
                    predictor=model,
                    overlap=0.25,
                )

                probs = torch.sigmoid(logits)
                dice_val = float(dice_from_probs(probs, labels, threshold=0.5).item())
                case_dices.append(dice_val)
                writer.writerow([case_name, f"{dice_val:.6f}"])
                print(f"[Best Fold {fold_idx}] Test case: {case_name}, Dice={dice_val:.6f}")

                if SAVE_PREDICTIONS:
                    pred_mask = (probs >= 0.5).float().cpu()
                    image_meta = decollate_batch(batch)[0]["image"].meta

                    saver(pred_mask[0], meta_data=image_meta)
                    saved_src = os.path.join(pred_dir, f"{case_name}_0000_pred.nii.gz")
                    saved_dst = os.path.join(pred_dir, f"{case_name}.nii.gz")
                    if os.path.exists(saved_src):
                        if os.path.exists(saved_dst):
                            os.remove(saved_dst)
                        os.replace(saved_src, saved_dst)
                    else:
                        print(f"[Warning] Expected prediction file not found for rename: {saved_src}")

    mean_dice = float(np.mean(case_dices)) if case_dices else 0.0
    print(f"[Best Fold {fold_idx}] Test mean Dice: {mean_dice:.6f}")
    print(f"[Best Fold {fold_idx}] Test metrics saved to: {csv_path}")


def train_one_fold(
    fold_idx: int,
    train_files: List[Dict[str, str]],
    val_files: List[Dict[str, str]],
    device: torch.device,
) -> Tuple[float, str]:
    fold_dir = os.path.join(RUNS_ROOT, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    print(f"\n========== Fold {fold_idx} ==========")
    print(f"Train samples: {len(train_files)} | Val samples: {len(val_files)}")

    train_ds = Dataset(data=train_files, transform=get_train_transforms())
    val_ds = Dataset(data=val_files, transform=get_eval_transforms())

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = DiceCELoss(sigmoid=True)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_dice = -1.0
    best_ckpt = os.path.join(fold_dir, f"best_model_fold{fold_idx}.pth")
    last_ckpt = os.path.join(fold_dir, f"last_model_fold{fold_idx}.pth")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses: List[float] = []

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(images)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(float(loss.item()))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_dice = validate_one_epoch(model, val_loader, device)

        torch.save(model.state_dict(), last_ckpt)
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_ckpt)

        print(
            f"[Fold {fold_idx}] Epoch {epoch:03d}/{EPOCHS} "
            f"| Train Loss={train_loss:.6f} | Val Dice={val_dice:.6f} | Best Dice={best_dice:.6f}"
        )

    return best_dice, best_ckpt


def main() -> None:
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RUNS_ROOT, exist_ok=True)

    print("=" * 80)
    print("MONAI VNet 3D Segmentation Training")
    print(f"Device: {device}")
    print(f"Train images: {IMAGES_TR_DIR}")
    print(f"Train labels: {LABELS_TR_DIR}")
    print(f"Test images:  {IMAGES_TS_DIR}")
    print(f"Test labels:  {LABELS_TS_DIR}")
    print("=" * 80)

    train_all_files = collect_train_files(IMAGES_TR_DIR, LABELS_TR_DIR)
    test_files = collect_test_files(IMAGES_TS_DIR, LABELS_TS_DIR)

    print(f"Total train cases: {len(train_all_files)}")
    print(f"Total test cases:  {len(test_files)}")

    split_pairs = build_kfold_indices(len(train_all_files), NUM_FOLDS, SEED)
    fold_summaries: List[Dict[str, object]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(split_pairs):
        fold_train_files = [train_all_files[i] for i in train_idx]
        fold_val_files = [train_all_files[i] for i in val_idx]
        best_dice, best_ckpt = train_one_fold(
            fold_idx=fold_idx,
            train_files=fold_train_files,
            val_files=fold_val_files,
            device=device,
        )
        fold_summaries.append(
            {
                "fold_idx": fold_idx,
                "best_dice": best_dice,
                "best_ckpt": best_ckpt,
            }
        )

    selected = max(fold_summaries, key=lambda x: x["best_dice"])
    selected_fold = int(selected["fold_idx"])
    selected_dice = float(selected["best_dice"])
    selected_ckpt = str(selected["best_ckpt"])

    print(
        f"\nSelected best fold: {selected_fold} "
        f"(best validation Dice={selected_dice:.6f})"
    )

    test_output_dir = os.path.join(RUNS_ROOT, "best_fold_test")
    os.makedirs(test_output_dir, exist_ok=True)
    test_model = build_model(device)
    run_test_for_fold(
        fold_idx=selected_fold,
        best_ckpt=selected_ckpt,
        output_dir=test_output_dir,
        model=test_model,
        test_files=test_files,
        device=device,
    )

    print("\nAll folds finished.")


if __name__ == "__main__":
    main()
