import csv
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from monai.data import DataLoader, Dataset, decollate_batch, list_data_collate
from monai.transforms import SaveImage

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from pipeline import (  # noqa: E402
    build_kfold_indices,
    build_model,
    collect_files,
    dice_bce_loss,
    dice_from_probs,
    get_eval_transforms,
    get_train_transforms,
    infer_volume_probs,
    set_seed,
    volume_to_2d_batches,
)


# =========================
# Hard-coded configuration
# =========================
DATA_ROOT = r"D:\project\Monai_model\data"
IMAGES_TR_DIR = os.path.join(DATA_ROOT, "imagesTr")
LABELS_TR_DIR = os.path.join(DATA_ROOT, "labelsTr")
IMAGES_TS_DIR = os.path.join(DATA_ROOT, "imagesTs")
LABELS_TS_DIR = os.path.join(DATA_ROOT, "labelsTs")

RUNS_ROOT = os.path.join(".", "runs", "DeepLabV3Plus")

SEED = 42
NUM_FOLDS = 5
EPOCHS = 200
BATCH_SIZE = 1
NUM_WORKERS = 4
LEARNING_RATE = 1e-4

INPUT_SHAPE = (512, 512)
SLICE_BATCH_SIZE = 8

BACKBONE = "xception"
DOWNSAMPLE_FACTOR = 16
PRETRAINED_BACKBONE = False

SAVE_PREDICTIONS = True


def validate_one_epoch(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    dices: List[float] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device).float()   # [1, 1, D, H, W]
            labels = (batch["label"] > 0.5).float()      # [1, 1, D, H, W]

            probs = infer_volume_probs(
                model=model,
                volume=images,
                device=device,
                input_shape=INPUT_SHAPE,
                slice_batch_size=SLICE_BATCH_SIZE,
            )
            batch_dice = dice_from_probs(probs, labels.cpu())
            dices.extend(batch_dice.numpy().tolist())

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
                images = batch["image"].to(device).float()
                labels = (batch["label"] > 0.5).float()
                case_name = batch["case_name"][0]

                probs = infer_volume_probs(
                    model=model,
                    volume=images,
                    device=device,
                    input_shape=INPUT_SHAPE,
                    slice_batch_size=SLICE_BATCH_SIZE,
                )
                dice_val = float(dice_from_probs(probs, labels.cpu()).item())
                case_dices.append(dice_val)
                writer.writerow([case_name, f"{dice_val:.6f}"])
                print(f"[Best Fold {fold_idx}] Test case: {case_name}, Dice={dice_val:.6f}")

                if SAVE_PREDICTIONS:
                    pred_mask = (probs >= 0.5).float()
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

    model = build_model(
        device=device,
        backbone=BACKBONE,
        num_classes=1,
        downsample_factor=DOWNSAMPLE_FACTOR,
        pretrained_backbone=PRETRAINED_BACKBONE,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_dice = -1.0
    best_ckpt = os.path.join(fold_dir, f"best_model_fold{fold_idx}.pth")
    last_ckpt = os.path.join(fold_dir, f"last_model_fold{fold_idx}.pth")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses: List[float] = []

        for batch in train_loader:
            images = batch["image"].to(device).float()   # [B, 1, D, H, W]
            labels = (batch["label"] > 0.5).to(device).float()

            optimizer.zero_grad(set_to_none=True)
            volume_loss = 0.0
            volume_steps = 0

            for b in range(images.shape[0]):
                # [1, D, H, W]
                image_vol = images[b]
                label_vol = labels[b]

                for img_2d, lbl_2d in volume_to_2d_batches(
                    volume=image_vol,
                    label=label_vol,
                    slice_batch_size=SLICE_BATCH_SIZE,
                    input_shape=INPUT_SHAPE,
                ):
                    img_2d = img_2d.to(device).repeat(1, 3, 1, 1)
                    lbl_2d = lbl_2d.to(device)

                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        logits = model(img_2d)
                        loss = dice_bce_loss(logits, lbl_2d)

                    scaler.scale(loss).backward()
                    volume_loss += float(loss.item())
                    volume_steps += 1

            if volume_steps > 0:
                scaler.step(optimizer)
                scaler.update()
                epoch_losses.append(volume_loss / volume_steps)

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
    print("DeepLabV3+ Slice-wise 3D Segmentation Training")
    print(f"Device: {device}")
    print(f"Train images: {IMAGES_TR_DIR}")
    print(f"Train labels: {LABELS_TR_DIR}")
    print(f"Test images:  {IMAGES_TS_DIR}")
    print(f"Test labels:  {LABELS_TS_DIR}")
    print("=" * 80)

    train_all_files = collect_files(IMAGES_TR_DIR, LABELS_TR_DIR)
    test_files = collect_files(IMAGES_TS_DIR, LABELS_TS_DIR)

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
    test_model = build_model(
        device=device,
        backbone=BACKBONE,
        num_classes=1,
        downsample_factor=DOWNSAMPLE_FACTOR,
        pretrained_backbone=False,
    )
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

