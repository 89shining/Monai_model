import csv
import os
from glob import glob
from typing import Dict, List

import numpy as np
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, SaveImage, ScaleIntensityd

from ecau_net_3d import ECAUNet3D


# =========================
# Hard-coded configuration
# =========================
DATA_ROOT = r"D:\project\Monai_model\data"
IMAGES_TS_DIR = os.path.join(DATA_ROOT, "imagesTs")
LABELS_TS_DIR = os.path.join(DATA_ROOT, "labelsTs")

RUNS_ROOT = os.path.join(r"D:\project\Monai_model\Eso_CTV\ECAU_Net", "runs")
PATCH_SIZE = (96, 96, 96)
SW_BATCH_SIZE = 1
NUM_WORKERS = 4
MODEL_CHANNELS = (32, 64, 128, 256, 512)
SAVE_PREDICTIONS = True


def strip_nii_gz(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return os.path.splitext(filename)[0]


def case_key_from_image_path(image_path: str) -> str:
    return strip_nii_gz(os.path.basename(image_path)).replace("_0000", "")


def dice_from_probs(probs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5, eps: float = 1e-8) -> torch.Tensor:
    preds = (probs >= threshold).float()
    labels = (labels > 0.5).float()

    dims = tuple(range(2, preds.ndim))
    intersection = torch.sum(preds * labels, dim=dims)
    denominator = torch.sum(preds, dim=dims) + torch.sum(labels, dim=dims)

    dice = (2.0 * intersection + eps) / (denominator + eps)
    both_empty = denominator == 0
    dice = torch.where(both_empty, torch.ones_like(dice), dice)
    return dice.squeeze(1)


def get_eval_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"]),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


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
        raise FileNotFoundError(f"Some test labels are missing. Examples:\n{sample}")

    return files


def load_best_fold_info() -> Dict[str, str]:
    summary_csv = os.path.join(RUNS_ROOT, "fold_summary.csv")
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"fold_summary.csv not found: {summary_csv}")

    records = []
    with open(summary_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    if not records:
        raise RuntimeError("fold_summary.csv is empty")

    best = max(records, key=lambda r: float(r["best_val_dice"]))
    return {
        "fold": best["fold"],
        "best_val_dice": best["best_val_dice"],
        "best_model_path": best["best_model_path"],
    }


def build_model(device: torch.device) -> torch.nn.Module:
    return ECAUNet3D(in_channels=1, out_channels=1, channels=MODEL_CHANNELS).to(device)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = load_best_fold_info()

    best_model_path = info["best_model_path"]
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"best model does not exist: {best_model_path}")

    print("=" * 80)
    print("ECAU-Net standalone test")
    print(f"Device: {device}")
    print(f"Best fold: {info['fold']}, val dice={float(info['best_val_dice']):.6f}")
    print(f"Model path: {best_model_path}")
    print("=" * 80)

    model = build_model(device)
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt)
    model.eval()

    test_files = collect_test_files(IMAGES_TS_DIR, LABELS_TS_DIR)
    test_ds = Dataset(data=test_files, transform=get_eval_transforms())
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    output_dir = os.path.join(RUNS_ROOT, "best_fold_test")
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "test_metrics.csv")

    saver = SaveImage(
        output_dir=pred_dir,
        output_postfix="pred",
        output_ext=".nii.gz",
        separate_folder=False,
        print_log=False,
    )

    case_dices: List[float] = []

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
                dice = float(dice_from_probs(probs, labels).item())
                case_dices.append(dice)
                writer.writerow([case_name, f"{dice:.6f}"])
                print(f"Test case: {case_name}, Dice={dice:.6f}")

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

    mean_dice = float(np.mean(case_dices)) if case_dices else 0.0
    print(f"Mean Dice: {mean_dice:.6f}")
    print(f"Saved metrics: {csv_path}")


if __name__ == "__main__":
    main()
