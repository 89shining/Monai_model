import csv
import os
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.transforms import LoadImage

from ddunet_model import DDUnet


# =========================
# Hard-coded configuration
# =========================
DATA_ROOT = r"D:\project\Monai_model\data"
IMAGES_TS_DIR = os.path.join(DATA_ROOT, "imagesTs")
LABELS_TS_DIR = os.path.join(DATA_ROOT, "labelsTs")

RUNS_ROOT = os.path.join(r"D:\project\Monai_model\Eso_CTV\DDUnet", "runs")
OUTPUT_DIR = os.path.join(RUNS_ROOT, "best_fold_test")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")

CLIP_MIN = -150.0
CLIP_MAX = 200.0
INPUT_SIZE = (256, 256)
INFER_BATCH_SIZE = 32
SAVE_PRED = True


try:
    import nibabel as nib

    HAS_NIB = True
except Exception:
    HAS_NIB = False


def strip_nii_gz(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return os.path.splitext(filename)[0]


def case_key_from_image_path(image_path: str) -> str:
    return strip_nii_gz(os.path.basename(image_path)).replace("_0000", "")


def collect_cases(images_dir: str, labels_dir: str) -> List[Dict[str, str]]:
    image_paths = sorted(glob(os.path.join(images_dir, "*.nii*")))
    if not image_paths:
        raise FileNotFoundError(f"No test images found in: {images_dir}")

    cases: List[Dict[str, str]] = []
    missing = []
    for image_path in image_paths:
        case_name = case_key_from_image_path(image_path)
        label_path = os.path.join(labels_dir, f"{case_name}.nii.gz")
        if not os.path.exists(label_path):
            alt = os.path.join(labels_dir, f"{case_name}.nii")
            if os.path.exists(alt):
                label_path = alt
            else:
                missing.append((image_path, label_path))
                continue
        cases.append({"case_name": case_name, "image": image_path, "label": label_path})

    if missing:
        examples = "\n".join([f"image={m[0]}, expected={m[1]}" for m in missing[:5]])
        raise FileNotFoundError(f"Missing test labels:\n{examples}")

    return cases


def _to_depth_first(volume: np.ndarray) -> Tuple[np.ndarray, int]:
    axis = int(np.argmin(volume.shape))
    return np.moveaxis(volume, axis, 0), axis


def _from_depth_first(volume_dhw: np.ndarray, depth_axis: int) -> np.ndarray:
    return np.moveaxis(volume_dhw, 0, depth_axis)


def _resize_slices(volume_dhw: np.ndarray, mode: str) -> np.ndarray:
    tensor = torch.from_numpy(volume_dhw).unsqueeze(1).float()
    resized = F.interpolate(tensor, size=INPUT_SIZE, mode=mode, align_corners=False if mode == "bilinear" else None)
    return resized.squeeze(1).numpy()


def preprocess_case(image_path: str, label_path: str, loader: LoadImage):
    image, image_meta = loader(image_path)
    label, _ = loader(label_path)

    image = np.asarray(image, dtype=np.float32)
    label = np.asarray(label, dtype=np.float32)

    if image.ndim == 4:
        image = image[..., 0]
    if label.ndim == 4:
        label = label[..., 0]

    image_dhw, depth_axis = _to_depth_first(image)
    label_dhw, _ = _to_depth_first(label)

    image_dhw = np.clip(image_dhw, CLIP_MIN, CLIP_MAX)
    image_dhw = (image_dhw - CLIP_MIN) / (CLIP_MAX - CLIP_MIN)

    label_dhw = (label_dhw > 0).astype(np.float32)

    image_256 = _resize_slices(image_dhw, mode="bilinear").astype(np.float32)
    label_256 = (_resize_slices(label_dhw, mode="nearest") > 0.5).astype(np.float32)

    return {
        "image_256": image_256,
        "label_256": label_256,
        "raw_label_shape": label.shape,
        "depth_axis": depth_axis,
        "affine": image_meta.get("affine", None),
    }


@torch.no_grad()
def infer_case(model: nn.Module, image_dhw: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    d = image_dhw.shape[0]

    for start in range(0, d, INFER_BATCH_SIZE):
        end = min(start + INFER_BATCH_SIZE, d)
        batch = torch.from_numpy(image_dhw[start:end]).unsqueeze(1).to(device)
        logits = model(batch)
        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).float().cpu().numpy()[:, 0]
        preds.append(pred)

    return np.concatenate(preds, axis=0)


def dice_binary(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred = (pred > 0).astype(np.float32)
    target = (target > 0).astype(np.float32)
    inter = float((pred * target).sum())
    denom = float(pred.sum() + target.sum())
    if denom == 0:
        return 1.0
    return (2.0 * inter + eps) / (denom + eps)


def load_best_model_path() -> Tuple[int, str, float]:
    summary_csv = os.path.join(RUNS_ROOT, "fold_summary.csv")
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"Fold summary not found: {summary_csv}. Please run train_ddunet.py first.")

    records = []
    with open(summary_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                {
                    "fold": int(row["fold"]),
                    "best_dice": float(row["best_val_dice"]),
                    "best_model": row["best_model"],
                }
            )

    if not records:
        raise RuntimeError("No fold records found in fold_summary.csv")

    best = max(records, key=lambda x: x["best_dice"])
    return best["fold"], best["best_model"], best["best_dice"]


def save_prediction(case_name: str, pred_256_dhw: np.ndarray, raw_shape: Tuple[int, ...], depth_axis: int, affine):
    os.makedirs(PRED_DIR, exist_ok=True)

    # Resize back to raw in-plane size for convenient comparison with label.
    h_raw, w_raw = [raw_shape[i] for i in range(3) if i != depth_axis]
    pred_tensor = torch.from_numpy(pred_256_dhw).unsqueeze(1).float()
    pred_resized = F.interpolate(pred_tensor, size=(h_raw, w_raw), mode="nearest").squeeze(1).numpy()
    pred_raw = _from_depth_first((pred_resized > 0.5).astype(np.uint8), depth_axis)

    if HAS_NIB and affine is not None:
        out_path = os.path.join(PRED_DIR, f"{case_name}.nii.gz")
        nii = nib.Nifti1Image(pred_raw.astype(np.uint8), affine)
        nib.save(nii, out_path)
    else:
        out_path = os.path.join(PRED_DIR, f"{case_name}.npy")
        np.save(out_path, pred_raw.astype(np.uint8))


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_fold, best_model_path, best_val_dice = load_best_model_path()

    print("=" * 80)
    print("DDUnet test")
    print(f"Device: {device}")
    print(f"Selected best fold: {best_fold}, val Dice={best_val_dice:.6f}")
    print(f"Best model: {best_model_path}")
    print("=" * 80)

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model does not exist: {best_model_path}")

    model = DDUnet(in_channels=1, out_channels=1).to(device)
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader = LoadImage(image_only=False)
    test_cases = collect_cases(IMAGES_TS_DIR, LABELS_TS_DIR)

    rows = []
    dices = []
    for case in test_cases:
        proc = preprocess_case(case["image"], case["label"], loader)
        pred = infer_case(model, proc["image_256"], device)
        dice = dice_binary(pred, proc["label_256"])

        rows.append((case["case_name"], dice))
        dices.append(dice)

        print(f"Test case: {case['case_name']}, Dice={dice:.6f}")

        if SAVE_PRED:
            save_prediction(
                case_name=case["case_name"],
                pred_256_dhw=pred,
                raw_shape=proc["raw_label_shape"],
                depth_axis=proc["depth_axis"],
                affine=proc["affine"],
            )

    mean_dice = float(np.mean(dices)) if dices else 0.0

    csv_path = os.path.join(OUTPUT_DIR, "test_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["case_name", "dice"])
        for case_name, dice in rows:
            writer.writerow([case_name, f"{dice:.6f}"])
        writer.writerow(["MEAN", f"{mean_dice:.6f}"])

    print(f"Mean Dice: {mean_dice:.6f}")
    print(f"Saved metrics: {csv_path}")
    if SAVE_PRED:
        print(f"Saved predictions: {PRED_DIR}")


if __name__ == "__main__":
    main()
