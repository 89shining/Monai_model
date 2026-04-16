import csv
import os
import random
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.transforms import LoadImage
from torch.utils.data import DataLoader, Dataset

from ddunet_model import DDUnet


# =========================
# Hard-coded configuration
# =========================
DATA_ROOT = r"D:\project\Monai_model\data"
IMAGES_TR_DIR = os.path.join(DATA_ROOT, "imagesTr")
LABELS_TR_DIR = os.path.join(DATA_ROOT, "labelsTr")
IMAGES_TS_DIR = os.path.join(DATA_ROOT, "imagesTs")
LABELS_TS_DIR = os.path.join(DATA_ROOT, "labelsTs")

RUNS_ROOT = os.path.join(r"D:\project\Monai_model\Eso_CTV\DDUnet", "runs")

SEED = 42
NUM_FOLDS = 5
EPOCHS = 40
BATCH_SIZE = 6
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
ADAM_BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.0

CLIP_MIN = -150.0
CLIP_MAX = 200.0
INPUT_SIZE = (256, 256)
INFER_BATCH_SIZE = 32


@dataclass
class CaseData:
    case_name: str
    image_path: str
    label_path: str


class DiceLossBinary(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strip_nii_gz(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return os.path.splitext(filename)[0]


def case_key_from_image_path(image_path: str) -> str:
    return strip_nii_gz(os.path.basename(image_path)).replace("_0000", "")


def collect_cases(images_dir: str, labels_dir: str) -> List[CaseData]:
    image_paths = sorted(glob(os.path.join(images_dir, "*.nii*")))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    cases: List[CaseData] = []
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
        cases.append(CaseData(case_name=case_name, image_path=image_path, label_path=label_path))

    if missing:
        examples = "\n".join([f"image={m[0]}, expected={m[1]}" for m in missing[:5]])
        raise FileNotFoundError(f"Missing labels:\n{examples}")

    return cases


def build_kfold_indices(num_samples: int, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if num_samples < n_splits:
        raise ValueError(f"num_samples={num_samples} must be >= n_splits={n_splits}")

    rng = np.random.RandomState(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_splits)

    split_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(n_splits):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(n_splits) if i != fold_idx])
        split_pairs.append((train_idx, val_idx))
    return split_pairs


def _to_depth_first(volume: np.ndarray) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={volume.shape}")
    depth_axis = int(np.argmin(volume.shape))
    return np.moveaxis(volume, depth_axis, 0)


def _resize_slices(volume_dhw: np.ndarray, mode: str) -> np.ndarray:
    tensor = torch.from_numpy(volume_dhw).unsqueeze(1).float()  # [D,1,H,W]
    resized = F.interpolate(tensor, size=INPUT_SIZE, mode=mode, align_corners=False if mode == "bilinear" else None)
    return resized.squeeze(1).numpy()


def preprocess_case(image_path: str, label_path: str, loader: LoadImage) -> Tuple[np.ndarray, np.ndarray]:
    image = loader(image_path)
    label = loader(label_path)

    if isinstance(image, tuple):
        image = image[0]
    if isinstance(label, tuple):
        label = label[0]

    image = np.asarray(image, dtype=np.float32)
    label = np.asarray(label, dtype=np.float32)

    if image.ndim == 4:
        image = image[..., 0]
    if label.ndim == 4:
        label = label[..., 0]

    image = _to_depth_first(image)
    label = _to_depth_first(label)

    image = np.clip(image, CLIP_MIN, CLIP_MAX)
    image = (image - CLIP_MIN) / (CLIP_MAX - CLIP_MIN)
    label = (label > 0).astype(np.float32)

    image = _resize_slices(image, mode="bilinear")
    label = _resize_slices(label, mode="nearest")

    image = image.astype(np.float32)
    label = (label > 0.5).astype(np.float32)
    return image, label


class SliceDataset(Dataset):
    def __init__(self, cases: List[CaseData]):
        self.loader = LoadImage(image_only=True)
        self.case_names: List[str] = []
        self.images: List[np.ndarray] = []  # [D,H,W]
        self.labels: List[np.ndarray] = []  # [D,H,W]
        self.slice_index: List[Tuple[int, int]] = []

        for case in cases:
            image, label = preprocess_case(case.image_path, case.label_path, self.loader)
            case_i = len(self.images)
            self.case_names.append(case.case_name)
            self.images.append(image)
            self.labels.append(label)
            for s in range(image.shape[0]):
                self.slice_index.append((case_i, s))

    def __len__(self) -> int:
        return len(self.slice_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case_i, slice_i = self.slice_index[idx]
        image = self.images[case_i][slice_i]
        label = self.labels[case_i][slice_i]
        return {
            "image": torch.from_numpy(image).unsqueeze(0),
            "label": torch.from_numpy(label).unsqueeze(0),
        }


def dice_binary(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred = (pred > 0).astype(np.float32)
    target = (target > 0).astype(np.float32)
    intersection = float((pred * target).sum())
    denom = float(pred.sum() + target.sum())
    if denom == 0:
        return 1.0
    return (2.0 * intersection + eps) / (denom + eps)


@torch.no_grad()
def infer_case(model: nn.Module, image_dhw: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    preds: List[np.ndarray] = []

    d = image_dhw.shape[0]
    for start in range(0, d, INFER_BATCH_SIZE):
        end = min(start + INFER_BATCH_SIZE, d)
        batch = torch.from_numpy(image_dhw[start:end]).unsqueeze(1).to(device)  # [N,1,H,W]
        logits = model(batch)
        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).float().cpu().numpy()[:, 0]
        preds.append(pred)

    return np.concatenate(preds, axis=0)


@torch.no_grad()
def validate(model: nn.Module, val_cases: List[CaseData], device: torch.device) -> float:
    loader = LoadImage(image_only=True)
    case_dices = []
    for case in val_cases:
        image, label = preprocess_case(case.image_path, case.label_path, loader)
        pred = infer_case(model, image, device)
        case_dices.append(dice_binary(pred, label))
    if not case_dices:
        return 0.0
    return float(np.mean(case_dices))


def train_fold(fold_idx: int, train_cases: List[CaseData], val_cases: List[CaseData], device: torch.device) -> Dict[str, object]:
    fold_dir = os.path.join(RUNS_ROOT, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    train_ds = SliceDataset(train_cases)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = DDUnet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = DiceLossBinary()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_dice = -1.0
    best_path = os.path.join(fold_dir, f"best_model_fold{fold_idx}.pth")
    last_path = os.path.join(fold_dir, f"last_model_fold{fold_idx}.pth")
    log_csv = os.path.join(fold_dir, "train_log.csv")

    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_dice", "best_val_dice"])

        for epoch in range(1, EPOCHS + 1):
            model.train()
            losses = []

            for batch in train_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = model(images)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                losses.append(float(loss.item()))

            train_loss = float(np.mean(losses)) if losses else 0.0
            val_dice = validate(model, val_cases, device)

            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_dice": val_dice,
            }
            torch.save(ckpt, last_path)
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(ckpt, best_path)

            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_dice:.6f}", f"{best_dice:.6f}"])
            print(
                f"[Fold {fold_idx}] Epoch {epoch:03d}/{EPOCHS} | "
                f"TrainLoss={train_loss:.6f} | ValDice={val_dice:.6f} | Best={best_dice:.6f}"
            )

    return {
        "fold": fold_idx,
        "best_dice": best_dice,
        "best_model": best_path,
        "last_model": last_path,
    }


def main() -> None:
    set_seed(SEED)
    os.makedirs(RUNS_ROOT, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("DDUnet (paper-style) 2D training")
    print(f"Device: {device}")
    print(f"Train images: {IMAGES_TR_DIR}")
    print(f"Train labels: {LABELS_TR_DIR}")
    print(f"Epochs: {EPOCHS}, Batch size: {BATCH_SIZE}")
    print("=" * 80)

    train_cases = collect_cases(IMAGES_TR_DIR, LABELS_TR_DIR)
    split_pairs = build_kfold_indices(len(train_cases), NUM_FOLDS, SEED)

    fold_records: List[Dict[str, object]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(split_pairs):
        fold_train = [train_cases[i] for i in train_idx]
        fold_val = [train_cases[i] for i in val_idx]
        record = train_fold(fold_idx, fold_train, fold_val, device)
        fold_records.append(record)

    fold_csv = os.path.join(RUNS_ROOT, "fold_summary.csv")
    with open(fold_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "best_val_dice", "best_model", "last_model"])
        for r in fold_records:
            writer.writerow([r["fold"], f"{float(r['best_dice']):.6f}", r["best_model"], r["last_model"]])

    best_fold_record = max(fold_records, key=lambda x: float(x["best_dice"]))
    print("\nTraining finished.")
    print(
        f"Best fold: {best_fold_record['fold']} with val Dice={float(best_fold_record['best_dice']):.6f}"
    )
    print(f"Fold summary saved: {fold_csv}")
    print("Run test next: python test_ddunet.py")


if __name__ == "__main__":
    main()
