import csv
import glob
import os
import subprocess
import sys
import time
from typing import Dict, List

from sklearn.model_selection import KFold

# =========================
# Run Config (edit here)
# =========================
DATA_ROOT = r"D:\data\your_dataset_root"
SAVE_ROOT = r"D:\project\Monai_model\AttentionUNet\output"
RUNS_ROOT = os.path.join(SAVE_ROOT, "TrainResults")
PRED_SAVE_DIR = os.path.join(SAVE_ROOT, "TestResults")

CUDA_VISIBLE_DEVICES = "0"   # e.g. "0" / "1" / "0,1"

NUM_FOLDS = 5
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 30
TRAIN_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
LR = 1e-4
SEED = 42
TRAIN_WORKERS = 4
VAL_WORKERS = 2
TEST_WORKERS = 2
ROI_X, ROI_Y, ROI_Z = 96, 96, 96
THRESHOLD = 0.5

AUTO_RESUME = True
# =========================


def run_cmd(cmd: List[str], env: Dict[str, str]):
    print("\n[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def read_best_from_fold_epoch_csv(fold_dir: str):
    epoch_csv = os.path.join(fold_dir, "epoch_metrics.csv")
    if not os.path.exists(epoch_csv):
        raise FileNotFoundError(f"epoch_metrics.csv not found: {epoch_csv}")

    with open(epoch_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows in: {epoch_csv}")

    best_row = max(rows, key=lambda r: float(r["val_dice_fg"]))
    return int(best_row["epoch"]), float(best_row["val_dice_fg"])


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


def collect_case_ids(data_root: str):
    images = sorted(glob.glob(os.path.join(data_root, "imagesTr", "*_0000.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_root, "labelsTr", "*.nii.gz")))

    if not images:
        raise FileNotFoundError(f"No training images found under: {os.path.join(data_root, 'imagesTr')}")
    if not labels:
        raise FileNotFoundError(f"No training labels found under: {os.path.join(data_root, 'labelsTr')}")

    image_map = {image_case_id(p): p for p in images}
    label_map = {label_case_id(p): p for p in labels}

    image_ids = set(image_map.keys())
    label_ids = set(label_map.keys())
    if image_ids != label_ids:
        missing_labels = sorted(image_ids - label_ids)
        missing_images = sorted(label_ids - image_ids)
        raise ValueError(
            "Image/label case IDs are inconsistent. "
            f"missing_labels={len(missing_labels)}, missing_images={len(missing_images)}"
        )

    return sorted(image_ids)


def write_cv_results(runs_root: str, num_folds: int, seed: int, data_root: str):
    case_ids = collect_case_ids(data_root)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    split_sizes = {}
    for fold, (train_idx, val_idx) in enumerate(kf.split(case_ids)):
        split_sizes[fold] = (len(train_idx), len(val_idx))

    rows = []
    for fold in range(num_folds):
        fold_dir = os.path.join(runs_root, f"fold_{fold}")
        ckpt = os.path.join(fold_dir, f"best_model_fold{fold}.pth")
        done_marker = os.path.join(fold_dir, "fold_done.flag")

        if not os.path.exists(done_marker):
            raise FileNotFoundError(f"Missing done marker: {done_marker}")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

        best_epoch, best_dice = read_best_from_fold_epoch_csv(fold_dir)
        ntr, nval = split_sizes[fold]
        rows.append(
            {
                "fold": fold,
                "best_epoch": best_epoch,
                "best_dice_fg": best_dice,
                "num_train": ntr,
                "num_val": nval,
                "checkpoint": ckpt,
            }
        )

    cv_path = os.path.join(runs_root, "cv_results.csv")
    with open(cv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fold", "best_epoch", "best_dice_fg", "num_train", "num_val", "checkpoint"],
        )
        writer.writeheader()
        for r in rows:
            r = dict(r)
            r["best_dice_fg"] = f"{r['best_dice_fg']:.6f}"
            writer.writerow(r)

    best_row = max(rows, key=lambda x: x["best_dice_fg"])
    print("\n===== CV Summary =====")
    print(f"Saved: {cv_path}")
    print(f"Best fold: {best_row['fold']} (DiceFG={best_row['best_dice_fg']:.4f})")


def main():
    os.makedirs(RUNS_ROOT, exist_ok=True)
    os.makedirs(PRED_SAVE_DIR, exist_ok=True)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    train_py = os.path.join(os.path.dirname(__file__), "train.py")
    test_py = os.path.join(os.path.dirname(__file__), "test.py")

    for fold in range(NUM_FOLDS):
        fold_dir = os.path.join(RUNS_ROOT, f"fold_{fold}")
        done_marker = os.path.join(fold_dir, "fold_done.flag")
        ckpt = os.path.join(fold_dir, f"best_model_fold{fold}.pth")

        if AUTO_RESUME and os.path.exists(done_marker) and os.path.exists(ckpt):
            print(f"[Resume] Skip completed fold {fold}")
            continue

        cmd = [
            sys.executable,
            train_py,
            "--data_root",
            DATA_ROOT,
            "--runs_root",
            RUNS_ROOT,
            "--num_folds",
            str(NUM_FOLDS),
            "--max_epochs",
            str(MAX_EPOCHS),
            "--early_stop_patience",
            str(EARLY_STOP_PATIENCE),
            "--batch_size",
            str(TRAIN_BATCH_SIZE),
            "--lr",
            str(LR),
            "--seed",
            str(SEED),
            "--train_workers",
            str(TRAIN_WORKERS),
            "--val_workers",
            str(VAL_WORKERS),
            "--roi_x",
            str(ROI_X),
            "--roi_y",
            str(ROI_Y),
            "--roi_z",
            str(ROI_Z),
            "--only_fold",
            str(fold),
        ]
        if AUTO_RESUME:
            cmd.append("--resume")
        run_cmd(cmd, env)

        os.makedirs(fold_dir, exist_ok=True)
        with open(done_marker, "w", encoding="utf-8") as f:
            f.write(f"done_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    write_cv_results(RUNS_ROOT, NUM_FOLDS, SEED, DATA_ROOT)

    test_cmd = [
        sys.executable,
        test_py,
        "--data_root",
        DATA_ROOT,
        "--runs_root",
        RUNS_ROOT,
        "--save_dir",
        PRED_SAVE_DIR,
        "--batch_size",
        str(TEST_BATCH_SIZE),
        "--workers",
        str(TEST_WORKERS),
        "--roi_x",
        str(ROI_X),
        "--roi_y",
        str(ROI_Y),
        "--roi_z",
        str(ROI_Z),
        "--threshold",
        str(THRESHOLD),
    ]
    run_cmd(test_cmd, env)


if __name__ == "__main__":
    main()
