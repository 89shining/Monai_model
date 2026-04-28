import csv
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List

from sklearn.model_selection import KFold

# =========================
# Run Config (edit here)
# =========================
NII_DATA_ROOT = r"D:\data\your_dataset_root"  # raw NIfTI root: imagesTr/labelsTr/imagesTs/labelsTs
SAVE_ROOT = r"D:\project\Monai_model\Deeplabv3+\output"
DATA2D_ROOT = os.path.join(SAVE_ROOT, "Data2DCache")  # generated PNG slices
RUNS_ROOT = os.path.join(SAVE_ROOT, "TrainResults")
PRED_SAVE_DIR = os.path.join(SAVE_ROOT, "TestResults")

CUDA_VISIBLE_DEVICES = "0"

NUM_CLASSES = 2
BACKBONE = "mobilenet"
DOWNSAMPLE_FACTOR = 16

NUM_FOLDS = 5
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 20
TRAIN_BATCH_SIZE = 8
TEST_BATCH_SIZE = 4
LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42
WORKERS = 4
INPUT_H, INPUT_W = 512, 512

PREPROCESS_AXIS = "z"      # z/y/x
PREPROCESS_FORCE = False
KEEP_2D_CACHE = False      # False: delete DATA2D_ROOT after test
AUTO_RESUME = True
PRETRAINED_BACKBONE = False
# =========================


def run_cmd(cmd: List[str], env: Dict[str, str]):
    print("\n[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def read_case_ids_from_manifest(data2d_root: str):
    manifest = os.path.join(data2d_root, "train_manifest.csv")
    if not os.path.exists(manifest):
        raise FileNotFoundError(f"Missing manifest: {manifest}")

    case_ids = set()
    with open(manifest, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            case_ids.add(r["case_id"])

    case_ids = sorted(case_ids)
    if not case_ids:
        raise ValueError("No case IDs in train_manifest.csv")
    return case_ids


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


def write_cv_results(runs_root: str, num_folds: int, seed: int, data2d_root: str):
    case_ids = read_case_ids_from_manifest(data2d_root)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

    split_sizes = {}
    for fold, (tr_idx, va_idx) in enumerate(kf.split(case_ids)):
        split_sizes[fold] = (len(tr_idx), len(va_idx))

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
        rows.append({"fold": fold, "best_epoch": best_epoch, "best_dice_fg": best_dice, "num_train": ntr, "num_val": nval, "checkpoint": ckpt})

    cv_path = os.path.join(runs_root, "cv_results.csv")
    with open(cv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "best_epoch", "best_dice_fg", "num_train", "num_val", "checkpoint"])
        writer.writeheader()
        for r in rows:
            rr = dict(r)
            rr["best_dice_fg"] = f"{rr['best_dice_fg']:.6f}"
            writer.writerow(rr)

    best_row = max(rows, key=lambda x: x["best_dice_fg"])
    print("\n===== CV Summary =====")
    print(f"Saved: {cv_path}")
    print(f"Best fold: {best_row['fold']} (DiceFG={best_row['best_dice_fg']:.4f})")


def main():
    os.makedirs(DATA2D_ROOT, exist_ok=True)
    os.makedirs(RUNS_ROOT, exist_ok=True)
    os.makedirs(PRED_SAVE_DIR, exist_ok=True)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    preprocess_py = os.path.join(os.path.dirname(__file__), "preprocess_nii_to_2d.py")
    train_py = os.path.join(os.path.dirname(__file__), "train.py")
    test_py = os.path.join(os.path.dirname(__file__), "test.py")

    preprocess_cmd = [
        sys.executable,
        preprocess_py,
        "--source_root", NII_DATA_ROOT,
        "--output_root", DATA2D_ROOT,
        "--num_classes", str(NUM_CLASSES),
        "--axis", PREPROCESS_AXIS,
    ]
    if PREPROCESS_FORCE:
        preprocess_cmd.append("--force")
    run_cmd(preprocess_cmd, env)

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
            "--data_root", DATA2D_ROOT,
            "--runs_root", RUNS_ROOT,
            "--num_classes", str(NUM_CLASSES),
            "--backbone", BACKBONE,
            "--downsample_factor", str(DOWNSAMPLE_FACTOR),
            "--num_folds", str(NUM_FOLDS),
            "--max_epochs", str(MAX_EPOCHS),
            "--early_stop_patience", str(EARLY_STOP_PATIENCE),
            "--batch_size", str(TRAIN_BATCH_SIZE),
            "--lr", str(LR),
            "--weight_decay", str(WEIGHT_DECAY),
            "--seed", str(SEED),
            "--workers", str(WORKERS),
            "--input_h", str(INPUT_H),
            "--input_w", str(INPUT_W),
            "--only_fold", str(fold),
        ]
        if PRETRAINED_BACKBONE:
            cmd.append("--pretrained_backbone")
        if AUTO_RESUME:
            cmd.append("--resume")

        run_cmd(cmd, env)

        os.makedirs(fold_dir, exist_ok=True)
        with open(done_marker, "w", encoding="utf-8") as f:
            f.write(f"done_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    write_cv_results(RUNS_ROOT, NUM_FOLDS, SEED, DATA2D_ROOT)

    test_cmd = [
        sys.executable,
        test_py,
        "--data_root", DATA2D_ROOT,
        "--runs_root", RUNS_ROOT,
        "--save_dir", PRED_SAVE_DIR,
        "--num_classes", str(NUM_CLASSES),
        "--backbone", BACKBONE,
        "--downsample_factor", str(DOWNSAMPLE_FACTOR),
        "--batch_size", str(TEST_BATCH_SIZE),
        "--workers", str(WORKERS),
        "--input_h", str(INPUT_H),
        "--input_w", str(INPUT_W),
    ]
    run_cmd(test_cmd, env)

    if not KEEP_2D_CACHE and os.path.isdir(DATA2D_ROOT):
        shutil.rmtree(DATA2D_ROOT, ignore_errors=True)
        print('[Cleanup] Removed 2D cache: ' + DATA2D_ROOT)

if __name__ == "__main__":
    main()

