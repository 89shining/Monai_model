import os
import subprocess
import sys
import time
from typing import Dict, List

# =========================
# Run Config (edit here)
# =========================
DATA_ROOT = r"D:\data\your_dataset_root"
SAVE_ROOT = r"D:\project\Monai_model\VNet\output"
RUNS_ROOT = os.path.join(SAVE_ROOT, "TrainResults")
PRED_SAVE_DIR = os.path.join(SAVE_ROOT, "TestResults")

CUDA_VISIBLE_DEVICES = "0"

NUM_FOLDS = 5
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 30
TRAIN_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
LR = 3e-4
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
            "--data_root", DATA_ROOT,
            "--runs_root", RUNS_ROOT,
            "--num_folds", str(NUM_FOLDS),
            "--max_epochs", str(MAX_EPOCHS),
            "--early_stop_patience", str(EARLY_STOP_PATIENCE),
            "--batch_size", str(TRAIN_BATCH_SIZE),
            "--lr", str(LR),
            "--seed", str(SEED),
            "--train_workers", str(TRAIN_WORKERS),
            "--val_workers", str(VAL_WORKERS),
            "--roi_x", str(ROI_X),
            "--roi_y", str(ROI_Y),
            "--roi_z", str(ROI_Z),
            "--only_fold", str(fold),
        ]
        if AUTO_RESUME:
            cmd.append("--resume")

        run_cmd(cmd, env)

        os.makedirs(fold_dir, exist_ok=True)
        with open(done_marker, "w", encoding="utf-8") as f:
            f.write(f"done_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    test_cmd = [
        sys.executable,
        test_py,
        "--data_root", DATA_ROOT,
        "--runs_root", RUNS_ROOT,
        "--save_dir", PRED_SAVE_DIR,
        "--batch_size", str(TEST_BATCH_SIZE),
        "--workers", str(TEST_WORKERS),
        "--roi_x", str(ROI_X),
        "--roi_y", str(ROI_Y),
        "--roi_z", str(ROI_Z),
        "--threshold", str(THRESHOLD),
    ]

    run_cmd(test_cmd, env)


if __name__ == "__main__":
    main()
