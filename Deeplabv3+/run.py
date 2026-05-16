import argparse
import csv
import os
import subprocess
import sys
from typing import List

from sklearn.model_selection import KFold


DATA_ROOT = "/home/wusi/Project_crop/Data/Rectal_146/RectalCTV_All"
SAVE_ROOT = "/home/wusi/Project_crop/Data/Rectal_146/Networks/Deeplabv3+/RectalCTV_All"
RUNS_ROOT = os.path.join(SAVE_ROOT, "TrainResults")
PRED_SAVE_DIR = os.path.join(SAVE_ROOT, "TestResults")

SEED = 42
NUM_FOLDS = 5
EPOCHS = 100
EARLY_STOP = 15
BATCH_SIZE = 8
NUM_WORKERS = 0
LR = 1e-4
BACKBONE = "xception"
AUTO_RESUME = True


def run_cmd(cmd: List[str], env: dict):
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def read_best_fold_from_cv(cv_path: str) -> int:
    with open(cv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    best = max(rows, key=lambda r: float(r["best_dice_fg"]))
    return int(best["fold"])


def write_cv_results(runs_root: str):
    rows = []
    for fold in range(NUM_FOLDS):
        fold_dir = os.path.join(runs_root, f"fold_{fold}")
        metrics = os.path.join(fold_dir, "epoch_metrics.csv")
        best_model = os.path.join(fold_dir, f"best_model_fold{fold}.pth")
        if not (os.path.exists(metrics) and os.path.exists(best_model)):
            raise FileNotFoundError(f"Missing fold outputs: {fold_dir}")

        with open(metrics, "r", encoding="utf-8") as f:
            mrows = list(csv.DictReader(f))
        best_row = max(mrows, key=lambda r: float(r["val_dice_fg"]))
        rows.append(
            {
                "fold": fold,
                "best_epoch": int(best_row["epoch"]),
                "best_dice_fg": float(best_row["val_dice_fg"]),
                "checkpoint": best_model,
            }
        )

    cv_path = os.path.join(runs_root, "cv_results.csv")
    with open(cv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "best_epoch", "best_dice_fg", "checkpoint"])
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            r2["best_dice_fg"] = f"{r['best_dice_fg']:.6f}"
            w.writerow(r2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--save_root", type=str, default=SAVE_ROOT)
    args = parser.parse_args()

    runs_root = os.path.join(args.save_root, "TrainResults")
    pred_save_dir = os.path.join(args.save_root, "TestResults")
    os.makedirs(runs_root, exist_ok=True)
    os.makedirs(pred_save_dir, exist_ok=True)

    env = dict(os.environ)
    train_py = os.path.join(os.path.dirname(__file__), "train.py")
    test_py = os.path.join(os.path.dirname(__file__), "test.py")

    for fold in range(NUM_FOLDS):
        fold_dir = os.path.join(runs_root, f"fold_{fold}")
        done = os.path.join(fold_dir, "fold_done.flag")
        best_model = os.path.join(fold_dir, f"best_model_fold{fold}.pth")
        if AUTO_RESUME and os.path.exists(done) and os.path.exists(best_model):
            print(f"[Resume] skip fold {fold}")
            continue

        cmd = [
            sys.executable,
            train_py,
            "--data_root",
            args.data_root,
            "--runs_root",
            runs_root,
            "--fold",
            str(fold),
            "--num_folds",
            str(NUM_FOLDS),
            "--seed",
            str(SEED),
            "--epochs",
            str(EPOCHS),
            "--early_stop",
            str(EARLY_STOP),
            "--batch_size",
            str(BATCH_SIZE),
            "--num_workers",
            str(NUM_WORKERS),
            "--lr",
            str(LR),
            "--backbone",
            BACKBONE,
        ]
        if AUTO_RESUME:
            cmd.append("--resume")
        run_cmd(cmd, env)

    write_cv_results(runs_root)
    best_fold = read_best_fold_from_cv(os.path.join(runs_root, "cv_results.csv"))
    print(f"[Test] best fold={best_fold}")

    test_cmd = [
        sys.executable,
        test_py,
        "--data_root",
        args.data_root,
        "--runs_root",
        runs_root,
        "--save_dir",
        pred_save_dir,
        "--fold",
        str(best_fold),
        "--backbone",
        BACKBONE,
    ]
    run_cmd(test_cmd, env)


if __name__ == "__main__":
    main()
