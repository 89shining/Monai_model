import argparse
import os
import sys

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from train_attention_unet import (
    IMAGES_TS_DIR,
    LABELS_TS_DIR,
    RUNS_ROOT,
    SEED,
    build_model,
    collect_test_files,
    run_test_for_fold,
    set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone test for AttentionUnet with same logic as training script.")
    parser.add_argument("--fold_idx", type=int, default=0, help="Fold index used only for logging and default checkpoint name.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to checkpoint .pth. If empty, uses ../runs/AttentionUnet/fold_{fold_idx}/best_model_fold{fold_idx}.pth",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(RUNS_ROOT, "manual_test"),
        help="Directory to save test_metrics.csv and predictions/.",
    )
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = args.checkpoint or os.path.join(
        RUNS_ROOT, f"fold_{args.fold_idx}", f"best_model_fold{args.fold_idx}.pth"
    )

    print("=" * 80)
    print("AttentionUnet Standalone Testing")
    print(f"Device: {device}")
    print(f"Test images: {IMAGES_TS_DIR}")
    print(f"Test labels: {LABELS_TS_DIR}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 80)

    test_files = collect_test_files(IMAGES_TS_DIR, LABELS_TS_DIR)
    model = build_model(device)
    os.makedirs(args.output_dir, exist_ok=True)

    run_test_for_fold(
        fold_idx=args.fold_idx,
        best_ckpt=checkpoint,
        output_dir=args.output_dir,
        model=model,
        test_files=test_files,
        device=device,
    )


if __name__ == "__main__":
    main()
