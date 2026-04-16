import argparse
import csv
import os
import sys
from typing import List

import numpy as np
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.transforms import SaveImage

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from pipeline import (  # noqa: E402
    build_model,
    collect_files,
    dice_from_probs,
    get_eval_transforms,
    infer_volume_probs,
)


DATA_ROOT = r"D:\project\Monai_model\data"
IMAGES_TS_DIR = os.path.join(DATA_ROOT, "imagesTs")
LABELS_TS_DIR = os.path.join(DATA_ROOT, "labelsTs")

INPUT_SHAPE = (512, 512)
SLICE_BATCH_SIZE = 8
NUM_WORKERS = 4

BACKBONE = "xception"
DOWNSAMPLE_FACTOR = 16
SAVE_PREDICTIONS = True


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepLabV3+ test on imagesTs/labelsTs.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(".", "runs", "DeepLabV3Plus", "fold_0", "best_model_fold0.pth"),
        help="Path to .pth checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(".", "runs", "DeepLabV3Plus", "manual_test"),
        help="Directory to save metrics and predictions.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print("=" * 80)
    print("DeepLabV3+ Testing")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test images: {IMAGES_TS_DIR}")
    print(f"Test labels: {LABELS_TS_DIR}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 80)

    test_files = collect_files(IMAGES_TS_DIR, LABELS_TS_DIR)
    test_ds = Dataset(data=test_files, transform=get_eval_transforms())
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    model = build_model(
        device=device,
        backbone=BACKBONE,
        num_classes=1,
        downsample_factor=DOWNSAMPLE_FACTOR,
        pretrained_backbone=False,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    pred_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "test_metrics.csv")

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
                print(f"Test case: {case_name}, Dice={dice_val:.6f}")

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
    print(f"Test mean Dice: {mean_dice:.6f}")
    print(f"Metrics saved to: {csv_path}")


if __name__ == "__main__":
    main()

