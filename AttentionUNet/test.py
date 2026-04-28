import argparse
import csv
import glob
import os

import SimpleITK as sitk
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.networks.nets import AttentionUnet
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, ScaleIntensityd


def parse_args():
    parser = argparse.ArgumentParser(description="AttentionUNet testing with best-fold auto selection.")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root directory.")
    parser.add_argument("--runs_root", type=str, default="./runs/AttentionUNet", help="Training runs root.")
    parser.add_argument("--save_dir", type=str, default="./predictions", help="Prediction output directory.")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size.")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers.")
    parser.add_argument("--roi_x", type=int, default=96, help="Sliding-window ROI x.")
    parser.add_argument("--roi_y", type=int, default=96, help="Sliding-window ROI y.")
    parser.add_argument("--roi_z", type=int, default=96, help="Sliding-window ROI z.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold.")
    parser.add_argument("--fold", type=int, default=None, help="Force a specific fold. If omitted, choose best fold from cv_results.csv.")
    return parser.parse_args()


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


def build_test_data(data_root: str):
    images = sorted(glob.glob(os.path.join(data_root, "imagesTs", "*_0000.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_root, "labelsTs", "*.nii.gz")))

    if not images:
        raise FileNotFoundError(f"No test images found under: {os.path.join(data_root, 'imagesTs')}")
    if not labels:
        raise FileNotFoundError(f"No test labels found under: {os.path.join(data_root, 'labelsTs')}")

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

    case_ids = sorted(image_ids)
    return [{"image": image_map[cid], "label": label_map[cid]} for cid in case_ids]


def select_best_fold_from_cv(runs_root: str) -> int:
    cv_path = os.path.join(runs_root, "cv_results.csv")
    if not os.path.exists(cv_path):
        raise FileNotFoundError(f"cv_results.csv not found: {cv_path}")

    with open(cv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"cv_results.csv is empty: {cv_path}")

    required = {"fold", "best_dice_fg"}
    if not required.issubset(set(rows[0].keys())):
        raise ValueError("cv_results.csv must contain columns: fold,best_dice_fg")

    best_row = max(rows, key=lambda r: float(r["best_dice_fg"]))
    return int(best_row["fold"])


def load_model(runs_root: str, fold: int, device: torch.device):
    model_path = os.path.join(runs_root, f"fold_{fold}", f"best_model_fold{fold}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, model_path


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    fold = args.fold if args.fold is not None else select_best_fold_from_cv(args.runs_root)
    model, model_path = load_model(args.runs_root, fold, device)

    data_dicts = build_test_data(args.data_root)
    test_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys="image"),
        EnsureTyped(keys=["image", "label"]),
    ])
    test_ds = Dataset(data_dicts, test_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    print("\n===== AttentionUNet Test (best fold only) =====")
    print(f"Device: {device}")
    print(f"Selected fold: {fold}")
    print(f"Model: {model_path}")
    print(f"Save dir: {args.save_dir}")

    roi_size = (args.roi_x, args.roi_y, args.roi_z)
    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(device)
            output = sliding_window_inference(image, roi_size=roi_size, sw_batch_size=1, predictor=model)
            output = torch.sigmoid(output)
            output = (output > args.threshold).float()

            bs = output.shape[0]
            for i in range(bs):
                pred = output[i, 0].cpu().numpy().astype("uint8")

                image_path = batch["image_meta_dict"]["filename_or_obj"][i]
                label_path = batch["label_meta_dict"]["filename_or_obj"][i]
                save_name = os.path.basename(label_path)

                itk_img = sitk.ReadImage(image_path)
                pred_itk = sitk.GetImageFromArray(pred)
                pred_itk.CopyInformation(itk_img)

                out_path = os.path.join(args.save_dir, save_name)
                sitk.WriteImage(pred_itk, out_path)
                print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
