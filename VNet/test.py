import argparse
import csv
import glob
import os

import SimpleITK as sitk
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.networks.nets import VNet
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, ScaleIntensityRanged


def parse_args():
    parser = argparse.ArgumentParser(description="VNet testing with best-fold auto selection.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--runs_root", type=str, default="./runs/VNet")
    parser.add_argument("--save_dir", type=str, default="./predictions")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--roi_x", type=int, default=128)
    parser.add_argument("--roi_y", type=int, default=128)
    parser.add_argument("--roi_z", type=int, default=96)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fold", type=int, default=None)
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

    ids = sorted(image_map.keys())
    missing = [i for i in ids if i not in label_map]
    if missing:
        raise ValueError(f"Missing labels for {len(missing)} test cases, e.g. {missing[:5]}")
    return [{"image": image_map[i], "label": label_map[i]} for i in ids]


def select_best_fold_from_cv(runs_root: str) -> int:
    cv_path = os.path.join(runs_root, "cv_results.csv")
    if not os.path.exists(cv_path):
        raise FileNotFoundError(f"cv_results.csv not found: {cv_path}")

    with open(cv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"cv_results.csv is empty: {cv_path}")

    best_row = max(rows, key=lambda r: float(r["best_dice_fg"]))
    return int(best_row["fold"])


def load_model(runs_root: str, fold: int, device: torch.device):
    model_path = os.path.join(runs_root, f"fold_{fold}", f"best_model_fold{fold}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = VNet(spatial_dims=3, in_channels=1, out_channels=1).to(device)
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
    tf = Compose([
        LoadImaged(["image", "label"]),
        EnsureChannelFirstd(["image", "label"]),
        ScaleIntensityRanged("image", -1000, 1000, 0, 1, True),
        EnsureTyped(["image", "label"]),
    ])

    loader = DataLoader(
        Dataset(data_dicts, tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    print(f"Selected fold: {fold}")
    print(f"Model: {model_path}")

    roi_size = (args.roi_x, args.roi_y, args.roi_z)
    with torch.no_grad():
        for b in loader:
            x = b["image"].to(device)

            pred = sliding_window_inference(
                x, roi_size, 1, model,
                sw_device=device,
                device=torch.device("cpu")
            )

            pred = (torch.sigmoid(pred) > args.threshold).float()
            bs = pred.shape[0]
            for i in range(bs):
                arr = pred[i, 0].cpu().numpy().astype("uint8")
                image_path = b["image_meta_dict"]["filename_or_obj"][i]
                label_path = b["label_meta_dict"]["filename_or_obj"][i]
                out_name = os.path.basename(label_path)

                img = sitk.ReadImage(image_path)
                out = sitk.GetImageFromArray(arr)
                out.CopyInformation(img)

                out_path = os.path.join(args.save_dir, out_name)
                sitk.WriteImage(out, out_path)
                print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
