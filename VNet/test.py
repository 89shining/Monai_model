import argparse
import glob
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.transforms import (
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    Lambdad,
    LoadImaged,
    Orientationd,
    Resized,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
)

from train import VNet3DSeg


def parse_args():
    parser = argparse.ArgumentParser(description="VNet testing with full-volume inference and inverse transform.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--runs_root", type=str, default="./runs/VNet")
    parser.add_argument("--save_dir", type=str, default="./predictions")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)

    parser.add_argument("--roi_x", type=int, default=256)
    parser.add_argument("--roi_y", type=int, default=256)
    parser.add_argument("--roi_z", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=None, help="Use one fold only. If omitted, ensemble all folds.")

    parser.add_argument("--a_min", type=float, default=-160.0)
    parser.add_argument("--a_max", type=float, default=240.0)
    parser.add_argument("--pixdim_x", type=float, default=1.0)
    parser.add_argument("--pixdim_y", type=float, default=1.0)
    parser.add_argument("--pixdim_z", type=float, default=1.0)
    parser.add_argument("--axcodes", type=str, default="RAS")

    parser.add_argument("--image_dir", type=str, default="imagesTs")
    parser.add_argument("--label_dir", type=str, default="labelsTs")
    parser.add_argument("--save_prob", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def binarize_label(x):
    if isinstance(x, torch.Tensor):
        return (x > 0).float()
    return (x > 0).astype(np.float32)


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


def _find_images(folder: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(folder, "*_0000.nii.gz")) + glob.glob(os.path.join(folder, "*_0000.nii")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(folder, "*.nii.gz")) + glob.glob(os.path.join(folder, "*.nii")))
    return paths


def build_test_data(data_root: str, image_dir: str, label_dir: str) -> List[Dict[str, str]]:
    image_folder = os.path.join(data_root, image_dir)
    label_folder = os.path.join(data_root, label_dir)

    images = _find_images(image_folder)
    if not images:
        raise FileNotFoundError(f"No test images found under: {image_folder}")

    image_map = {image_case_id(p): p for p in images}
    data: List[Dict[str, str]] = []

    labels = sorted(glob.glob(os.path.join(label_folder, "*.nii.gz")) + glob.glob(os.path.join(label_folder, "*.nii")))
    if labels:
        label_map = {label_case_id(p): p for p in labels}
        image_ids = set(image_map.keys())
        label_ids = set(label_map.keys())
        if image_ids != label_ids:
            missing_labels = sorted(image_ids - label_ids)
            missing_images = sorted(label_ids - image_ids)
            raise ValueError(
                "Image/label case IDs are inconsistent. "
                f"missing_labels={len(missing_labels)}, missing_images={len(missing_images)}\n"
                f"First missing labels: {missing_labels[:5]}\n"
                f"First missing images: {missing_images[:5]}"
            )
        for cid in sorted(image_ids):
            data.append({"case_id": cid, "image": image_map[cid], "label": label_map[cid]})
    else:
        for cid in sorted(image_map.keys()):
            data.append({"case_id": cid, "image": image_map[cid]})

    return data


def load_model(runs_root: str, fold: int, device: torch.device):
    model_path = os.path.join(runs_root, f"fold_{fold}", f"best_model_fold{fold}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = VNet3DSeg().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()
    return model, model_path


def get_test_transforms(args, has_label: bool):
    keys = ["image", "label"] if has_label else ["image"]
    modes = ("bilinear", "nearest") if has_label else "bilinear"

    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes=args.axcodes),
        Spacingd(keys=keys, pixdim=(args.pixdim_x, args.pixdim_y, args.pixdim_z), mode=modes),
        ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=0.0, b_max=1.0, clip=True),
        Resized(keys=keys, spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode=modes),
    ]
    if has_label:
        transforms.append(Lambdad(keys=["label"], func=binarize_label))
    transforms.append(EnsureTyped(keys=keys, track_meta=True))
    return Compose(transforms)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    data_dicts = build_test_data(args.data_root, args.image_dir, args.label_dir)
    has_label = "label" in data_dicts[0]

    folds = [args.fold] if args.fold is not None else list(range(args.num_folds))
    models = []
    model_paths = []
    for fold in folds:
        model, model_path = load_model(args.runs_root, fold, device)
        models.append(model)
        model_paths.append(model_path)

    test_transforms = get_test_transforms(args, has_label)

    post_transforms = [
        Invertd(
            keys=["pred_prob"],
            transform=test_transforms,
            orig_keys=["image"],
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys=["pred"], threshold=args.threshold),
        Invertd(
            keys=["pred"],
            transform=test_transforms,
            orig_keys=["image"],
            nearest_interp=True,
            to_tensor=True,
        ),
        SaveImaged(
            keys=["pred"],
            meta_keys=["pred_meta_dict"],
            output_dir=args.save_dir,
            output_postfix="",
            output_ext=".nii.gz",
            separate_folder=False,
            output_dtype="uint8",
            resample=False,
            print_log=False,
        ),
    ]
    if args.save_prob:
        post_transforms.append(
            SaveImaged(
                keys=["pred_prob"],
                meta_keys=["pred_prob_meta_dict"],
                output_dir=args.save_dir,
                output_postfix="prob",
                output_ext=".nii.gz",
                separate_folder=False,
                output_dtype="float32",
                resample=False,
                print_log=False,
            )
        )
    post_transforms = Compose(post_transforms)

    test_ds = Dataset(data_dicts, test_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    print("\n===== VNet Test: full-volume inference =====")
    print(f"Device: {device}")
    print(f"Cases: {len(data_dicts)} | Labels available: {has_label}")
    print(f"Folds: {folds}")
    print(f"ROI size: {(args.roi_x, args.roi_y, args.roi_z)}")
    for p in model_paths:
        print(f"Model: {p}")
    print(f"Save dir: {args.save_dir}")

    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(device)
            case_id = batch.get("case_id", ["unknown"])[0] if isinstance(batch.get("case_id", None), list) else batch.get("case_id", "unknown")

            prob_sum: Optional[torch.Tensor] = None
            for model in models:
                logits = model(image)
                probs = torch.sigmoid(logits)
                prob_sum = probs if prob_sum is None else (prob_sum + probs)

            prob_mean = torch.clamp(prob_sum / float(len(models)), 0.0, 1.0)
            batch["pred_prob"] = prob_mean.detach().cpu()
            batch["pred"] = prob_mean.detach().cpu()

            if args.debug:
                print(
                    f"Case {case_id} | image={tuple(image.shape)} | "
                    f"prob={tuple(prob_mean.shape)} | "
                    f"prob_min={prob_mean.min().item():.4f}, prob_max={prob_mean.max().item():.4f}"
                )

            for item in decollate_batch(batch):
                out = post_transforms(item)
                meta = out.get("pred_meta_dict") if isinstance(out, dict) else None
                saved_to = None if meta is None else meta.get("saved_to")
                if saved_to:
                    print(f"Saved: {saved_to}")
                else:
                    print("Saved one prediction file.")

    print("Done.")


if __name__ == "__main__":
    main()
