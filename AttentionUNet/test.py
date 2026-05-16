import argparse
import glob
import os

import SimpleITK as sitk
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)

from train import AttentionUNet3D


def parse_args():
    parser = argparse.ArgumentParser(description="AttentionUNet testing with fold ensemble.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--runs_root", type=str, default="./runs/AttentionUNet")
    parser.add_argument("--save_dir", type=str, default="./predictions")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--roi_x", type=int, default=96)
    parser.add_argument("--roi_y", type=int, default=96)
    parser.add_argument("--roi_z", type=int, default=64)
    parser.add_argument("--sw_batch_size", type=int, default=2)
    parser.add_argument("--infer_overlap", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=None)

    parser.add_argument("--a_min", type=float, default=-160.0)
    parser.add_argument("--a_max", type=float, default=240.0)
    parser.add_argument("--pixdim_x", type=float, default=1.0)
    parser.add_argument("--pixdim_y", type=float, default=1.0)
    parser.add_argument("--pixdim_z", type=float, default=1.0)
    parser.add_argument("--axcodes", type=str, default="RAS")

    parser.add_argument("--f1", type=int, default=32)
    parser.add_argument("--f2", type=int, default=64)
    parser.add_argument("--f3", type=int, default=128)
    parser.add_argument("--f4", type=int, default=256)
    parser.add_argument("--f5", type=int, default=512)
    return parser.parse_args()


def _binarize_label(x):
    return (x > 0).astype(x.dtype)


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
    images = sorted(glob.glob(os.path.join(data_root, "imagesTs", "*_0000.nii.gz")) + glob.glob(os.path.join(data_root, "imagesTs", "*_0000.nii")))
    labels = sorted(glob.glob(os.path.join(data_root, "labelsTs", "*.nii.gz")) + glob.glob(os.path.join(data_root, "labelsTs", "*.nii")))

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


def load_model(runs_root: str, fold: int, device: torch.device, args):
    model_path = os.path.join(runs_root, f"fold_{fold}", f"best_model_fold{fold}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = AttentionUNet3D(
        in_channels=1,
        out_channels=1,
        features=(args.f1, args.f2, args.f3, args.f4, args.f5),
    ).to(device)

    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model, model_path


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    folds = [args.fold] if args.fold is not None else list(range(args.num_folds))
    models = []
    model_paths = []
    for fold in folds:
        model, model_path = load_model(args.runs_root, fold, device, args)
        models.append(model)
        model_paths.append(model_path)

    test_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes=args.axcodes),
        Spacingd(
            keys=["image", "label"],
            pixdim=(args.pixdim_x, args.pixdim_y, args.pixdim_z),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=0.0, b_max=1.0, clip=True),
        Lambdad(keys="label", func=_binarize_label),
        EnsureTyped(keys=["image", "label"]),
    ])

    data_dicts = build_test_data(args.data_root)
    test_ds = Dataset(data_dicts, test_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    print("\n===== AttentionUNet Test (probability-map ensemble) =====")
    print(f"Device: {device}")
    print(f"Folds: {folds}")
    for p in model_paths:
        print(f"Model: {p}")
    print(f"Save dir: {args.save_dir}")

    roi_size = (args.roi_x, args.roi_y, args.roi_z)
    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(device)
            prob_sum = None
            for model in models:
                logits = sliding_window_inference(
                    image,
                    roi_size=roi_size,
                    sw_batch_size=args.sw_batch_size,
                    predictor=model,
                    overlap=args.infer_overlap,
                )
                probs = torch.sigmoid(logits)
                prob_sum = probs if prob_sum is None else (prob_sum + probs)

            output = prob_sum / float(len(models))
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
