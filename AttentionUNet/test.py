import argparse
import glob
import os

import SimpleITK as sitk
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.networks.nets import AttentionUnet
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, Lambdad, LoadImaged, ScaleIntensityRanged


def parse_args():
    parser = argparse.ArgumentParser(description="AttentionUNet testing with 5-fold ensemble.")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root directory.")
    parser.add_argument("--runs_root", type=str, default="./runs/AttentionUNet", help="Training runs root.")
    parser.add_argument("--save_dir", type=str, default="./predictions", help="Prediction output directory.")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size.")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers.")
    parser.add_argument("--roi_x", type=int, default=96, help="Sliding-window ROI x.")
    parser.add_argument("--roi_y", type=int, default=96, help="Sliding-window ROI y.")
    parser.add_argument("--roi_z", type=int, default=96, help="Sliding-window ROI z.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold.")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds for ensemble loading.")
    parser.add_argument("--fold", type=int, default=None, help="Force single fold inference and disable ensemble.")
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


def _binarize_label(x):
    return (x > 0).astype(x.dtype)


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

    folds = [args.fold] if args.fold is not None else list(range(args.num_folds))
    models = []
    model_paths = []
    for fold in folds:
        model, model_path = load_model(args.runs_root, fold, device)
        models.append(model)
        model_paths.append(model_path)

    data_dicts = build_test_data(args.data_root)
    test_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys="image", a_min=-160.0, a_max=240.0, b_min=0.0, b_max=1.0, clip=True),
        Lambdad(keys="label", func=_binarize_label),
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
                logits = sliding_window_inference(image, roi_size=roi_size, sw_batch_size=1, predictor=model)
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
                # Keep shape/spacing/origin/direction exactly aligned to original CT geometry.
                pred_itk.CopyInformation(itk_img)

                out_path = os.path.join(args.save_dir, save_name)
                sitk.WriteImage(pred_itk, out_path)
                print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
