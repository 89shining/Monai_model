import argparse
import glob
import os

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from ddunet import DDUNet2D


WINDOW_WIDTH = 400.0
WINDOW_LEVEL = 40.0
WINDOW_MIN = WINDOW_LEVEL - WINDOW_WIDTH / 2.0
WINDOW_MAX = WINDOW_LEVEL + WINDOW_WIDTH / 2.0


def strip_nii_ext(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return os.path.splitext(name)[0]


def image_case_id(path: str) -> str:
    name = strip_nii_ext(os.path.basename(path))
    return name[:-5] if name.endswith("_0000") else name


def window_and_norm(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, WINDOW_MIN, WINDOW_MAX)
    arr = (arr - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN)
    return arr.astype(np.float32)


def build_model(ckpt_path: str, device: torch.device):
    model = DDUNet2D(in_channels=1, out_channels=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--runs_root", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--fold", type=int, required=True)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(args.runs_root, f"fold_{args.fold}", f"best_model_fold{args.fold}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    model = build_model(ckpt_path, device)

    images = sorted(glob.glob(os.path.join(args.data_root, "imagesTs", "*.nii*")))
    if not images:
        raise FileNotFoundError("No test images found in imagesTs")

    for image_path in images:
        itk_img = sitk.ReadImage(image_path)
        vol = sitk.GetArrayFromImage(itk_img).astype(np.float32)
        vol = window_and_norm(vol)

        pred_slices = []
        with torch.no_grad():
            for z in range(vol.shape[0]):
                sl = torch.from_numpy(vol[z])[None, None, ...]
                sl = F.interpolate(sl, size=(512, 512), mode="bilinear", align_corners=False).to(device)
                prob = model(sl)
                mask = (prob > 0.5).float()
                mask = F.interpolate(mask, size=vol[z].shape, mode="nearest")
                pred_slices.append(mask[0, 0].cpu().numpy().astype(np.uint8))

        pred_vol = np.stack(pred_slices, axis=0)
        pred_itk = sitk.GetImageFromArray(pred_vol)
        pred_itk.CopyInformation(itk_img)

        cid = image_case_id(image_path)
        out_path = os.path.join(args.save_dir, f"{cid}.nii.gz")
        sitk.WriteImage(pred_itk, out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
