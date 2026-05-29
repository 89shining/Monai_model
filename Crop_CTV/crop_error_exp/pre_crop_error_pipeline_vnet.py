"""
Crop error experiment pipeline for VNet (.pth inference, z-direction only):
1) Generate cropped test images from full-size images by GT-derived z bounds
2) Run VNet prediction on each crop setting using a specified .pth checkpoint
3) Restore cropped predictions back to full-size volume

Notes:
- `image_dir_fullsize` and `gt_dir_fullsize` must be full-size volumes (not pre-cropped).
- Z boundary convention in this script:
  low index = lower boundary (smaller z index)
  high index = upper boundary (larger z index)
- Four modes for each K in {1,2,3}:
  inward   : [low+K, high-K]
  outward  : [low-K, high+K]
  upshift  : [low+K, high+K]
  downshift: [low-K, high-K]
"""

import glob
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


MODES = ("inward", "outward", "upshift", "downshift")
KS = (1, 2, 3)

# -----------------------------
# Fixed experiment configuration
# -----------------------------
IMAGE_DIR_FULLSIZE = "/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset009_EsoCTV73pAll/imagesTs"
GT_DIR_FULLSIZE = "/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset009_EsoCTV73pAll/labelsTs"

# Cropped test inputs root (contains K*_*/imagesTs)
CROP_ROOT = "/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset008_EsoCTV73p/testdata_28p/Crop_error_cropsize"

# Prediction roots
PRED_ROOT_CROPSIZE = "/home/wusi/Project_crop/Data/Rectal_146/Networks/VNet/RectalCTV_All/TestResults_fold0/Pre_crop_error_cropsize"
PRED_ROOT_FULLSIZE = "/home/wusi/Project_crop/Data/Rectal_146/Networks/VNet/RectalCTV_All/TestResults_fold0/Pre_crop_error_fullsize"

# VNet inference config
# Prefer explicit checkpoint path. If empty, fallback to runs_root + fold naming.
VNET_MODEL_PATH = ""
VNET_RUNS_ROOT = "/home/wusi/Project_crop/Data/Rectal_146/Networks/VNet/RectalCTV_All/TrainResults"
VNET_FOLD = 0

ROI_X, ROI_Y, ROI_Z = 96, 96, 64
SW_BATCH_SIZE = 2
INFER_OVERLAP = 0.5
THRESHOLD = 0.5
A_MIN, A_MAX = -160.0, 240.0
PIXDIM_X, PIXDIM_Y, PIXDIM_Z = 1.0, 1.0, 1.0
AXCODES = "RAS"
NUM_WORKERS = 2

IMAGE_SUFFIX = "_0000"
GT_THRESHOLD = 0.0

# Stage toggles
RUN_CROP = True
RUN_PREDICT = True
RUN_RESTORE = True


def get_io_backend() -> Tuple[str, Any]:
    try:
        import SimpleITK as sitk  # type: ignore

        return "sitk", sitk
    except Exception:
        pass
    try:
        import nibabel as nib  # type: ignore

        return "nib", nib
    except Exception:
        pass
    raise ImportError("Neither SimpleITK nor nibabel is installed.")


def read_nii_as_zyx(path: str, backend_name: str, backend_mod: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    if backend_name == "sitk":
        itk = backend_mod.ReadImage(path)
        arr_zyx = backend_mod.GetArrayFromImage(itk)  # (Z, Y, X)
        return arr_zyx, {"itk_image": itk}

    img = backend_mod.load(path)
    arr_xyz = np.asarray(img.dataobj)
    if arr_xyz.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={arr_xyz.shape}, file={path}")
    arr_zyx = np.transpose(arr_xyz, (2, 1, 0))
    return arr_zyx, {"affine": img.affine, "header": img.header.copy()}


def write_nii_from_zyx(
    arr_zyx: np.ndarray,
    out_path: str,
    ref_meta: Dict[str, Any],
    backend_name: str,
    backend_mod: Any,
    z_start: int = 0,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if backend_name == "sitk":
        out_img = backend_mod.GetImageFromArray(arr_zyx)
        ref_img = ref_meta["itk_image"]
        out_img.SetSpacing(ref_img.GetSpacing())
        out_img.SetDirection(ref_img.GetDirection())
        new_origin = ref_img.TransformIndexToPhysicalPoint((0, 0, int(z_start)))
        out_img.SetOrigin(new_origin)
        backend_mod.WriteImage(out_img, out_path)
        return

    arr_xyz = np.transpose(arr_zyx, (2, 1, 0))
    affine = ref_meta["affine"].copy()
    new_origin_h = affine @ np.array([0.0, 0.0, float(z_start), 1.0])
    affine[:3, 3] = new_origin_h[:3]
    out_img = backend_mod.Nifti1Image(arr_xyz, affine, header=ref_meta["header"])
    backend_mod.save(out_img, out_path)


def strip_nii_ext(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


def list_nii(folder: str) -> List[str]:
    a = sorted(glob.glob(os.path.join(folder, "*.nii.gz")))
    b = sorted(glob.glob(os.path.join(folder, "*.nii")))
    return sorted(set(a + b))


def find_nii_by_stem(folder: str, stem: str) -> str:
    cands = [os.path.join(folder, stem + ".nii.gz"), os.path.join(folder, stem + ".nii")]
    for c in cands:
        if os.path.exists(c):
            return c
    return ""


def find_gt_path(gt_dir_fullsize: str, case_id: str) -> str:
    return find_nii_by_stem(gt_dir_fullsize, case_id)


def find_image_path(image_dir_fullsize: str, case_id: str, image_suffix: str) -> str:
    p = find_nii_by_stem(image_dir_fullsize, case_id + image_suffix)
    if p:
        return p
    return find_nii_by_stem(image_dir_fullsize, case_id)


def image_path_to_case_id(image_path: str, image_suffix: str) -> str:
    stem = strip_nii_ext(os.path.basename(image_path))
    if stem.endswith(image_suffix):
        return stem[: -len(image_suffix)]
    return stem


def group_name(k: int, mode: str) -> str:
    return f"K{k}_{mode}"


def compute_bounds_from_gt(gt_zyx: np.ndarray, k: int, mode: str, gt_threshold: float) -> Tuple[int, int]:
    z_indices = np.where(np.any(gt_zyx > gt_threshold, axis=(1, 2)))[0]
    if len(z_indices) == 0:
        raise ValueError("GT has no foreground slices.")

    low = int(z_indices[0])
    high = int(z_indices[-1])

    if mode == "inward":
        low_new, high_new = low + k, high - k
    elif mode == "outward":
        low_new, high_new = low - k, high + k
    elif mode == "upshift":
        low_new, high_new = low + k, high + k
    elif mode == "downshift":
        low_new, high_new = low - k, high - k
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return low_new, high_new


def clip_bounds(low: int, high: int, z_size: int) -> Tuple[int, int]:
    return max(0, low), min(z_size - 1, high)


def generate_crops(
    image_dir_fullsize: str,
    gt_dir_fullsize: str,
    crop_root: str,
    image_suffix: str,
    gt_threshold: float,
    backend_name: str,
    backend_mod: Any,
) -> None:
    os.makedirs(crop_root, exist_ok=True)
    image_paths = list_nii(image_dir_fullsize)
    if not image_paths:
        raise FileNotFoundError(f"No images in {image_dir_fullsize}")

    for k in KS:
        for mode in MODES:
            g = group_name(k, mode)
            out_images_ts = os.path.join(crop_root, g, "imagesTs")
            os.makedirs(out_images_ts, exist_ok=True)

            ok, fail = 0, 0
            print(f"\n[Crop] {g}")
            for image_path in image_paths:
                case_id = image_path_to_case_id(image_path, image_suffix)
                gt_path = find_gt_path(gt_dir_fullsize, case_id)
                if not gt_path:
                    print(f"[Skip] GT missing: {case_id}")
                    fail += 1
                    continue

                try:
                    image_arr, image_meta = read_nii_as_zyx(image_path, backend_name, backend_mod)
                    gt_arr, _ = read_nii_as_zyx(gt_path, backend_name, backend_mod)
                    if image_arr.shape != gt_arr.shape:
                        raise ValueError(f"shape mismatch image={image_arr.shape}, gt={gt_arr.shape}")

                    low, high = compute_bounds_from_gt(gt_arr, k, mode, gt_threshold)
                    low, high = clip_bounds(low, high, image_arr.shape[0])
                    if low > high:
                        raise ValueError(f"invalid range after clip: [{low},{high}]")

                    crop_arr = image_arr[low : high + 1, :, :]
                    out_name = f"{case_id}{image_suffix}.nii.gz"
                    out_path = os.path.join(out_images_ts, out_name)
                    write_nii_from_zyx(
                        crop_arr,
                        out_path,
                        image_meta,
                        backend_name,
                        backend_mod,
                        z_start=low,
                    )
                    ok += 1
                except Exception as e:
                    print(f"[Fail] {case_id}: {e}")
                    fail += 1

            print(f"[Crop Done] {g}: OK={ok}, Fail={fail}")


def _resolve_vnet_model_path(vnet_model_path: str, vnet_runs_root: str, vnet_fold: int) -> str:
    if vnet_model_path:
        return vnet_model_path
    return os.path.join(vnet_runs_root, f"fold_{vnet_fold}", f"best_model_fold{vnet_fold}.pth")


def run_predict_vnet(
    crop_root: str,
    pred_root_cropsize: str,
    vnet_model_path: str,
    vnet_runs_root: str,
    vnet_fold: int,
    roi_x: int,
    roi_y: int,
    roi_z: int,
    sw_batch_size: int,
    infer_overlap: float,
    threshold: float,
    a_min: float,
    a_max: float,
    pixdim_x: float,
    pixdim_y: float,
    pixdim_z: float,
    axcodes: str,
    num_workers: int,
) -> None:
    import torch
    from monai.data import DataLoader, Dataset, decollate_batch
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        AsDiscreted,
        Compose,
        EnsureChannelFirstd,
        EnsureTyped,
        Invertd,
        LoadImaged,
        Orientationd,
        SaveImaged,
        ScaleIntensityRanged,
        Spacingd,
    )

    vnet_dir = Path(__file__).resolve().parents[2] / "VNet"
    if str(vnet_dir) not in sys.path:
        sys.path.insert(0, str(vnet_dir))
    from train import VNet3DSeg  # type: ignore

    model_path = _resolve_vnet_model_path(vnet_model_path, vnet_runs_root, vnet_fold)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"VNet model not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VNet3DSeg().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()

    os.makedirs(pred_root_cropsize, exist_ok=True)
    print(f"[VNet] Model: {model_path}")
    print(f"[VNet] Device: {device}")

    tf = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes=axcodes),
            Spacingd(keys=["image"], pixdim=(pixdim_x, pixdim_y, pixdim_z), mode=("bilinear",)),
            ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),
            EnsureTyped(keys=["image"], track_meta=True),
        ]
    )

    post_transforms = Compose(
        [
            AsDiscreted(keys="pred", threshold=threshold),
            Invertd(keys="pred", transform=tf, orig_keys="image", nearest_interp=True, to_tensor=False),
            SaveImaged(
                keys="pred",
                meta_keys="pred_meta_dict",
                output_dir="",
                output_postfix="",
                output_ext=".nii.gz",
                separate_folder=False,
                output_dtype="uint8",
                resample=False,
                print_log=False,
            ),
        ]
    )

    for k in KS:
        for mode in MODES:
            g = group_name(k, mode)
            in_dir = os.path.join(crop_root, g, "imagesTs")
            out_dir = os.path.join(pred_root_cropsize, g)
            os.makedirs(out_dir, exist_ok=True)

            image_paths = sorted(
                glob.glob(os.path.join(in_dir, "*_0000.nii.gz")) +
                glob.glob(os.path.join(in_dir, "*_0000.nii"))
            )
            if not image_paths:
                print(f"[Predict Skip] {g}: no input in {in_dir}")
                continue

            data = [{"image": p} for p in image_paths]
            loader = DataLoader(
                Dataset(data=data, transform=tf),
                batch_size=1,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
            )

            print(f"\n[Predict] {g}")
            with torch.no_grad():
                for batch in loader:
                    image = batch["image"].to(device)
                    probs = sliding_window_inference(
                        image,
                        roi_size=(roi_x, roi_y, roi_z),
                        sw_batch_size=sw_batch_size,
                        predictor=model,
                        overlap=infer_overlap,
                    )
                    batch["pred"] = probs
                    for item in decollate_batch(batch):
                        post_transforms.transforms[-1].output_dir = out_dir
                        out = post_transforms(item)
                        meta = out.get("pred_meta_dict") if isinstance(out, dict) else None
                        saved_to = None if meta is None else meta.get("saved_to")
                        print(f"Saved: {saved_to}" if saved_to else "Saved one prediction file.")


def restore_predictions(
    pred_root_cropsize: str,
    pred_root_fullsize: str,
    image_dir_fullsize: str,
    gt_dir_fullsize: str,
    image_suffix: str,
    gt_threshold: float,
    backend_name: str,
    backend_mod: Any,
) -> None:
    os.makedirs(pred_root_fullsize, exist_ok=True)

    for k in KS:
        for mode in MODES:
            g = group_name(k, mode)
            pred_dir = os.path.join(pred_root_cropsize, g)
            out_dir = os.path.join(pred_root_fullsize, g)
            os.makedirs(out_dir, exist_ok=True)

            pred_files = list_nii(pred_dir) if os.path.isdir(pred_dir) else []
            if not pred_files:
                print(f"[Restore Skip] {g}: no predictions in {pred_dir}")
                continue

            ok, fail = 0, 0
            print(f"\n[Restore] {g}")
            for pred_path in pred_files:
                pred_name = os.path.basename(pred_path)
                case_id = strip_nii_ext(pred_name)
                gt_path = find_gt_path(gt_dir_fullsize, case_id)
                image_path = find_image_path(image_dir_fullsize, case_id, image_suffix)

                if not gt_path:
                    print(f"[Skip] GT missing: {case_id}")
                    fail += 1
                    continue
                if not image_path:
                    print(f"[Skip] image missing: {case_id}")
                    fail += 1
                    continue

                try:
                    pred_arr, _ = read_nii_as_zyx(pred_path, backend_name, backend_mod)
                    gt_arr, _ = read_nii_as_zyx(gt_path, backend_name, backend_mod)
                    image_arr, image_meta = read_nii_as_zyx(image_path, backend_name, backend_mod)

                    if image_arr.shape != gt_arr.shape:
                        raise ValueError(f"shape mismatch image={image_arr.shape}, gt={gt_arr.shape}")
                    if pred_arr.ndim != 3:
                        raise ValueError(f"pred is not 3D, shape={pred_arr.shape}")
                    if pred_arr.shape[1:] != image_arr.shape[1:]:
                        raise ValueError(
                            f"Only z-crop restore supported, pred={pred_arr.shape}, image={image_arr.shape}"
                        )

                    low, high = compute_bounds_from_gt(gt_arr, k, mode, gt_threshold)
                    low, high = clip_bounds(low, high, image_arr.shape[0])
                    if low > high:
                        raise ValueError(f"invalid range after clip: [{low},{high}]")

                    expected_depth = high - low + 1
                    if pred_arr.shape[0] != expected_depth:
                        raise ValueError(
                            f"pred z-depth mismatch: pred={pred_arr.shape[0]}, expected={expected_depth}, "
                            f"range=[{low},{high}]"
                        )

                    restored = np.zeros_like(image_arr, dtype=pred_arr.dtype)
                    restored[low : high + 1, :, :] = pred_arr
                    out_path = os.path.join(out_dir, pred_name)
                    write_nii_from_zyx(restored, out_path, image_meta, backend_name, backend_mod)
                    ok += 1
                except Exception as e:
                    print(f"[Fail] {g}/{pred_name}: {e}")
                    fail += 1

            print(f"[Restore Done] {g}: OK={ok}, Fail={fail}")


def main() -> None:
    backend_name, backend_mod = get_io_backend()
    print(f"IO backend: {backend_name}")

    if RUN_CROP:
        generate_crops(
            image_dir_fullsize=IMAGE_DIR_FULLSIZE,
            gt_dir_fullsize=GT_DIR_FULLSIZE,
            crop_root=CROP_ROOT,
            image_suffix=IMAGE_SUFFIX,
            gt_threshold=GT_THRESHOLD,
            backend_name=backend_name,
            backend_mod=backend_mod,
        )

    if RUN_PREDICT:
        run_predict_vnet(
            crop_root=CROP_ROOT,
            pred_root_cropsize=PRED_ROOT_CROPSIZE,
            vnet_model_path=VNET_MODEL_PATH,
            vnet_runs_root=VNET_RUNS_ROOT,
            vnet_fold=VNET_FOLD,
            roi_x=ROI_X,
            roi_y=ROI_Y,
            roi_z=ROI_Z,
            sw_batch_size=SW_BATCH_SIZE,
            infer_overlap=INFER_OVERLAP,
            threshold=THRESHOLD,
            a_min=A_MIN,
            a_max=A_MAX,
            pixdim_x=PIXDIM_X,
            pixdim_y=PIXDIM_Y,
            pixdim_z=PIXDIM_Z,
            axcodes=AXCODES,
            num_workers=NUM_WORKERS,
        )

    if RUN_RESTORE:
        restore_predictions(
            pred_root_cropsize=PRED_ROOT_CROPSIZE,
            pred_root_fullsize=PRED_ROOT_FULLSIZE,
            image_dir_fullsize=IMAGE_DIR_FULLSIZE,
            gt_dir_fullsize=GT_DIR_FULLSIZE,
            image_suffix=IMAGE_SUFFIX,
            gt_threshold=GT_THRESHOLD,
            backend_name=backend_name,
            backend_mod=backend_mod,
        )

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
