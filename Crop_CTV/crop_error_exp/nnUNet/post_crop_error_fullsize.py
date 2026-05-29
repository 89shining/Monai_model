"""
Post-process full-volume predictions with GT-based z-boundary perturbations.

What this script does:
1) Read full-size prediction volumes (already inferred).
2) For each case, compute GT z-range from full-size labels.
3) Generate 12 variants: K1/K2/K3 x (inward/outward/upshift/downshift).
4) Keep prediction only in the target z-range and zero out all other slices.
5) Save outputs in full-size shape (same shape as original prediction).

Fixed paths (hard-coded):
- Input full prediction dir:
  /home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_results/Dataset009_EsoCTV73pAll/nnUNetTrainer__nnUNetPlans__3d_fullres/testResult_28p_fold4
- Full-size GT dir:
  /home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset009_EsoCTV73pAll/labelsTs
- Output root:
  /home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_results/Dataset009_EsoCTV73pAll/nnUNetTrainer__nnUNetPlans__3d_fullres/TestResults_28p_fold4/Post_crop_error_fullsize

Output subfolders:
- K1_inward, K1_outward, K1_upshift, K1_downshift
- K2_inward, K2_outward, K2_upshift, K2_downshift
- K3_inward, K3_outward, K3_upshift, K3_downshift
"""

import glob
import os
from typing import Any, Dict, List, Tuple

import numpy as np


# -----------------------------
# Fixed experiment paths
# -----------------------------
PRED_DIR_FULLSIZE = (
    "/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_results/"
    "Dataset009_EsoCTV73pAll/nnUNetTrainer__nnUNetPlans__3d_fullres/TestResults_28p_fold4/rawPred"
)
GT_DIR_FULLSIZE = "/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset009_EsoCTV73pAll/labelsTs"
OUT_ROOT_FULLSIZE = (
    "/home/wusi/nnUNet/nnUNetFrame/DATASET/nnUNet_results/"
    "Dataset009_EsoCTV73pAll/nnUNetTrainer__nnUNetPlans__3d_fullres/"
    "TestResults_28p_fold4/Post_crop_error_fullsize"
)

GT_THRESHOLD = 0.0
MODES = ("inward", "outward", "upshift", "downshift")
KS = (1, 2, 3)


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

    raise ImportError("Neither SimpleITK nor nibabel is available.")


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
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if backend_name == "sitk":
        out_img = backend_mod.GetImageFromArray(arr_zyx)
        out_img.CopyInformation(ref_meta["itk_image"])
        backend_mod.WriteImage(out_img, out_path)
        return

    arr_xyz = np.transpose(arr_zyx, (2, 1, 0))
    out_img = backend_mod.Nifti1Image(arr_xyz, ref_meta["affine"], header=ref_meta["header"])
    backend_mod.save(out_img, out_path)


def strip_nii_ext(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return filename


def list_nii_files(folder: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(folder, "*.nii.gz")))
    paths.extend(sorted(glob.glob(os.path.join(folder, "*.nii"))))
    return sorted(set(paths))


def find_gt_path(case_id: str) -> str:
    cands = [
        os.path.join(GT_DIR_FULLSIZE, f"{case_id}.nii.gz"),
        os.path.join(GT_DIR_FULLSIZE, f"{case_id}.nii"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return ""


def compute_bounds_from_gt(gt_zyx: np.ndarray, k: int, mode: str, gt_threshold: float) -> Tuple[int, int]:
    z_idx = np.where(np.any(gt_zyx > gt_threshold, axis=(1, 2)))[0]
    if len(z_idx) == 0:
        raise ValueError("GT has no foreground slices.")

    low = int(z_idx[0])   # lower boundary: smaller z index
    high = int(z_idx[-1]) # upper boundary: larger z index

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


def main() -> None:
    backend_name, backend_mod = get_io_backend()
    print(f"IO backend: {backend_name}")

    pred_paths = list_nii_files(PRED_DIR_FULLSIZE)
    if not pred_paths:
        raise FileNotFoundError(f"No prediction files found: {PRED_DIR_FULLSIZE}")

    os.makedirs(OUT_ROOT_FULLSIZE, exist_ok=True)
    print(f"Found {len(pred_paths)} prediction files.")

    total_ok = 0
    total_fail = 0

    for k in KS:
        for mode in MODES:
            group = f"K{k}_{mode}"
            out_dir = os.path.join(OUT_ROOT_FULLSIZE, group)
            os.makedirs(out_dir, exist_ok=True)
            ok = 0
            fail = 0
            print(f"\n[Post] {group}")

            for pred_path in pred_paths:
                pred_name = os.path.basename(pred_path)
                case_id = strip_nii_ext(pred_name)
                gt_path = find_gt_path(case_id)
                if not gt_path:
                    print(f"[Skip] GT not found: {case_id}")
                    fail += 1
                    continue

                out_path = os.path.join(out_dir, pred_name)
                try:
                    pred_zyx, pred_meta = read_nii_as_zyx(pred_path, backend_name, backend_mod)
                    gt_zyx, _ = read_nii_as_zyx(gt_path, backend_name, backend_mod)

                    if pred_zyx.ndim != 3:
                        raise ValueError(f"Prediction is not 3D: shape={pred_zyx.shape}")
                    if pred_zyx.shape != gt_zyx.shape:
                        raise ValueError(f"Prediction/GT shape mismatch: pred={pred_zyx.shape}, gt={gt_zyx.shape}")

                    low, high = compute_bounds_from_gt(gt_zyx, k, mode, GT_THRESHOLD)
                    low, high = clip_bounds(low, high, pred_zyx.shape[0])
                    if low > high:
                        raise ValueError(f"Invalid range after clipping: low={low}, high={high}")

                    # Keep full-size shape and zero out slices outside [low, high]
                    out_zyx = np.zeros_like(pred_zyx)
                    out_zyx[low : high + 1, :, :] = pred_zyx[low : high + 1, :, :]

                    write_nii_from_zyx(out_zyx, out_path, pred_meta, backend_name, backend_mod)
                    ok += 1
                    print(f"[OK] {pred_name}")
                except Exception as e:
                    fail += 1
                    print(f"[Fail] {pred_name}: {e}")

            print(f"[Done] {group}: OK={ok}, Fail={fail}")
            total_ok += ok
            total_fail += fail

    print(f"\nAll done. Total OK={total_ok}, Total Fail={total_fail}")


if __name__ == "__main__":
    main()
