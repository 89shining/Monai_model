"""
 将预测结果的非GT层mask做清除后处理
Post-process full-volume nnUNet predictions:
remove predictions on non-GT slices while keeping original full volume size.

Default paths are written in code for direct run.
"""

import argparse
import glob
import os
from typing import Any, Dict, List, Tuple

import numpy as np


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

    raise ImportError(
        "Neither SimpleITK nor nibabel is available. "
        "Please install one of them: pip install SimpleITK  or  pip install nibabel"
    )


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


def find_gt_path(gt_dir: str, case_id: str) -> str:
    cands = [
        os.path.join(gt_dir, f"{case_id}.nii.gz"),
        os.path.join(gt_dir, f"{case_id}.nii"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return ""


def postprocess_one_case(
    pred_path: str,
    gt_path: str,
    out_path: str,
    gt_threshold: float,
    backend_name: str,
    backend_mod: Any,
) -> None:
    pred, pred_meta = read_nii_as_zyx(pred_path, backend_name, backend_mod)
    gt, _ = read_nii_as_zyx(gt_path, backend_name, backend_mod)

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}")

    # Keep only GT-positive slices along Z, zero out other slices.
    z_keep = np.any(gt > gt_threshold, axis=(1, 2))
    out = np.zeros_like(pred)
    out[z_keep, :, :] = pred[z_keep, :, :]

    write_nii_from_zyx(out, out_path, pred_meta, backend_name, backend_mod)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process full-volume predictions by removing non-GT-slice predictions"
    )
    parser.add_argument(
        "--pred_dir",
        default=r"C:\Users\dell\Desktop\Eso_83\nnUNet_all_volume",
        help="Full-volume prediction folder (*.nii / *.nii.gz)",
    )
    parser.add_argument(
        "--gt_dir",
        default=r"C:\Users\dell\Desktop\Eso_83\labelsTs",
        help="Full-size GT folder",
    )
    parser.add_argument(
        "--out_dir",
        default=r"C:\Users\dell\Desktop\Eso_83\nnUNet_all_postprocess",
        help="Output folder",
    )
    parser.add_argument(
        "--gt_threshold",
        type=float,
        default=0.0,
        help="GT foreground threshold, default > 0",
    )
    args = parser.parse_args()

    backend_name, backend_mod = get_io_backend()
    print(f"IO backend: {backend_name}")

    pred_paths = list_nii_files(args.pred_dir)
    if not pred_paths:
        raise FileNotFoundError(f"No NIfTI files found in pred_dir: {args.pred_dir}")

    ok_cnt = 0
    fail_cnt = 0
    print(f"Found {len(pred_paths)} prediction files, start post-process...")

    for pred_path in pred_paths:
        pred_name = os.path.basename(pred_path)
        case_id = strip_nii_ext(pred_name)
        gt_path = find_gt_path(args.gt_dir, case_id)

        if not gt_path:
            print(f"[Skip] GT not found: case={case_id}")
            fail_cnt += 1
            continue

        out_path = os.path.join(args.out_dir, pred_name)
        try:
            postprocess_one_case(
                pred_path=pred_path,
                gt_path=gt_path,
                out_path=out_path,
                gt_threshold=args.gt_threshold,
                backend_name=backend_name,
                backend_mod=backend_mod,
            )
            ok_cnt += 1
            print(f"[OK] {pred_name}")
        except Exception as exc:
            fail_cnt += 1
            print(f"[Fail] {pred_name}: {exc}")

    print(f"Done. Success={ok_cnt}, Failed/Skipped={fail_cnt}")


if __name__ == "__main__":
    main()
