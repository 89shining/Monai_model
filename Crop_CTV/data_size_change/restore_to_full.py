"""
剪裁预测结果恢复至原图尺寸
Restore prediction volumes (cropped by GT slices) back to full image size.

Example:
python restore_pred_to_full_by_gt.py ^
  --pred_dir D:\\data\\pred ^
  --gt_dir D:\\data\\labelsTr ^
  --image_dir D:\\data\\imagesTr ^
  --out_dir D:\\data\\pred_restored
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


def remove_suffix_from_pred_filenames(pred_dir: str, suffix: str) -> None:
    if not suffix:
        return

    pred_paths = list_nii_files(pred_dir)
    renamed = 0
    skipped = 0
    for src in pred_paths:
        name = os.path.basename(src)
        stem = strip_nii_ext(name)
        ext = ".nii.gz" if name.endswith(".nii.gz") else ".nii"
        if not stem.endswith(suffix):
            skipped += 1
            continue

        new_stem = stem[: -len(suffix)]
        if not new_stem:
            skipped += 1
            continue
        dst = os.path.join(pred_dir, new_stem + ext)
        if os.path.abspath(src) == os.path.abspath(dst):
            skipped += 1
            continue
        if os.path.exists(dst):
            print(f"[Rename Skip] target exists: {os.path.basename(dst)}")
            skipped += 1
            continue

        os.rename(src, dst)
        renamed += 1
        print(f"[Rename] {name} -> {os.path.basename(dst)}")
    print(f"[Rename Done] Renamed={renamed}, Skipped={skipped}")


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
        meta = {"itk_image": itk}
        return arr_zyx, meta

    # nibabel backend: data is usually (X, Y, Z), convert to (Z, Y, X)
    img = backend_mod.load(path)
    arr_xyz = np.asarray(img.dataobj)
    if arr_xyz.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={arr_xyz.shape}, file={path}")
    arr_zyx = np.transpose(arr_xyz, (2, 1, 0))
    meta = {"affine": img.affine, "header": img.header.copy()}
    return arr_zyx, meta


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

    # nibabel backend: convert back to (X, Y, Z)
    arr_xyz = np.transpose(arr_zyx, (2, 1, 0))
    header = ref_meta["header"]
    out_img = backend_mod.Nifti1Image(arr_xyz, ref_meta["affine"], header=header)
    backend_mod.save(out_img, out_path)


def find_image_path(image_dir: str, case_id: str, image_suffix: str) -> str:
    cands = [
        os.path.join(image_dir, f"{case_id}{image_suffix}.nii.gz"),
        os.path.join(image_dir, f"{case_id}{image_suffix}.nii"),
        os.path.join(image_dir, f"{case_id}.nii.gz"),
        os.path.join(image_dir, f"{case_id}.nii"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return ""


def find_gt_path(gt_dir: str, case_id: str) -> str:
    cands = [
        os.path.join(gt_dir, f"{case_id}.nii.gz"),
        os.path.join(gt_dir, f"{case_id}.nii"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return ""


def restore_one_case(
    pred_path: str,
    gt_path: str,
    image_path: str,
    out_path: str,
    gt_threshold: float,
    backend_name: str,
    backend_mod: Any,
) -> None:
    pred, _ = read_nii_as_zyx(pred_path, backend_name, backend_mod)
    image_arr, image_meta = read_nii_as_zyx(image_path, backend_name, backend_mod)
    gt, _ = read_nii_as_zyx(gt_path, backend_name, backend_mod)

    if image_arr.shape != gt.shape:
        raise ValueError(f"GT shape != image shape: gt={gt.shape}, image={image_arr.shape}")
    if pred.ndim != 3:
        raise ValueError(f"Prediction is not 3D: shape={pred.shape}")
    if pred.shape[1:] != image_arr.shape[1:]:
        raise ValueError(
            f"Only Z-crop is supported in this script. pred={pred.shape}, image={image_arr.shape}"
        )

    z_indices = np.where(np.any(gt > gt_threshold, axis=(1, 2)))[0]
    if len(z_indices) == 0:
        raise ValueError("GT has no foreground slices.")

    restored = np.zeros(image_arr.shape, dtype=pred.dtype)

    # Mode A: pred only contains GT-positive slices (exact slice list)
    if pred.shape[0] == len(z_indices):
        restored[z_indices, :, :] = pred
    else:
        # Mode B: pred contains the contiguous range [z_min, z_max]
        z_min, z_max = int(z_indices[0]), int(z_indices[-1])
        span = z_max - z_min + 1
        if pred.shape[0] != span:
            raise ValueError(
                f"Cannot match pred depth to GT slices: pred_z={pred.shape[0]}, gt_nonzero={len(z_indices)}, gt_span={span}"
            )
        restored[z_min : z_max + 1, :, :] = pred

    write_nii_from_zyx(restored, out_path, image_meta, backend_name, backend_mod)


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore GT-slice-cropped predictions to full volume size")
    parser.add_argument(
        "--pred_dir",
        default=r"D:\SAM\Crop\Rectal_146\VNet\VNet_pre_crop",
        help="Prediction folder (*.nii / *.nii.gz)",
    )
    parser.add_argument(
        "--gt_dir",
        default=r"D:\SAM\Crop\Rectal_146\labelsTs",
        help="Full-size GT folder",
    )
    parser.add_argument(
        "--image_dir",
        default=r"D:\SAM\Crop\Rectal_146\imagesTs",
        help="Original image folder",
    )
    parser.add_argument(
        "--out_dir",
        default=r"D:\SAM\Crop\Rectal_146\VNet\VNet_all_preprocess",
        help="Output folder",
    )
    parser.add_argument(
        "--image_suffix",
        default="_0000",
        help="Image suffix relative to case_id, default: _0000",
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

    # Rename prediction files in pred_dir: remove trailing image suffix (e.g., "_0000")
    remove_suffix_from_pred_filenames(args.pred_dir, args.image_suffix)

    pred_paths = list_nii_files(args.pred_dir)
    if not pred_paths:
        raise FileNotFoundError(f"No NIfTI files found in pred_dir: {args.pred_dir}")

    ok_cnt = 0
    fail_cnt = 0
    print(f"Found {len(pred_paths)} prediction files, start restoring...")

    for pred_path in pred_paths:
        pred_name = os.path.basename(pred_path)
        case_id_raw = strip_nii_ext(pred_name)
        case_id = (
            case_id_raw[: -len(args.image_suffix)]
            if args.image_suffix and case_id_raw.endswith(args.image_suffix)
            else case_id_raw
        )
        gt_path = find_gt_path(args.gt_dir, case_id)
        image_path = find_image_path(args.image_dir, case_id, args.image_suffix)

        if not gt_path:
            print(f"[Skip] GT not found: case={case_id}")
            fail_cnt += 1
            continue
        if not image_path:
            print(f"[Skip] image not found: case={case_id}")
            fail_cnt += 1
            continue

        out_path = os.path.join(args.out_dir, pred_name)
        try:
            restore_one_case(
                pred_path=pred_path,
                gt_path=gt_path,
                image_path=image_path,
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
