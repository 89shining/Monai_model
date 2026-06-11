import argparse
import csv
import glob
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

MODES = ("inward", "outward", "upshift", "downshift")
KS = (1, 2, 3)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    crop_root: str
    runs_root: str
    save_root: str
    restored_save_root: str
    full_image_dir: str
    full_gt_dir: str
    test_script: Path
    extra_args: List[str]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run crop-error inference for multiple models using each model's native test.py logic."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["AttentionUNet", "DDUnet", "Deeplabv3+", "VNet"],
        choices=["AttentionUNet", "DDUnet", "Deeplabv3+", "VNet"],
        help="Models to run.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to invoke model test scripts.",
    )
    parser.add_argument(
        "--cuda",
        type=str,
        default=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        help="CUDA_VISIBLE_DEVICES passed to subprocesses. Empty keeps current environment.",
    )
    parser.add_argument(
        "--repo_root",
        type=str,
        default="",
        help="Repository root containing AttentionUNet/DDUnet/Deeplabv3+/VNet. Auto-detected if omitted.",
    )
    return parser.parse_args()


def default_model_configs(repo_root: Path) -> Dict[str, ModelConfig]:
    return {
        "AttentionUNet": ModelConfig(
            name="AttentionUNet",
            crop_root="/home/wusi/Project_crop/Data/Eso_83/EsoCTV_CropError",
            runs_root="/home/wusi/Project_crop/Data/Eso_83/Networks/AttentionUNet/EsoCTV_Crop/TrainResults",
            save_root="/home/wusi/Project_crop/Data/Eso_83/Networks/AttentionUNet/EsoCTV_Error_CropSize",
            restored_save_root="/home/wusi/Project_crop/Data/Eso_83/Networks/AttentionUNet/EsoCTV_Error_FullSize",
            full_image_dir="/home/wusi/Project_crop/Data/Eso_83/EsoCTV_All/imagesTs",
            full_gt_dir="/home/wusi/Project_crop/Data/Eso_83/EsoCTV_All/labelsTs",
            test_script=repo_root / "AttentionUNet" / "test.py",
            extra_args=[],
        ),
        "DDUnet": ModelConfig(
            name="DDUnet",
            crop_root="/home/wusi/Project_crop/Data/Eso_83/EsoCTV_CropError",
            runs_root="/home/wusi/Project_crop/Data/Eso_83/Networks/DDUnet/EsoCTV_Crop/TrainResults",
            save_root="/home/wusi/Project_crop/Data/Eso_83/Networks/DDUnet/EsoCTV_Error_CropSize",
            restored_save_root="/home/wusi/Project_crop/Data/Eso_83/Networks/DDUnet/EsoCTV_Error_FullSize",
            full_image_dir="/home/wusi/Project_crop/Data/Eso_83/EsoCTV_All/imagesTs",
            full_gt_dir="/home/wusi/Project_crop/Data/Eso_83/EsoCTV_All/labelsTs",
            test_script=repo_root / "DDUnet" / "test.py",
            extra_args=[],
        ),
        "Deeplabv3+": ModelConfig(
            name="Deeplabv3+",
            crop_root="/home/wusi/Project_crop/Data/Rectal_146/RectalCTV_CropError",
            runs_root="/home/wusi/Project_crop/Data/Rectal_146/Networks/Deeplabv3+/RectalCTV_Crop/TrainResults",
            save_root="/home/wusi/Project_crop/Data/Rectal_146/Networks/Deeplabv3+/RectalCTV_Error_CropSize",
            restored_save_root="/home/wusi/Project_crop/Data/Rectal_146/Networks/Deeplabv3+/RectalCTV_Error_FullSize",
            full_image_dir="/home/wusi/Project_crop/Data/Rectal_146/RectalCTV_All/imagesTs",
            full_gt_dir="/home/wusi/Project_crop/Data/Rectal_146/RectalCTV_All/labelsTs",
            test_script=repo_root / "Deeplabv3+" / "test.py",
            extra_args=["--backbone", "xception"],
        ),
        "VNet": ModelConfig(
            name="VNet",
            crop_root="/home/wusi/Project_crop/Data/Rectal_146/RectalCTV_CropError",
            runs_root="/home/wusi/Project_crop/Data/Rectal_146/Networks/VNet/RectalCTV_Crop/TrainResults",
            save_root="/home/wusi/Project_crop/Data/Rectal_146/Networks/VNet/RectalCTV_Error_CropSize",
            restored_save_root="/home/wusi/Project_crop/Data/Rectal_146/Networks/VNet/RectalCTV_Error_FullSize",
            full_image_dir="/home/wusi/Project_crop/Data/Rectal_146/RectalCTV_All/imagesTs",
            full_gt_dir="/home/wusi/Project_crop/Data/Rectal_146/RectalCTV_All/labelsTs",
            test_script=repo_root / "VNet" / "test.py",
            extra_args=[],
        ),
    }


def detect_repo_root(script_path: Path, user_repo_root: str) -> Path:
    if user_repo_root:
        repo_root = Path(user_repo_root).resolve()
        if not repo_root.exists():
            raise FileNotFoundError(f"repo_root does not exist: {repo_root}")
        return repo_root

    candidates = [script_path.resolve().parent] + list(script_path.resolve().parents)
    for candidate in candidates:
        if all((candidate / name).exists() for name in ["AttentionUNet", "DDUnet", "Deeplabv3+", "VNet"]):
            return candidate

    raise FileNotFoundError(
        "Could not auto-detect repo_root. Please pass --repo_root, for example: "
        "/home/wusi/Project_crop/Monai_model"
    )


def read_best_fold_from_cv(cv_path: str) -> int:
    with open(cv_path, "r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows found in {cv_path}")
    best = max(rows, key=lambda row: float(row["best_dice_fg"]))
    return int(best["fold"])


def write_cv_results_if_needed(runs_root: str) -> str:
    cv_path = os.path.join(runs_root, "cv_results.csv")
    if os.path.exists(cv_path):
        return cv_path

    rows = []
    for fold_dir_name in sorted(os.listdir(runs_root)):
        fold_dir = os.path.join(runs_root, fold_dir_name)
        if not os.path.isdir(fold_dir) or not fold_dir_name.startswith("fold_"):
            continue

        fold = int(fold_dir_name.split("_", 1)[1])
        metrics_path = os.path.join(fold_dir, "epoch_metrics.csv")
        model_path = os.path.join(fold_dir, f"best_model_fold{fold}.pth")
        if not (os.path.exists(metrics_path) and os.path.exists(model_path)):
            continue

        with open(metrics_path, "r", encoding="utf-8") as file:
            metric_rows = list(csv.DictReader(file))
        if not metric_rows:
            continue

        best_row = max(metric_rows, key=lambda row: float(row["val_dice_fg"]))
        rows.append(
            {
                "fold": fold,
                "best_epoch": int(best_row["epoch"]),
                "best_dice_fg": float(best_row["val_dice_fg"]),
                "checkpoint": model_path,
            }
        )

    if not rows:
        raise FileNotFoundError(f"No usable fold results found under {runs_root}")

    with open(cv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["fold", "best_epoch", "best_dice_fg", "checkpoint"])
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["fold"]):
            writer.writerow(
                {
                    "fold": row["fold"],
                    "best_epoch": row["best_epoch"],
                    "best_dice_fg": f"{row['best_dice_fg']:.6f}",
                    "checkpoint": row["checkpoint"],
                }
            )
    return cv_path


def list_crop_groups(crop_root: str) -> List[str]:
    groups = []
    for name in sorted(os.listdir(crop_root)):
        group_dir = os.path.join(crop_root, name)
        image_dir = os.path.join(group_dir, "imagesTs")
        if os.path.isdir(group_dir) and os.path.isdir(image_dir):
            groups.append(name)
    if not groups:
        raise FileNotFoundError(f"No crop-error groups with imagesTs found under {crop_root}")
    return groups


def parse_group_name(group_name: str) -> Tuple[int, str]:
    prefix, mode = group_name.split("_", 1)
    if not prefix.startswith("K"):
        raise ValueError(f"Invalid group name: {group_name}")
    k = int(prefix[1:])
    if k not in KS or mode not in MODES:
        raise ValueError(f"Invalid group name: {group_name}")
    return k, mode


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


def normalize_prediction_filenames(pred_dir: str, suffix: str = "_0000") -> None:
    if not suffix:
        return

    renamed = 0
    for src in list_nii_files(pred_dir):
        name = os.path.basename(src)
        stem = strip_nii_ext(name)
        if not stem.endswith(suffix):
            continue

        new_stem = stem[: -len(suffix)]
        if not new_stem:
            continue
        ext = ".nii.gz" if name.endswith(".nii.gz") else ".nii"
        dst = os.path.join(pred_dir, f"{new_stem}{ext}")
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        if os.path.exists(dst):
            continue
        os.rename(src, dst)
        renamed += 1

    if renamed:
        print(f"[Rename] {pred_dir} | removed suffix from {renamed} files", flush=True)


def get_io_backend() -> Tuple[str, Any]:
    try:
        import SimpleITK as sitk

        return "sitk", sitk
    except Exception:
        pass

    try:
        import nibabel as nib

        return "nib", nib
    except Exception:
        pass

    raise ImportError("Neither SimpleITK nor nibabel is installed.")


def read_nii_as_zyx(path: str, backend_name: str, backend_mod: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    if backend_name == "sitk":
        itk = backend_mod.ReadImage(path)
        return backend_mod.GetArrayFromImage(itk), {"itk_image": itk}

    image = backend_mod.load(path)
    arr_xyz = np.asarray(image.dataobj)
    if arr_xyz.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape={arr_xyz.shape}, file={path}")
    return np.transpose(arr_xyz, (2, 1, 0)), {"affine": image.affine, "header": image.header.copy()}


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


def find_image_path(image_dir: str, case_id: str, image_suffix: str) -> str:
    candidates = [
        os.path.join(image_dir, f"{case_id}{image_suffix}.nii.gz"),
        os.path.join(image_dir, f"{case_id}{image_suffix}.nii"),
        os.path.join(image_dir, f"{case_id}.nii.gz"),
        os.path.join(image_dir, f"{case_id}.nii"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def find_gt_path(gt_dir: str, case_id: str) -> str:
    candidates = [
        os.path.join(gt_dir, f"{case_id}.nii.gz"),
        os.path.join(gt_dir, f"{case_id}.nii"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def compute_bounds_from_gt(gt_zyx: np.ndarray, k: int, mode: str, gt_threshold: float) -> Tuple[int, int]:
    z_indices = np.where(np.any(gt_zyx > gt_threshold, axis=(1, 2)))[0]
    if len(z_indices) == 0:
        raise ValueError("GT has no foreground slices.")

    low = int(z_indices[0])
    high = int(z_indices[-1])

    if mode == "inward":
        return low + k, high - k
    if mode == "outward":
        return low - k, high + k
    if mode == "upshift":
        return low + k, high + k
    if mode == "downshift":
        return low - k, high - k
    raise ValueError(f"Unknown mode: {mode}")


def clip_bounds(low: int, high: int, z_size: int) -> Tuple[int, int]:
    return max(0, low), min(z_size - 1, high)


def restore_one_case(
    pred_path: str,
    gt_path: str,
    image_path: str,
    out_path: str,
    k: int,
    mode: str,
    gt_threshold: float,
    backend_name: str,
    backend_mod: Any,
) -> None:
    pred_arr, _ = read_nii_as_zyx(pred_path, backend_name, backend_mod)
    image_arr, image_meta = read_nii_as_zyx(image_path, backend_name, backend_mod)
    gt_arr, _ = read_nii_as_zyx(gt_path, backend_name, backend_mod)

    if image_arr.shape != gt_arr.shape:
        raise ValueError(f"GT shape != image shape: gt={gt_arr.shape}, image={image_arr.shape}")
    if pred_arr.ndim != 3:
        raise ValueError(f"Prediction is not 3D: shape={pred_arr.shape}")
    if pred_arr.shape[1:] != image_arr.shape[1:]:
        raise ValueError(f"Only Z-crop restore is supported. pred={pred_arr.shape}, image={image_arr.shape}")

    low, high = compute_bounds_from_gt(gt_arr, k, mode, gt_threshold)
    low, high = clip_bounds(low, high, image_arr.shape[0])
    if low > high:
        raise ValueError(f"invalid range after clip: [{low},{high}]")

    expected_depth = high - low + 1
    if pred_arr.shape[0] != expected_depth:
        raise ValueError(
            f"pred z-depth mismatch: pred={pred_arr.shape[0]}, expected={expected_depth}, range=[{low},{high}]"
        )

    restored = np.zeros(image_arr.shape, dtype=pred_arr.dtype)
    restored[low : high + 1, :, :] = pred_arr

    write_nii_from_zyx(restored, out_path, image_meta, backend_name, backend_mod)


def restore_group_predictions(
    group_name: str,
    pred_dir: str,
    out_dir: str,
    full_image_dir: str,
    full_gt_dir: str,
    image_suffix: str = "_0000",
    gt_threshold: float = 0.0,
) -> None:
    backend_name, backend_mod = get_io_backend()
    k, mode = parse_group_name(group_name)

    pred_paths = list_nii_files(pred_dir)
    if not pred_paths:
        raise FileNotFoundError(f"No prediction files found in {pred_dir}")

    ok_count = 0
    fail_count = 0
    for pred_path in pred_paths:
        pred_name = os.path.basename(pred_path)
        case_id_raw = strip_nii_ext(pred_name)
        case_id = (
            case_id_raw[: -len(image_suffix)]
            if image_suffix and case_id_raw.endswith(image_suffix)
            else case_id_raw
        )
        gt_path = find_gt_path(full_gt_dir, case_id)
        image_path = find_image_path(full_image_dir, case_id, image_suffix)
        if not gt_path or not image_path:
            fail_count += 1
            continue

        out_path = os.path.join(out_dir, pred_name)
        try:
            restore_one_case(
                pred_path,
                gt_path,
                image_path,
                out_path,
                k,
                mode,
                gt_threshold,
                backend_name,
                backend_mod,
            )
            ok_count += 1
        except Exception as exc:
            fail_count += 1
            print(f"[Restore Fail] {group_name}/{pred_name}: {exc}", flush=True)

    print(f"[Restore] {pred_dir} -> {out_dir} | OK={ok_count}, Fail={fail_count}", flush=True)


def run_cmd(cmd: List[str], env: Dict[str, str]) -> None:
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def run_model(config: ModelConfig, python_exe: str, cuda_value: str) -> None:
    if not config.test_script.exists():
        raise FileNotFoundError(f"Missing test script: {config.test_script}")

    cv_path = write_cv_results_if_needed(config.runs_root)
    best_fold = read_best_fold_from_cv(cv_path)
    groups = list_crop_groups(config.crop_root)

    os.makedirs(config.save_root, exist_ok=True)
    os.makedirs(config.restored_save_root, exist_ok=True)
    env = dict(os.environ)
    if cuda_value:
        env["CUDA_VISIBLE_DEVICES"] = cuda_value

    print(f"\n===== {config.name} =====", flush=True)
    print(f"runs_root: {config.runs_root}", flush=True)
    print(f"best_fold: {best_fold}", flush=True)
    print(f"save_root: {config.save_root}", flush=True)
    print(f"restored_save_root: {config.restored_save_root}", flush=True)

    for group in groups:
        data_root = os.path.join(config.crop_root, group)
        save_dir = os.path.join(config.save_root, group)
        restored_dir = os.path.join(config.restored_save_root, group)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(restored_dir, exist_ok=True)

        cmd = [
            python_exe,
            str(config.test_script),
            "--data_root",
            data_root,
            "--runs_root",
            config.runs_root,
            "--save_dir",
            save_dir,
            "--fold",
            str(best_fold),
            *config.extra_args,
        ]
        run_cmd(cmd, env)
        normalize_prediction_filenames(save_dir, "_0000")
        restore_group_predictions(
            group_name=group,
            pred_dir=save_dir,
            out_dir=restored_dir,
            full_image_dir=config.full_image_dir,
            full_gt_dir=config.full_gt_dir,
        )


def main():
    args = parse_args()
    repo_root = detect_repo_root(Path(__file__), args.repo_root)
    configs = default_model_configs(repo_root)

    for model_name in args.models:
        run_model(configs[model_name], args.python, args.cuda)

    print("\nAll crop-error tests finished.", flush=True)


if __name__ == "__main__":
    main()
