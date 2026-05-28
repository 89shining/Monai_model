"""
Batch rename prediction NIfTI files by removing a trailing suffix in filename stem.
"""

import glob
import os
from typing import List

# -----------------------------
# Edit here: set your paths/values directly in code
# -----------------------------
PRED_DIR = r"C:\Users\WS\Desktop\Crop\Rectal_146\VNet\VNet_all_rawpred"
SUFFIX = "_0000"


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


def main() -> None:
    pred_dir = PRED_DIR
    suffix = SUFFIX

    if not os.path.isdir(pred_dir):
        raise NotADirectoryError(f"PRED_DIR not found: {pred_dir}")
    if not suffix:
        raise ValueError("suffix cannot be empty")

    pred_paths = list_nii_files(pred_dir)
    if not pred_paths:
        print(f"No NIfTI files found in: {pred_dir}")
        return

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
            raise FileExistsError(f"Rename target already exists: {dst}")

        os.rename(src, dst)
        renamed += 1
        print(f"[Rename] {name} -> {os.path.basename(dst)}")

    print(f"Done. Renamed={renamed}, Skipped={skipped}")


if __name__ == "__main__":
    main()
