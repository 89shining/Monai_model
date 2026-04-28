import argparse
import csv
import glob
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Convert NIfTI dataset (imagesTr/labelsTr/imagesTs/labelsTs) to 2D PNG slices.")
    parser.add_argument("--source_root", type=str, required=True, help="NIfTI dataset root.")
    parser.add_argument("--output_root", type=str, required=True, help="Output 2D dataset root.")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes including background.")
    parser.add_argument("--axis", type=str, default="z", choices=["z", "y", "x"], help="Slice axis in image array space.")
    parser.add_argument("--force", action="store_true", help="Re-generate even if output files exist.")
    return parser.parse_args()


def strip_nii_ext(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return os.path.splitext(name)[0]


def image_case_id(path: str) -> str:
    n = strip_nii_ext(os.path.basename(path))
    return n[:-5] if n.endswith("_0000") else n


def label_case_id(path: str) -> str:
    return strip_nii_ext(os.path.basename(path))


def find_nii_pairs(root: str, image_subdir: str, label_subdir: str) -> List[Tuple[str, str, str]]:
    imgs = sorted(glob.glob(os.path.join(root, image_subdir, "*_0000.nii.gz")) + glob.glob(os.path.join(root, image_subdir, "*_0000.nii")))
    lbls = sorted(glob.glob(os.path.join(root, label_subdir, "*.nii.gz")) + glob.glob(os.path.join(root, label_subdir, "*.nii")))

    if not imgs:
        raise FileNotFoundError(f"No NIfTI images found under: {os.path.join(root, image_subdir)}")
    if not lbls:
        raise FileNotFoundError(f"No NIfTI labels found under: {os.path.join(root, label_subdir)}")

    imap = {image_case_id(p): p for p in imgs}
    lmap = {label_case_id(p): p for p in lbls}

    iids = set(imap.keys())
    lids = set(lmap.keys())
    if iids != lids:
        raise ValueError(
            f"ID mismatch for {image_subdir}/{label_subdir}: "
            f"missing_labels={len(iids - lids)}, missing_images={len(lids - iids)}"
        )

    case_ids = sorted(iids)
    return [(cid, imap[cid], lmap[cid]) for cid in case_ids]


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)
    p1, p99 = np.percentile(vol, [0.5, 99.5])
    if p99 <= p1:
        p1 = float(vol.min())
        p99 = float(vol.max())
    if p99 <= p1:
        return np.zeros_like(vol, dtype=np.uint8)
    vol = np.clip(vol, p1, p99)
    vol = (vol - p1) / (p99 - p1)
    return (vol * 255.0).astype(np.uint8)


def slice_count(arr: np.ndarray, axis: int) -> int:
    return arr.shape[axis]


def get_slice(arr: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        return arr[idx, :, :]
    if axis == 1:
        return arr[:, idx, :]
    return arr[:, :, idx]


def axis_to_index(axis: str) -> int:
    return {"z": 0, "y": 1, "x": 2}[axis]


def ensure_dirs(out_root: str):
    for d in ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]:
        os.makedirs(os.path.join(out_root, d), exist_ok=True)


def save_rgb_png(gray2d: np.ndarray, out_path: str):
    rgb = np.stack([gray2d, gray2d, gray2d], axis=-1)
    Image.fromarray(rgb, mode="RGB").save(out_path)


def save_label_png(lbl2d: np.ndarray, out_path: str, num_classes: int):
    lbl2d = np.asarray(lbl2d)
    lbl2d = np.clip(lbl2d, 0, num_classes - 1).astype(np.uint8)
    Image.fromarray(lbl2d, mode="L").save(out_path)


def convert_split(pairs, out_root: str, image_out_subdir: str, label_out_subdir: str, axis_idx: int, num_classes: int, force: bool):
    rows = []
    for case_id, img_nii, lbl_nii in pairs:
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_nii))
        lbl = sitk.GetArrayFromImage(sitk.ReadImage(lbl_nii))

        if img.shape != lbl.shape:
            raise ValueError(f"Shape mismatch for case {case_id}: image={img.shape}, label={lbl.shape}")

        img_u8 = normalize_volume(img)
        n = slice_count(img_u8, axis_idx)

        for i in range(n):
            s_img = get_slice(img_u8, axis_idx, i)
            s_lbl = get_slice(lbl, axis_idx, i)

            img_name = f"{case_id}_slice{i:04d}.png"
            lbl_name = f"{case_id}_slice{i:04d}.png"
            img_out = os.path.join(out_root, image_out_subdir, img_name)
            lbl_out = os.path.join(out_root, label_out_subdir, lbl_name)

            if force or not os.path.exists(img_out):
                save_rgb_png(s_img, img_out)
            if force or not os.path.exists(lbl_out):
                save_label_png(s_lbl, lbl_out, num_classes)

            rows.append(
                {
                    "case_id": case_id,
                    "slice_idx": i,
                    "image": os.path.join(image_out_subdir, img_name),
                    "label": os.path.join(label_out_subdir, lbl_name),
                }
            )

    return rows


def write_manifest(path: str, rows: List[Dict[str, object]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "slice_idx", "image", "label"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    args = parse_args()
    ensure_dirs(args.output_root)
    axis_idx = axis_to_index(args.axis)

    tr_pairs = find_nii_pairs(args.source_root, "imagesTr", "labelsTr")
    ts_pairs = find_nii_pairs(args.source_root, "imagesTs", "labelsTs")

    tr_rows = convert_split(tr_pairs, args.output_root, "imagesTr", "labelsTr", axis_idx, args.num_classes, args.force)
    ts_rows = convert_split(ts_pairs, args.output_root, "imagesTs", "labelsTs", axis_idx, args.num_classes, args.force)

    write_manifest(os.path.join(args.output_root, "train_manifest.csv"), tr_rows)
    write_manifest(os.path.join(args.output_root, "test_manifest.csv"), ts_rows)

    print("=" * 88)
    print("NIfTI -> 2D preprocessing done")
    print(f"Source: {args.source_root}")
    print(f"Output: {args.output_root}")
    print(f"Axis: {args.axis}")
    print(f"Train slices: {len(tr_rows)}")
    print(f"Test slices: {len(ts_rows)}")
    print("=" * 88)


if __name__ == "__main__":
    main()
