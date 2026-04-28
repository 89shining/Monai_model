import argparse
import csv
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Convert NIfTI dataset to 2D PNG slices for DDUNet.")
    parser.add_argument("--source_root", type=str, required=True, help="NIfTI dataset root.")
    parser.add_argument("--output_root", type=str, required=True, help="Output 2D dataset root.")
    parser.add_argument("--axis", type=str, default="z", choices=["z", "y", "x"], help="Slice axis.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing PNG files.")
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


def axis_to_index(axis: str) -> int:
    return {"z": 0, "y": 1, "x": 2}[axis]


def get_slice(arr: np.ndarray, axis: int, idx: int) -> np.ndarray:
    if axis == 0:
        return arr[idx, :, :]
    if axis == 1:
        return arr[:, idx, :]
    return arr[:, :, idx]


def normalize_to_u8(vol: np.ndarray) -> np.ndarray:
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


def find_nii_pairs(root: str, image_subdir: str, label_subdir: str) -> List[Tuple[str, str, str]]:
    imgs = sorted(glob.glob(os.path.join(root, image_subdir, "*_0000.nii.gz")) + glob.glob(os.path.join(root, image_subdir, "*_0000.nii")))
    lbls = sorted(glob.glob(os.path.join(root, label_subdir, "*.nii.gz")) + glob.glob(os.path.join(root, label_subdir, "*.nii")))
    if not imgs:
        raise FileNotFoundError(f"No NIfTI images found under: {os.path.join(root, image_subdir)}")
    if not lbls:
        raise FileNotFoundError(f"No NIfTI labels found under: {os.path.join(root, label_subdir)}")

    imap = {image_case_id(p): p for p in imgs}
    lmap = {label_case_id(p): p for p in lbls}
    if set(imap) != set(lmap):
        raise ValueError(
            f"ID mismatch in {image_subdir}/{label_subdir}: "
            f"missing_labels={len(set(imap) - set(lmap))}, missing_images={len(set(lmap) - set(imap))}"
        )
    ids = sorted(imap.keys())
    return [(cid, imap[cid], lmap[cid]) for cid in ids]


def ensure_dirs(root: str):
    for d in ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]:
        os.makedirs(os.path.join(root, d), exist_ok=True)


def write_manifest(path: str, rows: List[Dict[str, object]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "slice_idx", "image", "label"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def convert_split(pairs, out_root: str, img_subdir: str, lbl_subdir: str, axis_idx: int, force: bool):
    rows = []
    for cid, ip, lp in pairs:
        img = sitk.GetArrayFromImage(sitk.ReadImage(ip))
        lbl = sitk.GetArrayFromImage(sitk.ReadImage(lp))
        if img.shape != lbl.shape:
            raise ValueError(f"Shape mismatch for {cid}: image={img.shape}, label={lbl.shape}")

        img_u8 = normalize_to_u8(img)
        n = img_u8.shape[axis_idx]
        for i in range(n):
            s_img = get_slice(img_u8, axis_idx, i)
            s_lbl = get_slice(lbl, axis_idx, i).astype(np.uint8)

            fn = f"{cid}_slice{i:04d}.png"
            img_out = os.path.join(out_root, img_subdir, fn)
            lbl_out = os.path.join(out_root, lbl_subdir, fn)

            if force or not os.path.exists(img_out):
                Image.fromarray(s_img, mode="L").save(img_out)
            if force or not os.path.exists(lbl_out):
                Image.fromarray(s_lbl, mode="L").save(lbl_out)

            rows.append({"case_id": cid, "slice_idx": i, "image": os.path.join(img_subdir, fn), "label": os.path.join(lbl_subdir, fn)})
    return rows


def main():
    args = parse_args()
    ensure_dirs(args.output_root)
    axis_idx = axis_to_index(args.axis)

    tr_pairs = find_nii_pairs(args.source_root, "imagesTr", "labelsTr")
    ts_pairs = find_nii_pairs(args.source_root, "imagesTs", "labelsTs")

    tr_rows = convert_split(tr_pairs, args.output_root, "imagesTr", "labelsTr", axis_idx, args.force)
    ts_rows = convert_split(ts_pairs, args.output_root, "imagesTs", "labelsTs", axis_idx, args.force)

    write_manifest(os.path.join(args.output_root, "train_manifest.csv"), tr_rows)
    write_manifest(os.path.join(args.output_root, "test_manifest.csv"), ts_rows)

    print("=" * 88)
    print("DDUNet preprocessing done")
    print(f"Source: {args.source_root}")
    print(f"Output: {args.output_root}")
    print(f"Axis: {args.axis}")
    print(f"Train slices: {len(tr_rows)}")
    print(f"Test slices: {len(ts_rows)}")
    print("=" * 88)


if __name__ == "__main__":
    main()

