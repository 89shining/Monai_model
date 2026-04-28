import argparse
import csv
import os
import re

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from ddunet import DDUNet2D


def parse_args():
    parser = argparse.ArgumentParser(description="DDUNet best-fold test on 2D slices.")
    parser.add_argument("--data_root", type=str, required=True, help="2D data root.")
    parser.add_argument("--runs_root", type=str, default="./runs/DDUNet")
    parser.add_argument("--save_dir", type=str, default="./predictions")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--input_h", type=int, default=256)
    parser.add_argument("--input_w", type=int, default=256)
    parser.add_argument("--fold", type=int, default=None, help="Force fold index. None: choose best from cv_results.")
    return parser.parse_args()


def case_from_slice_name(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    m = re.match(r"(.+)_slice\d+$", stem)
    return m.group(1) if m else stem


def load_test_samples(data_root: str):
    manifest = os.path.join(data_root, "test_manifest.csv")
    samples = []
    if os.path.exists(manifest):
        with open(manifest, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                samples.append(
                    {
                        "case_id": r["case_id"],
                        "slice_idx": int(r["slice_idx"]),
                        "image": os.path.join(data_root, r["image"]),
                        "label": os.path.join(data_root, r["label"]),
                    }
                )
    else:
        img_dir = os.path.join(data_root, "imagesTs")
        lbl_dir = os.path.join(data_root, "labelsTs")
        for fn in sorted(os.listdir(img_dir)):
            if not fn.lower().endswith(".png"):
                continue
            lp = os.path.join(lbl_dir, fn)
            if not os.path.exists(lp):
                continue
            samples.append({"case_id": case_from_slice_name(fn), "slice_idx": 0, "image": os.path.join(img_dir, fn), "label": lp})
    if not samples:
        raise ValueError("No test samples found.")
    return samples


class DDUNetTestDataset(Dataset):
    def __init__(self, samples, input_h: int, input_w: int):
        self.samples = samples
        self.input_h = input_h
        self.input_w = input_w

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s["image"]).convert("L")
        ow, oh = image.size
        image = image.resize((self.input_w, self.input_h), Image.BILINEAR)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_np = np.expand_dims(image_np, axis=0)
        return {"image": torch.from_numpy(image_np), "label_path": s["label"], "orig_w": ow, "orig_h": oh}


def select_best_fold(runs_root: str) -> int:
    cv_csv = os.path.join(runs_root, "cv_results.csv")
    if not os.path.exists(cv_csv):
        raise FileNotFoundError(f"cv_results.csv not found: {cv_csv}")
    with open(cv_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"cv_results.csv is empty: {cv_csv}")
    return int(max(rows, key=lambda r: float(r["best_dice_fg"]))["fold"])


def load_model(args, fold: int, device: torch.device):
    ckpt = os.path.join(args.runs_root, f"fold_{fold}", f"best_model_fold{fold}.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Model not found: {ckpt}")
    model = DDUNet2D(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, ckpt


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold = args.fold if args.fold is not None else select_best_fold(args.runs_root)
    model, ckpt = load_model(args, fold, device)

    samples = load_test_samples(args.data_root)
    ds = DDUNetTestDataset(samples, args.input_h, args.input_w)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    print("=" * 88)
    print("DDUNet Test")
    print(f"Device: {device}")
    print(f"Selected fold: {fold}")
    print(f"Checkpoint: {ckpt}")
    print(f"Save dir: {args.save_dir}")
    print("=" * 88)

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device, non_blocking=True)
            pred = model(imgs)
            pred = (pred > 0.5).float().cpu().numpy().astype(np.uint8)

            bs = pred.shape[0]
            for i in range(bs):
                label_path = batch["label_path"][i]
                save_name = os.path.basename(label_path)
                ow = int(batch["orig_w"][i])
                oh = int(batch["orig_h"][i])
                p = Image.fromarray((pred[i, 0] * 255).astype(np.uint8), mode="L").resize((ow, oh), Image.NEAREST)
                out_path = os.path.join(args.save_dir, save_name)
                p.save(out_path)
                print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

