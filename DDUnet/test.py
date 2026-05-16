import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np
import SimpleITK as sitk
import torch
from monai.data import DataLoader
from monai.transforms import Compose, EnsureTyped, Lambdad, Resized
from torch.utils.data import Dataset


def parse_args(default_runs_root: str):
    p = argparse.ArgumentParser(description='2D segmentation testing with best fold.')
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--runs_root', type=str, default=default_runs_root)
    p.add_argument('--save_dir', type=str, default='./predictions')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--input_h', type=int, default=512)
    p.add_argument('--input_w', type=int, default=512)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--fold', type=int, default=None)
    p.add_argument('--a_min', type=float, default=-160.0)
    p.add_argument('--a_max', type=float, default=240.0)
    return p.parse_args()


def strip_nii_ext(filename: str) -> str:
    if filename.endswith('.nii.gz'): return filename[:-7]
    if filename.endswith('.nii'): return filename[:-4]
    return os.path.splitext(filename)[0]


def image_case_id(path: str) -> str:
    n = strip_nii_ext(os.path.basename(path)); return n[:-5] if n.endswith('_0000') else n


def label_case_id(path: str) -> str:
    return strip_nii_ext(os.path.basename(path))


def build_test_cases(data_root: str):
    imgs = sorted(glob.glob(os.path.join(data_root, 'imagesTs', '*_0000.nii.gz')) + glob.glob(os.path.join(data_root, 'imagesTs', '*_0000.nii')))
    lbls = sorted(glob.glob(os.path.join(data_root, 'labelsTs', '*.nii.gz')) + glob.glob(os.path.join(data_root, 'labelsTs', '*.nii')))
    im = {image_case_id(p): p for p in imgs}; lb = {label_case_id(p): p for p in lbls}
    ids = sorted(set(im).intersection(set(lb)))
    return [{'case_id': c, 'image': im[c], 'label': lb[c]} for c in ids]


def pick_best_fold(runs_root: str) -> int:
    with open(os.path.join(runs_root, 'cv_results.csv'), 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return int(max(rows, key=lambda r: float(r['best_dice_fg']))['fold'])


class NiftiTestSliceDataset(Dataset):
    def __init__(self, case_items, transform=None):
        self.transform = transform
        self.records = []
        self.cache = {}
        for item in case_items:
            img_itk = sitk.ReadImage(item['image'])
            lbl_itk = sitk.ReadImage(item['label'])
            img = sitk.GetArrayFromImage(img_itk).astype(np.float32)
            lbl = (sitk.GetArrayFromImage(lbl_itk) > 0).astype(np.float32)
            self.cache[item['case_id']] = {'img': img, 'lbl': lbl, 'img_itk': img_itk, 'label_name': os.path.basename(item['label'])}
            for z in range(img.shape[0]):
                self.records.append((item['case_id'], z))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        cid, z = self.records[idx]
        v = self.cache[cid]
        sample = {'image': v['img'][z][None, ...], 'label': v['lbl'][z][None, ...], 'case_id': cid, 'slice_idx': z}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

from ddunet import DDUNet2D


def main():
    args = parse_args('./runs/DDUNet'); os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fold = args.fold if args.fold is not None else pick_best_fold(args.runs_root)
    model_path = os.path.join(args.runs_root, f'fold_{fold}', 'best_epoch_weights.pth')

    model = DDUNet2D(in_channels=1, out_channels=1).to(device)
    st = torch.load(model_path, map_location=device)
    model.load_state_dict(st['model'] if isinstance(st, dict) and 'model' in st else st)
    model.eval()

    def norm(x):
        x = np.clip(x, args.a_min, args.a_max)
        return (x - args.a_min) / max(args.a_max - args.a_min, 1e-6)

    tf = Compose([
        Lambdad(keys=['image'], func=norm),
        Resized(keys=['image', 'label'], spatial_size=(args.input_h, args.input_w), mode=('bilinear', 'nearest')),
        EnsureTyped(keys=['image', 'label'], dtype=torch.float32),
    ])

    cases = build_test_cases(args.data_root)
    ds = NiftiTestSliceDataset(cases, tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type=='cuda')

    preds = defaultdict(list)
    with torch.no_grad():
        for b in loader:
            x = b['image'].to(device)
            p = model(x)
            y = (p > args.threshold).float().cpu().numpy()
            for i in range(y.shape[0]):
                cid = b['case_id'][i]
                z = int(b['slice_idx'][i])
                preds[cid].append((z, y[i, 0]))

    for cid, items in preds.items():
        items = sorted(items, key=lambda t: t[0])
        vol = np.stack([m for _, m in items], axis=0).astype(np.uint8)
        meta = ds.cache[cid]
        out = sitk.GetImageFromArray(vol)
        out.CopyInformation(meta['img_itk'])
        sitk.WriteImage(out, os.path.join(args.save_dir, meta['label_name']))


if __name__ == '__main__':
    main()
