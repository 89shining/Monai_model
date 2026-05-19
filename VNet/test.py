import argparse
import csv
import glob
import os

import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscreted, Compose, EnsureChannelFirstd, EnsureTyped, Invertd, Lambdad, LoadImaged, Orientationd, SaveImaged, ScaleIntensityRanged, Spacingd

from train import VNet3DSeg


def parse_args():
    p = argparse.ArgumentParser(description='VNet testing using best fold.')
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--runs_root', type=str, default='./runs/VNet')
    p.add_argument('--save_dir', type=str, default='./predictions')
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--roi_x', type=int, default=96)
    p.add_argument('--roi_y', type=int, default=96)
    p.add_argument('--roi_z', type=int, default=64)
    p.add_argument('--sw_batch_size', type=int, default=2)
    p.add_argument('--infer_overlap', type=float, default=0.5)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--fold', type=int, default=None)
    p.add_argument('--a_min', type=float, default=-160.0)
    p.add_argument('--a_max', type=float, default=240.0)
    p.add_argument('--pixdim_x', type=float, default=1.0)
    p.add_argument('--pixdim_y', type=float, default=1.0)
    p.add_argument('--pixdim_z', type=float, default=1.0)
    p.add_argument('--axcodes', type=str, default='RAS')
    return p.parse_args()


def _bin(x): return (x > 0).astype(x.dtype)

def strip_nii_ext(filename: str) -> str:
    if filename.endswith('.nii.gz'): return filename[:-7]
    if filename.endswith('.nii'): return filename[:-4]
    return os.path.splitext(filename)[0]

def image_case_id(path: str) -> str:
    n = strip_nii_ext(os.path.basename(path)); return n[:-5] if n.endswith('_0000') else n

def label_case_id(path: str) -> str:
    return strip_nii_ext(os.path.basename(path))


def build_test_data(data_root: str):
    images = sorted(glob.glob(os.path.join(data_root, 'imagesTs', '*_0000.nii.gz')) + glob.glob(os.path.join(data_root, 'imagesTs', '*_0000.nii')))
    labels = sorted(glob.glob(os.path.join(data_root, 'labelsTs', '*.nii.gz')) + glob.glob(os.path.join(data_root, 'labelsTs', '*.nii')))
    image_map = {image_case_id(p): p for p in images}; label_map = {label_case_id(p): p for p in labels}
    case_ids = sorted(set(image_map).intersection(set(label_map)))
    return [{'image': image_map[c], 'label': label_map[c]} for c in case_ids]


def pick_best_fold(runs_root: str) -> int:
    with open(os.path.join(runs_root, 'cv_results.csv'), 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    best = max(rows, key=lambda r: float(r['best_dice_fg']))
    return int(best['fold'])


def main():
    args = parse_args(); os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fold = args.fold if args.fold is not None else pick_best_fold(args.runs_root)

    model_path = os.path.join(args.runs_root, f'fold_{fold}', f'best_model_fold{fold}.pth')
    model = VNet3DSeg().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['model'] if isinstance(state, dict) and 'model' in state else state)
    model.eval()

    tf = Compose([
        LoadImaged(keys=['image', 'label']), EnsureChannelFirstd(keys=['image', 'label']),
        Orientationd(keys=['image', 'label'], axcodes=args.axcodes),
        Spacingd(keys=['image', 'label'], pixdim=(args.pixdim_x, args.pixdim_y, args.pixdim_z), mode=('bilinear', 'nearest')),
        ScaleIntensityRanged(keys=['image'], a_min=args.a_min, a_max=args.a_max, b_min=0.0, b_max=1.0, clip=True),
        Lambdad(keys='label', func=_bin), EnsureTyped(keys=['image', 'label'], track_meta=True),
    ])
    post_transforms = Compose([
        AsDiscreted(keys='pred', threshold=args.threshold),
        Invertd(
            keys='pred',
            transform=tf,
            orig_keys='image',
            nearest_interp=True,
            to_tensor=False,
        ),
        SaveImaged(
            keys='pred',
            meta_keys='pred_meta_dict',
            output_dir=args.save_dir,
            output_postfix='',
            output_ext='.nii.gz',
            separate_folder=False,
            output_dtype='uint8',
            resample=False,
            print_log=False,
        ),
    ])

    loader = DataLoader(Dataset(build_test_data(args.data_root), tf), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == 'cuda')

    with torch.no_grad():
        for batch in loader:
            image = batch['image'].to(device)
            probs = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), args.sw_batch_size, model, overlap=args.infer_overlap)
            batch['pred'] = probs
            for item in decollate_batch(batch):
                out = post_transforms(item)
                meta = out.get('pred_meta_dict') if isinstance(out, dict) else None
                saved_to = None if meta is None else meta.get('saved_to')
                if saved_to:
                    print(f"Saved: {saved_to}")
                else:
                    print("Saved one prediction file.")


if __name__ == '__main__':
    main()
