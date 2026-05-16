import argparse
import csv
import glob
import os
import random
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from monai.data import DataLoader, Dataset, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)

from vnet import VNet


def parse_args():
    parser = argparse.ArgumentParser(description="3D VNet training with K-fold CV.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--runs_root", type=str, default="./runs/VNet")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--early_stop_patience", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=12.0)
    parser.add_argument("--roi_x", type=int, default=96)
    parser.add_argument("--roi_y", type=int, default=96)
    parser.add_argument("--roi_z", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--sw_batch_size", type=int, default=2)
    parser.add_argument("--infer_overlap", type=float, default=0.5)
    parser.add_argument("--a_min", type=float, default=-160.0)
    parser.add_argument("--a_max", type=float, default=240.0)
    parser.add_argument("--pixdim_x", type=float, default=1.0)
    parser.add_argument("--pixdim_y", type=float, default=1.0)
    parser.add_argument("--pixdim_z", type=float, default=1.0)
    parser.add_argument("--axcodes", type=str, default="RAS")
    parser.add_argument("--only_fold", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def binarize_label(x):
    if isinstance(x, torch.Tensor):
        return (x > 0).float()
    return (x > 0).astype(np.float32)


class VNet3DSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = VNet(elu=True, nll=False)

    def forward(self, x):
        n, _, d, h, w = x.shape
        out = self.net(x)
        out = out.view(n, d, h, w, 2).permute(0, 4, 1, 2, 3).contiguous()
        return out[:, 1:2]


class ProbDiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(sigmoid=False)
        self.bce = nn.BCELoss()

    def forward(self, pred_prob, target):
        return self.dice(pred_prob, target) + self.bce(pred_prob, target)


def strip_nii_ext(filename: str) -> str:
    if filename.endswith('.nii.gz'):
        return filename[:-7]
    if filename.endswith('.nii'):
        return filename[:-4]
    return os.path.splitext(filename)[0]


def image_case_id(path: str) -> str:
    name = strip_nii_ext(os.path.basename(path))
    return name[:-5] if name.endswith('_0000') else name


def collect_data_list(data_root: str) -> List[Dict[str, str]]:
    image_dir = os.path.join(data_root, 'imagesTr')
    label_dir = os.path.join(data_root, 'labelsTr')
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.nii.gz')) + glob.glob(os.path.join(image_dir, '*.nii')))
    if not image_paths:
        raise RuntimeError(f'No NIfTI files found in: {image_dir}')
    data_list = []
    for image_path in image_paths:
        case_id = image_case_id(image_path)
        p1 = os.path.join(label_dir, f'{case_id}.nii.gz')
        p2 = os.path.join(label_dir, f'{case_id}.nii')
        label_path = p1 if os.path.exists(p1) else p2
        if not os.path.exists(label_path):
            raise FileNotFoundError(f'Label not found for case: {case_id}')
        data_list.append({'case_id': case_id, 'image': image_path, 'label': label_path})
    return data_list


def make_kfold_splits(data_list: List[Dict[str, str]], num_folds: int, seed: int):
    indices = list(range(len(data_list)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    fold_sizes = [len(indices) // num_folds] * num_folds
    for i in range(len(indices) % num_folds):
        fold_sizes[i] += 1
    folds = []
    start = 0
    for fs in fold_sizes:
        folds.append(indices[start:start + fs])
        start += fs
    splits = []
    for fold_idx in range(num_folds):
        val_indices = folds[fold_idx]
        train_indices = [j for i, f in enumerate(folds) if i != fold_idx for j in f]
        splits.append(([data_list[i] for i in train_indices], [data_list[i] for i in val_indices]))
    return splits


def get_transforms(args):
    roi_size = (args.roi_x, args.roi_y, args.roi_z)
    pixdim = (args.pixdim_x, args.pixdim_y, args.pixdim_z)
    train_transforms = Compose([
        LoadImaged(keys=['image', 'label']), EnsureChannelFirstd(keys=['image', 'label']),
        Orientationd(keys=['image', 'label'], axcodes=args.axcodes),
        Spacingd(keys=['image', 'label'], pixdim=pixdim, mode=('bilinear', 'nearest')),
        ScaleIntensityRanged(keys=['image'], a_min=args.a_min, a_max=args.a_max, b_min=0.0, b_max=1.0, clip=True),
        Lambdad(keys=['label'], func=binarize_label), SpatialPadd(keys=['image', 'label'], spatial_size=roi_size),
        RandCropByPosNegLabeld(keys=['image', 'label'], label_key='label', spatial_size=roi_size, pos=1, neg=1,
                               num_samples=args.num_samples, image_key='image', image_threshold=0),
        RandFlipd(keys=['image', 'label'], spatial_axis=0, prob=0.5),
        RandFlipd(keys=['image', 'label'], spatial_axis=1, prob=0.5),
        RandAffined(keys=['image', 'label'], prob=0.2, rotate_range=(0.1, 0.1, 0.05), scale_range=(0.1, 0.1, 0.05),
                    mode=('bilinear', 'nearest'), padding_mode='border'),
        RandShiftIntensityd(keys=['image'], offsets=0.1, prob=0.3),
        RandScaleIntensityd(keys=['image'], factors=0.1, prob=0.3),
        RandGaussianNoised(keys=['image'], prob=0.15, mean=0.0, std=0.01),
        EnsureTyped(keys=['image', 'label'], dtype=torch.float32),
    ])
    val_transforms = Compose([
        LoadImaged(keys=['image', 'label']), EnsureChannelFirstd(keys=['image', 'label']),
        Orientationd(keys=['image', 'label'], axcodes=args.axcodes),
        Spacingd(keys=['image', 'label'], pixdim=pixdim, mode=('bilinear', 'nearest')),
        ScaleIntensityRanged(keys=['image'], a_min=args.a_min, a_max=args.a_max, b_min=0.0, b_max=1.0, clip=True),
        Lambdad(keys=['label'], func=binarize_label),
        EnsureTyped(keys=['image', 'label'], dtype=torch.float32),
    ])
    return train_transforms, val_transforms


def train_one_epoch(model, loader, optimizer, loss_func, device, args):
    model.train(); total = 0.0; steps = 0
    for batch in loader:
        steps += 1
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad(set_to_none=True)
        probs = model(images)
        loss = loss_func(probs, labels)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        total += loss.item()
    return total / max(steps, 1)


@torch.no_grad()
def validate(model, loader, loss_func, dice_metric, post_pred, post_label, device, args):
    model.eval(); dice_metric.reset(); val_loss = 0.0; steps = 0
    for batch in loader:
        steps += 1
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        probs = sliding_window_inference(images, (args.roi_x, args.roi_y, args.roi_z), args.sw_batch_size, model, overlap=args.infer_overlap)
        probs = torch.clamp(probs, 0.0, 1.0)
        val_loss += loss_func(probs, labels).item()
        preds = [post_pred(x) for x in decollate_batch(probs)]
        gts = [post_label(x) for x in decollate_batch(labels)]
        dice_metric(y_pred=preds, y=gts)
    return val_loss / max(steps, 1), dice_metric.aggregate().item()


def main():
    args = parse_args()
    os.makedirs(args.runs_root, exist_ok=True)
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_list = collect_data_list(args.data_root)
    splits = make_kfold_splits(data_list, args.num_folds, args.seed)
    all_results = []

    for fold_idx, (train_files, val_files) in enumerate(splits):
        if args.only_fold is not None and fold_idx != args.only_fold:
            continue
        fold_dir = os.path.join(args.runs_root, f'fold_{fold_idx}')
        os.makedirs(fold_dir, exist_ok=True)

        train_tf, val_tf = get_transforms(args)
        train_ds = Dataset(data=train_files, transform=train_tf)
        val_ds = Dataset(data=val_files, transform=val_tf)
        g = torch.Generator(); g.manual_seed(args.seed + fold_idx)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=device.type == 'cuda', collate_fn=list_data_collate,
                                  worker_init_fn=seed_worker, generator=g)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                pin_memory=device.type == 'cuda', worker_init_fn=seed_worker, generator=g)

        model = VNet3DSeg().to(device)
        loss_func = ProbDiceBCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.eta_min)
        dice_metric = DiceMetric(include_background=True, reduction='mean', get_not_nans=False)
        post_pred = Compose([AsDiscrete(threshold=0.5)])
        post_label = Compose([AsDiscrete(threshold=0.5)])

        log_path = os.path.join(fold_dir, 'epoch_metrics.csv')
        best_path = os.path.join(fold_dir, f'best_model_fold{fold_idx}.pth')
        last_state_path = os.path.join(fold_dir, f'last_state_fold{fold_idx}.pt')

        best_dice, best_epoch, no_improve, start_epoch = -1.0, -1, 0, 1
        if args.resume and os.path.exists(last_state_path):
            state = torch.load(last_state_path, map_location=device)
            model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            best_dice = float(state.get('best_dice', -1.0)); best_epoch = int(state.get('best_epoch', -1))
            no_improve = int(state.get('epochs_no_improve', 0)); start_epoch = int(state.get('epoch', 0)) + 1

        with open(log_path, 'a' if (args.resume and os.path.exists(log_path) and start_epoch > 1) else 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_dice_fg', 'lr', 'is_best'])
            for epoch in range(start_epoch, args.max_epochs + 1):
                train_loss = train_one_epoch(model, train_loader, optimizer, loss_func, device, args)
                val_loss, val_dice = validate(model, val_loader, loss_func, dice_metric, post_pred, post_label, device, args)
                lr_now = optimizer.param_groups[0]['lr']; scheduler.step()
                is_best = int(val_dice > best_dice)
                writer.writerow([epoch, f'{train_loss:.6f}', f'{val_loss:.6f}', f'{val_dice:.6f}', f'{lr_now:.8f}', is_best]); f.flush()
                if is_best:
                    best_dice, best_epoch, no_improve = val_dice, epoch, 0
                    torch.save({'fold': fold_idx, 'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(), 'best_dice': best_dice, 'best_epoch': best_epoch, 'args': vars(args)}, best_path)
                else:
                    no_improve += 1
                torch.save({'fold': fold_idx, 'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(), 'best_dice': best_dice, 'best_epoch': best_epoch,
                            'epochs_no_improve': no_improve, 'args': vars(args)}, last_state_path)
                if no_improve >= args.early_stop_patience:
                    break

        all_results.append({'fold': fold_idx, 'best_epoch': best_epoch, 'best_dice_fg': best_dice,
                            'num_train': len(train_files), 'num_val': len(val_files), 'checkpoint': best_path})

    with open(os.path.join(args.runs_root, 'cv_results.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['fold', 'best_epoch', 'best_dice_fg', 'num_train', 'num_val', 'checkpoint'])
        writer.writeheader()
        for r in sorted(all_results, key=lambda x: x['fold']):
            writer.writerow({'fold': r['fold'], 'best_epoch': r['best_epoch'], 'best_dice_fg': f"{r['best_dice_fg']:.6f}",
                             'num_train': r['num_train'], 'num_val': r['num_val'], 'checkpoint': r['checkpoint']})


if __name__ == '__main__':
    main()
