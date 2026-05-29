import argparse
import csv
import glob
import os
import subprocess
import sys
from typing import Dict, List, Optional

from sklearn.model_selection import KFold


def parse_args():
    p = argparse.ArgumentParser(description="Train + test pipeline for VNet.")
    p.add_argument("--data_root", type=str, default="/home/wusi/Project_crop/Data/Rectal_146/RectalCTV_All")
    p.add_argument("--save_root", type=str, default="/home/wusi/Project_crop/Data/Rectal_146/Networks/VNet/RectalCTV_All")
    p.add_argument("--cuda", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", "1"))

    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--only_fold", type=int, default=-1, help="Train only one fold. Use -1 to train all folds (default).")

    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--early_stop_patience", type=int, default=15)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--test_batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eta_min", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=12.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--roi_x", type=int, default=256)
    p.add_argument("--roi_y", type=int, default=256)
    p.add_argument("--roi_z", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.5)

    p.add_argument("--a_min", type=float, default=-160.0)
    p.add_argument("--a_max", type=float, default=240.0)
    p.add_argument("--pixdim_x", type=float, default=1.0)
    p.add_argument("--pixdim_y", type=float, default=1.0)
    p.add_argument("--pixdim_z", type=float, default=1.0)
    p.add_argument("--axcodes", type=str, default="RAS")

    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Resume from last_state if available (default: True).")
    p.add_argument("--strong_aug", action="store_true")
    p.add_argument("--save_prob", action="store_true")
    return p.parse_args()


def run_cmd(cmd: List[str], env: Dict[str, str]):
    print("\n[CMD]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def strip_nii_ext(filename: str) -> str:
    if filename.endswith('.nii.gz'):
        return filename[:-7]
    if filename.endswith('.nii'):
        return filename[:-4]
    return os.path.splitext(filename)[0]


def image_case_id(path: str) -> str:
    name = strip_nii_ext(os.path.basename(path))
    return name[:-5] if name.endswith('_0000') else name


def label_case_id(path: str) -> str:
    return strip_nii_ext(os.path.basename(path))


def collect_case_ids(data_root: str) -> List[str]:
    images = sorted(glob.glob(os.path.join(data_root, 'imagesTr', '*_0000.nii.gz')))
    labels = sorted(glob.glob(os.path.join(data_root, 'labelsTr', '*.nii.gz')))

    if not images:
        raise FileNotFoundError(f"No training images found under: {os.path.join(data_root, 'imagesTr')}")
    if not labels:
        raise FileNotFoundError(f"No training labels found under: {os.path.join(data_root, 'labelsTr')}")

    image_map = {image_case_id(p): p for p in images}
    label_map = {label_case_id(p): p for p in labels}
    image_ids = set(image_map.keys())
    label_ids = set(label_map.keys())
    if image_ids != label_ids:
        missing_labels = sorted(image_ids - label_ids)
        missing_images = sorted(label_ids - image_ids)
        raise ValueError(
            "Image/label case IDs are inconsistent. "
            f"missing_labels={len(missing_labels)}, missing_images={len(missing_images)}"
        )
    return sorted(image_ids)


def read_best_from_fold_epoch_csv(fold_dir: str):
    with open(os.path.join(fold_dir, 'epoch_metrics.csv'), 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in epoch_metrics.csv under: {fold_dir}")
    best = max(rows, key=lambda r: float(r['val_dice_fg']))
    return int(best['epoch']), float(best['val_dice_fg'])


def is_fold_completed(runs_root: str, fold: int) -> bool:
    fold_dir = os.path.join(runs_root, f'fold_{fold}')
    ckpt = os.path.join(fold_dir, f'best_model_fold{fold}.pth')
    epoch_csv = os.path.join(fold_dir, 'epoch_metrics.csv')
    if not (os.path.exists(ckpt) and os.path.exists(epoch_csv)):
        return False
    with open(epoch_csv, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return len(rows) > 0


def write_cv_results(runs_root: str, num_folds: int, seed: int, data_root: str, trained_folds: List[int]):
    case_ids = collect_case_ids(data_root)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    split_sizes = {fold: (len(tr), len(val)) for fold, (tr, val) in enumerate(kf.split(case_ids))}

    rows = []
    for fold in trained_folds:
        fold_dir = os.path.join(runs_root, f'fold_{fold}')
        ckpt = os.path.join(fold_dir, f'best_model_fold{fold}.pth')
        if not is_fold_completed(runs_root, fold):
            continue
        best_epoch, best_dice = read_best_from_fold_epoch_csv(fold_dir)
        ntr, nval = split_sizes[fold]
        rows.append({
            'fold': fold,
            'best_epoch': best_epoch,
            'best_dice_fg': best_dice,
            'num_train': ntr,
            'num_val': nval,
            'checkpoint': ckpt,
        })

    if not rows:
        raise RuntimeError("No completed folds found to summarize in cv_results.csv.")

    cv_path = os.path.join(runs_root, 'cv_results.csv')
    with open(cv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['fold', 'best_epoch', 'best_dice_fg', 'num_train', 'num_val', 'checkpoint'])
        w.writeheader()
        for r in sorted(rows, key=lambda x: x['fold']):
            rec = dict(r)
            rec['best_dice_fg'] = f"{rec['best_dice_fg']:.6f}"
            w.writerow(rec)
    return cv_path


def pick_best_fold(cv_path: str) -> int:
    with open(cv_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return int(max(rows, key=lambda r: float(r['best_dice_fg']))['fold'])


def main():
    args = parse_args()

    runs_root = os.path.join(args.save_root, 'TrainResults')
    pred_save_dir = os.path.join(args.save_root, 'TestResults')
    os.makedirs(runs_root, exist_ok=True)
    os.makedirs(pred_save_dir, exist_ok=True)

    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = args.cuda

    this_dir = os.path.dirname(os.path.abspath(__file__))
    train_py = os.path.join(this_dir, 'train.py')
    test_py = os.path.join(this_dir, 'test.py')

    if args.only_fold >= 0:
        folds_to_train = [args.only_fold]
    else:
        folds_to_train = list(range(args.num_folds))

    for fold in folds_to_train:
        if is_fold_completed(runs_root, fold):
            print(f"[Skip] fold_{fold} already completed: {os.path.join(runs_root, f'fold_{fold}')}", flush=True)
            continue

        cmd = [
            sys.executable, train_py,
            '--data_root', args.data_root,
            '--runs_root', runs_root,
            '--num_folds', str(args.num_folds),
            '--seed', str(args.seed),
            '--max_epochs', str(args.max_epochs),
            '--early_stop_patience', str(args.early_stop_patience),
            '--batch_size', str(args.train_batch_size),
            '--num_workers', str(args.num_workers),
            '--lr', str(args.lr),
            '--eta_min', str(args.eta_min),
            '--weight_decay', str(args.weight_decay),
            '--grad_clip', str(args.grad_clip),
            '--roi_x', str(args.roi_x),
            '--roi_y', str(args.roi_y),
            '--roi_z', str(args.roi_z),
            '--a_min', str(args.a_min),
            '--a_max', str(args.a_max),
            '--pixdim_x', str(args.pixdim_x),
            '--pixdim_y', str(args.pixdim_y),
            '--pixdim_z', str(args.pixdim_z),
            '--axcodes', args.axcodes,
            '--only_fold', str(fold),
        ]
        if args.resume:
            cmd.append('--resume')
        if args.strong_aug:
            cmd.append('--strong_aug')
        run_cmd(cmd, env)

    cv_path = write_cv_results(runs_root, args.num_folds, args.seed, args.data_root, folds_to_train)

    test_fold: Optional[int]
    if args.only_fold >= 0:
        test_fold = args.only_fold
    else:
        test_fold = pick_best_fold(cv_path)

    print(f"\n[Test] fold={test_fold}", flush=True)

    test_cmd = [
        sys.executable, test_py,
        '--data_root', args.data_root,
        '--runs_root', runs_root,
        '--save_dir', pred_save_dir,
        '--batch_size', str(args.test_batch_size),
        '--workers', str(args.num_workers),
        '--roi_x', str(args.roi_x),
        '--roi_y', str(args.roi_y),
        '--roi_z', str(args.roi_z),
        '--threshold', str(args.threshold),
        '--num_folds', str(args.num_folds),
        '--fold', str(test_fold),
        '--a_min', str(args.a_min),
        '--a_max', str(args.a_max),
        '--pixdim_x', str(args.pixdim_x),
        '--pixdim_y', str(args.pixdim_y),
        '--pixdim_z', str(args.pixdim_z),
        '--axcodes', args.axcodes,
    ]
    if args.save_prob:
        test_cmd.append('--save_prob')
    run_cmd(test_cmd, env)

    print("\nAll done.", flush=True)
    print(f"TrainResults: {runs_root}", flush=True)
    print(f"TestResults: {pred_save_dir}", flush=True)


if __name__ == '__main__':
    main()
