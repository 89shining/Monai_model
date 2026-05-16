import csv
import glob
import os
import subprocess
import sys

from sklearn.model_selection import KFold

DATA_ROOT = '/home/wusi/Project_crop/Data/Rectal_146/RectalCTV_All'
SAVE_ROOT = '/home/wusi/Project_crop/Data/Rectal_146/Networks/DeepLabV3Plus/RectalCTV_All'
TRAIN_RESULTS = os.path.join(SAVE_ROOT, 'TrainResults')
TEST_RESULTS = os.path.join(SAVE_ROOT, 'TestResults')

NUM_FOLDS = 5
SEED = 42
CUDA_VISIBLE_DEVICES = '5'


def strip_nii_ext(name: str):
    if name.endswith('.nii.gz'):
        return name[:-7]
    if name.endswith('.nii'):
        return name[:-4]
    return os.path.splitext(name)[0]


def case_id_from_image(path: str):
    n = strip_nii_ext(os.path.basename(path))
    return n[:-5] if n.endswith('_0000') else n


def collect_case_ids(data_root: str):
    images = sorted(glob.glob(os.path.join(data_root, 'imagesTr', '*_0000.nii.gz')) + glob.glob(os.path.join(data_root, 'imagesTr', '*_0000.nii')))
    return [case_id_from_image(p) for p in images]


def read_best_dice(fold_dir: str):
    p = os.path.join(fold_dir, 'best_dice.txt')
    if os.path.exists(p):
        return float(open(p, 'r', encoding='utf-8').read().strip())

    # fallback for legacy runs: parse epoch_metrics.csv val_dice_fg
    em = os.path.join(fold_dir, 'epoch_metrics.csv')
    if os.path.exists(em):
        import csv
        rows = list(csv.DictReader(open(em, 'r', encoding='utf-8')) )
        if rows and 'val_dice_fg' in rows[0]:
            return max(float(r['val_dice_fg']) for r in rows)
    return -1.0


def resolve_best_checkpoint(fold_dir: str, fold: int):
    cands = [
        os.path.join(fold_dir, f'best_model_fold{fold}.pth'),
        os.path.join(fold_dir, 'best_dice_weights.pth'),
        os.path.join(fold_dir, 'best_epoch_weights.pth'),
        os.path.join(fold_dir, f'last_model_fold{fold}.pth'),
        os.path.join(fold_dir, 'last_epoch_weights.pth'),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    return cands[0]


def main():
    os.makedirs(TRAIN_RESULTS, exist_ok=True)
    os.makedirs(TEST_RESULTS, exist_ok=True)

    case_ids = collect_case_ids(DATA_ROOT)
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    train_py = os.path.join(os.path.dirname(__file__), 'train.py')
    test_py = os.path.join(os.path.dirname(__file__), 'test.py')

    rows = []
    for fold, _ in enumerate(kf.split(case_ids)):
        fold_dir = os.path.join(TRAIN_RESULTS, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        env = dict(os.environ)
        env['PYTHONUNBUFFERED'] = '1'
        env['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
        env['DEEPLAB_DATA_ROOT'] = DATA_ROOT
        env['DEEPLAB_SAVE_DIR'] = fold_dir
        env['DEEPLAB_FOLD'] = str(fold)

        cmd = [sys.executable, train_py]
        print(f'[Fold {fold}] start training...')
        subprocess.run(cmd, check=True, env=env)

        best_dice = read_best_dice(fold_dir)
        rows.append({'fold': fold, 'best_epoch': '', 'best_dice_fg': f'{best_dice:.6f}', 'checkpoint': resolve_best_checkpoint(fold_dir, fold)})

    cv_csv = os.path.join(TRAIN_RESULTS, 'cv_results.csv')
    with open(cv_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['fold', 'best_epoch', 'best_dice_fg', 'checkpoint'])
        w.writeheader()
        for r in rows:
            # fill best_epoch from epoch_metrics
            em = os.path.join(TRAIN_RESULTS, f"fold_{r['fold']}", 'epoch_metrics.csv')
            if os.path.exists(em):
                em_rows = list(csv.DictReader(open(em, 'r', encoding='utf-8')))
                if em_rows:
                    best_row = max(em_rows, key=lambda x: float(x['val_dice_fg']))
                    r['best_epoch'] = best_row['epoch']
            w.writerow(r)

    best = max(rows, key=lambda r: float(r['best_dice_fg']))
    print(f"[Test] use best fold={best['fold']} dice={best['best_dice_fg']}")

    test_cmd = [
        sys.executable, test_py,
        '--data_root', DATA_ROOT,
        '--model_path', best['checkpoint'],
        '--save_dir', os.path.join(TEST_RESULTS, f"best_fold_{best['fold']}"),
    ]
    subprocess.run(test_cmd, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES, PYTHONUNBUFFERED='1'))


if __name__ == '__main__':
    main()
