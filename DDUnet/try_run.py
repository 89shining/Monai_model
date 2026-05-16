import argparse
import os
import subprocess
import sys

DATA_ROOT = '/home/wusi/Project_crop/Data/Eso_83/EsoCTV_All'
SAVE_ROOT = '/home/wusi/Project_crop/Data/Eso_83/Networks/DDUNet/EsoCTV_All'
TRAIN_RESULTS = os.path.join(SAVE_ROOT, 'TrainResults')
TEST_RESULTS = os.path.join(SAVE_ROOT, 'TestResults')

CUDA_VISIBLE_DEVICES = '0'


def parse_args():
    p = argparse.ArgumentParser(description='Run one fold training + test for DDUNet quick check.')
    p.add_argument('--fold', type=int, default=0, help='Fold index to run (0-4).')
    p.add_argument('--data_root', type=str, default=DATA_ROOT)
    p.add_argument('--train_results', type=str, default=TRAIN_RESULTS)
    p.add_argument('--test_results', type=str, default=TEST_RESULTS)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--input_h', type=int, default=512)
    p.add_argument('--input_w', type=int, default=512)
    p.add_argument('--a_min', type=float, default=-160.0)
    p.add_argument('--a_max', type=float, default=240.0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.train_results, exist_ok=True)
    os.makedirs(args.test_results, exist_ok=True)

    train_py = os.path.join(os.path.dirname(__file__), 'train.py')
    test_py = os.path.join(os.path.dirname(__file__), 'test.py')

    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

    print(f'[TryRun-DDUNet] Train fold={args.fold}', flush=True)
    train_cmd = [
        sys.executable, train_py,
        '--data_root', args.data_root,
        '--runs_root', args.train_results,
        '--num_folds', '5',
        '--only_fold', str(args.fold),
        '--resume',
        '--batch_size', str(args.batch_size),
        '--num_workers', str(args.num_workers),
        '--input_h', str(args.input_h),
        '--input_w', str(args.input_w),
        '--a_min', str(args.a_min),
        '--a_max', str(args.a_max),
    ]
    subprocess.run(train_cmd, check=True, env=env)

    fold_dir = os.path.join(args.train_results, f'fold_{args.fold}')
    best_ckpt = os.path.join(fold_dir, f'best_model_fold{args.fold}.pth')
    latest_ckpt = os.path.join(fold_dir, f'latest_model_fold{args.fold}.pth')

    if os.path.exists(best_ckpt):
        ckpt = best_ckpt
    elif os.path.exists(latest_ckpt):
        ckpt = latest_ckpt
    else:
        raise FileNotFoundError(f'No checkpoint found in {fold_dir}')

    save_dir = os.path.join(args.test_results, f'try_fold_{args.fold}')
    os.makedirs(save_dir, exist_ok=True)

    print(f'[TryRun-DDUNet] Test fold={args.fold} using {ckpt}', flush=True)
    test_cmd = [
        sys.executable, test_py,
        '--data_root', args.data_root,
        '--runs_root', args.train_results,
        '--save_dir', save_dir,
        '--batch_size', str(args.batch_size),
        '--workers', str(args.num_workers),
        '--input_h', str(args.input_h),
        '--input_w', str(args.input_w),
        '--a_min', str(args.a_min),
        '--a_max', str(args.a_max),
        '--fold', str(args.fold),
    ]
    subprocess.run(test_cmd, check=True, env=env)

    print(f'[TryRun-DDUNet] Done. Predictions saved to: {save_dir}', flush=True)


if __name__ == '__main__':
    main()
