import argparse
import os
import subprocess
import sys

DATA_ROOT = '/home/wusi/Project_crop/Data/Rectal_146/RectalCTV_All'
SAVE_ROOT = '/home/wusi/Project_crop/Data/Rectal_146/Networks/DeepLabV3Plus/RectalCTV_All'
TRAIN_RESULTS = os.path.join(SAVE_ROOT, 'TrainResults')
TEST_RESULTS = os.path.join(SAVE_ROOT, 'TestResults')

CUDA_VISIBLE_DEVICES = '5'


def parse_args():
    p = argparse.ArgumentParser(description='Run one fold training + test for quick check.')
    p.add_argument('--fold', type=int, default=0, help='Fold index to run (0-4).')
    p.add_argument('--data_root', type=str, default=DATA_ROOT)
    p.add_argument('--train_results', type=str, default=TRAIN_RESULTS)
    p.add_argument('--test_results', type=str, default=TEST_RESULTS)
    p.add_argument('--resume', action='store_true', help='Resume from latest checkpoint if exists.')
    p.add_argument('--no_debug', action='store_true', help='Disable training debug logs.')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.train_results, exist_ok=True)
    os.makedirs(args.test_results, exist_ok=True)

    train_py = os.path.join(os.path.dirname(__file__), 'train.py')
    test_py = os.path.join(os.path.dirname(__file__), 'test.py')

    fold_dir = os.path.join(args.train_results, f'fold_{args.fold}')
    os.makedirs(fold_dir, exist_ok=True)

    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    env['DEEPLAB_DATA_ROOT'] = args.data_root
    env['DEEPLAB_SAVE_DIR'] = fold_dir
    env['DEEPLAB_FOLD'] = str(args.fold)
    env['DEEPLAB_RESUME'] = '1' if args.resume else '0'
    env['DEEPLAB_DEBUG'] = '0' if args.no_debug else '1'

    print(f'[TryRun] Train fold={args.fold}', flush=True)
    subprocess.run([sys.executable, train_py], check=True, env=env)

    # Prefer best checkpoint, fallback to latest checkpoint.
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

    print(f'[TryRun] Test fold={args.fold} using {ckpt}', flush=True)
    subprocess.run(
        [
            sys.executable,
            test_py,
            '--data_root',
            args.data_root,
            '--model_path',
            ckpt,
            '--save_dir',
            save_dir,
        ],
        check=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES, PYTHONUNBUFFERED='1'),
    )

    print(f'[TryRun] Done. Predictions saved to: {save_dir}', flush=True)


if __name__ == '__main__':
    main()
