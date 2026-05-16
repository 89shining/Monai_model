import argparse
import glob
import os

import numpy as np
import SimpleITK as sitk
import torch

from nets.deeplabv3_plus import DeepLab


def parse_args():
    p = argparse.ArgumentParser(description='DeepLabv3+ test on 3D NIfTI by 2D slicing.')
    p.add_argument('--data_root', type=str, default=r'/home/wusi/Project_crop/Data/Rectal_146/RectalCTV_All')
    p.add_argument('--model_path', type=str, required=True)
    p.add_argument('--save_dir', type=str, default='/home/wusi/Project_crop/Data/Rectal_146/Networks/DeepLabV3Plus/RectalCTV_All/TestResults')
    p.add_argument('--backbone', type=str, default='xception', choices=['xception', 'mobilenet'])
    p.add_argument('--downsample_factor', type=int, default=16)
    p.add_argument('--input_h', type=int, default=512)
    p.add_argument('--input_w', type=int, default=512)
    return p.parse_args()


def strip_nii_ext(filename: str) -> str:
    if filename.endswith('.nii.gz'):
        return filename[:-7]
    if filename.endswith('.nii'):
        return filename[:-4]
    return os.path.splitext(filename)[0]


def image_case_id(path: str) -> str:
    name = strip_nii_ext(os.path.basename(path))
    return name[:-5] if name.endswith('_0000') else name


def window_norm(slice2d: np.ndarray):
    x = np.clip(slice2d, -160.0, 240.0)
    x = (x + 160.0) / 400.0
    return x.astype(np.float32)


def resize2d(img: np.ndarray, h: int, w: int, mode: str):
    import cv2
    interp = cv2.INTER_LINEAR if mode == 'linear' else cv2.INTER_NEAREST
    return cv2.resize(img, (w, h), interpolation=interp)


def infer_num_classes_from_state(state_dict) -> int:
    w = state_dict.get('cls_conv.weight', None)
    if w is None:
        return 1
    return int(w.shape[0])


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    state = torch.load(args.model_path, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']

    num_classes = infer_num_classes_from_state(state)
    print(f'[Test] inferred num_classes from checkpoint: {num_classes}', flush=True)

    model = DeepLab(num_classes=num_classes, backbone=args.backbone, downsample_factor=args.downsample_factor, pretrained=False)
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    images = sorted(glob.glob(os.path.join(args.data_root, 'imagesTs', '*_0000.nii.gz')) + glob.glob(os.path.join(args.data_root, 'imagesTs', '*_0000.nii')))
    if not images:
        raise FileNotFoundError('No imagesTs NIfTI found.')

    for ip in images:
        itk = sitk.ReadImage(ip)
        vol = sitk.GetArrayFromImage(itk).astype(np.float32)
        pred_slices = []

        for z in range(vol.shape[0]):
            sl = window_norm(vol[z])
            raw_h, raw_w = sl.shape
            sl = resize2d(sl, args.input_h, args.input_w, 'linear')

            x = torch.from_numpy(sl[None, None, ...]).float().to(device)
            x = x.repeat(1, 3, 1, 1)
            logits = model(x)

            if num_classes == 1:
                pred = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()[0, 0]
            else:
                pred = torch.argmax(logits, dim=1).cpu().numpy()[0].astype(np.float32)

            pred = resize2d(pred, raw_h, raw_w, 'nearest')
            pred_slices.append(pred)

        pred_vol = np.stack(pred_slices, axis=0).astype(np.uint8)
        out_itk = sitk.GetImageFromArray(pred_vol)
        out_itk.CopyInformation(itk)

        cid = image_case_id(ip)
        save_path = os.path.join(args.save_dir, f'{cid}.nii.gz')
        sitk.WriteImage(out_itk, save_path)
        print(f'Saved: {save_path}', flush=True)


if __name__ == '__main__':
    main()
