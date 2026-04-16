import os
import random
from glob import glob
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityd,
)
from monai.utils import set_determinism

from networks import DeepLab


def strip_nii_gz(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return os.path.splitext(filename)[0]


def case_key_from_image_path(image_path: str) -> str:
    name = strip_nii_gz(os.path.basename(image_path))
    return name.replace("_0000", "")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


def collect_files(images_dir: str, labels_dir: str) -> List[Dict[str, str]]:
    image_paths = sorted(glob(os.path.join(images_dir, "*.nii*")))
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {images_dir}")

    files: List[Dict[str, str]] = []
    missing_labels = []
    for image_path in image_paths:
        case_key = case_key_from_image_path(image_path)
        label_path = os.path.join(labels_dir, f"{case_key}.nii.gz")
        if not os.path.exists(label_path):
            alt_label_path = os.path.join(labels_dir, f"{case_key}.nii")
            if os.path.exists(alt_label_path):
                label_path = alt_label_path
            else:
                missing_labels.append((image_path, label_path))
                continue
        files.append({"image": image_path, "label": label_path, "case_name": case_key})

    if missing_labels:
        sample = "\n".join([f"image={i}, expected_label={l}" for i, l in missing_labels[:5]])
        raise FileNotFoundError(
            "Some labels are missing after applying the '_0000' matching rule. "
            f"Examples:\n{sample}"
        )
    return files


def build_kfold_indices(num_samples: int, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if num_samples < n_splits:
        raise ValueError(f"num_samples ({num_samples}) must be >= n_splits ({n_splits}).")

    rng = np.random.RandomState(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_splits)

    split_pairs = []
    for fold_idx in range(n_splits):
        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[i] for i in range(n_splits) if i != fold_idx])
        split_pairs.append((train_idx, val_idx))
    return split_pairs


def get_train_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"]),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_eval_transforms() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"]),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def build_model(
    device: torch.device,
    backbone: str = "xception",
    num_classes: int = 1,
    downsample_factor: int = 16,
    pretrained_backbone: bool = False,
) -> torch.nn.Module:
    model = DeepLab(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained_backbone,
        downsample_factor=downsample_factor,
    )
    return model.to(device)


def resize_2d_tensor(tensor: torch.Tensor, size_hw: Sequence[int], mode: str) -> torch.Tensor:
    if tensor.shape[-2] == size_hw[0] and tensor.shape[-1] == size_hw[1]:
        return tensor
    if mode in ("bilinear", "bicubic", "trilinear"):
        return F.interpolate(tensor, size=size_hw, mode=mode, align_corners=False)
    return F.interpolate(tensor, size=size_hw, mode=mode)


def dice_from_probs(probs: torch.Tensor, labels: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    preds = (probs >= 0.5).float()
    labels = (labels > 0.5).float()
    dims = tuple(range(2, preds.ndim))
    intersection = torch.sum(preds * labels, dim=dims)
    denominator = torch.sum(preds, dim=dims) + torch.sum(labels, dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    both_empty = denominator == 0
    dice = torch.where(both_empty, torch.ones_like(dice), dice)
    return dice.squeeze(1)


def dice_bce_loss(logits: torch.Tensor, labels: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    labels = (labels > 0.5).float()
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    probs = torch.sigmoid(logits)
    inter = torch.sum(probs * labels, dim=(1, 2, 3))
    denom = torch.sum(probs, dim=(1, 2, 3)) + torch.sum(labels, dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    dice_loss = 1.0 - dice.mean()
    return bce + dice_loss


def volume_to_2d_batches(
    volume: torch.Tensor,
    label: torch.Tensor,
    slice_batch_size: int,
    input_shape: Sequence[int],
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    # volume/label: [1, D, H, W]
    if volume.ndim != 4 or label.ndim != 4:
        raise ValueError("volume and label must have shape [1, D, H, W].")

    slices_img = volume[0]  # [D, H, W]
    slices_lbl = label[0]   # [D, H, W]
    depth = slices_img.shape[0]

    for start in range(0, depth, slice_batch_size):
        end = min(start + slice_batch_size, depth)
        img_2d = slices_img[start:end].unsqueeze(1)  # [N, 1, H, W]
        lbl_2d = slices_lbl[start:end].unsqueeze(1)  # [N, 1, H, W]

        img_2d = resize_2d_tensor(img_2d, input_shape, mode="bilinear")
        lbl_2d = resize_2d_tensor(lbl_2d, input_shape, mode="nearest")
        yield img_2d, lbl_2d


def infer_volume_probs(
    model: torch.nn.Module,
    volume: torch.Tensor,
    device: torch.device,
    input_shape: Sequence[int],
    slice_batch_size: int,
) -> torch.Tensor:
    # volume: [1, 1, D, H, W]
    if volume.ndim != 5 or volume.shape[0] != 1 or volume.shape[1] != 1:
        raise ValueError("volume must have shape [1, 1, D, H, W].")

    _, _, depth, h, w = volume.shape
    all_probs: List[torch.Tensor] = []
    vol = volume[0]  # [1, D, H, W]

    for img_2d, _ in volume_to_2d_batches(vol, vol, slice_batch_size=slice_batch_size, input_shape=input_shape):
        img_2d = img_2d.to(device)
        img_2d = img_2d.repeat(1, 3, 1, 1)
        logits = model(img_2d)
        logits = resize_2d_tensor(logits, (h, w), mode="bilinear")
        probs = torch.sigmoid(logits).detach().cpu()  # [N, 1, H, W]
        all_probs.append(probs)

    probs_2d = torch.cat(all_probs, dim=0)[:depth]          # [D, 1, H, W]
    probs_3d = probs_2d.permute(1, 0, 2, 3).unsqueeze(0)    # [1, 1, D, H, W]
    return probs_3d
