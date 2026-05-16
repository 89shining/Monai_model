import csv
import glob
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from monai.transforms import Compose, EnsureTyped, RandAffined, RandFlipd, RandGaussianNoised, RandScaleIntensityd, RandShiftIntensityd
from torch.utils.data import DataLoader, Dataset

from nets.deeplabv3_plus import DeepLab


@dataclass
class Config:
    data_root: str = "/home/wusi/Project_crop/Data/Rectal_146/RectalCTV_All"
    save_root: str = "/home/wusi/Project_crop/Data/Rectal_146/Networks/DeepLabV3Plus/RectalCTV_All/TrainResults"
    fold: int = 0
    num_folds: int = 5
    seed: int = 42

    epochs: int = 100
    early_stopping_patience: int = 15
    batch_size: int = 8
    num_workers: int = 0

    lr: float = 1e-4
    eta_min: float = 1e-6
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    pos_weight: float = 50.0
    eval_threshold: float = 0.5

    a_min: float = -160.0
    a_max: float = 240.0
    input_h: int = 512
    input_w: int = 512

    backbone: str = "xception"
    downsample_factor: int = 16
    num_classes: int = 1
    pretrained: bool = False
    model_path: str = "model_data/xception_pytorch_imagenet.pth"
    resume: bool = True
    debug: bool = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def strip_nii_ext(filename: str) -> str:
    if filename.endswith('.nii.gz'):
        return filename[:-7]
    if filename.endswith('.nii'):
        return filename[:-4]
    return os.path.splitext(filename)[0]


def case_id_from_image(path: str) -> str:
    n = strip_nii_ext(os.path.basename(path))
    return n[:-5] if n.endswith('_0000') else n


def load_case_volumes(data_root: str, split: str = "imagesTr") -> Dict[str, str]:
    image_dir = os.path.join(data_root, split)
    images = sorted(glob.glob(os.path.join(image_dir, '*_0000.nii.gz')) + glob.glob(os.path.join(image_dir, '*_0000.nii')))
    return {case_id_from_image(p): p for p in images}


def load_label_volumes(data_root: str, split: str = "labelsTr") -> Dict[str, str]:
    label_dir = os.path.join(data_root, split)
    labels = sorted(glob.glob(os.path.join(label_dir, '*.nii.gz')) + glob.glob(os.path.join(label_dir, '*.nii')))
    return {strip_nii_ext(os.path.basename(p)): p for p in labels}


def make_patient_splits(case_ids: List[str], num_folds: int, seed: int):
    idx = list(range(len(case_ids)))
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    fold_sizes = [len(idx) // num_folds] * num_folds
    for i in range(len(idx) % num_folds):
        fold_sizes[i] += 1
    folds = []
    s = 0
    for fs in fold_sizes:
        folds.append(idx[s:s+fs])
        s += fs
    splits = []
    for f in range(num_folds):
        val_idx = folds[f]
        train_idx = [j for i, fold in enumerate(folds) if i != f for j in fold]
        splits.append(([case_ids[i] for i in train_idx], [case_ids[i] for i in val_idx]))
    return splits


def window_norm(slice2d: np.ndarray, a_min: float, a_max: float) -> np.ndarray:
    x = np.clip(slice2d, a_min, a_max)
    x = (x - a_min) / max(a_max - a_min, 1e-6)
    return x.astype(np.float32)


def resize2d(img: np.ndarray, h: int, w: int, mode: str) -> np.ndarray:
    import cv2
    interp = cv2.INTER_LINEAR if mode == "linear" else cv2.INTER_NEAREST
    return cv2.resize(img, (w, h), interpolation=interp)


class SliceTrainDataset(Dataset):
    def __init__(self, case_ids: List[str], image_map: Dict[str, str], label_map: Dict[str, str], cfg: Config):
        self.cfg = cfg
        self.records = []
        self.cache = {}
        for cid in case_ids:
            if cid not in image_map or cid not in label_map:
                continue
            vol = sitk.GetArrayFromImage(sitk.ReadImage(image_map[cid])).astype(np.float32)
            lab = sitk.GetArrayFromImage(sitk.ReadImage(label_map[cid])).astype(np.float32)
            if vol.shape != lab.shape:
                continue
            self.cache[cid] = (vol, (lab > 0).astype(np.float32))
            for z in range(vol.shape[0]):
                self.records.append((cid, z))

        self.aug = Compose([
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
            RandAffined(
                keys=["image", "label"], prob=0.2,
                rotate_range=(0.1,), scale_range=(0.1, 0.1),
                mode=("bilinear", "nearest"), padding_mode="border"
            ),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
            RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        cid, z = self.records[idx]
        vol, lab = self.cache[cid]
        img = window_norm(vol[z], self.cfg.a_min, self.cfg.a_max)
        msk = lab[z]
        img = resize2d(img, self.cfg.input_h, self.cfg.input_w, "linear")
        msk = resize2d(msk, self.cfg.input_h, self.cfg.input_w, "nearest")

        d = {"image": img[None, ...], "label": msk[None, ...]}
        d = self.aug(d)
        image = d["image"].repeat(3, 1, 1)
        label = d["label"]
        return image, label


class SimpleBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor):
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight, reduction="mean"
        )


def patient_3d_dice(
    model: nn.Module,
    case_ids: List[str],
    image_map: Dict[str, str],
    label_map: Dict[str, str],
    cfg: Config,
    device: torch.device,
) -> Tuple[float, float, float, float, float, float]:
    model.eval()
    hard_dices = []
    best_hard_dices = []
    best_thresholds = []
    soft_dices = []
    pred_fg_ratios = []
    gt_fg_ratios = []
    thr_grid = [0.10, 0.20, 0.30, 0.40, 0.50]
    with torch.no_grad():
        for cid in case_ids:
            if cid not in image_map or cid not in label_map:
                continue
            vol = sitk.GetArrayFromImage(sitk.ReadImage(image_map[cid])).astype(np.float32)
            lab = (sitk.GetArrayFromImage(sitk.ReadImage(label_map[cid])) > 0).astype(np.float32)
            probs_vol = []
            for z in range(vol.shape[0]):
                img = window_norm(vol[z], cfg.a_min, cfg.a_max)
                img = resize2d(img, cfg.input_h, cfg.input_w, "linear")
                x = torch.from_numpy(img[None, None, ...]).float().to(device)
                x = x.repeat(1, 3, 1, 1)
                logits = model(x)
                probs = torch.sigmoid(logits)
                probs_vol.append(probs.cpu().numpy()[0, 0])
            probs_vol = np.stack(probs_vol, axis=0)
            probs_vol = np.stack([resize2d(s, lab.shape[1], lab.shape[2], "linear") for s in probs_vol], axis=0)

            pred_vol = (probs_vol > cfg.eval_threshold).astype(np.float32)
            inter = float((pred_vol * lab).sum())
            den = float(pred_vol.sum() + lab.sum())
            hard_dice = (2.0 * inter + 1e-6) / (den + 1e-6)

            case_best_dice = -1.0
            case_best_thr = cfg.eval_threshold
            for thr in thr_grid:
                pred_thr = (probs_vol > thr).astype(np.float32)
                inter_thr = float((pred_thr * lab).sum())
                den_thr = float(pred_thr.sum() + lab.sum())
                dice_thr = (2.0 * inter_thr + 1e-6) / (den_thr + 1e-6)
                if dice_thr > case_best_dice:
                    case_best_dice = dice_thr
                    case_best_thr = thr

            soft_inter = float((probs_vol * lab).sum())
            soft_den = float(probs_vol.sum() + lab.sum())
            soft_dice = (2.0 * soft_inter + 1e-6) / (soft_den + 1e-6)
            hard_dices.append(hard_dice)
            best_hard_dices.append(case_best_dice)
            best_thresholds.append(case_best_thr)
            soft_dices.append(soft_dice)
            pred_fg_ratios.append(float(pred_vol.mean()))
            gt_fg_ratios.append(float(lab.mean()))
    if not hard_dices:
        return 0.0, 0.0, 0.0, 0.0, 0.0, cfg.eval_threshold
    return (
        float(np.mean(hard_dices)),
        float(np.mean(best_hard_dices)),
        float(np.mean(soft_dices)),
        float(np.mean(pred_fg_ratios)),
        float(np.mean(gt_fg_ratios)),
        float(np.mean(best_thresholds)),
    )


def build_model(cfg: Config, device: torch.device) -> nn.Module:
    model = DeepLab(num_classes=cfg.num_classes, backbone=cfg.backbone, downsample_factor=cfg.downsample_factor, pretrained=cfg.pretrained)
    if cfg.model_path and os.path.exists(cfg.model_path):
        state = torch.load(cfg.model_path, map_location=device)
        model_dict = model.state_dict()
        load_dict = {k: v for k, v in state.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(load_dict)
        model.load_state_dict(model_dict)
    return model.to(device)


def main():
    cfg = Config()
    cfg.fold = int(os.environ.get("DEEPLAB_FOLD", "0"))
    cfg.data_root = os.environ.get("DEEPLAB_DATA_ROOT", cfg.data_root)
    cfg.save_root = os.environ.get("DEEPLAB_SAVE_DIR", os.path.join(cfg.save_root, f"fold_{cfg.fold}"))
    cfg.resume = os.environ.get("DEEPLAB_RESUME", "1").strip().lower() in {"1", "true", "yes", "y"}
    cfg.debug = os.environ.get("DEEPLAB_DEBUG", "1").strip().lower() in {"1", "true", "yes", "y"}
    os.makedirs(cfg.save_root, exist_ok=True)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_map = load_case_volumes(cfg.data_root, "imagesTr")
    label_map = load_label_volumes(cfg.data_root, "labelsTr")
    case_ids = sorted(set(image_map.keys()).intersection(set(label_map.keys())))
    splits = make_patient_splits(case_ids, cfg.num_folds, cfg.seed)
    train_ids, val_ids = splits[cfg.fold]
    print(
        f"[Split] fold={cfg.fold} | total_cases={len(case_ids)} | train_cases={len(train_ids)} | val_cases={len(val_ids)}",
        flush=True,
    )

    train_ds = SliceTrainDataset(train_ids, image_map, label_map, cfg)
    print(f"[Data] train_slices={len(train_ds)}", flush=True)
    if len(val_ids) == 0:
        raise RuntimeError(f"Validation set is empty for fold {cfg.fold}. Check num_folds or data_root.")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=(device.type=="cuda"), drop_last=True)

    model = build_model(cfg, device)
    loss_fn = SimpleBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min)

    best_ckpt_path = os.path.join(cfg.save_root, f"best_model_fold{cfg.fold}.pth")
    latest_ckpt_path = os.path.join(cfg.save_root, f"latest_model_fold{cfg.fold}.pth")
    csv_path = os.path.join(cfg.save_root, "epoch_metrics.csv")

    best_dice = -1.0
    best_epoch = -1
    patience_count = 0
    start_epoch = 1
    log_rows = []

    if cfg.resume and os.path.exists(latest_ckpt_path):
        st = torch.load(latest_ckpt_path, map_location=device)
        if isinstance(st, dict) and 'model' in st:
            model.load_state_dict(st['model'])
            if 'optimizer' in st:
                optimizer.load_state_dict(st['optimizer'])
            if 'scheduler' in st:
                scheduler.load_state_dict(st['scheduler'])
            best_dice = float(st.get('best_dice_fg', -1.0))
            best_epoch = int(st.get('best_epoch', -1))
            patience_count = int(st.get('patience_count', 0))
            start_epoch = int(st.get('epoch', 0)) + 1
        # resume existing csv rows if present
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                log_rows = list(csv.DictReader(f))
        print(f"[Resume] fold={cfg.fold} from epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        for x, y in train_loader:
            steps += 1
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            pos_weight_t = torch.tensor([cfg.pos_weight], device=device, dtype=logits.dtype)
            loss = loss_fn(logits, y, pos_weight_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / max(steps, 1)
        val_dice, val_dice_best_thr, val_soft_dice, pred_fg_ratio, gt_fg_ratio, best_thr_mean = patient_3d_dice(
            model, val_ids, image_map, label_map, cfg, device
        )
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()

        is_best = val_dice_best_thr > best_dice
        if is_best:
            best_dice = val_dice_best_thr
            best_epoch = epoch
            patience_count = 0
            ckpt = {
                "model": model.state_dict(),
                "epoch": epoch,
                "best_dice_fg": best_dice,
                "best_epoch": best_epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": cfg.__dict__,
            }
            torch.save(ckpt, best_ckpt_path)
        else:
            patience_count += 1

        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "best_dice_fg": best_dice,
                "best_epoch": best_epoch,
                "patience_count": patience_count,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": cfg.__dict__,
            },
            latest_ckpt_path,
        )

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_dice_fg": f"{val_dice:.6f}",
                "val_dice_best_thr": f"{val_dice_best_thr:.6f}",
                "best_thr_mean": f"{best_thr_mean:.2f}",
                "lr": f"{lr_now:.8f}",
                "is_best": int(is_best),
            }
        )
        print(
            f"Fold {cfg.fold} | Epoch {epoch}/{cfg.epochs} | TrainLoss {train_loss:.4f} | "
            f"ValDice3D@{cfg.eval_threshold:.2f} {val_dice:.4f} | ValDice3D@BestThr {val_dice_best_thr:.4f} | LR {lr_now:.2e}",
            flush=True,
        )
        if cfg.debug:
            print(
                f"[Debug] Epoch {epoch} | SoftDice3D {val_soft_dice:.4f} | PredFG {pred_fg_ratio:.5f} | "
                f"GTFG {gt_fg_ratio:.5f} | PosW {cfg.pos_weight:.2f} | BestThrMean {best_thr_mean:.2f}",
                flush=True,
            )

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["epoch", "train_loss", "val_dice_fg", "val_dice_best_thr", "best_thr_mean", "lr", "is_best"],
            )
            w.writeheader()
            for r in log_rows:
                w.writerow(r)

        with open(os.path.join(cfg.save_root, "best_dice.txt"), "w", encoding="utf-8") as f:
            f.write(f"{best_dice:.6f}")

        if patience_count >= cfg.early_stopping_patience:
            print(f"Early stopping at epoch {epoch}, best epoch {best_epoch}, best dice {best_dice:.4f}", flush=True)
            break


if __name__ == "__main__":
    main()
