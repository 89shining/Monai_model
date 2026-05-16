import os
import cv2
import numpy as np
import torch
import SimpleITK as sitk
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


def _strip_nii_ext(name: str) -> str:
    if name.endswith('.nii.gz'):
        return name[:-7]
    if name.endswith('.nii'):
        return name[:-4]
    return os.path.splitext(name)[0]


def _case_id_from_image_name(name: str) -> str:
    base = _strip_nii_ext(name)
    return base[:-5] if base.endswith('_0000') else base


class DeeplabDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(DeeplabDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

        self.image_dir = os.path.join(dataset_path, 'imagesTr')
        self.label_dir = os.path.join(dataset_path, 'labelsTr')

        self.case_ids = [line.strip().split()[0] for line in annotation_lines if line.strip()]
        self._cache = {}
        self._slices = []

        for cid in self.case_ids:
            ip = os.path.join(self.image_dir, f'{cid}_0000.nii.gz')
            if not os.path.exists(ip):
                ip = os.path.join(self.image_dir, f'{cid}_0000.nii')
            lp = os.path.join(self.label_dir, f'{cid}.nii.gz')
            if not os.path.exists(lp):
                lp = os.path.join(self.label_dir, f'{cid}.nii')
            if not (os.path.exists(ip) and os.path.exists(lp)):
                continue

            img = sitk.GetArrayFromImage(sitk.ReadImage(ip)).astype(np.float32)
            lbl = sitk.GetArrayFromImage(sitk.ReadImage(lp)).astype(np.float32)
            if img.shape != lbl.shape:
                continue

            self._cache[cid] = (img, (lbl > 0).astype(np.uint8))
            for z in range(img.shape[0]):
                self._slices.append((cid, z))

        self.length = len(self._slices)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        cid, z = self._slices[index]
        vol, lab = self._cache[cid]
        image_slice = vol[z]
        label_slice = lab[z]

        # Fixed windowing: WW=400, WL=40 => [-160, 240].
        lo, hi = -160.0, 240.0
        image_slice = np.clip((image_slice - lo) / (hi - lo), 0.0, 1.0)
        image_slice = (image_slice * 255.0).astype(np.uint8)

        jpg = Image.fromarray(image_slice).convert('RGB')
        png = Image.fromarray(label_slice.astype(np.uint8), mode='L')

        jpg, png = self.get_random_data(jpg, png, self.input_shape, random=self.train)

        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])
        png = np.array(png, dtype=np.uint8)
        png[png >= self.num_classes] = self.num_classes
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        iw, ih = image.size
        h, w = input_shape

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            return new_image, new_label

        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        image_data = np.array(image, np.uint8)

        blur = self.rand() < 0.25
        if blur:
            image_data = cv2.GaussianBlur(image_data, (5, 5), 0)

        rotate = self.rand() < 0.25
        if rotate:
            center = (w // 2, h // 2)
            rotation = np.random.randint(-10, 11)
            M = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data = cv2.warpAffine(image_data, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(128, 128, 128))
            label = cv2.warpAffine(np.array(label, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))

        # No HSV color jitter for medical grayscale-derived images.
        return image_data, label


def deeplab_dataset_collate(batch):
    images, pngs, seg_labels = [], [], []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels
