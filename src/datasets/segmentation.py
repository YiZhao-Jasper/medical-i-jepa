import os

from logging import getLogger

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

logger = getLogger()


class LungSegmentationDataset(Dataset):
    """Montgomery / JSRT lung segmentation dataset."""

    def __init__(
        self,
        root_path,
        dataset='montgomery',
        split='train',
        img_size=224,
        train_ratio=0.8,
        seed=42,
    ):
        super().__init__()
        self.img_size = img_size
        self.split = split

        self.images = []
        self.masks = []

        if dataset == 'montgomery':
            self._load_montgomery(root_path)
        elif dataset == 'jsrt':
            self._load_jsrt(root_path)
        elif dataset == 'montgomery_jsrt':
            self._load_montgomery(root_path)
            self._load_jsrt(root_path)
        else:
            raise ValueError(f'Unknown segmentation dataset: {dataset}')

        rng = random.Random(seed)
        indices = list(range(len(self.images)))
        rng.shuffle(indices)
        n_train = int(len(indices) * train_ratio)

        if split == 'train':
            indices = indices[:n_train]
        else:
            indices = indices[n_train:]

        self.images = [self.images[i] for i in indices]
        self.masks = [self.masks[i] for i in indices]
        logger.info(f'LungSegmentation [{dataset}/{split}]: {len(self.images)} images')

    def _load_montgomery(self, root_path):
        img_dir = os.path.join(root_path, 'montgomery', 'CXR_png')
        mask_l_dir = os.path.join(root_path, 'montgomery', 'ManualMask', 'leftMask')
        mask_r_dir = os.path.join(root_path, 'montgomery', 'ManualMask', 'rightMask')

        if not os.path.isdir(img_dir):
            logger.warning(f'Montgomery image dir not found: {img_dir}')
            return

        for fname in sorted(os.listdir(img_dir)):
            if not fname.endswith('.png'):
                continue
            img_path = os.path.join(img_dir, fname)
            left_path = os.path.join(mask_l_dir, fname)
            right_path = os.path.join(mask_r_dir, fname)
            if os.path.exists(left_path) and os.path.exists(right_path):
                self.images.append(img_path)
                self.masks.append((left_path, right_path))

    def _load_jsrt(self, root_path):
        img_dir = os.path.join(root_path, 'jsrt', 'images')
        mask_dir = os.path.join(root_path, 'jsrt', 'masks')

        if not os.path.isdir(img_dir):
            logger.warning(f'JSRT image dir not found: {img_dir}')
            return

        for fname in sorted(os.listdir(img_dir)):
            base = os.path.splitext(fname)[0]
            img_path = os.path.join(img_dir, fname)

            mask_path = None
            for ext in ['.png', '.gif', '.bmp']:
                candidate = os.path.join(mask_dir, base + ext)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break

            if mask_path is not None:
                self.images.append(img_path)
                self.masks.append(mask_path)

    def _load_mask(self, mask_info):
        if isinstance(mask_info, tuple):
            left = Image.open(mask_info[0]).convert('L')
            right = Image.open(mask_info[1]).convert('L')
            import numpy as np
            mask = Image.fromarray(
                (np.array(left) > 127).astype(np.uint8) * 255
                | (np.array(right) > 127).astype(np.uint8) * 255,
                mode='L',
            )
        else:
            mask = Image.open(mask_info).convert('L')
        return mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = self._load_mask(self.masks[index])

        if self.split == 'train':
            img, mask = self._train_transform(img, mask)
        else:
            img, mask = self._val_transform(img, mask)

        return img, mask

    def _train_transform(self, img, mask):
        i, j, h, w = T.RandomResizedCrop.get_params(
            img, scale=(0.7, 1.0), ratio=(0.9, 1.1))
        img = TF.resized_crop(img, i, j, h, w, [self.img_size, self.img_size])
        mask = TF.resized_crop(mask, i, j, h, w, [self.img_size, self.img_size],
                               interpolation=T.InterpolationMode.NEAREST)

        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        mask = (TF.to_tensor(mask) > 0.5).float()
        return img, mask.squeeze(0)

    def _val_transform(self, img, mask):
        img = TF.resize(img, [self.img_size, self.img_size])
        mask = TF.resize(mask, [self.img_size, self.img_size],
                         interpolation=T.InterpolationMode.NEAREST)

        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        mask = (TF.to_tensor(mask) > 0.5).float()
        return img, mask.squeeze(0)
