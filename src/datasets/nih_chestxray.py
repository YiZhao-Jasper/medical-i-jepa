import os
import csv
import zipfile
import io
from logging import getLogger
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

logger = getLogger()

NIH_14_LABELS = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Effusion',
    'Emphysema',
    'Fibrosis',
    'Hernia',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pleural_Thickening',
    'Pneumonia',
    'Pneumothorax',
]

ZIP_NAMES = [
    'images_123.zip',
    'images_45.zip',
    'images_67.zip',
    'images_89.zip',
    'images_1011.zip',
    'images_12.zip',
]


def _build_zip_index(root_path):
    """Build a mapping from image filename → (zip_path, path_inside_zip)."""
    index = {}
    for zname in ZIP_NAMES:
        zpath = os.path.join(root_path, zname)
        if not os.path.exists(zpath):
            logger.warning(f'ZIP not found (still downloading?): {zpath}')
            continue
        try:
            zf = zipfile.ZipFile(zpath, 'r')
            for entry in zf.namelist():
                if entry.endswith('.png'):
                    basename = os.path.basename(entry)
                    index[basename] = (zpath, entry)
            zf.close()
        except (zipfile.BadZipFile, Exception) as e:
            logger.warning(f'Cannot read {zpath}: {e}')
    return index


def _try_extracted_path(root_path, filename):
    """Check if image exists in extracted directories."""
    flat = os.path.join(root_path, 'images', filename)
    if os.path.exists(flat):
        return flat
    for i in range(1, 13):
        candidate = os.path.join(root_path, f'images_{i:03d}', 'images', filename)
        if os.path.exists(candidate):
            return candidate
    return None


def _load_verified_set(root_path):
    """Load the set of verified-clean image filenames."""
    vpath = os.path.join(root_path, 'verified_images.txt')
    if os.path.exists(vpath):
        with open(vpath) as f:
            return set(line.strip() for line in f if line.strip())
    return None


def _parse_csv(csv_path, verified_set=None):
    """Parse Data_Entry_2017.csv, return list of (filename, multi-hot label vector).
    If verified_set is provided, only include filenames in that set."""
    label_to_idx = {l: i for i, l in enumerate(NIH_14_LABELS)}
    entries = []
    skipped = 0
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row['Image Index']
            if verified_set is not None and fname not in verified_set:
                skipped += 1
                continue
            findings = row['Finding Labels']
            label_vec = [0.0] * len(NIH_14_LABELS)
            if findings.strip() != 'No Finding':
                for disease in findings.split('|'):
                    disease = disease.strip()
                    if disease in label_to_idx:
                        label_vec[label_to_idx[disease]] = 1.0
            entries.append((fname, label_vec))
    if skipped > 0:
        logger.info(f'Skipped {skipped} unverified/corrupted images from CSV')
    return entries


class NIHChestXrayPretraining(Dataset):
    """NIH ChestX-ray14 for self-supervised pretraining (images only, no labels)."""

    def __init__(self, root_path, transform=None, use_zip=True):
        super().__init__()
        self.root_path = root_path
        self.transform = transform
        self.use_zip = use_zip

        csv_path = os.path.join(root_path, 'Data_Entry_2017.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'Data_Entry_2017.csv not found at {csv_path}')

        verified = _load_verified_set(root_path)
        self.entries = _parse_csv(csv_path, verified_set=verified)
        self.filenames = [e[0] for e in self.entries]

        if use_zip:
            logger.info('Building ZIP index for NIH images...')
            self.zip_index = _build_zip_index(root_path)
            self._zip_handles = {}
            self.filenames = [f for f in self.filenames if f in self.zip_index]
            logger.info(f'ZIP index: {len(self.filenames)} images available')
        else:
            self.zip_index = None

        logger.info(f'NIHChestXrayPretraining: {len(self.filenames)} verified images')

    def _open_image(self, filename):
        if self.use_zip and self.zip_index is not None and filename in self.zip_index:
            zpath, inner = self.zip_index[filename]
            pid = os.getpid()
            key = (pid, zpath)
            if key not in self._zip_handles:
                self._zip_handles[key] = zipfile.ZipFile(zpath, 'r')
            zf = self._zip_handles[key]
            data = zf.read(inner)
            return Image.open(io.BytesIO(data)).convert('RGB')

        extracted = _try_extracted_path(self.root_path, filename)
        if extracted:
            return Image.open(extracted).convert('RGB')

        raise FileNotFoundError(f'Image not found: {filename}')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        fname = self.filenames[index]
        try:
            img = self._open_image(fname)
        except Exception:
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform is not None:
            img = self.transform(img)
        return img, index


class NIHChestXrayClassification(Dataset):
    """NIH ChestX-ray14 for downstream multi-label classification."""

    def __init__(
        self,
        root_path,
        split='train',
        transform=None,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
        use_zip=True,
    ):
        super().__init__()
        self.root_path = root_path
        self.transform = transform
        self.use_zip = use_zip
        self.num_classes = len(NIH_14_LABELS)

        csv_path = os.path.join(root_path, 'Data_Entry_2017.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'Data_Entry_2017.csv not found at {csv_path}')

        verified = _load_verified_set(root_path)
        all_entries = _parse_csv(csv_path, verified_set=verified)

        patients = {}
        for fname, label in all_entries:
            pid = int(fname.split('_')[0])
            if pid not in patients:
                patients[pid] = []
            patients[pid].append((fname, label))

        import random
        rng = random.Random(seed)
        pids = sorted(patients.keys())
        rng.shuffle(pids)

        n_train = int(len(pids) * train_ratio)
        n_val = int(len(pids) * val_ratio)

        if split == 'train':
            selected_pids = set(pids[:n_train])
        elif split in ('valid', 'val'):
            selected_pids = set(pids[n_train:n_train + n_val])
        else:
            selected_pids = set(pids[n_train + n_val:])

        if use_zip:
            self.zip_index = _build_zip_index(root_path)
            self._zip_handles = {}
        else:
            self.zip_index = None
            self._zip_handles = {}

        self.filenames = []
        self.labels = []
        for pid in sorted(selected_pids):
            for fname, label in patients[pid]:
                self.filenames.append(fname)
                self.labels.append(label)

        self.labels = torch.tensor(self.labels, dtype=torch.float32)

        logger.info(
            f'NIHChestXrayClassification [{split}]: '
            f'{len(self.filenames)} images, {self.num_classes} classes'
        )

    def _open_image(self, filename):
        if self.use_zip and self.zip_index is not None and filename in self.zip_index:
            zpath, inner = self.zip_index[filename]
            pid = os.getpid()
            key = (pid, zpath)
            if key not in self._zip_handles:
                self._zip_handles[key] = zipfile.ZipFile(zpath, 'r')
            zf = self._zip_handles[key]
            data = zf.read(inner)
            return Image.open(io.BytesIO(data)).convert('RGB')

        extracted = _try_extracted_path(self.root_path, filename)
        if extracted:
            return Image.open(extracted).convert('RGB')

        raise FileNotFoundError(f'Image not found: {filename}')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        fname = self.filenames[index]
        label = self.labels[index]
        img = self._open_image(fname)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def make_nih_pretrain(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    drop_last=True,
    use_zip=True,
    **kwargs,
):
    dataset = NIHChestXrayPretraining(
        root_path=root_path,
        transform=transform,
        use_zip=use_zip,
    )
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    logger.info('NIH ChestX-ray14 pretraining data loader created')
    return dataset, data_loader, dist_sampler


def make_nih_classify(
    root_path,
    split='train',
    transform=None,
    batch_size=64,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    drop_last=True,
    use_zip=True,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42,
):
    dataset = NIHChestXrayClassification(
        root_path=root_path,
        split=split,
        transform=transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        use_zip=use_zip,
    )
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
        )
    else:
        sampler = None

    shuffle = (sampler is None) and (split == 'train')
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last and (split == 'train'),
        pin_memory=pin_mem,
        num_workers=num_workers,
    )
    return dataset, data_loader, sampler
