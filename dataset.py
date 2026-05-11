# DATASET.py
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
NUM_CLASSES = 37


def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        # transforms.ColorJitter(
        #     brightness=0.1, 
        #     contrast=0.1,
        #     saturation=0.05,
        #     hue=0.1,
        # ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.01, 0.1)),
    ])

def get_val_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _resolve_oxford_pet_dirs(root: str) -> Tuple[Path, Path, Path]:
    root_path = Path(root)
    candidates = [
        root_path / "oxford-iiit-pet",
        root_path,
    ]
    for base in candidates:
        img_dir = base / "images"
        ann_dir = base / "annotations"
        trainval_txt = ann_dir / "trainval.txt"
        test_txt = ann_dir / "test.txt"
        if img_dir.is_dir() and trainval_txt.is_file() and test_txt.is_file():
            return img_dir, trainval_txt, test_txt
    raise FileNotFoundError(
        "Cannot find Oxford-IIIT Pet folders. Expect one of:\n"
        "1) <data_root>/oxford-iiit-pet/images + annotations/\n"
        "2) <data_root>/images + annotations/"
    )


def _read_split(split_file: Path) -> list[tuple[str, int]]:
    samples: list[tuple[str, int]] = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        stem = parts[0]
        label_1based = int(parts[1])
        label_0based = label_1based - 1
        samples.append((stem, label_0based))
    if not samples:
        raise RuntimeError(f"Empty split file: {split_file}")
    return samples


def _split_indices(n: int, val_ratio: float, seed: int) -> Tuple[list[int], list[int]]:
    if n < 2 or val_ratio <= 0:
        return list(range(n)), []
    val_len = int(n * val_ratio)
    val_len = max(1, min(val_len, n - 1))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    return perm[val_len:], perm[:val_len]


class PetsDataset(Dataset):
    def __init__(
        self,
        img_dir: Path,
        samples: list[tuple[str, int]],
        transform=None,
        is_train: bool = True,
    ):
        self.img_dir = img_dir
        self.samples = samples
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stem, label = self.samples[idx]
        img_path = self.img_dir / f"{stem}.jpg"
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

def get_train_val_loaders(
    root: str,
    batch_size: int,
    val_ratio: float,
    num_workers: int,
    download: bool,
    pin_memory: bool,
    seed: int = 42,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    del download  # Kept for API compatibility with train.py
    img_dir, trainval_txt, _ = _resolve_oxford_pet_dirs(root)
    all_samples = _read_split(trainval_txt)
    train_idx, val_idx = _split_indices(len(all_samples), val_ratio=val_ratio, seed=seed)

    train_samples = [all_samples[i] for i in train_idx]
    train_ds = PetsDataset(
        img_dir=img_dir,
        samples=train_samples,
        transform=get_train_transform(),
        is_train=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if not val_idx:
        return train_loader, None

    val_samples = [all_samples[i] for i in val_idx]
    val_ds = PetsDataset(
        img_dir=img_dir,
        samples=val_samples,
        transform=get_val_transform(),
        is_train=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def get_test_loader(
    root: str,
    batch_size: int,
    num_workers: int,
    download: bool,
    pin_memory: bool,
) -> DataLoader:
    del download  # Kept for API compatibility with train.py
    img_dir, _, test_txt = _resolve_oxford_pet_dirs(root)
    test_samples = _read_split(test_txt)
    test_ds = PetsDataset(
        img_dir=img_dir,
        samples=test_samples,
        transform=get_val_transform(),
        is_train=False,
    )
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

def get_train_eval_loader(
    root: str,
    batch_size: int,
    val_ratio: float,
    num_workers: int,
    download: bool,
    pin_memory: bool,
    seed: int = 42,
) -> DataLoader:
    del download
    img_dir, trainval_txt, _ = _resolve_oxford_pet_dirs(root)
    all_samples = _read_split(trainval_txt)

    train_idx, _ = _split_indices(
        len(all_samples),
        val_ratio=val_ratio,
        seed=seed
    )

    train_samples = [all_samples[i] for i in train_idx]

    train_eval_ds = PetsDataset(
        img_dir=img_dir,
        samples=train_samples,
        transform=get_val_transform(),
        is_train=False,
    )

    return DataLoader(
        train_eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
