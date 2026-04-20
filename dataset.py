"""
1.2 Dataset: Oxford-IIIT Pet
- 使用 torchvision.datasets.OxfordIIITPet 加载
- trainval 用于训练（可再划分为 train/val）；test 仅用于最终评估
- 数据增强仅应用于训练集（符合作业要求）
"""
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms

# 数据存放目录（与 .gitignore 中 data/ 一致）
DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# Oxford-IIIT Pet 共 37 类（breed）
NUM_CLASSES = 37


def _train_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),         
        transforms.ColorJitter(0.15, 0.15, 0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _eval_transforms():
    """验证/测试集：仅 resize 与归一化，不做随机增强"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_train_val_loaders(root, batch_size, val_ratio, num_workers, download, pin_memory):
    # 分别创建两个数据集，各自设置 transform，互不干扰
    full_train = datasets.OxfordIIITPet(
        root=root, split="trainval", target_types="category", download=download,
        transform=_train_transforms()
    )
    full_val = datasets.OxfordIIITPet(
        root=root, split="trainval", target_types="category", download=download,
        transform=_eval_transforms()
    )

    n = len(full_train)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    train_idx, val_idx = random_split(range(n), [n_train, n_val])

    train_subset = torch.utils.data.Subset(full_train, train_idx)
    val_subset = torch.utils.data.Subset(full_val, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader

def get_train_loader_only(root=DEFAULT_ROOT, batch_size=32, num_workers=0, download=True):
    """仅返回 trainval 的 DataLoader（不划分 val），用于训练。"""
    trainval = datasets.OxfordIIITPet(
        root=root,
        split="trainval",
        target_types="category",
        download=download,
        transform=_train_transforms(),
    )
    return DataLoader(trainval, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def get_test_loader(root=DEFAULT_ROOT, batch_size=32, num_workers=0, download=True, pin_memory=False):
    """
    仅用于最终评估的 test 集（文档要求 test 必须仅用于最终评估）。
    """
    test_set = datasets.OxfordIIITPet(
        root=root,
        split="test",
        target_types="category",
        download=download,
        transform=_eval_transforms(),
    )
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
