"""
1.2 Dataset: Oxford-IIIT Pet
- 使用 torchvision.datasets.OxfordIIITPet 加载
- trainval 用于训练（可再划分为 train/val）；test 仅用于最终评估
- 数据增强仅应用于训练集（符合作业要求）
"""
import os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms

# 数据存放目录（与 .gitignore 中 data/ 一致）
DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# Oxford-IIIT Pet 共 37 类（breed）
NUM_CLASSES = 37


def _train_transforms():
    """训练集数据增强（仅对 Oxford-IIIT Pet 训练图像做变换，符合要求）"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
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


def get_train_val_loaders(
    root=DEFAULT_ROOT,
    batch_size=32,
    val_ratio=0.1,
    num_workers=0,
    download=True,
):
    """
    加载 trainval 并划分为 train / validation。
    - root: 数据集根目录，数据会下载到 root/oxford-iiit-pet
    - val_ratio: 从 trainval 中取多少比例作为验证集（0 表示不划分，全部做训练）
    """
    trainval = datasets.OxfordIIITPet(
        root=root,
        split="trainval",
        target_types="category",
        download=download,
        transform=_train_transforms(),
    )
    n = len(trainval)
    if val_ratio <= 0 or val_ratio >= 1:
        train_loader = DataLoader(trainval, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return train_loader, None
    n_val = int(n * val_ratio)
    n_train = n - n_val
    train_ds, val_ds = random_split(trainval, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
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


def get_test_loader(root=DEFAULT_ROOT, batch_size=32, num_workers=0, download=True):
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
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
