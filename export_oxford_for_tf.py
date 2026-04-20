"""
用 PyTorch 下载 Oxford-IIIT Pet，并导出为「按类别分文件夹」结构，
供 TensorFlow 的 image_dataset_from_directory 使用。
- trainval 拆成 train / val；test 单独，仅最终评估用（符合作业 1.2）
"""
import os
from torchvision import datasets
from PIL import Image
import numpy as np

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
OUT_ROOT = os.path.join(DATA_ROOT, "oxford_pet_folders")  # train / val / test 各含 0..36
VAL_RATIO = 0.1


def main():
    from torch.utils.data import random_split

    print("Downloading Oxford-IIIT Pet (trainval)...")
    trainval = datasets.OxfordIIITPet(root=DATA_ROOT, split="trainval", target_types="category", download=True)
    print("Downloading Oxford-IIIT Pet (test)...")
    test_set = datasets.OxfordIIITPet(root=DATA_ROOT, split="test", target_types="category", download=True)

    n = len(trainval)
    n_val = int(n * VAL_RATIO)
    n_train = n - n_val
    train_ds, val_ds = random_split(trainval, [n_train, n_val])

    train_dir = os.path.join(OUT_ROOT, "train")
    val_dir = os.path.join(OUT_ROOT, "val")
    test_dir = os.path.join(OUT_ROOT, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    for c in range(37):
        for d in (train_dir, val_dir, test_dir):
            os.makedirs(os.path.join(d, str(c)), exist_ok=True)

    # 导出 train
    train_indices = train_ds.indices
    for idx in train_indices:
        img, label = trainval[idx]
        if not isinstance(img, Image.Image):
            img = Image.fromarray((img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        class_dir = os.path.join(train_dir, str(int(label)))
        img.save(os.path.join(class_dir, f"train_{idx:05d}.jpg"))
    print(f"Exported train: {len(train_indices)} images -> {train_dir}")

    # 导出 val
    val_indices = val_ds.indices
    for idx in val_indices:
        img, label = trainval[idx]
        if not isinstance(img, Image.Image):
            img = Image.fromarray((img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        class_dir = os.path.join(val_dir, str(int(label)))
        img.save(os.path.join(class_dir, f"val_{idx:05d}.jpg"))
    print(f"Exported val: {len(val_indices)} images -> {val_dir}")

    # 导出 test
    for i in range(len(test_set)):
        img, label = test_set[i]
        if not isinstance(img, Image.Image):
            img = Image.fromarray((img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        class_dir = os.path.join(test_dir, str(int(label)))
        img.save(os.path.join(class_dir, f"test_{i:05d}.jpg"))
    print(f"Exported test: {len(test_set)} images -> {test_dir}")
    print("Done. Use train_tf_oxford.py to train.")


if __name__ == "__main__":
    main()
