"""
1.4 训练与评估
- 最多训练 30 个 epoch（文档要求）
- 使用 CrossEntropyLoss 与优化器从零训练
- 仅在训练结束后对 test 集做一次最终评估（test 集训练过程中不参与）
"""
import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import get_train_val_loaders, get_test_loader, NUM_CLASSES
from model import build_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
        
        optimizer.step()
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data", help="数据集根目录")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数，最多 30（作业要求）。试跑可加 --epochs 2 先确认能跑通")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="从 trainval 中取比例做验证，0 表示不划分")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="保存权重目录")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 权重衰减，减轻过拟合")
    parser.add_argument("--patience", type=int, default=6, help="早停：验证损失连续多少轮不降则停止")
    args = parser.parse_args()

    if args.epochs > 30:
        print("Warning: 作业要求最多 30 epochs，已自动限制为 30")
        args.epochs = 30

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu" and args.epochs >= 10:
        print("Tip: CPU 训练较慢，可先用 python train.py --epochs 2 试跑确认无误再跑满 30 轮。")

    print("Loading dataset (first run may download ~800MB)...")
    pin_memory = device.type == "cuda"  # CPU 时关闭 pin_memory 避免警告
    train_loader, val_loader = get_train_val_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        download=True,
        pin_memory=pin_memory,
    )
    n_batches = len(train_loader)
    print(f"Train batches per epoch: {n_batches} (one epoch may take several min on CPU)")

    # 模型创建必须在「函数里」或「类外面」，不能写在 class PetClassifier 内部
    model = build_model(num_classes=NUM_CLASSES, device=device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs} starting...")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            print(f"Epoch {epoch}/{args.epochs}  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")
            # 按验证损失保存最佳（损失最低 = 泛化最好，避免过拟合）
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pth"))
                print(f"          -> 新的最佳 (val_loss={val_loss:.4f})，已保存")
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\n早停：验证损失连续 {args.patience} 轮未改善，停止训练。")
                break
        else:
            print(f"Epoch {epoch}/{args.epochs}  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
            if train_acc > best_val_acc:
                best_val_acc = train_acc
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pth"))

    # 最终评估：仅使用 test 集（文档要求 test 仅用于最终评估）
    print("\n--- Final evaluation on test set only ---")
    test_loader = get_test_loader(
        root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=pin_memory
    )
    if os.path.exists(os.path.join(args.save_dir, "best.pth")):
        model.load_state_dict(torch.load(os.path.join(args.save_dir, "best.pth"), map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}")
    torch.save(model.state_dict(), os.path.join(args.save_dir, "final.pth"))
    print("Done.")


if __name__ == "__main__":
    main()
