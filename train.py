"""
1.4 训练与评估
- 最多训练 30 个 epoch（文档要求）
- 使用 CrossEntropyLoss 与优化器从零训练
- 仅在训练结束后对 test 集做一次最终评估（test 集训练过程中不参与）
"""
import argparse
import os
import random
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import get_train_val_loaders, get_test_loader, NUM_CLASSES
from model import build_model


def build_warmup_cosine_scheduler(optimizer, total_epochs, warmup_epochs, base_lr):
    warmup_epochs = max(0, min(warmup_epochs, total_epochs - 1))
    cosine_epochs = max(1, total_epochs - warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=base_lr * 0.01
    )
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    return cosine_scheduler


def build_sgd_optimizer(params, lr, momentum, weight_decay):
    return torch.optim.SGD(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )


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
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="从 trainval 中取比例做验证，0 表示不划分")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="保存权重目录")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="L2 权重衰减，减轻过拟合")
    parser.add_argument("--patience", type=int, default=6, help="早停：验证损失连续多少轮不降则停止")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD 动量系数")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="学习率 warmup 轮数（SGD 常用）")
    parser.add_argument("--two_stage", action=argparse.BooleanOptionalAction, default=True, help="启用两阶段训练：先 AdamW 再 SGD（默认开启）")
    parser.add_argument("--stage1_epochs", type=int, default=3, help="两阶段中第 1 阶段（AdamW）训练轮数")
    parser.add_argument("--stage1_lr", type=float, default=3e-4, help="两阶段第 1 阶段学习率")
    parser.add_argument("--stage1_weight_decay", type=float, default=1e-4, help="两阶段第 1 阶段权重衰减")
    parser.add_argument("--stage1_warmup_epochs", type=int, default=1, help="两阶段第 1 阶段 warmup 轮数")
    args = parser.parse_args()

    if args.epochs > 30:
        print("Warning: 作业要求最多 30 epochs，已自动限制为 30")
        args.epochs = 30

    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    if args.two_stage and args.epochs < 2:
        print("Warning: 两阶段训练至少需要 2 个 epoch，已退回单阶段 SGD。")
        args.two_stage = False

    stage1_epochs = 0
    if args.two_stage:
        stage1_epochs = max(1, min(args.stage1_epochs, args.epochs - 1))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.stage1_lr, weight_decay=args.stage1_weight_decay
        )
        scheduler = build_warmup_cosine_scheduler(
            optimizer=optimizer,
            total_epochs=stage1_epochs,
            warmup_epochs=args.stage1_warmup_epochs,
            base_lr=args.stage1_lr,
        )
        print(
            f"Two-stage training enabled | Stage1: AdamW ({stage1_epochs} epochs, lr={args.stage1_lr}) | "
            f"Stage2: SGD ({args.epochs - stage1_epochs} epochs, lr={args.lr})"
        )
    else:
        optimizer = build_sgd_optimizer(
            params=model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        if args.lr < 1e-2:
            print("Warning: 当前 SGD 学习率偏小，建议 --lr 0.02~0.05。")
        scheduler = build_warmup_cosine_scheduler(
            optimizer=optimizer,
            total_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            base_lr=args.lr,
        )
    print(f"Optimizer: {optimizer.__class__.__name__}, lr={optimizer.param_groups[0]['lr']}")

    # 2. 检查数据加载器返回的图像和标签
    images, labels = next(iter(train_loader))
    print(images.min(), images.max(), images.mean())  # 应该大约 -2~2 之间（归一化后）
    print(labels.unique())  # 应该包含 0~36 之间的整数

    best_val_loss = float("inf")
    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        if args.two_stage and epoch == stage1_epochs + 1:
            optimizer = build_sgd_optimizer(
                params=model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
            scheduler = build_warmup_cosine_scheduler(
                optimizer=optimizer,
                total_epochs=args.epochs - stage1_epochs,
                warmup_epochs=args.warmup_epochs,
                base_lr=args.lr,
            )
            print(
                f"\nSwitching to Stage2 at epoch {epoch}: "
                f"SGD(lr={args.lr}, momentum={args.momentum}, wd={args.weight_decay})"
            )

        print(f"Epoch {epoch}/{args.epochs} starting...")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch}/{args.epochs}  LR: {current_lr:.6f}  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")
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
