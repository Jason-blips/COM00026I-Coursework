import os
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from dataset import NUM_CLASSES, get_test_loader, get_train_val_loaders, get_train_eval_loader
from model import build_model

# Warmup + cosine decay learning rate scheduler
def build_warmup_cosine_scheduler(optimizer, total_epochs, warmup_epochs, base_lr):
    warmup_epochs = max(0, min(warmup_epochs, total_epochs - 1))
    cosine_epochs = max(1, total_epochs - warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=base_lr * 0.01,
    )

    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

    return cosine_scheduler

# SGD with momentum and weight decay
def build_sgd_optimizer(params, lr, momentum, weight_decay):
    return torch.optim.SGD(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )

# Train model for one epoch
def train_one_epoch(model, loader, criterion, optimizer, device):
    # Enable training mode
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = targets.long().to(device)

        # Clear previous gradients
        optimizer.zero_grad(set_to_none=True)
        # Forward pass
        logits = model(images)
        # Compute classification loss
        loss = criterion(logits, labels)
        # Backpropagation
        loss.backward()
        # Stabilise training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Update model parameters
        optimizer.step()
        # Compute training accuracy
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()

        total_loss += loss.item()
        total += labels.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)


# Evaluate model on Validation or test set
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        labels = targets.long().to(device)
        
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)


def main():       
    parser = argparse.ArgumentParser()   
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (maximum 30)")
    parser.add_argument("--lr", type=float, default=0.075)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio from the official trainval set")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=3.5e-3)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--label_smoothing", type=float, default=0.06)
    args = parser.parse_args([])   

    if args.epochs > 30:
        args.epochs = 30

    os.makedirs(args.save_dir, exist_ok=True)

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading dataset (first run may download ~800MB)...")
    pin_memory = device.type == "cuda"
    train_loader, val_loader = get_train_val_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        download=True,
        pin_memory=pin_memory,
        seed=args.seed,

    )
    print(f"Train batches per epoch: {len(train_loader)}")

    all_labels = []

    for _, labels in train_loader:
        all_labels.append(labels)

    all_labels = torch.cat(all_labels)

    print("Global min:", all_labels.min().item())
    print("Global max:", all_labels.max().item())
    print("Num unique:", len(torch.unique(all_labels)))
    print("Unique labels:", torch.unique(all_labels))

    images, labels = next(iter(train_loader))

    print("First batch label range:", labels.min().item(), labels.max().item())
    print("Unique labels:", torch.unique(labels))

    for _, labels in train_loader:
        print("Another batch label range:", labels.min().item(), labels.max().item())
        break
    
    model = build_model(num_classes=NUM_CLASSES, device=device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing
    )

    optimizer = build_sgd_optimizer(
        params=model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        base_lr=args.lr,
    )

    best_val_acc = 0.0
    epochs_no_improve = 0

    lr_history = []

    for epoch in range(1, args.epochs + 1):

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        lr_history.append(current_lr)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} "
            f"Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} "
            f"Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0

            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, "best.pth")
            )

            print("  -> Best model saved")

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print("\nEarly stopping triggered due to no validation improvement.")
            break

        scheduler.step()

    print("\nLearning rate used at each epoch:")
    print([round(lr, 8) for lr in lr_history])    

    print("\n--- Final Test Evaluation ---")
    test_loader = get_test_loader(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=True,
        pin_memory=pin_memory,
    )
    best_path = os.path.join(args.save_dir, "best.pth")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.to(device)

    train_eval_loader = get_train_eval_loader(
        root=args.data_root,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        download=True,
        pin_memory=pin_memory,
        seed=args.seed,
    )
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    final_train_loss, final_train_acc = evaluate(
        model,
        train_eval_loader,
        criterion,
        device
    )
    print(f"Final Train Loss: {final_train_loss:.4f} | Final Train Acc: {final_train_acc * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}%")
    torch.save(model.state_dict(), os.path.join(args.save_dir, "final.pth"))
    print("Done.")


if __name__ == "__main__":
    main()
