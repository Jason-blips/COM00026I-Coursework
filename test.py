import torch
import os

from dataset import get_test_loader, NUM_CLASSES
from model import build_model

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        pred = logits.argmax(dim=1)

        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return correct / total


def main():
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    model = build_model(NUM_CLASSES).to(device)

    model_path = os.path.join(
        'checkpoints',
        'best.pth'
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    test_loader = get_test_loader(
        root='data',
        batch_size=64,
        num_workers=4,
        download=False,
        pin_memory=True,
    )

    acc = evaluate(model, test_loader, device)

    print(f'Final Test Accuracy: {acc:.4f}')


if __name__ == '__main__':
    main()