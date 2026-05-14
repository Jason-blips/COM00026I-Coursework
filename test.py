import os
import torch

from dataset import get_test_loader, NUM_CLASSES
from model import build_model

# Evaluate model accuracy on the official test set
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

    # Load official Oxford-IIIT Pet test set
    test_loader = get_test_loader(
        root='data',
        batch_size=64,
        num_workers=4,
        download=True,
        pin_memory=torch.cuda.is_available(),
    )

    acc = evaluate(model, test_loader, device)

    print(f'Final Test Accuracy: {acc * 100:.2f}%')


if __name__ == '__main__':
    main()