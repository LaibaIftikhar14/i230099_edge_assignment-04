import os
import time
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from models.teacher import ResNet50Teacher


def get_cifar100_loaders(data_dir: str = "./data", batch_size: int = 256, num_workers: int = 2):
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_ds = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_tf)
    pw = num_workers > 0
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=pw),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=pw),
    )


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    use_amp = device.type == "cuda"
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def train_teacher(epochs: int = 30, batch_size: int = 256, max_lr: float = 0.2, save_path: str = "checkpoints/teacher/resnet50_cifar100.pth"):
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_loader, test_loader = get_cifar100_loaders(batch_size=batch_size)
    model = ResNet50Teacher(num_classes=100).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=max_lr / 10.0, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=len(train_loader), pct_start=0.2
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()

        acc = evaluate(model, test_loader, device)
        history.append({"epoch": epoch, "loss": running_loss / len(train_loader), "val_acc": acc})
        if acc >= best_acc:
            best_acc = acc
            torch.save({"model_state_dict": model.state_dict(), "top1_acc": best_acc}, save_path)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            elapsed = time.time() - t_start
            print(f"[Teacher Ep {epoch:02d}/{epochs}] Loss: {running_loss/len(train_loader):.4f} | Val Top-1: {acc:.2f}% | Best: {best_acc:.2f}% | {elapsed:.1f}s")
    return best_acc, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--save-path", type=str, default="checkpoints/teacher/resnet50_cifar100.pth")
    args = parser.parse_args()
    train_teacher(epochs=args.epochs, batch_size=args.batch_size, max_lr=args.lr, save_path=args.save_path)
