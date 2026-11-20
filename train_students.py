import os
import time
import argparse
import torch
import torch.nn as nn
from models.teacher import ResNet50Teacher
from models.students import get_student_model
from distillation.kd_loss import DistillationLoss
from train_teacher import get_cifar100_loaders, evaluate


def train_student(
    student_name: str,
    mode: str,
    epochs: int = 25,
    batch_size: int = 256,
    max_lr: float = 0.1,
    teacher_ckpt: str = "checkpoints/teacher/resnet50_cifar100.pth",
    temperature: float = 4.0,
    alpha: float = 0.7,
    beta: float = 0.5,
):
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    train_loader, test_loader = get_cifar100_loaders(batch_size=batch_size)
    student = get_student_model(student_name, num_classes=100).to(device)

    if mode == "kd":
        teacher = ResNet50Teacher(num_classes=100).to(device)
        ckpt = torch.load(teacher_ckpt, map_location=device)
        teacher.load_state_dict(ckpt["model_state_dict"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        kd_criterion = DistillationLoss(
            student_channels=student.feature_channels,
            teacher_channels=teacher.feature_channels,
            temperature=temperature,
            alpha=alpha,
            beta=beta,
        ).to(device)
        params = list(student.parameters()) + list(kd_criterion.parameters())
        save_path = f"checkpoints/kd/{student_name}_kd_best.pth"
    else:
        ce_criterion = nn.CrossEntropyLoss()
        params = student.parameters()
        save_path = f"checkpoints/baseline/{student_name}_baseline_best.pth"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    optimizer = torch.optim.SGD(params, lr=max_lr / 10.0, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=len(train_loader), pct_start=0.2
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_acc = 0.0
    history = []
    for epoch in range(1, epochs + 1):
        student.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                if mode == "kd":
                    with torch.inference_mode():
                        t_logits, t_feats = teacher(images, return_features=True)
                    s_logits, s_feats = student(images, return_features=True)
                    loss, _ = kd_criterion(s_logits, t_logits, labels, s_feats, t_feats)
                else:
                    loss = ce_criterion(student(images), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()

        acc = evaluate(student, test_loader, device)
        history.append({"epoch": epoch, "loss": running_loss / len(train_loader), "val_acc": acc})
        if acc >= best_acc:
            best_acc = acc
            torch.save(
                {
                    "student_name": student_name,
                    "mode": mode,
                    "model_state_dict": student.state_dict(),
                    "top1_acc": best_acc,
                },
                save_path,
            )
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            elapsed = time.time() - t_start
            print(f"[{student_name.upper()} | {mode.upper()} | Ep {epoch:02d}/{epochs}] Acc: {acc:.2f}% | Best: {best_acc:.2f}% | {elapsed:.1f}s")
    return best_acc, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=str, default="all", choices=["all", "mobilenetv2", "squeezenet", "efficientnet_lite0"])
    parser.add_argument("--mode", type=str, required=True, choices=["baseline", "kd"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--teacher-ckpt", type=str, default="checkpoints/teacher/resnet50_cifar100.pth")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.5)
    args = parser.parse_args()

    students = ["mobilenetv2", "squeezenet", "efficientnet_lite0"] if args.student == "all" else [args.student]
    for s in students:
        train_student(
            student_name=s,
            mode=args.mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_lr=args.lr,
            teacher_ckpt=args.teacher_ckpt,
            temperature=args.temperature,
            alpha=args.alpha,
            beta=args.beta,
        )
