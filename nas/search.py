import os
import time
import json
import argparse
import pandas as pd
import optuna
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from nas.search_space import ArchitectureGenerator
from nas.profiler import profile_model_on_cpu, check_deployment_constraints


def get_cifar10_loaders(resolution: int, batch_size: int = 256, search_subset: bool = False):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = T.Compose([
        T.Resize((resolution, resolution)),
        T.RandomCrop(resolution, padding=max(2, resolution // 8)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([
        T.Resize((resolution, resolution)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    if search_subset:
        train_ds = Subset(train_ds, list(range(0, 10000)))
        test_ds = Subset(test_ds, list(range(0, 3000)))

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True),
    )


@torch.inference_mode()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
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


def run_nas(
    n_trials: int = 15,
    search_epochs: int = 3,
    retrain_epochs: int = 25,
    max_latency_ms: float = 8.0,
    max_storage_mb: float = 2.5,
    max_peak_mem_mb: float = 25.0,
):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints/nas", exist_ok=True)

    history_records = []
    cached_loaders = {}

    def objective(trial: optuna.Trial) -> float:
        config = ArchitectureGenerator.sample_config(trial)
        model = ArchitectureGenerator.build_model(config, num_classes=10)

        hw = profile_model_on_cpu(model, input_resolution=config["input_resolution"])
        valid, reason = check_deployment_constraints(
            hw,
            max_latency_ms=max_latency_ms,
            max_storage_mb=max_storage_mb,
            max_peak_mem_mb=max_peak_mem_mb,
        )

        if not valid:
            record = {
                "trial": trial.number,
                "status": f"REJECTED ({reason})",
                "val_acc": 0.0,
                "score": -100.0,
                **hw,
                "config": json.dumps(config),
            }
            history_records.append(record)
            trial.set_user_attr("config", config)
            trial.set_user_attr("hw", hw)
            trial.set_user_attr("valid", False)
            print(f"[Trial {trial.number:02d}] REJECTED -> {reason}")
            return -100.0

        res = config["input_resolution"]
        if res not in cached_loaders:
            cached_loaders[res] = get_cifar10_loaders(resolution=res, batch_size=256, search_subset=True)
        train_loader, val_loader = cached_loaders[res]

        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=5e-3, epochs=search_epochs, steps_per_epoch=len(train_loader)
        )
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        criterion = nn.CrossEntropyLoss()

        for _ in range(search_epochs):
            model.train()
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

        val_acc = evaluate_accuracy(model, val_loader, device)
        hw_penalty = 1.5 * (hw["latency_mean_ms"] / max_latency_ms) + 1.0 * (hw["storage_mb"] / max_storage_mb)
        score = val_acc - hw_penalty

        record = {
            "trial": trial.number,
            "status": "VALID",
            "val_acc": round(val_acc, 2),
            "score": round(score, 4),
            **hw,
            "config": json.dumps(config),
        }
        history_records.append(record)
        trial.set_user_attr("config", config)
        trial.set_user_attr("hw", hw)
        trial.set_user_attr("valid", True)
        trial.set_user_attr("val_acc", val_acc)
        print(
            f"[Trial {trial.number:02d}] VALID | Res={res} | "
            f"Acc={val_acc:.2f}% | Lat={hw['latency_mean_ms']:.2f}ms | Size={hw['storage_mb']:.2f}MB | Score={score:.2f}"
        )
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    history_df = pd.DataFrame(history_records)
    history_df.to_csv("results/nas_history.csv", index=False)

    valid_trials = [t for t in study.trials if t.user_attrs.get("valid", False)]
    if not valid_trials:
        raise RuntimeError("No candidate satisfied hardware constraints. Relax max_latency_ms / max_storage_mb.")

    best_trial = max(valid_trials, key=lambda t: t.value)
    best_config = best_trial.user_attrs["config"]
    best_hw = best_trial.user_attrs["hw"]

    with open("results/best_nas_config.json", "w") as f:
        json.dump(
            {
                "best_trial": best_trial.number,
                "proxy_val_acc": best_trial.user_attrs["val_acc"],
                "hardware_metrics": best_hw,
                "architecture_config": best_config,
            },
            f,
            indent=2,
        )
    print(f"\n[NAS Search Complete] Best Trial #{best_trial.number} | Config: {best_config}")

    # Full training from scratch on complete CIFAR-10 dataset
    t_retrain = time.time()
    print(f"\n[Retraining Best NAS Architecture From Scratch for {retrain_epochs} Epochs]")
    best_model = ArchitectureGenerator.build_model(best_config, num_classes=10).to(device)
    train_loader, test_loader = get_cifar10_loaders(
        resolution=best_config["input_resolution"], batch_size=256, search_subset=False
    )
    optimizer = torch.optim.SGD(best_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.1, epochs=retrain_epochs, steps_per_epoch=len(train_loader), pct_start=0.2
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0.0
    for epoch in range(1, retrain_epochs + 1):
        best_model.train()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(best_model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        test_acc = evaluate_accuracy(best_model, test_loader, device)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(
                {
                    "architecture_config": best_config,
                    "hardware_metrics": best_hw,
                    "model_state_dict": best_model.state_dict(),
                    "top1_acc": best_test_acc,
                },
                "checkpoints/nas/nas_best_retrained.pth",
            )
        if epoch == 1 or epoch % 5 == 0 or epoch == retrain_epochs:
            print(f"[NAS Retrain Ep {epoch:02d}/{retrain_epochs}] Test Acc: {test_acc:.2f}% | Best: {best_test_acc:.2f}% | {time.time()-t_retrain:.1f}s")
    return best_config, best_test_acc, history_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--search-epochs", type=int, default=3)
    parser.add_argument("--retrain-epochs", type=int, default=25)
    parser.add_argument("--max-latency-ms", type=float, default=8.0)
    parser.add_argument("--max-storage-mb", type=float, default=2.5)
    parser.add_argument("--max-peak-mem-mb", type=float, default=25.0)
    args = parser.parse_args()
    run_nas(
        n_trials=args.n_trials,
        search_epochs=args.search_epochs,
        retrain_epochs=args.retrain_epochs,
        max_latency_ms=args.max_latency_ms,
        max_storage_mb=args.max_storage_mb,
        max_peak_mem_mb=args.max_peak_mem_mb,
    )
