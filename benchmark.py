import os
import pandas as pd
import torch
from models.teacher import ResNet50Teacher
from models.students import get_student_model
from nas.search_space import ArchitectureGenerator
from nas.profiler import profile_model_on_cpu
from onnx_export_and_infer import export_and_verify_onnx
from train_teacher import get_cifar100_loaders, evaluate as eval_cifar100
from nas.search import get_cifar10_loaders, evaluate_accuracy as eval_cifar10


def run_benchmarks() -> pd.DataFrame:
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, c100_test = get_cifar100_loaders(batch_size=256)
    rows = []

    teacher_ckpt = "checkpoints/teacher/resnet50_cifar100.pth"
    if os.path.exists(teacher_ckpt):
        teacher = ResNet50Teacher(num_classes=100)
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location="cpu")["model_state_dict"])
        hw = profile_model_on_cpu(teacher, input_resolution=32, timed_runs=50)
        acc = eval_cifar100(teacher.to(device), c100_test, device)
        rows.append({
            "Model": "ResNet-50 (Teacher)",
            "Dataset": "CIFAR-100",
            "Top1_Acc_%": round(acc, 2),
            "Params_M": hw["params_m"],
            "PT_Size_MB": hw["storage_mb"],
            "PT_CPU_Latency_ms": hw["latency_mean_ms"],
            "Peak_Mem_MB": hw["peak_mem_mb"],
            "ONNX_Size_MB": None,
            "ORT_CPU_Latency_ms": None,
            "ONNX_Max_Diff": None,
        })

    for mode in ["baseline", "kd"]:
        for s in ["mobilenetv2", "squeezenet", "efficientnet_lite0"]:
            ckpt_path = f"checkpoints/{mode}/{s}_{mode}_best.pth"
            if not os.path.exists(ckpt_path):
                continue
            model = get_student_model(s, num_classes=100)
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model_state_dict"])
            hw = profile_model_on_cpu(model, input_resolution=32, timed_runs=50)
            onnx_res = export_and_verify_onnx(model, f"onnx_models/{s}_{mode}.onnx", input_resolution=32, num_runs=50)
            acc = eval_cifar100(model.to(device), c100_test, device)
            rows.append({
                "Model": f"{s} ({mode.upper()})",
                "Dataset": "CIFAR-100",
                "Top1_Acc_%": round(acc, 2),
                "Params_M": hw["params_m"],
                "PT_Size_MB": hw["storage_mb"],
                "PT_CPU_Latency_ms": hw["latency_mean_ms"],
                "Peak_Mem_MB": hw["peak_mem_mb"],
                "ONNX_Size_MB": onnx_res["onnx_size_mb"],
                "ORT_CPU_Latency_ms": onnx_res["ort_cpu_latency_ms"],
                "ONNX_Max_Diff": onnx_res["max_abs_diff"],
            })

    nas_ckpt = "checkpoints/nas/nas_best_retrained.pth"
    if os.path.exists(nas_ckpt):
        ckpt = torch.load(nas_ckpt, map_location="cpu")
        cfg = ckpt["architecture_config"]
        res_px = int(cfg["input_resolution"])
        nas_model = ArchitectureGenerator.build_model(cfg, num_classes=10)
        nas_model.load_state_dict(ckpt["model_state_dict"])
        hw = profile_model_on_cpu(nas_model, input_resolution=res_px, timed_runs=50)
        onnx_res = export_and_verify_onnx(nas_model, "onnx_models/nas_best.onnx", input_resolution=res_px, num_runs=50)
        _, c10_test = get_cifar10_loaders(resolution=res_px, batch_size=256, search_subset=False)
        acc = eval_cifar10(nas_model.to(device), c10_test, device)
        rows.append({
            "Model": f"NAS-Discovered ({res_px}x{res_px})",
            "Dataset": "CIFAR-10",
            "Top1_Acc_%": round(acc, 2),
            "Params_M": hw["params_m"],
            "PT_Size_MB": hw["storage_mb"],
            "PT_CPU_Latency_ms": hw["latency_mean_ms"],
            "Peak_Mem_MB": hw["peak_mem_mb"],
            "ONNX_Size_MB": onnx_res["onnx_size_mb"],
            "ORT_CPU_Latency_ms": onnx_res["ort_cpu_latency_ms"],
            "ONNX_Max_Diff": onnx_res["max_abs_diff"],
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/benchmark_summary.csv", index=False)
    return df


if __name__ == "__main__":
    df = run_benchmarks()
    print(df.to_string(index=False))
