import os
import time
import numpy as np
import onnx
import onnxruntime as ort
import torch
from models.students import get_student_model
from nas.search_space import ArchitectureGenerator


def export_and_verify_onnx(
    model: torch.nn.Module,
    onnx_path: str,
    input_resolution: int = 32,
    num_runs: int = 50,
) -> dict:
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    model = model.to("cpu").eval()
    dummy_input = torch.randn(1, 3, input_resolution, input_resolution, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    with torch.inference_mode():
        pt_out = model(dummy_input).numpy()

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])

    ort_inputs = {"input": dummy_input.numpy()}
    ort_out = session.run(["logits"], ort_inputs)[0]
    max_diff = float(np.max(np.abs(pt_out - ort_out)))

    for _ in range(8):
        session.run(["logits"], ort_inputs)

    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        session.run(["logits"], ort_inputs)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    onnx_size_mb = os.path.getsize(onnx_path) / (1024 ** 2)
    return {
        "onnx_path": onnx_path,
        "onnx_size_mb": round(onnx_size_mb, 4),
        "ort_cpu_latency_ms": round(float(np.mean(latencies)), 4),
        "max_abs_diff": round(max_diff, 7),
    }


def export_all_models():
    results = {}
    for mode in ["baseline", "kd"]:
        for s in ["mobilenetv2", "squeezenet", "efficientnet_lite0"]:
            ckpt_path = f"checkpoints/{mode}/{s}_{mode}_best.pth"
            if not os.path.exists(ckpt_path):
                continue
            model = get_student_model(s, num_classes=100)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            res = export_and_verify_onnx(model, f"onnx_models/{s}_{mode}.onnx", input_resolution=32)
            results[f"{s}_{mode}"] = res
            print(f"[ONNX Exported] {s}_{mode}: {res}")

    nas_ckpt_path = "checkpoints/nas/nas_best_retrained.pth"
    if os.path.exists(nas_ckpt_path):
        ckpt = torch.load(nas_ckpt_path, map_location="cpu")
        cfg = ckpt["architecture_config"]
        nas_model = ArchitectureGenerator.build_model(cfg, num_classes=10)
        nas_model.load_state_dict(ckpt["model_state_dict"])
        res = export_and_verify_onnx(nas_model, "onnx_models/nas_best.onnx", input_resolution=cfg["input_resolution"])
        results["nas_best"] = res
        print(f"[ONNX Exported] nas_best: {res}")
    return results


if __name__ == "__main__":
    export_all_models()
