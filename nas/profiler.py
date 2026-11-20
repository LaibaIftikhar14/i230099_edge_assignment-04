import io
import time
import tracemalloc
from typing import Dict, Tuple
import numpy as np
import psutil
import torch
import torch.nn as nn


def profile_model_on_cpu(
    model: nn.Module,
    input_resolution: int,
    warmup_runs: int = 6,
    timed_runs: int = 25,
) -> Dict[str, float]:
    """Measures CPU latency (ms), serialized model storage size (MB), and peak memory (MB)."""
    cpu_model = model.to("cpu").eval()
    dummy_input = torch.randn(1, 3, input_resolution, input_resolution)

    buffer = io.BytesIO()
    torch.save(cpu_model.state_dict(), buffer)
    storage_mb = len(buffer.getvalue()) / (1024 ** 2)
    params_m = sum(p.numel() for p in cpu_model.parameters()) / 1e6

    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = cpu_model(dummy_input)

    process = psutil.Process()
    rss_before = process.memory_info().rss
    tracemalloc.start()
    with torch.inference_mode():
        _ = cpu_model(dummy_input)
    _, py_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = process.memory_info().rss
    peak_mem_mb = max(py_peak_bytes, max(0, rss_after - rss_before)) / (1024 ** 2)

    latencies_ms = []
    with torch.inference_mode():
        for _ in range(timed_runs):
            t0 = time.perf_counter()
            _ = cpu_model(dummy_input)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "params_m": round(params_m, 4),
        "storage_mb": round(storage_mb, 4),
        "latency_mean_ms": round(float(np.mean(latencies_ms)), 4),
        "latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 4),
        "peak_mem_mb": round(float(peak_mem_mb), 4),
    }


def check_deployment_constraints(
    hw_metrics: Dict[str, float],
    max_latency_ms: float = 8.0,
    max_storage_mb: float = 2.5,
    max_peak_mem_mb: float = 25.0,
) -> Tuple[bool, str]:
    if hw_metrics["latency_mean_ms"] > max_latency_ms:
        return False, f"Latency {hw_metrics['latency_mean_ms']:.2f}ms > {max_latency_ms}ms"
    if hw_metrics["storage_mb"] > max_storage_mb:
        return False, f"Size {hw_metrics['storage_mb']:.2f}MB > {max_storage_mb}MB"
    if hw_metrics["peak_mem_mb"] > max_peak_mem_mb:
        return False, f"PeakMem {hw_metrics['peak_mem_mb']:.2f}MB > {max_peak_mem_mb}MB"
    return True, "PASS"
