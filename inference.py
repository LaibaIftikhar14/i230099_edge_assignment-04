import time
import argparse
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as T
from PIL import Image

from models.students import get_student_model
from nas.search_space import ArchitectureGenerator


class UnifiedEdgeInference:
    """Unified inference API supporting PyTorch & ONNX Runtime across Students and NAS models."""

    def __init__(self, model_type: str, backend: str = "onnx", mode: str = "kd"):
        self.model_type = model_type.lower()
        self.backend = backend.lower()
        self.mode = mode.lower()

        if self.model_type == "nas":
            ckpt = torch.load("checkpoints/nas/nas_best_retrained.pth", map_location="cpu")
            self.cfg = ckpt["architecture_config"]
            self.resolution = int(self.cfg["input_resolution"])
            self.mean, self.std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
            self.onnx_path = "onnx_models/nas_best.onnx"
            self.pt_model = ArchitectureGenerator.build_model(self.cfg, num_classes=10)
            self.pt_model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.resolution = 32
            self.mean, self.std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            self.onnx_path = f"onnx_models/{self.model_type}_{self.mode}.onnx"
            ckpt = torch.load(f"checkpoints/{self.mode}/{self.model_type}_{self.mode}_best.pth", map_location="cpu")
            self.pt_model = get_student_model(self.model_type, num_classes=100)
            self.pt_model.load_state_dict(ckpt["model_state_dict"])

        self.transform = T.Compose([
            T.Resize((self.resolution, self.resolution)),
            T.ToTensor(),
            T.Normalize(self.mean, self.std),
        ])

        if self.backend == "onnx":
            self.session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
        else:
            self.pt_model.eval()

    def predict_tensor(self, tensor: torch.Tensor) -> dict:
        t0 = time.perf_counter()
        if self.backend == "onnx":
            logits = self.session.run(["logits"], {"input": tensor.numpy()})[0]
            probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = probs / np.sum(probs, axis=1, keepdims=True)
        else:
            with torch.inference_mode():
                logits = self.pt_model(tensor)
                probs = torch.softmax(logits, dim=1).numpy()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        top1_idx = int(np.argmax(probs[0]))
        return {
            "model": self.model_type,
            "mode": self.mode if self.model_type != "nas" else "nas_discovered",
            "backend": self.backend,
            "predicted_class": top1_idx,
            "confidence": round(float(probs[0, top1_idx]), 4),
            "latency_ms": round(latency_ms, 4),
        }

    def predict_image(self, image_path: str) -> dict:
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0)
        return self.predict_tensor(tensor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["mobilenetv2", "squeezenet", "efficientnet_lite0", "nas"])
    parser.add_argument("--mode", type=str, default="kd", choices=["baseline", "kd"])
    parser.add_argument("--backend", type=str, default="onnx", choices=["pytorch", "onnx"])
    parser.add_argument("--image", type=str, default=None)
    args = parser.parse_args()

    engine = UnifiedEdgeInference(model_type=args.model, backend=args.backend, mode=args.mode)
    if args.image:
        print(engine.predict_image(args.image))
    else:
        dummy = torch.randn(1, 3, engine.resolution, engine.resolution)
        print(engine.predict_tensor(dummy))
