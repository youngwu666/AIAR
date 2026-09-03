
import csv
import time
from pathlib import Path

import torch

def _backend_name(device):
    return str(device).split(":", 1)[0]

def synchronize(device):

    backend = _backend_name(device)

    if backend == "cuda":
        torch.cuda.synchronize(device)
    elif backend == "mps":
        sync = getattr(torch.mps, "synchronize", None)
        if sync is not None:
            sync()

def _memory_snapshot(device):

    backend = _backend_name(device)

    if backend == "cuda":
        return {
            "gpu_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
            "gpu_memory_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
            "gpu_memory_measurement": "peak",
        }
    if backend == "mps":
        current = getattr(torch.mps, "current_allocated_memory", None)
        driver = getattr(torch.mps, "driver_allocated_memory", None)
        return {
            "gpu_memory_allocated_gb": (current() if current else 0) / 1024**3,
            "gpu_memory_reserved_gb": (driver() if driver else 0) / 1024**3,
            "gpu_memory_measurement": "current",
        }
    return {
        "gpu_memory_allocated_gb": 0.0,
        "gpu_memory_reserved_gb": 0.0,
        "gpu_memory_measurement": "unavailable_on_cpu",
    }

def parameter_counts(models):
    counts = {}
    total = 0
    trainable = 0
    for name, model in models.items():
        if model is None or not hasattr(model, "parameters"):
            count = trainable_count = 0
        else:
            params = list(model.parameters())
            count = sum(parameter.numel() for parameter in params)
            trainable_count = sum(
                parameter.numel() for parameter in params if parameter.requires_grad
            )
        counts[name] = {
            "total_m": count / 1e6,
            "trainable_m": trainable_count / 1e6,
        }
        total += count
        trainable += trainable_count
    counts["total"] = {
        "total_m": total / 1e6,
        "trainable_m": trainable / 1e6,
    }
    return counts

class EfficiencyTracker:
    def __init__(self, device, models=None):
        self.device = device
        self.models = models or {}
        self.decoder_time_s = 0.0
        self._start = None
        self._elapsed = 0.0
        self._started = False

    def start(self):
        synchronize(self.device)
        if _backend_name(self.device) == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._elapsed = 0.0
        self._started = True
        self._start = time.perf_counter()

    def pause(self):
        if self._start is None:
            return
        synchronize(self.device)
        self._elapsed += time.perf_counter() - self._start
        self._start = None

    def resume(self):
        if self._start is not None:
            return
        synchronize(self.device)
        self._start = time.perf_counter()

    def measure_decoder(self, decoder, *args, **kwargs):
        synchronize(self.device)
        start = time.perf_counter()
        output = decoder(*args, **kwargs)
        synchronize(self.device)
        self.decoder_time_s += time.perf_counter() - start
        return output

    def finish(self):
        if not self._started:
            raise RuntimeError("EfficiencyTracker.start() must be called first.")
        self.pause()
        result = {
            "inference_time_s": self._elapsed,
            "decoder_time_s": self.decoder_time_s,
            "device": str(self.device),
            "parameter_counts_m": parameter_counts(self.models),
        }
        result.update(_memory_snapshot(self.device))
        return result

class TimedModule(torch.nn.Module):
    def __init__(self, module, tracker):
        super().__init__()
        self.module = module
        self.tracker = tracker

    def forward(self, *args, **kwargs):
        return self.tracker.measure_decoder(self.module, *args, **kwargs)

def save_efficiency(path, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        (key, value)
        for key, value in metrics.items()
        if key != "parameter_counts_m"
    ]
    for model_name, counts in metrics.get("parameter_counts_m", {}).items():
        rows.append((f"{model_name}_parameters_m", counts["total_m"]))
        rows.append(
            (f"{model_name}_trainable_parameters_m", counts["trainable_m"])
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)
