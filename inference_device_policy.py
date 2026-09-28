"""Central inference-only CPU/CUDA admission for the sentiment application.

This repository does not train models. The helper only resolves the device for
retained pretrained Transformer inference. A scheduler CPU admission is
authoritative; a GPU-admitted child must prove a usable Torch CUDA device.
"""
from __future__ import annotations

import os
from typing import Mapping, Any

_TRUE = {"1", "true", "yes", "on", "y"}
_CPU_FLAGS = (
    "CPU_ONLY",
    "TRAINING_CONTROL_CPU_ONLY",
    "OPF_ADP_DISABLE_GPU_ACCELERATORS",
)


def cpu_admission_requested(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    visibility = str(env.get("CUDA_VISIBLE_DEVICES", "")).strip()
    return (
        str(env.get("TRAINING_CONTROL_BACKEND", "")).strip().lower() == "cpu"
        or any(str(env.get(name, "")).strip().lower() in _TRUE for name in _CPU_FLAGS)
        or ("CUDA_VISIBLE_DEVICES" in env and visibility in {"", "-1"})
    )


def gpu_admission_requested(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get("TRAINING_CONTROL_BACKEND", "")).strip().lower() == "gpu"


def _check_admission(environ: Mapping[str, str] | None = None) -> bool:
    cpu = cpu_admission_requested(environ)
    if cpu and gpu_admission_requested(environ):
        raise RuntimeError("conflicting CPU and GPU scheduler admission")
    return cpu


def _cuda_probe(torch_module: Any, index: int) -> bool:
    device = f"cuda:{index}"
    try:
        probe = torch_module.empty((1,), device=device)
        probe.fill_(1)
        torch_module.cuda.synchronize(index)
        del probe
        return True
    except (RuntimeError, OSError, AttributeError, TypeError, ValueError):
        return False


def best_cuda_index(torch_module: Any, environ: Mapping[str, str] | None = None) -> int:
    """Return the best *local* usable CUDA index, or -1 for standalone CPU."""
    if _check_admission(environ):
        return -1
    try:
        if not torch_module.cuda.is_available():
            count = 0
        else:
            count = int(torch_module.cuda.device_count())
    except (RuntimeError, OSError, AttributeError, TypeError, ValueError):
        count = 0

    best_index = -1
    best_free = -1
    for index in range(max(0, count)):
        if not _cuda_probe(torch_module, index):
            continue
        try:
            props = torch_module.cuda.get_device_properties(index)
            used = int(torch_module.cuda.memory_reserved(index))
            free = max(0, int(props.total_memory) - used)
        except (RuntimeError, OSError, AttributeError, TypeError, ValueError):
            free = 0
        if best_index < 0 or free > best_free:
            best_index, best_free = index, free

    if best_index < 0 and gpu_admission_requested(environ):
        raise RuntimeError("GPU-admitted inference worker has no usable Torch CUDA device")
    return best_index


def resolve_torch_device(torch_module: Any, environ: Mapping[str, str] | None = None) -> Any:
    index = best_cuda_index(torch_module, environ)
    return torch_module.device("cpu" if index < 0 else f"cuda:{index}")


def transformers_pipeline_device(device: Any) -> int:
    """Transformers Pipeline's stable integer convention: -1 CPU, >=0 CUDA."""
    if getattr(device, "type", None) != "cuda":
        return -1
    index = getattr(device, "index", None)
    return 0 if index is None else int(index)
