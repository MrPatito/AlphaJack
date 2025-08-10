import os
import random
from contextlib import nullcontext
from typing import Any, Dict, Iterable, Tuple, Union

import numpy as np
import torch

TensorOrNest = Union[torch.Tensor, Dict[str, Any], Iterable[Any]]

def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism vs speed trade-off
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.benchmark = not deterministic

def get_device(prefer: str = "auto") -> Tuple[torch.device, torch.dtype]:
    if prefer == "auto":
        prefer = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(prefer)
    # Prefer bf16 on capable GPUs, else fp16
    use_bf16 = device.type == "cuda" and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    # Improve matmul perf on Ampere+
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    return device, dtype

def move_to_device(batch: TensorOrNest, device: torch.device, non_blocking: bool = True) -> TensorOrNest:
    if isinstance(batch, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(x, device, non_blocking) for x in batch)
    if hasattr(batch, "to"):
        kwargs = {"non_blocking": non_blocking} if device.type == "cuda" else {}
        return batch.to(device, **kwargs)
    return batch

def maybe_autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=dtype)
    return nullcontext()

def log_device() -> None:
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"Using device: cuda:0 ({p.name}, cc={p.major}.{p.minor}, {p.total_memory/1e9:.1f} GB)")
    else:
        print("Using device: cpu")
