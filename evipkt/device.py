from __future__ import annotations

import torch


def resolve_device(device: str = "auto") -> torch.device:
    """
    Resolve training device.

    - auto: cuda if available else cpu
    - cuda / cuda:0: require GPU (raise if unavailable)
    - cpu: force CPU
    """
    text = (device or "auto").strip().lower()
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if text.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable. "
                "Run outside the sandbox and check `nvidia-smi` / driver install."
            )
        return torch.device(device)
    if text == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device='{device}'. Use auto, cpu, or cuda[:index].")


def dataloader_kwargs(device: torch.device) -> dict:
    use_cuda = device.type == "cuda"
    return {
        "pin_memory": use_cuda,
        "num_workers": 2 if use_cuda else 0,
    }
