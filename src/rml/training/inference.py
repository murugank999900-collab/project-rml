"""Batched inference on in-memory arrays."""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def predict_logits(
    model: torch.nn.Module,
    X: np.ndarray | torch.Tensor,
    device: torch.device,
    batch_size: int = 1024,
    amp: bool = False,
) -> torch.Tensor:
    """Return float32 logits on the CPU, shape (N, num_classes)."""
    model.eval()
    X = torch.as_tensor(X)
    use_amp = amp and device.type == "cuda"
    out = []
    for start in range(0, X.shape[0], batch_size):
        xb = X[start : start + batch_size].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out.append(model(xb).float().cpu())
    return torch.cat(out)


def predict(model, X, device, batch_size: int = 1024, amp: bool = False) -> np.ndarray:
    return predict_logits(model, X, device, batch_size, amp).argmax(dim=1).numpy()
