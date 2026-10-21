from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_metric": best_metric,
        "config": config,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def load_model_state(checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.6
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint
    return {"model_state_dict": checkpoint, "config": {}}
