from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def capture_rng_state(dataloader_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "dataloader_generator": dataloader_generator.get_state(),
    }


def restore_rng_state(state: dict[str, Any], dataloader_generator: torch.Generator) -> None:
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch_cpu_state = state["torch_cpu"]
    if not isinstance(torch_cpu_state, torch.Tensor):
        raise TypeError("Checkpoint torch_cpu RNG state is not a tensor.")
    torch.set_rng_state(torch_cpu_state.detach().to(device="cpu", dtype=torch.uint8))
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        cuda_states = state["torch_cuda"]
        if not isinstance(cuda_states, (list, tuple)) or not all(isinstance(value, torch.Tensor) for value in cuda_states):
            raise TypeError("Checkpoint torch_cuda RNG state is not a tensor sequence.")
        torch.cuda.set_rng_state_all(
            [value.detach().to(device="cpu", dtype=torch.uint8) for value in cuda_states]
        )
    dataloader_state = state["dataloader_generator"]
    if not isinstance(dataloader_state, torch.Tensor):
        raise TypeError("Checkpoint dataloader RNG state is not a tensor.")
    dataloader_generator.set_state(dataloader_state.detach().to(device="cpu", dtype=torch.uint8))


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
    dataloader_generator: torch.Generator,
) -> None:
    payload = {
        "checkpoint_format": "paper_recipe_v1",
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_metric": float(best_metric),
        "config": config,
        "rng_state": capture_rng_state(dataloader_generator),
    }
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.6
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or payload.get("checkpoint_format") != "paper_recipe_v1":
        raise ValueError(f"Not a paper_recipe_v1 checkpoint: {path}")
    return payload


__all__ = ["capture_rng_state", "load_checkpoint", "restore_rng_state", "save_training_checkpoint"]
