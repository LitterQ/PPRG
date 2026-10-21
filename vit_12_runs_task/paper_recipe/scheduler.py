from __future__ import annotations

from typing import Any

import torch

from .config import lr_for_epoch


class PaperEpochLRScheduler:
    """Small stateful scheduler with the official fractional-epoch semantics."""

    def __init__(self, optimizer: torch.optim.Optimizer, *, epochs: int = 40, lr_max: float = 0.1) -> None:
        self.optimizer = optimizer
        self.epochs = int(epochs)
        self.lr_max = float(lr_max)
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> float:
        lr = lr_for_epoch(epoch, epochs=self.epochs, lr_max=self.lr_max)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.current_epoch = int(epoch)
        return lr

    def state_dict(self) -> dict[str, Any]:
        return {"epochs": self.epochs, "lr_max": self.lr_max, "current_epoch": self.current_epoch}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["epochs"]) != self.epochs or float(state["lr_max"]) != self.lr_max:
            raise ValueError("Scheduler state conflicts with the resolved paper recipe.")
        current_epoch = int(state["current_epoch"])
        if not 0 <= current_epoch <= self.epochs:
            raise ValueError(f"Invalid scheduler current_epoch={current_epoch}")
        self.current_epoch = current_epoch
        if current_epoch:
            self.set_epoch(current_epoch)


__all__ = ["PaperEpochLRScheduler"]
