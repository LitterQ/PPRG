from __future__ import annotations

import torch


def erm_loss(losses: dict[str, torch.Tensor]) -> torch.Tensor:
    if "clean" not in losses:
        raise KeyError("ERM requires a clean loss.")
    return losses["clean"]
