from __future__ import annotations

import torch


def rex_loss(losses: dict[str, torch.Tensor], beta: float = 1.0) -> torch.Tensor:
    if not losses:
        raise ValueError("REx requires at least one domain loss.")
    risks = torch.stack(list(losses.values()))
    return risks.mean() + beta * risks.var(unbiased=False)
