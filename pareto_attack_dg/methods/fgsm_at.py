from __future__ import annotations

import torch


def fgsm_at_loss(losses: dict[str, torch.Tensor]) -> torch.Tensor:
    if "fgsm_linf" not in losses:
        raise KeyError("FGSM-AT requires fgsm_linf loss.")
    return losses["fgsm_linf"]
