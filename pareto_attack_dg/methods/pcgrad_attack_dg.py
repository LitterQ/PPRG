from __future__ import annotations

import torch

from pareto_attack_dg.optim.pcgrad import pcgrad_backward


def pcgrad_attack_dg_backward(model: torch.nn.Module, losses: dict[str, torch.Tensor]) -> dict[str, object]:
    return pcgrad_backward(
        list(losses.values()),
        model,
        objective_names=list(losses.keys()),
    )
