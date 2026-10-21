from __future__ import annotations

import torch

from pareto_attack_dg.optim.mgda import mgda_backward


def mgda_attack_dg_backward(model: torch.nn.Module, losses: dict[str, torch.Tensor]) -> dict[str, object]:
    return mgda_backward(
        list(losses.values()),
        model,
        objective_names=list(losses.keys()),
    )
