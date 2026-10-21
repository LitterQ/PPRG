from __future__ import annotations

import torch
import torch.nn.functional as F

from .projection import (
    normalize_l2_per_sample,
    project_l2_ball,
    project_linf_ball,
    random_l2_delta_like,
)


def pgd_linf_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    random_start: bool = True,
) -> torch.Tensor:
    x_clean = x.detach()
    if random_start:
        x_adv = x_clean + torch.empty_like(x_clean).uniform_(-eps, eps)
        x_adv = x_adv.clamp(0.0, 1.0)
    else:
        x_adv = x_clean.clone()

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = project_linf_ball(x_adv, x_clean, eps).clamp(0.0, 1.0)

    return x_adv.detach()


def pgd_l2_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    random_start: bool = True,
) -> torch.Tensor:
    x_clean = x.detach()
    if random_start:
        x_adv = (x_clean + random_l2_delta_like(x_clean, eps)).clamp(0.0, 1.0)
    else:
        x_adv = x_clean.clone()

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * normalize_l2_per_sample(grad)
            delta = project_l2_ball(x_adv - x_clean, eps)
            x_adv = (x_clean + delta).clamp(0.0, 1.0)

    return x_adv.detach()
