from __future__ import annotations

import torch

from .projection import normalize_l2_per_sample, project_l2_ball, project_linf_ball, random_l2_delta_like


def cw_margin_loss(logits: torch.Tensor, y: torch.Tensor, confidence: float = 0.0) -> torch.Tensor:
    true_logits = logits.gather(1, y.reshape(-1, 1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y.reshape(-1, 1), float("-inf"))
    other_logits = masked.max(dim=1).values
    return (other_logits - true_logits + confidence).mean()


def margin_pgd_l2_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    random_start: bool = True,
    confidence: float = 0.0,
) -> torch.Tensor:
    x_clean = x.detach()
    if random_start:
        x_adv = (x_clean + random_l2_delta_like(x_clean, eps)).clamp(0.0, 1.0)
    else:
        x_adv = x_clean.clone()

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        with torch.enable_grad():
            logits = model(x_adv)
            loss = cw_margin_loss(logits, y, confidence=confidence)
            grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * normalize_l2_per_sample(grad)
            delta = project_l2_ball(x_adv - x_clean, eps)
            x_adv = (x_clean + delta).clamp(0.0, 1.0)

    return x_adv.detach()


def margin_pgd_linf_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    random_start: bool = True,
    confidence: float = 0.0,
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
            logits = model(x_adv)
            loss = cw_margin_loss(logits, y, confidence=confidence)
            grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = project_linf_ball(x_adv, x_clean, eps).clamp(0.0, 1.0)

    return x_adv.detach()
