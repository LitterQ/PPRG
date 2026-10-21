from __future__ import annotations

import torch
import torch.nn.functional as F


def official_cw_loss(logits: torch.Tensor, target: torch.Tensor, *, reduction: bool = True) -> torch.Tensor:
    """Official repository's clipped untargeted CW margin objective."""

    sorted_logits, sorted_indices = logits.sort(dim=1)
    top_is_target = (sorted_indices[:, -1] == target).to(logits.dtype)
    max_other = sorted_logits[:, -2] * top_is_target + sorted_logits[:, -1] * (1.0 - top_is_target)
    target_logit = logits.gather(1, target[:, None]).squeeze(1)
    values = (max_other - target_logit).clamp_min(-10.0)
    return values.mean() if reduction else values


def official_linf_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float,
    alpha: float,
    steps: int,
    objective: str,
) -> torch.Tensor:
    """Pixel-space equivalent of official pgd.py attack_pgd/attack_cw.

    The official normalized-space code updates only samples still correctly
    classified, keeps the maximum-loss restart (one restart by parser default),
    and initializes the per-sample best loss at zero.
    """

    if objective not in {"ce", "cw"}:
        raise ValueError(f"Unsupported official objective: {objective}")
    x_clean = x.detach()
    max_loss = torch.zeros(target.shape[0], device=x.device, dtype=x.dtype)
    max_adv = x_clean.clone()
    delta = torch.empty_like(x_clean).uniform_(-eps, eps)
    delta = (x_clean + delta).clamp(0.0, 1.0) - x_clean

    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        logits = model(x_clean + delta)
        correct = logits.argmax(1) == target
        if not bool(correct.any()):
            break
        loss = F.cross_entropy(logits, target) if objective == "ce" else official_cw_loss(logits, target)
        grad = torch.autograd.grad(loss, delta, only_inputs=True)[0].detach()
        with torch.no_grad():
            updated = (delta[correct] + alpha * grad[correct].sign()).clamp(-eps, eps)
            updated = (x_clean[correct] + updated).clamp(0.0, 1.0) - x_clean[correct]
            delta[correct] = updated

    with torch.no_grad():
        logits = model(x_clean + delta)
        losses = (
            F.cross_entropy(logits, target, reduction="none")
            if objective == "ce"
            else official_cw_loss(logits, target, reduction=False)
        )
        selected = losses >= max_loss
        max_adv[selected] = (x_clean + delta)[selected]
    return max_adv.detach()


__all__ = ["official_cw_loss", "official_linf_attack"]
