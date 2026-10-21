from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn.functional as F

from pareto_attack_dg.attacks.projection import normalize_l2_per_sample, project_l2_ball


def soft_target_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 1:
        return F.cross_entropy(logits, target)
    if target.shape != logits.shape:
        raise ValueError(f"soft target shape {tuple(target.shape)} != logits shape {tuple(logits.shape)}")
    if not torch.is_floating_point(target):
        raise TypeError("soft targets must be floating point probabilities")
    return torch.sum(-target * F.log_softmax(logits, dim=-1), dim=-1).mean()


@contextmanager
def inner_attack_eval_mode(model: torch.nn.Module) -> Iterator[None]:
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def _random_l2_delta(x: torch.Tensor, eps: float, generator: torch.Generator | None) -> torch.Tensor:
    delta = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    delta = normalize_l2_per_sample(delta)
    # Match the existing attack's uniform radius sampling while allowing an
    # isolated deterministic generator per branch/batch.
    dim = max(1, x[0].numel())
    radius = torch.rand(
        (x.shape[0], 1, 1, 1), dtype=x.dtype, device=x.device, generator=generator
    ).pow(1.0 / dim)
    return delta * radius * eps


def pgd_linf_soft_target(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 10,
    random_start: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    x_clean = x.detach()
    if random_start:
        delta = torch.empty(x_clean.shape, dtype=x_clean.dtype, device=x_clean.device).uniform_(
            -eps, eps, generator=generator
        )
        x_adv = (x_clean + delta).clamp(0.0, 1.0)
    else:
        x_adv = x_clean.clone()

    with inner_attack_eval_mode(model):
        for _ in range(steps):
            x_adv = x_adv.detach().requires_grad_(True)
            with torch.enable_grad():
                loss = soft_target_cross_entropy(model(x_adv), target)
                grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
            with torch.no_grad():
                x_adv = x_adv + alpha * grad.sign()
                delta = (x_adv - x_clean).clamp(-eps, eps)
                x_adv = (x_clean + delta).clamp(0.0, 1.0)
    return x_adv.detach()


def pgd_l2_soft_target(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 0.5,
    alpha: float = 0.1,
    steps: int = 5,
    random_start: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    x_clean = x.detach()
    if random_start:
        x_adv = (x_clean + _random_l2_delta(x_clean, eps, generator)).clamp(0.0, 1.0)
    else:
        x_adv = x_clean.clone()

    with inner_attack_eval_mode(model):
        for _ in range(steps):
            x_adv = x_adv.detach().requires_grad_(True)
            with torch.enable_grad():
                loss = soft_target_cross_entropy(model(x_adv), target)
                grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
            with torch.no_grad():
                x_adv = x_adv + alpha * normalize_l2_per_sample(grad)
                delta = project_l2_ball(x_adv - x_clean, eps)
                x_adv = (x_clean + delta).clamp(0.0, 1.0)
    return x_adv.detach()


def tensor_sha256(tensor: torch.Tensor) -> str:
    import hashlib

    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


__all__ = ["pgd_l2_soft_target", "pgd_linf_soft_target", "soft_target_cross_entropy", "tensor_sha256"]
