from __future__ import annotations

import torch
import torch.nn.functional as F

from .projection import normalize_l2_per_sample


def fgsm_linf_attack(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    x_adv = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
    x_adv = x_adv + eps * grad.sign()
    return x_adv.clamp(0.0, 1.0).detach()


def fgm_l2_attack(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    x_adv = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
    x_adv = x_adv + eps * normalize_l2_per_sample(grad)
    return x_adv.clamp(0.0, 1.0).detach()
