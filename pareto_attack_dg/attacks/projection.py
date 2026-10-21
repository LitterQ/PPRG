from __future__ import annotations

import torch


def _batch_view(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], -1)


def l2_norm_per_sample(tensor: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    shape = (tensor.shape[0],) + (1,) * (tensor.dim() - 1)
    norms = _batch_view(tensor).norm(p=2, dim=1).reshape(shape)
    if eps > 0:
        norms = norms.clamp_min(eps)
    return norms


def normalize_l2_per_sample(tensor: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return tensor / l2_norm_per_sample(tensor, eps=eps)


def project_linf_ball(x_adv: torch.Tensor, x_clean: torch.Tensor, eps_linf: float) -> torch.Tensor:
    delta = torch.clamp(x_adv - x_clean, min=-eps_linf, max=eps_linf)
    return x_clean + delta


def project_l2_ball(delta: torch.Tensor, eps_l2: float) -> torch.Tensor:
    norms = l2_norm_per_sample(delta, eps=1e-12)
    scale = torch.clamp(eps_l2 / norms, max=1.0)
    return delta * scale


def random_l2_delta_like(x: torch.Tensor, radius: float) -> torch.Tensor:
    delta = normalize_l2_per_sample(torch.randn_like(x))
    sample_shape = (x.shape[0],) + (1,) * (x.dim() - 1)
    dim = max(1, x[0].numel())
    scales = torch.rand(sample_shape, device=x.device, dtype=x.dtype).pow(1.0 / dim)
    return delta * scales * radius
