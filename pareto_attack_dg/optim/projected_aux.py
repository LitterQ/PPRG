from __future__ import annotations

from typing import Any

import torch

from .grad_utils import get_trainable_params, grad_vector, write_grad_vector


def primary_projected_aux_backward(
    losses: dict[str, torch.Tensor],
    model: torch.nn.Module,
    primary_domain: str = "pgd_linf_10",
    aux_domain: str = "pgd_l2_5",
    aux_weight: float = 0.25,
    eps: float = 1e-12,
) -> dict[str, Any]:
    if primary_domain not in losses:
        raise KeyError(f"Primary domain {primary_domain!r} is not in selected losses: {list(losses)}")
    if aux_domain not in losses:
        raise KeyError(f"Auxiliary domain {aux_domain!r} is not in selected losses: {list(losses)}")

    params = get_trainable_params(model)
    primary_grad = grad_vector(losses[primary_domain], params, retain_graph=True)
    aux_grad = grad_vector(losses[aux_domain], params, retain_graph=True)

    primary_norm = primary_grad.norm().clamp_min(eps)
    aux_norm = aux_grad.norm().clamp_min(eps)
    dot = torch.dot(aux_grad, primary_grad)
    cosine = dot / (primary_norm * aux_norm).clamp_min(eps)

    projected = bool(dot < 0)
    projection_norm = 0.0
    aux_projected = aux_grad
    if projected:
        projection = dot / torch.dot(primary_grad, primary_grad).clamp_min(eps) * primary_grad
        aux_projected = aux_grad - projection
        projection_norm = float(projection.norm().detach().cpu().item())

    final_grad = primary_grad + aux_weight * aux_projected
    write_grad_vector(params, final_grad)

    scalar_proxy = losses[primary_domain].detach() + aux_weight * losses[aux_domain].detach()
    return {
        "primary_domain": primary_domain,
        "aux_domain": aux_domain,
        "aux_weight": aux_weight,
        "primary_grad_norm": float(primary_grad.norm().detach().cpu().item()),
        "aux_grad_norm": float(aux_grad.norm().detach().cpu().item()),
        "aux_projected_grad_norm": float(aux_projected.norm().detach().cpu().item()),
        "primary_aux_dot": float(dot.detach().cpu().item()),
        "primary_aux_cosine": float(cosine.detach().cpu().item()),
        "primary_aux_projected": projected,
        "primary_aux_projection_norm": projection_norm,
        "grad_norm": float(final_grad.norm().detach().cpu().item()),
        "loss_value": float(scalar_proxy.cpu().item()),
    }
