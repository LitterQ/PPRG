from __future__ import annotations

from typing import Any

import torch

from .grad_utils import apply_preference_weights, get_trainable_params, grad_matrix, write_grad_vector


def project_to_simplex(v: torch.Tensor) -> torch.Tensor:
    if v.dim() != 1:
        raise ValueError("Simplex projection expects a 1D tensor.")
    if v.numel() == 1:
        return torch.ones_like(v)

    sorted_v, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(sorted_v, dim=0) - 1.0
    ind = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
    cond = sorted_v - cssv / ind > 0
    if not torch.any(cond):
        return torch.full_like(v, 1.0 / v.numel())
    rho = torch.nonzero(cond, as_tuple=False)[-1, 0]
    theta = cssv[rho] / (rho.to(dtype=v.dtype) + 1.0)
    return torch.clamp(v - theta, min=0.0)


def solve_min_norm_weights(
    grad_matrix: torch.Tensor,
    max_iter: int = 250,
    tol: float = 1e-6,
) -> torch.Tensor:
    if grad_matrix.dim() != 2:
        raise ValueError("grad_matrix must have shape [num_objectives, num_parameters].")
    num_objectives = grad_matrix.shape[0]
    if num_objectives == 1:
        return torch.ones(1, device=grad_matrix.device, dtype=grad_matrix.dtype)

    q_mat = grad_matrix @ grad_matrix.T
    q_mat = 0.5 * (q_mat + q_mat.T)
    alpha = torch.full((num_objectives,), 1.0 / num_objectives, device=grad_matrix.device, dtype=grad_matrix.dtype)

    try:
        spectral_norm = torch.linalg.matrix_norm(q_mat, ord=2)
    except RuntimeError:
        spectral_norm = torch.trace(q_mat).abs()
    lipschitz = (2.0 * spectral_norm).clamp_min(1e-12)
    step = 1.0 / lipschitz

    for _ in range(max_iter):
        grad_alpha = 2.0 * (q_mat @ alpha)
        new_alpha = project_to_simplex(alpha - step * grad_alpha)
        if torch.norm(new_alpha - alpha) < tol:
            alpha = new_alpha
            break
        alpha = new_alpha
    return alpha


def mgda_gradient(
    losses: list[torch.Tensor],
    model: torch.nn.Module,
    objective_names: list[str] | None = None,
    preference_weights: list[float] | None = None,
    max_iter: int = 250,
    tol: float = 1e-6,
) -> tuple[list[torch.nn.Parameter], torch.Tensor, dict[str, Any]]:
    if not losses:
        raise ValueError("MGDA requires at least one loss.")
    params = get_trainable_params(model)
    names = objective_names or [f"objective_{idx}" for idx in range(len(losses))]
    raw_grad_mat = grad_matrix(losses, params, retain_graph=True)
    weighted_grad_mat = apply_preference_weights(raw_grad_mat, preference_weights)
    alpha = solve_min_norm_weights(weighted_grad_mat, max_iter=max_iter, tol=tol)
    final_grad = torch.sum(alpha.reshape(-1, 1) * weighted_grad_mat, dim=0)

    info: dict[str, Any] = {
        "mgda_alpha": {name: float(value.detach().cpu().item()) for name, value in zip(names, alpha)},
        "objective_names": names,
        "grad_norm": float(final_grad.norm().detach().cpu().item()),
        "objective_grad_norms": {
            name: float(grad.norm().detach().cpu().item()) for name, grad in zip(names, weighted_grad_mat)
        },
    }
    return params, final_grad, info


def mgda_backward(
    losses: list[torch.Tensor],
    model: torch.nn.Module,
    objective_names: list[str] | None = None,
    preference_weights: list[float] | None = None,
) -> dict[str, Any]:
    params, final_grad, info = mgda_gradient(
        losses=losses,
        model=model,
        objective_names=objective_names,
        preference_weights=preference_weights,
    )
    write_grad_vector(params, final_grad)
    info["loss_value"] = float(torch.stack([loss.detach() for loss in losses]).mean().cpu().item())
    return info
