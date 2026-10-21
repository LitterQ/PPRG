from __future__ import annotations

import torch


def get_trainable_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def flatten_grads(
    grads: list[torch.Tensor | None] | tuple[torch.Tensor | None, ...],
    params: list[torch.nn.Parameter],
) -> torch.Tensor:
    flat_parts: list[torch.Tensor] = []
    for grad, param in zip(grads, params):
        if grad is None:
            # Missing gradients are represented as zeros for this objective.
            # This keeps all objective vectors aligned in the same parameter space.
            flat_parts.append(torch.zeros_like(param).reshape(-1))
        else:
            flat_parts.append(grad.detach().reshape(-1))
    if not flat_parts:
        return torch.empty(0)
    return torch.cat(flat_parts)


def grad_vector(
    loss: torch.Tensor,
    params: list[torch.nn.Parameter],
    retain_graph: bool = True,
) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    return flatten_grads(grads, params)


def grad_matrix(
    losses: list[torch.Tensor],
    params: list[torch.nn.Parameter],
    retain_graph: bool = True,
) -> torch.Tensor:
    if not losses:
        raise ValueError("At least one loss is required to build a gradient matrix.")
    return torch.stack([grad_vector(loss, params, retain_graph=retain_graph) for loss in losses], dim=0)


def write_grad_vector(params: list[torch.nn.Parameter], flat_grad: torch.Tensor) -> None:
    offset = 0
    for param in params:
        numel = param.numel()
        param.grad = flat_grad[offset : offset + numel].view_as(param).clone()
        offset += numel


def apply_preference_weights(grad_mat: torch.Tensor, preference_weights: list[float] | None) -> torch.Tensor:
    if preference_weights is None:
        return grad_mat
    weights = torch.as_tensor(preference_weights, device=grad_mat.device, dtype=grad_mat.dtype)
    if weights.numel() != grad_mat.shape[0]:
        raise ValueError("Number of preference weights must match number of objectives.")
    # We apply user preference weights to gradients after autograd. This is
    # equivalent to scaling each scalar objective before differentiation.
    return grad_mat * weights.reshape(-1, 1)
