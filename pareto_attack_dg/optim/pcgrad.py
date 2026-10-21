from __future__ import annotations

import random
from itertools import combinations
from typing import Any

import torch

from .grad_utils import apply_preference_weights, get_trainable_params, grad_matrix, write_grad_vector


def pcgrad_aggregate(grad_mat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if grad_mat.shape[0] == 1:
        return grad_mat[0]

    corrected = grad_mat.clone()
    num_objectives = corrected.shape[0]
    for i in range(num_objectives):
        order = list(range(num_objectives))
        random.shuffle(order)
        for j in order:
            if i == j:
                continue
            dot = torch.dot(corrected[i], grad_mat[j])
            if dot < 0:
                denom = torch.dot(grad_mat[j], grad_mat[j]).clamp_min(eps)
                corrected[i] = corrected[i] - dot / denom * grad_mat[j]
    return corrected.mean(dim=0)


def pcgrad_gradient(
    losses: list[torch.Tensor],
    model: torch.nn.Module,
    objective_names: list[str] | None = None,
    preference_weights: list[float] | None = None,
) -> tuple[list[torch.nn.Parameter], torch.Tensor, dict[str, Any]]:
    if not losses:
        raise ValueError("PCGrad requires at least one loss.")
    params = get_trainable_params(model)
    names = objective_names or [f"objective_{idx}" for idx in range(len(losses))]
    raw_grad_mat = grad_matrix(losses, params, retain_graph=True)
    weighted_grad_mat = apply_preference_weights(raw_grad_mat, preference_weights)
    final_grad = pcgrad_aggregate(weighted_grad_mat)
    info: dict[str, Any] = {
        "objective_names": names,
        "grad_norm": float(final_grad.norm().detach().cpu().item()),
        "objective_grad_norms": {
            name: float(grad.norm().detach().cpu().item()) for name, grad in zip(names, weighted_grad_mat)
        },
    }
    return params, final_grad, info


def pcgrad_backward(
    losses: list[torch.Tensor],
    model: torch.nn.Module,
    objective_names: list[str] | None = None,
    preference_weights: list[float] | None = None,
) -> dict[str, Any]:
    params, final_grad, info = pcgrad_gradient(
        losses=losses,
        model=model,
        objective_names=objective_names,
        preference_weights=preference_weights,
    )
    write_grad_vector(params, final_grad)
    info["loss_value"] = float(torch.stack([loss.detach() for loss in losses]).mean().cpu().item())
    return info


def anchored_pcgrad_backward(
    losses: dict[str, torch.Tensor],
    model: torch.nn.Module,
    anchor_domain: str = "pgd_linf_10",
    anchor_weight: float = 1.0,
    project_auxiliaries: bool = True,
    aux_pcgrad: bool = True,
    preference_weights: dict[str, float] | None = None,
    eps: float = 1e-12,
) -> dict[str, Any]:
    if anchor_domain not in losses:
        raise KeyError(f"Anchor domain {anchor_domain!r} is not in selected losses: {list(losses)}")

    params = get_trainable_params(model)
    domains = list(losses.keys())
    raw_grads = grad_matrix([losses[domain] for domain in domains], params, retain_graph=True)

    if preference_weights:
        weights = [float(preference_weights.get(domain, 1.0)) for domain in domains]
        grads = apply_preference_weights(raw_grads, weights)
    else:
        grads = raw_grads

    anchor_idx = domains.index(anchor_domain)
    anchor_grad = grads[anchor_idx].clone()
    anchor_norm = anchor_grad.norm().clamp_min(eps)
    anchor_norm_sq = torch.dot(anchor_grad, anchor_grad).clamp_min(eps)

    aux_domains: list[str] = []
    aux_grads: list[torch.Tensor] = []
    conflict_rows: list[dict[str, float | str | bool]] = []
    projection_norms: list[float] = []
    conflicts = 0

    for idx, domain in enumerate(domains):
        if idx == anchor_idx:
            continue
        grad = grads[idx].clone()
        dot = torch.dot(grad, anchor_grad)
        cosine = dot / (grad.norm() * anchor_norm).clamp_min(eps)
        projected = bool(project_auxiliaries and dot < 0)
        projection_norm = 0.0
        if projected:
            projection = dot / anchor_norm_sq * anchor_grad
            grad = grad - projection
            projection_norm = float(projection.norm().detach().cpu().item())
            conflicts += 1
        projection_norms.append(projection_norm)
        conflict_rows.append(
            {
                "anchor_domain": anchor_domain,
                "domain": domain,
                "cosine_with_anchor": float(cosine.detach().cpu().item()),
                "dot_with_anchor": float(dot.detach().cpu().item()),
                "projected": projected,
                "projection_norm": projection_norm,
            }
        )
        aux_domains.append(domain)
        aux_grads.append(grad)

    if aux_grads:
        aux_matrix = torch.stack(aux_grads, dim=0)
        aux_agg = pcgrad_aggregate(aux_matrix) if aux_pcgrad else aux_matrix.mean(dim=0)
        # Existing PCGrad averages objective gradients. Treat the anchor as one
        # weighted objective and the auxiliary aggregate as num_aux objectives.
        final_grad = (anchor_weight * anchor_grad + len(aux_grads) * aux_agg) / (anchor_weight + len(aux_grads))
    else:
        final_grad = anchor_grad

    write_grad_vector(params, final_grad)
    cos_by_domain = {f"anchor_cos_{row['domain']}": row["cosine_with_anchor"] for row in conflict_rows}
    info: dict[str, Any] = {
        "anchor_domain": anchor_domain,
        "anchor_grad_norm": float(anchor_grad.norm().detach().cpu().item()),
        "anchor_conflict_fraction": conflicts / max(1, len(aux_grads)),
        "anchor_projection_norm_mean": sum(projection_norms) / max(1, len(projection_norms)),
        "anchor_conflict_rows": conflict_rows,
        "grad_norm": float(final_grad.norm().detach().cpu().item()),
        "objective_names": domains,
        "objective_grad_norms": {
            domain: float(grad.norm().detach().cpu().item()) for domain, grad in zip(domains, grads)
        },
        "loss_value": float(torch.stack([loss.detach() for loss in losses.values()]).mean().cpu().item()),
    }
    info.update(cos_by_domain)
    return info


def gradient_cosine_pairs(
    model: torch.nn.Module,
    losses_by_domain: dict[str, torch.Tensor],
    eps: float = 1e-12,
) -> list[dict[str, float | str]]:
    domains = list(losses_by_domain.keys())
    if len(domains) < 2:
        return []

    params = get_trainable_params(model)
    grads = grad_matrix([losses_by_domain[d] for d in domains], params, retain_graph=True)
    rows: list[dict[str, float | str]] = []
    for i, j in combinations(range(len(domains)), 2):
        gi = grads[i]
        gj = grads[j]
        cosine = torch.dot(gi, gj) / (gi.norm() * gj.norm()).clamp_min(eps)
        rows.append(
            {
                "domain_i": domains[i],
                "domain_j": domains[j],
                "cosine": float(cosine.detach().cpu().item()),
            }
        )
    return rows
