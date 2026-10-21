from __future__ import annotations

from contextlib import contextmanager

import torch

from .cw import margin_pgd_l2_attack, margin_pgd_linf_attack
from .fgsm import fgm_l2_attack, fgsm_linf_attack
from .pgd import pgd_l2_attack, pgd_linf_attack

__all__ = [
    "canonical_attack_name",
    "generate_attack_domain",
    "fgsm_linf_attack",
    "fgm_l2_attack",
    "pgd_linf_attack",
    "pgd_l2_attack",
    "margin_pgd_l2_attack",
    "margin_pgd_linf_attack",
]


def canonical_attack_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    aliases = {
        "fgsm": "fgsm_linf",
        "fgm": "fgm_l2",
        "pgd2": "pgd_linf_2",
        "pgd5": "pgd_linf_5",
        "pgd10": "pgd_linf_10",
        "pgd20": "pgd_linf_20",
        "pgd50": "pgd_linf_50",
        "cw": "cw_l2_20",
        "cw20": "cw_l2_20",
        "cw50": "cw_l2_50",
        "cw_linf": "cw_linf_20",
        "cw_linf20": "cw_linf_20",
        "cw_linf50": "cw_linf_50",
        "cwlinf20": "cw_linf_20",
        "cwlinf50": "cw_linf_50",
    }
    return aliases.get(normalized, normalized)


def _parse_pgd_steps(domain: str, prefix: str) -> int:
    suffix = domain.removeprefix(prefix)
    if not suffix.isdigit():
        raise ValueError(f"PGD domain must end with step count: {domain}")
    return int(suffix)


@contextmanager
def _attack_mode(model: torch.nn.Module, attack_eval_mode: bool):
    was_training = model.training
    if attack_eval_mode:
        model.eval()
    try:
        yield
    finally:
        if attack_eval_mode and was_training:
            model.train()


def generate_attack_domain(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    domain: str,
    eps_linf: float,
    eps_l2: float,
    alpha_linf: float,
    alpha_l2: float,
    random_start: bool = True,
    attack_eval_mode: bool = False,
) -> torch.Tensor:
    domain = canonical_attack_name(domain)
    if domain == "clean":
        return x.detach()

    with _attack_mode(model, attack_eval_mode):
        if domain == "fgsm_linf":
            return fgsm_linf_attack(model, x, y, eps=eps_linf)
        if domain == "fgm_l2":
            return fgm_l2_attack(model, x, y, eps=eps_l2)
        if domain.startswith("pgd_linf_"):
            steps = _parse_pgd_steps(domain, "pgd_linf_")
            return pgd_linf_attack(
                model,
                x,
                y,
                eps=eps_linf,
                alpha=alpha_linf,
                steps=steps,
                random_start=random_start,
            )
        if domain.startswith("pgd_l2_"):
            steps = _parse_pgd_steps(domain, "pgd_l2_")
            return pgd_l2_attack(
                model,
                x,
                y,
                eps=eps_l2,
                alpha=alpha_l2,
                steps=steps,
                random_start=random_start,
            )
        if domain == "cw_l2":
            return margin_pgd_l2_attack(
                model,
                x,
                y,
                eps=eps_l2,
                alpha=alpha_l2,
                steps=20,
                random_start=random_start,
            )
        if domain.startswith("cw_l2_"):
            steps = _parse_pgd_steps(domain, "cw_l2_")
            return margin_pgd_l2_attack(
                model,
                x,
                y,
                eps=eps_l2,
                alpha=alpha_l2,
                steps=steps,
                random_start=random_start,
            )
        if domain == "cw_linf":
            return margin_pgd_linf_attack(
                model,
                x,
                y,
                eps=eps_linf,
                alpha=alpha_linf,
                steps=20,
                random_start=random_start,
            )
        if domain.startswith("cw_linf_"):
            steps = _parse_pgd_steps(domain, "cw_linf_")
            return margin_pgd_linf_attack(
                model,
                x,
                y,
                eps=eps_linf,
                alpha=alpha_linf,
                steps=steps,
                random_start=random_start,
            )

    raise ValueError(f"Unsupported attack domain: {domain}")
