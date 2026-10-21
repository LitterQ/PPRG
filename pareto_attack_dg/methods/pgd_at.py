from __future__ import annotations

import torch

from pareto_attack_dg.attacks import canonical_attack_name


def pgd_at_loss(losses: dict[str, torch.Tensor], train_attack: str) -> torch.Tensor:
    domain = canonical_attack_name(train_attack)
    if domain not in losses:
        raise KeyError(f"PGD-AT requires {domain} loss.")
    return losses[domain]
