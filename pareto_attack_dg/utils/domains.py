from __future__ import annotations

from pareto_attack_dg.attacks import canonical_attack_name


def canonical_domains(domains: list[str]) -> list[str]:
    return [canonical_attack_name(domain) for domain in domains]


def metric_key(domain: str) -> str:
    domain = canonical_attack_name(domain)
    if domain == "clean":
        return "clean_acc"
    return f"{domain}_acc"


def log_alias(domain: str) -> str:
    domain = canonical_attack_name(domain)
    aliases = {
        "clean": "clean",
        "fgsm_linf": "fgsm",
        "pgd_linf_2": "pgd2",
        "pgd_linf_5": "pgd5",
        "pgd_linf_20": "pgd20",
    }
    return aliases.get(domain, domain)
