from __future__ import annotations

import argparse
import atexit
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision

from pareto_attack_dg.attacks import canonical_attack_name, generate_attack_domain
from pareto_attack_dg.methods import erm_loss, fgsm_at_loss, pgd_at_loss, rex_loss
from pareto_attack_dg.models import build_model
from pareto_attack_dg.optim import gradient_cosine_pairs
from pareto_attack_dg.optim.grad_utils import get_trainable_params, grad_vector, write_grad_vector
from pareto_attack_dg.optim.mgda import mgda_backward, mgda_gradient
from pareto_attack_dg.optim.pcgrad import anchored_pcgrad_backward, pcgrad_backward, pcgrad_gradient
from pareto_attack_dg.optim.projected_aux import primary_projected_aux_backward
from pareto_attack_dg.utils.checkpoint import save_checkpoint
from pareto_attack_dg.utils.config import parse_float_like
from pareto_attack_dg.utils.data import build_dataloaders, shutdown_dataloader_workers
from pareto_attack_dg.utils.domains import canonical_domains
from pareto_attack_dg.utils.logger import (
    append_csv,
    append_gradient_rows,
    ensure_dir,
    save_json,
    update_gradient_summary,
)
from pareto_attack_dg.utils.metrics import AverageMeter, accuracy_from_logits, num_correct
from pareto_attack_dg.utils.seed import set_seed
from vit_12_runs_task.model.deit_cifar import initial_trainable_parameters_hash, model_provenance

DEFAULT_ATTACK_DOMAINS = ["clean", "fgsm_linf", "pgd_linf_2", "pgd_linf_5", "pgd_l2_5"]
NEW_METHOD_DEFAULT_DOMAINS = {
    "avg_attack_dg": ["clean", "fgsm_linf", "pgd_linf_5", "pgd_l2_5"],
    "pcgrad_adv_only": ["clean", "fgsm_linf", "pgd_linf_5", "pgd_l2_5"],
    "weighted_pcgrad_linf": ["clean", "fgsm_linf", "pgd_linf_5", "pgd_l2_5"],
    "pcgrad_stronger_linf": ["clean", "fgsm_linf", "pgd_linf_5", "pgd_linf_10", "pgd_l2_5"],
    "mgda_attack_dg": ["clean", "fgsm_linf", "pgd_linf_5", "pgd_l2_5"],
    "mgda_adv_only": ["clean", "fgsm_linf", "pgd_linf_5", "pgd_l2_5"],
    "weighted_mgda_linf": ["clean", "fgsm_linf", "pgd_linf_5", "pgd_l2_5"],
    "mgda_stronger_linf": ["clean", "fgsm_linf", "pgd_linf_5", "pgd_linf_10", "pgd_l2_5"],
    "anchored_pcgrad_linf": ["clean", "fgsm_linf", "pgd_linf_5", "pgd_linf_10", "pgd_l2_5"],
    "curriculum_linf10_then_pcgrad_stronger": ["clean", "fgsm_linf", "pgd_linf_5", "pgd_linf_10", "pgd_l2_5"],
    "primary_projected_aux_l2": ["pgd_linf_10", "pgd_l2_5"],
}
DEFAULT_LINF_PREFERENCE_WEIGHTS = {
    "clean": 0.25,
    "fgsm_linf": 0.5,
    "pgd_linf_5": 2.0,
    "pgd_l2_5": 1.0,
}
COMMON_LOG_DOMAINS = ["clean", "fgsm_linf", "pgd_linf_2", "pgd_linf_5", "pgd_linf_10", "pgd_l2_5"]
TRAIN_LOG_FIELDS = [
    "epoch",
    "train_loss",
    "train_batches_seen",
    "train_samples_seen",
    "train_clean_loss",
    "train_fgsm_linf_loss",
    "train_fgsm_loss",
    "train_pgd2_loss",
    "train_pgd_linf_2_loss",
    "train_pgd5_loss",
    "train_pgd_linf_5_loss",
    "train_pgd_linf_10_loss",
    "train_pgd_l2_5_loss",
    "train_clean_acc",
    "train_fgsm_linf_acc",
    "train_fgsm_acc",
    "train_pgd2_acc",
    "train_pgd_linf_2_acc",
    "train_pgd5_acc",
    "train_pgd_linf_5_acc",
    "train_pgd_linf_10_acc",
    "train_pgd_l2_5_acc",
    "test_clean_acc",
    "test_pgd20_acc",
    "test_pgd_linf_20_acc",
    "test_pgd_l2_20_acc",
    "mgda_alpha_clean",
    "mgda_alpha_fgsm_linf",
    "mgda_alpha_pgd_linf_5",
    "mgda_alpha_pgd_linf_10",
    "mgda_alpha_pgd_l2_5",
    "pref_weight_clean",
    "pref_weight_fgsm_linf",
    "pref_weight_pgd_linf_5",
    "pref_weight_pgd_linf_10",
    "pref_weight_pgd_l2_5",
    "anchor_domain",
    "anchor_grad_norm",
    "anchor_conflict_fraction",
    "anchor_projection_norm_mean",
    "anchor_cos_clean",
    "anchor_cos_fgsm_linf",
    "anchor_cos_pgd_linf_5",
    "anchor_cos_pgd_l2_5",
    "curriculum_stage",
    "curriculum_warmup_epochs",
    "curriculum_stage1_attack",
    "curriculum_stage2_method",
    "primary_domain",
    "aux_domain",
    "aux_weight",
    "primary_grad_norm",
    "aux_grad_norm",
    "primary_aux_cosine",
    "primary_aux_projected",
    "primary_aux_projection_norm",
    "lr",
    "time_sec",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train attack-domain DG methods on CIFAR.")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load the official ImageNet initialization (DeiT only).",
    )
    parser.add_argument(
        "--pretrained_checkpoint",
        default="",
        help="Project-local source checkpoint to adapt with timm (implies --pretrained).",
    )
    parser.add_argument("--method", default="pcgrad_attack_dg")
    parser.add_argument("--attack_domains", nargs="+", default=None)
    parser.add_argument("--train_attack", default="pgd_linf_5")
    parser.add_argument("--clean_weight", type=float, default=0.25)
    parser.add_argument("--domain_weights", default="")
    parser.add_argument("--primary_domain", default="pgd_linf_10")
    parser.add_argument("--aux_domain", default="pgd_l2_5")
    parser.add_argument("--aux_weight", type=float, default=0.25)
    parser.add_argument(
        "--normalize_domain_weights",
        "--normalize-domain-weights",
        dest="normalize_domain_weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize weighted average losses by the sum of domain weights. Disable for exact weighted sums.",
    )
    parser.add_argument("--anchor_domain", default="pgd_linf_10")
    parser.add_argument("--anchor_weight", type=float, default=1.0)
    parser.add_argument("--anchor_project_auxiliaries", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--anchor_aux_pcgrad", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--curriculum_warmup_epochs", type=int, default=40)
    parser.add_argument("--curriculum_stage1_attack", default="pgd_linf_10")
    parser.add_argument("--curriculum_stage2_method", default="pcgrad_stronger_linf")
    parser.add_argument(
        "--curriculum_stage2_domains",
        nargs="+",
        default=["clean", "fgsm_linf", "pgd_linf_5", "pgd_linf_10", "pgd_l2_5"],
    )
    parser.add_argument("--curriculum_reset_optimizer", action="store_true")
    parser.add_argument("--curriculum_reset_scheduler", action="store_true")
    parser.add_argument("--eps_linf", default="8/255")
    parser.add_argument("--eps_l2", default="0.5")
    parser.add_argument("--alpha_linf", default="2/255")
    parser.add_argument("--alpha_l2", default="0.1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", default="sgd")
    parser.add_argument("--scheduler", default="cosine")
    parser.add_argument("--rex_beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--save_dir", default="runs/cifar10_resnet18_pcgrad")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--no_download", action="store_true")
    parser.add_argument("--use_fake_data", action="store_true")
    parser.add_argument("--fake_train_size", type=int, default=512)
    parser.add_argument("--fake_test_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--random_start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attack_eval_mode", action="store_true")
    parser.add_argument("--grad_diag_batches", type=int, default=1)
    parser.add_argument("--device", default="auto")
    return parser


def normalize_method(method: str) -> str:
    normalized = method.lower().replace("-", "_")
    aliases = {
        "attack_rex": "rex_attack_dg",
        "attack_pcgrad": "pcgrad_attack_dg",
        "pcgrad": "pcgrad_attack_dg",
        "rex": "rex_attack_dg",
    }
    return aliases.get(normalized, normalized)


def parse_domain_weights(value: str) -> dict[str, float]:
    if not value:
        return {}
    weights: dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid --domain_weights item {item!r}; expected domain:weight.")
        domain, raw_weight = item.split(":", 1)
        weights[canonical_attack_name(domain)] = float(raw_weight)
    return weights


def default_domain_weights_for_method(method: str) -> dict[str, float]:
    if method in {"weighted_pcgrad_linf", "weighted_mgda_linf"}:
        return dict(DEFAULT_LINF_PREFERENCE_WEIGHTS)
    return {}


def resolve_domain_weights(
    selected_domains: list[str],
    parsed_weights: dict[str, float],
    method: str,
) -> dict[str, float]:
    weights = default_domain_weights_for_method(method)
    weights.update(parsed_weights)
    selected = set(selected_domains)
    unknown = sorted(domain for domain in parsed_weights if domain not in selected)
    if unknown:
        raise ValueError(f"Domain weights were provided for domains not selected for training: {unknown}")
    return {domain: float(weights.get(domain, 1.0)) for domain in selected_domains}


def weights_for_domains(domains: list[str], domain_weights: dict[str, float]) -> list[float]:
    return [float(domain_weights.get(domain, 1.0)) for domain in domains]


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def resolve_train_domains(method: str, args: argparse.Namespace) -> list[str]:
    if method == "erm":
        return ["clean"]
    if method == "fgsm_at":
        return ["fgsm_linf"]
    if method == "pgd_at":
        return [canonical_attack_name(args.train_attack)]
    if args.attack_domains is not None:
        return unique_preserve_order(canonical_domains(args.attack_domains))
    if method in NEW_METHOD_DEFAULT_DOMAINS:
        return unique_preserve_order(NEW_METHOD_DEFAULT_DOMAINS[method])
    if method in {"rex_attack_dg", "pcgrad_attack_dg"}:
        return unique_preserve_order(DEFAULT_ATTACK_DOMAINS)
    raise ValueError(f"Unsupported method: {args.method}")


def curriculum_stage_for_epoch(args: argparse.Namespace, epoch: int) -> str:
    if epoch <= args.curriculum_warmup_epochs:
        return "linf10_warmup"
    return "pcgrad_stronger"


def domains_for_epoch(method: str, args: argparse.Namespace, train_domains: list[str], epoch: int) -> list[str]:
    if method != "curriculum_linf10_then_pcgrad_stronger":
        return train_domains
    if curriculum_stage_for_epoch(args, epoch) == "linf10_warmup":
        return [canonical_attack_name(args.curriculum_stage1_attack)]
    return unique_preserve_order(canonical_domains(args.curriculum_stage2_domains))


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    if args.optimizer.lower() != "sgd":
        raise ValueError("Only SGD is implemented for the first prototype.")
    return torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer) -> Any | None:
    scheduler = args.scheduler.lower()
    if scheduler == "none":
        return None
    if scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def make_domain_losses(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    domains: list[str],
    args: argparse.Namespace,
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, float]]:
    losses: OrderedDict[str, torch.Tensor] = OrderedDict()
    accuracies: dict[str, float] = {}

    for domain in domains:
        x_domain = generate_attack_domain(
            model,
            x,
            y,
            domain,
            eps_linf=args.eps_linf_float,
            eps_l2=args.eps_l2_float,
            alpha_linf=args.alpha_linf_float,
            alpha_l2=args.alpha_l2_float,
            random_start=args.random_start,
            attack_eval_mode=args.attack_eval_mode,
        )
        logits = model(x_domain)
        losses[domain] = F.cross_entropy(logits, y)
        accuracies[domain] = accuracy_from_logits(logits, y)

    return losses, accuracies


def compute_attack_domain_losses(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    domains: list[str],
    cfg: argparse.Namespace,
) -> OrderedDict[str, torch.Tensor]:
    losses, _ = make_domain_losses(model, x, y, domains, cfg)
    return losses


def weighted_domain_loss(
    losses: OrderedDict[str, torch.Tensor],
    domain_weights: dict[str, float],
    normalize: bool = True,
) -> torch.Tensor:
    domains = list(losses.keys())
    weights = torch.as_tensor(
        weights_for_domains(domains, domain_weights),
        device=next(iter(losses.values())).device,
        dtype=next(iter(losses.values())).dtype,
    )
    stacked = torch.stack(list(losses.values()))
    weighted = torch.sum(weights * stacked)
    if normalize:
        return weighted / weights.sum().clamp_min(1e-12)
    return weighted


def weighted_detached_loss(
    losses: OrderedDict[str, torch.Tensor],
    domain_weights: dict[str, float],
    normalize: bool = True,
) -> torch.Tensor:
    detached = OrderedDict((domain, loss.detach()) for domain, loss in losses.items())
    return weighted_domain_loss(detached, domain_weights, normalize=normalize)


def adv_only_backward(
    aggregator: str,
    model: torch.nn.Module,
    losses: OrderedDict[str, torch.Tensor],
    clean_weight: float,
) -> dict[str, Any]:
    if "clean" not in losses:
        raise KeyError(f"{aggregator}_adv_only requires a clean loss for the auxiliary gradient.")
    adv_items = [(domain, loss) for domain, loss in losses.items() if domain != "clean"]
    if not adv_items:
        raise ValueError(f"{aggregator}_adv_only requires at least one adversarial domain.")

    adv_domains = [domain for domain, _ in adv_items]
    adv_losses = [loss for _, loss in adv_items]
    if aggregator == "pcgrad":
        params, adv_grad, info = pcgrad_gradient(adv_losses, model, objective_names=adv_domains)
    elif aggregator == "mgda":
        params, adv_grad, info = mgda_gradient(adv_losses, model, objective_names=adv_domains)
    else:
        raise ValueError(f"Unsupported adv-only aggregator: {aggregator}")

    clean_grad = grad_vector(losses["clean"], params, retain_graph=True)
    final_grad = adv_grad + clean_weight * clean_grad
    write_grad_vector(params, final_grad)

    clean_loss = losses["clean"].detach()
    adv_loss = torch.stack([loss.detach() for loss in adv_losses]).mean()
    info.update(
        {
            "clean_auxiliary": True,
            "clean_weight": clean_weight,
            "grad_norm": float(final_grad.norm().detach().cpu().item()),
            "objective_grad_norms": {
                **info.get("objective_grad_norms", {}),
                "clean_auxiliary": float((clean_weight * clean_grad).norm().detach().cpu().item()),
            },
            "loss_value": float((adv_loss + clean_weight * clean_loss).cpu().item()),
        }
    )
    return info


def method_backward(
    method: str,
    model: torch.nn.Module,
    losses: OrderedDict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, Any]]:
    def scalar_info(loss: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        return loss.detach(), {"loss_value": float(loss.detach().cpu().item())}

    if method == "erm":
        loss = erm_loss(losses)
        loss.backward()
        return scalar_info(loss)
    if method == "fgsm_at":
        loss = fgsm_at_loss(losses)
        loss.backward()
        return scalar_info(loss)
    if method == "pgd_at":
        loss = pgd_at_loss(losses, args.train_attack)
        loss.backward()
        return scalar_info(loss)
    if method == "rex_attack_dg":
        loss = rex_loss(losses, beta=args.rex_beta)
        loss.backward()
        return scalar_info(loss)
    if method == "avg_attack_dg":
        loss = weighted_domain_loss(
            losses,
            args.parsed_domain_weights,
            normalize=args.normalize_domain_weights,
        )
        loss.backward()
        return scalar_info(loss)
    if method in {"pcgrad_attack_dg", "pcgrad_stronger_linf"}:
        info = pcgrad_backward(list(losses.values()), model, objective_names=list(losses.keys()))
        return torch.tensor(info["loss_value"], device=next(iter(losses.values())).device), info
    if method == "weighted_pcgrad_linf":
        domains = list(losses.keys())
        info = pcgrad_backward(
            list(losses.values()),
            model,
            objective_names=domains,
            preference_weights=weights_for_domains(domains, args.parsed_domain_weights),
        )
        weighted_loss = weighted_detached_loss(losses, args.parsed_domain_weights)
        info["loss_value"] = float(weighted_loss.cpu().item())
        return weighted_loss.to(next(iter(losses.values())).device), info
    if method == "pcgrad_adv_only":
        info = adv_only_backward("pcgrad", model, losses, args.clean_weight)
        return torch.tensor(info["loss_value"], device=next(iter(losses.values())).device), info
    if method in {"mgda_attack_dg", "mgda_stronger_linf"}:
        info = mgda_backward(list(losses.values()), model, objective_names=list(losses.keys()))
        return torch.tensor(info["loss_value"], device=next(iter(losses.values())).device), info
    if method == "weighted_mgda_linf":
        domains = list(losses.keys())
        info = mgda_backward(
            list(losses.values()),
            model,
            objective_names=domains,
            preference_weights=weights_for_domains(domains, args.parsed_domain_weights),
        )
        weighted_loss = weighted_detached_loss(losses, args.parsed_domain_weights)
        info["loss_value"] = float(weighted_loss.cpu().item())
        return weighted_loss.to(next(iter(losses.values())).device), info
    if method == "mgda_adv_only":
        info = adv_only_backward("mgda", model, losses, args.clean_weight)
        return torch.tensor(info["loss_value"], device=next(iter(losses.values())).device), info
    if method == "anchored_pcgrad_linf":
        info = anchored_pcgrad_backward(
            losses,
            model,
            anchor_domain=args.anchor_domain,
            anchor_weight=args.anchor_weight,
            project_auxiliaries=args.anchor_project_auxiliaries,
            aux_pcgrad=args.anchor_aux_pcgrad,
            preference_weights=args.parsed_domain_weights,
        )
        return torch.tensor(info["loss_value"], device=next(iter(losses.values())).device), info
    if method == "curriculum_linf10_then_pcgrad_stronger":
        if args.current_curriculum_stage == "linf10_warmup":
            domain = canonical_attack_name(args.curriculum_stage1_attack)
            if domain not in losses:
                raise KeyError(f"Curriculum warm-up loss {domain} is missing.")
            loss = losses[domain]
            loss.backward()
            return scalar_info(loss)
        if args.curriculum_stage2_method != "pcgrad_stronger_linf":
            raise ValueError("Only pcgrad_stronger_linf is supported as curriculum_stage2_method.")
        info = pcgrad_backward(list(losses.values()), model, objective_names=list(losses.keys()))
        return torch.tensor(info["loss_value"], device=next(iter(losses.values())).device), info
    if method == "primary_projected_aux_l2":
        info = primary_projected_aux_backward(
            losses,
            model,
            primary_domain=args.primary_domain,
            aux_domain=args.aux_domain,
            aux_weight=args.aux_weight,
        )
        return torch.tensor(info["loss_value"], device=next(iter(losses.values())).device), info
    raise ValueError(f"Unsupported method: {method}")


def evaluate_attack_accuracy(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    attack: str,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    max_batches = args.max_eval_batches

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_eval = generate_attack_domain(
            model,
            x,
            y,
            attack,
            eps_linf=args.eps_linf_float,
            eps_l2=args.eps_l2_float,
            alpha_linf=args.alpha_linf_float,
            alpha_l2=args.alpha_l2_float,
            random_start=True,
            attack_eval_mode=False,
        )
        with torch.no_grad():
            logits = model(x_eval)
        correct += num_correct(logits, y)
        total += y.numel()

    if was_training:
        model.train()
    if not max_batches:
        expected = len(loader.dataset)
        if total != expected:
            raise RuntimeError(
                f"Incomplete {attack} evaluation: processed {total} samples, expected {expected}."
            )
    return correct / max(1, total)


def metric_or_blank(meters: dict[str, AverageMeter], domain: str) -> float | str:
    meter = meters.get(domain)
    if meter is None or meter.count == 0:
        return ""
    return meter.avg


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    train_domains: list[str],
    method: str,
    args: argparse.Namespace,
    epoch: int,
    save_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    model.train()
    method_meter = AverageMeter()
    log_domains = list(train_domains)
    if method == "curriculum_linf10_then_pcgrad_stronger":
        log_domains = unique_preserve_order(log_domains + [canonical_attack_name(args.curriculum_stage1_attack)])
    epoch_domains = domains_for_epoch(method, args, train_domains, epoch)
    curriculum_stage = curriculum_stage_for_epoch(args, epoch) if method == "curriculum_linf10_then_pcgrad_stronger" else ""
    args.current_curriculum_stage = curriculum_stage
    loss_meters = {domain: AverageMeter() for domain in log_domains}
    acc_meters = {domain: AverageMeter() for domain in log_domains}
    diag_cosines: list[float] = []
    alpha_values: dict[str, list[float]] = {}
    anchor_values: dict[str, list[float]] = {}
    primary_aux_values: dict[str, list[float]] = {}
    primary_aux_projected: list[bool] = []
    batches_seen = 0
    samples_seen = 0

    for step, (x, y) in enumerate(train_loader):
        if args.max_train_batches and step >= args.max_train_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        batch_size = y.numel()
        batches_seen += 1
        samples_seen += batch_size

        optimizer.zero_grad(set_to_none=True)
        losses, accuracies = make_domain_losses(model, x, y, epoch_domains, args)

        if args.grad_diag_batches and step < args.grad_diag_batches and len(losses) > 1:
            rows = gradient_cosine_pairs(model, losses)
            for row in rows:
                row["epoch"] = epoch
                row["step"] = step
                diag_cosines.append(float(row["cosine"]))
            append_gradient_rows(save_dir / "gradient_cosine_epoch.csv", rows)

        method_loss, method_info = method_backward(method, model, losses, args)
        optimizer.step()
        method_meter.update(float(method_loss.item()), batch_size)
        for domain, alpha in method_info.get("mgda_alpha", {}).items():
            alpha_values.setdefault(domain, []).append(float(alpha))
        for key in [
            "anchor_grad_norm",
            "anchor_conflict_fraction",
            "anchor_projection_norm_mean",
            "anchor_cos_clean",
            "anchor_cos_fgsm_linf",
            "anchor_cos_pgd_linf_5",
            "anchor_cos_pgd_l2_5",
        ]:
            if key in method_info:
                anchor_values.setdefault(key, []).append(float(method_info[key]))
        for row in method_info.get("anchor_conflict_rows", []):
            row = dict(row)
            row["epoch"] = epoch
            row["step"] = step
            append_csv(
                save_dir / "anchor_conflict_epoch.csv",
                [
                    "epoch",
                    "step",
                    "anchor_domain",
                    "domain",
                    "cosine_with_anchor",
                    "dot_with_anchor",
                    "projected",
                    "projection_norm",
                ],
                row,
            )
        for key in [
            "primary_grad_norm",
            "aux_grad_norm",
            "primary_aux_cosine",
            "primary_aux_projection_norm",
        ]:
            if key in method_info:
                primary_aux_values.setdefault(key, []).append(float(method_info[key]))
        if "primary_aux_projected" in method_info:
            primary_aux_projected.append(bool(method_info["primary_aux_projected"]))

        for domain, loss in losses.items():
            loss_meters[domain].update(float(loss.detach().item()), batch_size)
            acc_meters[domain].update(accuracies[domain], batch_size)

    if not args.max_train_batches:
        expected_batches = len(train_loader)
        expected_samples = len(train_loader.dataset)
        if batches_seen != expected_batches or samples_seen != expected_samples:
            raise RuntimeError(
                "Incomplete training epoch: "
                f"processed {batches_seen}/{expected_batches} batches and "
                f"{samples_seen}/{expected_samples} samples."
            )

    if diag_cosines:
        negatives = sum(1 for value in diag_cosines if value < 0)
        update_gradient_summary(
            save_dir / "gradient_conflict_summary.json",
            {
                "epoch": epoch,
                "mean_cosine": sum(diag_cosines) / len(diag_cosines),
                "fraction_negative_cosine": negatives / len(diag_cosines),
            },
        )

    alpha_means: dict[str, float] = {}
    if alpha_values:
        for domain, values in alpha_values.items():
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            alpha_means[domain] = mean
            append_csv(
                save_dir / "mgda_alpha_epoch.csv",
                ["epoch", "domain", "alpha_mean", "alpha_std"],
                {
                    "epoch": epoch,
                    "domain": domain,
                    "alpha_mean": mean,
                    "alpha_std": variance**0.5,
                },
            )

    return {
        "train_loss": method_meter.avg,
        "train_batches_seen": batches_seen,
        "train_samples_seen": samples_seen,
        "loss_meters": loss_meters,
        "acc_meters": acc_meters,
        "mgda_alpha_means": alpha_means,
        "anchor_means": {key: sum(values) / len(values) for key, values in anchor_values.items()},
        "anchor_domain": args.anchor_domain if method == "anchored_pcgrad_linf" else "",
        "primary_aux_means": {key: sum(values) / len(values) for key, values in primary_aux_values.items()},
        "primary_aux_projected_fraction": (
            sum(1 for value in primary_aux_projected if value) / len(primary_aux_projected)
            if primary_aux_projected
            else ""
        ),
        "curriculum_stage": curriculum_stage,
    }


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = build_parser()
    args = parser.parse_args(argv)
    method = normalize_method(args.method)
    args.method = method
    args.eps_linf_float = parse_float_like(args.eps_linf)
    args.eps_l2_float = parse_float_like(args.eps_l2)
    args.alpha_linf_float = parse_float_like(args.alpha_linf)
    args.alpha_l2_float = parse_float_like(args.alpha_l2)
    args.domain_weights_user = parse_domain_weights(args.domain_weights)
    args.primary_domain = canonical_attack_name(args.primary_domain)
    args.aux_domain = canonical_attack_name(args.aux_domain)
    args.anchor_domain = canonical_attack_name(args.anchor_domain)
    args.curriculum_stage1_attack = canonical_attack_name(args.curriculum_stage1_attack)
    args.curriculum_stage2_method = normalize_method(args.curriculum_stage2_method)
    args.curriculum_stage2_domains = unique_preserve_order(canonical_domains(args.curriculum_stage2_domains))

    set_seed(args.seed, deterministic=args.deterministic)
    device = choose_device(args.device)
    save_dir = ensure_dir(args.save_dir)

    train_domains = resolve_train_domains(method, args)
    if method == "anchored_pcgrad_linf" and args.anchor_domain not in train_domains:
        raise ValueError(f"--anchor_domain {args.anchor_domain} must be in selected attack domains: {train_domains}")
    if method == "primary_projected_aux_l2":
        missing = [domain for domain in [args.primary_domain, args.aux_domain] if domain not in train_domains]
        if missing:
            raise ValueError(f"Primary/auxiliary domains {missing} must be in selected attack domains: {train_domains}")
    if method == "curriculum_linf10_then_pcgrad_stronger" and args.curriculum_stage2_method != "pcgrad_stronger_linf":
        raise ValueError("Only --curriculum_stage2_method pcgrad_stronger_linf is currently supported.")
    args.parsed_domain_weights = resolve_domain_weights(train_domains, args.domain_weights_user, method)
    train_loader, test_loader, num_classes = build_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_fake_data=args.use_fake_data,
        fake_train_size=args.fake_train_size,
        fake_test_size=args.fake_test_size,
        download=not args.no_download,
    )
    atexit.register(shutdown_dataloader_workers, train_loader, test_loader)

    pretrained_requested = bool(args.pretrained or args.pretrained_checkpoint)
    model = build_model(
        args.model,
        num_classes=num_classes,
        pretrained=pretrained_requested,
        pretrained_checkpoint=args.pretrained_checkpoint or None,
    ).to(device)
    initialization_hash = initial_trainable_parameters_hash(model)
    provenance = model_provenance(model)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    config = vars(args).copy()
    config["train_domains_resolved"] = train_domains
    config["model_alias"] = provenance.get("model_alias", args.model)
    config["upstream_timm_model_identifier"] = provenance.get("timm_model_id")
    config["pretrained_source"] = provenance.get("pretrained_source")
    config["pretrained_config"] = provenance.get("pretrained_config")
    config["input_size"] = provenance.get("input_size", [32, 32])
    config["patch_size"] = provenance.get("patch_size")
    config["normalization_mean"] = provenance.get("normalization_mean")
    config["normalization_std"] = provenance.get("normalization_std")
    config["pretrained_loaded"] = bool(provenance.get("pretrained_loaded", False))
    config["initial_trainable_parameters_hash"] = initialization_hash
    config["model_provenance"] = provenance
    config["library_versions"] = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": provenance.get("timm_version"),
    }
    config["dataloader_persistent_workers"] = args.num_workers > 0
    config["expected_train_batches_per_epoch"] = len(train_loader)
    config["expected_train_samples_per_epoch"] = len(train_loader.dataset)
    config["expected_eval_samples_per_attack"] = len(test_loader.dataset)
    config["resolved_command"] = ["python", *sys.argv]
    append_gradient_rows(save_dir / "gradient_cosine_epoch.csv", [])
    gradient_summary_path = save_dir / "gradient_conflict_summary.json"
    if not gradient_summary_path.exists():
        save_json(gradient_summary_path, {"epochs": []})
    save_json(save_dir / "config.json", config)
    if method in {"pcgrad_adv_only", "mgda_adv_only"}:
        print(f"{method}: clean loss is used only as auxiliary gradient with clean_weight={args.clean_weight}.")

    best_metric = -1.0
    for epoch in range(1, args.epochs + 1):
        if (
            method == "curriculum_linf10_then_pcgrad_stronger"
            and epoch == args.curriculum_warmup_epochs + 1
            and epoch <= args.epochs
        ):
            if args.curriculum_reset_optimizer:
                optimizer = build_optimizer(args, model)
            if args.curriculum_reset_scheduler:
                scheduler = build_scheduler(args, optimizer)
        start = time.time()
        train_stats = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            train_domains=train_domains,
            method=method,
            args=args,
            epoch=epoch,
            save_dir=save_dir,
            device=device,
        )
        if scheduler is not None:
            scheduler.step()

        test_clean_acc = evaluate_attack_accuracy(model, test_loader, "clean", device, args)
        test_pgd20_acc = evaluate_attack_accuracy(model, test_loader, "pgd_linf_20", device, args)
        test_pgd_l2_20_acc = evaluate_attack_accuracy(model, test_loader, "pgd_l2_20", device, args)
        elapsed = time.time() - start

        loss_meters: dict[str, AverageMeter] = train_stats["loss_meters"]
        acc_meters: dict[str, AverageMeter] = train_stats["acc_meters"]
        alpha_means: dict[str, float] = train_stats["mgda_alpha_means"]
        anchor_means: dict[str, float] = train_stats["anchor_means"]
        primary_aux_means: dict[str, float] = train_stats["primary_aux_means"]
        row = {
            "epoch": epoch,
            "train_loss": train_stats["train_loss"],
            "train_batches_seen": train_stats["train_batches_seen"],
            "train_samples_seen": train_stats["train_samples_seen"],
            "train_clean_loss": metric_or_blank(loss_meters, "clean"),
            "train_fgsm_linf_loss": metric_or_blank(loss_meters, "fgsm_linf"),
            "train_fgsm_loss": metric_or_blank(loss_meters, "fgsm_linf"),
            "train_pgd2_loss": metric_or_blank(loss_meters, "pgd_linf_2"),
            "train_pgd_linf_2_loss": metric_or_blank(loss_meters, "pgd_linf_2"),
            "train_pgd5_loss": metric_or_blank(loss_meters, "pgd_linf_5"),
            "train_pgd_linf_5_loss": metric_or_blank(loss_meters, "pgd_linf_5"),
            "train_pgd_linf_10_loss": metric_or_blank(loss_meters, "pgd_linf_10"),
            "train_pgd_l2_5_loss": metric_or_blank(loss_meters, "pgd_l2_5"),
            "train_clean_acc": metric_or_blank(acc_meters, "clean"),
            "train_fgsm_linf_acc": metric_or_blank(acc_meters, "fgsm_linf"),
            "train_fgsm_acc": metric_or_blank(acc_meters, "fgsm_linf"),
            "train_pgd2_acc": metric_or_blank(acc_meters, "pgd_linf_2"),
            "train_pgd_linf_2_acc": metric_or_blank(acc_meters, "pgd_linf_2"),
            "train_pgd5_acc": metric_or_blank(acc_meters, "pgd_linf_5"),
            "train_pgd_linf_5_acc": metric_or_blank(acc_meters, "pgd_linf_5"),
            "train_pgd_linf_10_acc": metric_or_blank(acc_meters, "pgd_linf_10"),
            "train_pgd_l2_5_acc": metric_or_blank(acc_meters, "pgd_l2_5"),
            "test_clean_acc": test_clean_acc,
            "test_pgd20_acc": test_pgd20_acc,
            "test_pgd_linf_20_acc": test_pgd20_acc,
            "test_pgd_l2_20_acc": test_pgd_l2_20_acc,
            "mgda_alpha_clean": alpha_means.get("clean", ""),
            "mgda_alpha_fgsm_linf": alpha_means.get("fgsm_linf", ""),
            "mgda_alpha_pgd_linf_5": alpha_means.get("pgd_linf_5", ""),
            "mgda_alpha_pgd_linf_10": alpha_means.get("pgd_linf_10", ""),
            "mgda_alpha_pgd_l2_5": alpha_means.get("pgd_l2_5", ""),
            "pref_weight_clean": args.parsed_domain_weights.get("clean", ""),
            "pref_weight_fgsm_linf": args.parsed_domain_weights.get("fgsm_linf", ""),
            "pref_weight_pgd_linf_5": args.parsed_domain_weights.get("pgd_linf_5", ""),
            "pref_weight_pgd_linf_10": args.parsed_domain_weights.get("pgd_linf_10", ""),
            "pref_weight_pgd_l2_5": args.parsed_domain_weights.get("pgd_l2_5", ""),
            "anchor_domain": train_stats.get("anchor_domain", ""),
            "anchor_grad_norm": anchor_means.get("anchor_grad_norm", ""),
            "anchor_conflict_fraction": anchor_means.get("anchor_conflict_fraction", ""),
            "anchor_projection_norm_mean": anchor_means.get("anchor_projection_norm_mean", ""),
            "anchor_cos_clean": anchor_means.get("anchor_cos_clean", ""),
            "anchor_cos_fgsm_linf": anchor_means.get("anchor_cos_fgsm_linf", ""),
            "anchor_cos_pgd_linf_5": anchor_means.get("anchor_cos_pgd_linf_5", ""),
            "anchor_cos_pgd_l2_5": anchor_means.get("anchor_cos_pgd_l2_5", ""),
            "curriculum_stage": train_stats.get("curriculum_stage", ""),
            "curriculum_warmup_epochs": args.curriculum_warmup_epochs if method == "curriculum_linf10_then_pcgrad_stronger" else "",
            "curriculum_stage1_attack": args.curriculum_stage1_attack if method == "curriculum_linf10_then_pcgrad_stronger" else "",
            "curriculum_stage2_method": args.curriculum_stage2_method if method == "curriculum_linf10_then_pcgrad_stronger" else "",
            "primary_domain": args.primary_domain if method == "primary_projected_aux_l2" else "",
            "aux_domain": args.aux_domain if method == "primary_projected_aux_l2" else "",
            "aux_weight": args.aux_weight if method == "primary_projected_aux_l2" else "",
            "primary_grad_norm": primary_aux_means.get("primary_grad_norm", ""),
            "aux_grad_norm": primary_aux_means.get("aux_grad_norm", ""),
            "primary_aux_cosine": primary_aux_means.get("primary_aux_cosine", ""),
            "primary_aux_projected": train_stats.get("primary_aux_projected_fraction", ""),
            "primary_aux_projection_norm": primary_aux_means.get("primary_aux_projection_norm", ""),
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": elapsed,
        }
        append_csv(save_dir / "train_log.csv", TRAIN_LOG_FIELDS, row)

        if test_pgd20_acc > best_metric:
            best_metric = test_pgd20_acc
            save_checkpoint(save_dir / "best.pt", model, optimizer, scheduler, epoch, best_metric, config)
        save_checkpoint(save_dir / "last.pt", model, optimizer, scheduler, epoch, best_metric, config)

        print(
            f"epoch={epoch} train_loss={train_stats['train_loss']:.4f} "
            f"clean={test_clean_acc:.4f} pgd20={test_pgd20_acc:.4f} "
            f"pgd_l2_20={test_pgd_l2_20_acc:.4f} "
            f"batches={train_stats['train_batches_seen']} "
            f"samples={train_stats['train_samples_seen']} time={elapsed:.1f}s"
        )

    shutdown_dataloader_workers(train_loader, test_loader)
    atexit.unregister(shutdown_dataloader_workers)
    print(f"done: best test_pgd20_acc={best_metric:.4f}, save_dir={save_dir}")


if __name__ == "__main__":
    main()
