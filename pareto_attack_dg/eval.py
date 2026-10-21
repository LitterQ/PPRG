from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

from pareto_attack_dg.attacks import canonical_attack_name, generate_attack_domain
from pareto_attack_dg.models import build_model
from pareto_attack_dg.utils.checkpoint import load_model_state
from pareto_attack_dg.utils.config import parse_float_like
from pareto_attack_dg.utils.data import (
    build_dataloaders,
    num_classes_for_dataset,
    shutdown_dataloader_workers,
)
from pareto_attack_dg.utils.domains import metric_key
from pareto_attack_dg.utils.logger import ensure_dir, save_json
from pareto_attack_dg.utils.metrics import num_correct

DEFAULT_EVAL_ATTACKS = [
    "clean",
    "fgsm_linf",
    "pgd_linf_20",
    "pgd_linf_50",
    "cw_linf_20",
    "cw_linf_50",
    "autoattack_linf",
    "pgd_l2_20",
    "pgd_l2_50",
    "cw_l2_20",
    "cw_l2_50",
    "autoattack_l2",
]
LINF_METRIC_KEYS = [
    "fgsm_linf_acc",
    "pgd_linf_20_acc",
    "pgd_linf_50_acc",
    "cw_linf_20_acc",
    "cw_linf_50_acc",
    "autoattack_linf_acc",
]
L2_METRIC_KEYS = [
    "pgd_l2_20_acc",
    "pgd_l2_50_acc",
    "cw_l2_20_acc",
    "cw_l2_50_acc",
    "autoattack_l2_acc",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CIFAR adversarial robustness.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument(
        "--model",
        default=None,
        help="Architecture override. Defaults to checkpoint metadata (or resnet18 for legacy raw checkpoints).",
    )
    parser.add_argument("--eval_attacks", nargs="+", default=DEFAULT_EVAL_ATTACKS)
    parser.add_argument("--eps_linf", default="8/255")
    parser.add_argument("--eps_l2", default="0.5")
    parser.add_argument("--alpha_linf", default="2/255")
    parser.add_argument("--alpha_l2", default="0.1")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--no_download", action="store_true")
    parser.add_argument("--use_fake_data", action="store_true")
    parser.add_argument("--fake_train_size", type=int, default=512)
    parser.add_argument("--fake_test_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=-1,
        help="Attack RNG seed. Negative uses the checkpoint seed (or zero for legacy checkpoints).",
    )
    parser.add_argument("--save_json", default="")
    parser.add_argument("--device", default="auto")
    return parser


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def stable_attack_seed(base_seed: int, attack: str) -> int:
    digest = hashlib.sha256(f"paper-eval-v1:{base_seed}:{attack}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def seed_attack(base_seed: int, attack: str) -> int:
    seed = stable_attack_seed(base_seed, attack)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def evaluate_attack(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    attack: str,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    model.eval()
    attack = canonical_attack_name(attack)
    args.attack_seeds[attack] = seed_attack(args.eval_seed_resolved, attack)
    correct = 0
    total = 0
    for batch_idx, (x, y) in enumerate(loader):
        if args.max_eval_batches and batch_idx >= args.max_eval_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if getattr(args, "paper_official_linf", False) and (
            attack.startswith("pgd_linf_") or attack.startswith("cw_linf_")
        ):
            from vit_12_runs_task.paper_recipe.eval_attacks import official_linf_attack

            prefix = "pgd_linf_" if attack.startswith("pgd_linf_") else "cw_linf_"
            steps = int(attack.removeprefix(prefix))
            x_adv = official_linf_attack(
                model,
                x,
                y,
                eps=args.eps_linf_float,
                alpha=args.alpha_linf_float,
                steps=steps,
                objective="ce" if prefix == "pgd_linf_" else "cw",
            )
        else:
            x_adv = generate_attack_domain(
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
            logits = model(x_adv)
        correct += num_correct(logits, y)
        total += y.numel()
    if not args.max_eval_batches:
        expected = len(loader.dataset)
        if total != expected:
            raise RuntimeError(
                f"Incomplete {attack} evaluation: processed {total} samples, expected {expected}."
            )
    args.evaluated_samples[attack] = total
    return correct / max(1, total)


def evaluate_autoattack(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    attack: str,
    device: torch.device,
    args: argparse.Namespace,
) -> float | None:
    try:
        from autoattack import AutoAttack
    except Exception as exc:
        print(f"warning: AutoAttack is unavailable ({exc}); {attack} accuracy set to null.")
        return None

    norm = "Linf" if attack in {"autoattack_linf", "square"} else "L2"
    eps = args.eps_linf_float if norm == "Linf" else args.eps_l2_float
    attack_seed = seed_attack(args.eval_seed_resolved, attack)
    args.attack_seeds[attack] = attack_seed
    adversary = AutoAttack(
        model,
        norm=norm,
        eps=eps,
        seed=attack_seed,
        version="standard",
        device=str(device),
    )
    if attack == "square":
        adversary.attacks_to_run = ["square"]

    model.eval()
    images: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for batch_idx, (x, y) in enumerate(loader):
        if args.max_eval_batches and batch_idx >= args.max_eval_batches:
            break
        images.append(x.cpu())
        targets.append(y.cpu())
    if not images:
        raise RuntimeError(f"No samples were available for {attack} evaluation.")
    x_all = torch.cat(images, dim=0)
    y_all = torch.cat(targets, dim=0)
    total = int(y_all.numel())
    if not args.max_eval_batches:
        expected = len(loader.dataset)
        if total != expected:
            raise RuntimeError(
                f"Incomplete {attack} evaluation: collected {total} samples, expected {expected}."
            )
    # Canonical AutoAttack usage: pass the complete evaluated set once and let
    # the suite apply its own robust-subset batching internally.
    x_adv = adversary.run_standard_evaluation(x_all, y_all, bs=args.batch_size)
    correct = 0
    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        with torch.no_grad():
            logits = model(x_adv[start:end].to(device, non_blocking=True))
        correct += num_correct(logits, y_all[start:end].to(device, non_blocking=True))
    args.evaluated_samples[attack] = total
    return correct / max(1, total)


def _mean_available(metrics: dict[str, float | None], keys: list[str]) -> float | None:
    values = [metrics[key] for key in keys if metrics.get(key) is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def compute_derived_metrics(metrics: dict[str, float | None]) -> dict[str, float | None]:
    robust_keys = [*LINF_METRIC_KEYS, *L2_METRIC_KEYS]
    robust_values = [metrics[key] for key in robust_keys if metrics.get(key) is not None]

    mean_linf = _mean_available(metrics, LINF_METRIC_KEYS)
    mean_l2 = _mean_available(metrics, L2_METRIC_KEYS)
    mean_robust = float(sum(robust_values) / len(robust_values)) if robust_values else None
    worst_robust = float(min(robust_values)) if robust_values else None
    gap = mean_l2 - mean_linf if mean_linf is not None and mean_l2 is not None else None
    return {
        "mean_linf_acc": mean_linf,
        "mean_l2_acc": mean_l2,
        "mean_robust_acc": mean_robust,
        "worst_robust_acc": worst_robust,
        "linf_l2_gap": gap,
    }


def evaluation_environment(*, include_autoattack: bool) -> dict[str, Any]:
    import timm
    import torchvision

    from vit_12_runs_task.paper_recipe.integrity import canonical_json, file_sha256

    payload: dict[str, Any] = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
    }
    if include_autoattack:
        import autoattack

        package_root = Path(autoattack.__file__).resolve().parent
        files = {
            str(path.relative_to(package_root).as_posix()): file_sha256(path)
            for path in sorted(package_root.rglob("*.py"))
        }
        if not files:
            raise RuntimeError(f"AutoAttack package contains no Python sources: {package_root}")
        payload["autoattack"] = {
            "package_root": str(package_root),
            "version": getattr(autoattack, "__version__", None),
            "files": files,
        }
    payload["environment_hash"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return payload


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = load_model_state(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    num_classes = num_classes_for_dataset(args.dataset)
    checkpoint_model = config.get("model_alias") or config.get("model")
    requested_model = args.model
    if requested_model and checkpoint_model:
        requested_normalized = requested_model.lower().replace("-", "_")
        checkpoint_normalized = str(checkpoint_model).lower().replace("-", "_")
        if requested_normalized != checkpoint_normalized:
            raise ValueError(
                f"--model {requested_model!r} conflicts with checkpoint architecture {checkpoint_model!r}."
            )
    resolved_model = requested_model or checkpoint_model or "resnet18"
    # Evaluation must never fetch or apply upstream pretrained weights: the local
    # checkpoint contains the complete wrapper and backbone state.
    model = build_model(str(resolved_model), num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    args.model_resolved = str(resolved_model)
    return model, config


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.eps_linf_float = parse_float_like(args.eps_linf)
    args.eps_l2_float = parse_float_like(args.eps_l2)
    args.alpha_linf_float = parse_float_like(args.alpha_linf)
    args.alpha_l2_float = parse_float_like(args.alpha_l2)
    device = choose_device(args.device)

    model, checkpoint_config = load_model(args, device)
    args.paper_official_linf = checkpoint_config.get("recipe") == "paper"
    args.eval_seed_resolved = int(
        args.eval_seed if args.eval_seed >= 0 else checkpoint_config.get("seed", 0)
    )
    args.attack_seeds: dict[str, int] = {}
    args.evaluated_samples: dict[str, int] = {}
    _, test_loader, _ = build_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_fake_data=args.use_fake_data,
        fake_train_size=args.fake_train_size,
        fake_test_size=args.fake_test_size,
        download=not args.no_download,
    )
    atexit.register(shutdown_dataloader_workers, test_loader)

    metrics: dict[str, float | None] = {metric_key(attack): None for attack in DEFAULT_EVAL_ATTACKS}
    for attack in args.eval_attacks:
        attack = canonical_attack_name(attack)
        key = metric_key(attack)
        if attack in {"autoattack_linf", "autoattack_l2", "square"}:
            metrics[key] = evaluate_autoattack(model, test_loader, attack, device, args)
        else:
            metrics[key] = evaluate_attack(model, test_loader, attack, device, args)
        print(f"{key}: {metrics[key]}")

    from vit_12_runs_task.paper_recipe.integrity import file_sha256, implementation_manifest

    manifest = implementation_manifest()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    environment = evaluation_environment(
        include_autoattack=any(
            canonical_attack_name(attack) in {"autoattack_linf", "autoattack_l2", "square"}
            for attack in args.eval_attacks
        )
    )
    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "dataset": args.dataset,
        "model": args.model_resolved,
        "eps_linf": args.eps_linf_float,
        "eps_l2": args.eps_l2_float,
        "alpha_linf": args.alpha_linf_float,
        "alpha_l2": args.alpha_l2_float,
        "checkpoint_config": checkpoint_config,
        "evaluation_implementation_hash": manifest["implementation_hash"],
        "evaluation_implementation_files": manifest["files"],
        "evaluation_environment": environment,
        "eval_seed": args.eval_seed_resolved,
        "attack_seeds": args.attack_seeds,
        "attacks_requested": [canonical_attack_name(attack) for attack in args.eval_attacks],
        "evaluation_semantics": {
            "paper_official_linf": args.paper_official_linf,
            "official_source_commit": checkpoint_config.get("paper_recipe", {})
            .get("official_source", {})
            .get("commit"),
            "autoattack_invocation": "single_complete_evaluated_set_call_with_internal_batching",
            "note": "Paper checkpoints use official correct-only, max-loss PGD/CW Linf semantics; legacy checkpoints are unchanged.",
        },
        "dataloader": {
            "num_workers": args.num_workers,
            "persistent_workers": args.num_workers > 0,
            "use_fake_data": bool(args.use_fake_data),
            "fake_test_size": args.fake_test_size if args.use_fake_data else None,
            "data_dir": str(Path(args.data_dir).expanduser().resolve()),
            "download_requested": not args.no_download,
            "expected_samples_per_attack": len(test_loader.dataset),
            "max_eval_batches": args.max_eval_batches,
            "actual_samples_per_attack": args.evaluated_samples,
        },
        "metrics": metrics,
        "derived_metrics": compute_derived_metrics(metrics),
    }

    output = args.save_json
    if output:
        ensure_dir(Path(output).parent)
        save_json(output, payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    shutdown_dataloader_workers(test_loader)
    atexit.unregister(shutdown_dataloader_workers)


if __name__ == "__main__":
    main()
