from __future__ import annotations

import argparse
import atexit
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision

from pareto_attack_dg.optim.projected_aux import primary_projected_aux_backward

from .checkpoint import load_checkpoint, restore_rng_state, save_training_checkpoint
from .config import (
    METHOD_SPECS,
    METHODS,
    PAPER_MODEL_ALIAS,
    PINNED_TIMM_VERSION,
    file_sha256,
    paper_recipe_hash,
    resolved_config,
)
from .data import build_paper_dataloaders, set_paper_loader_epoch, shutdown_dataloader_workers
from .integrity import implementation_hash
from .losses import pgd_l2_soft_target, pgd_linf_soft_target, soft_target_cross_entropy, tensor_sha256
from .model import build_paper_deit_tiny_patch4_32, initial_trainable_parameters_hash, model_provenance
from .scheduler import PaperEpochLRScheduler


TASK_ROOT = Path(__file__).resolve().parents[1]
RUN_KINDS = ("smoke", "validation_only", "manual_baseline", "formal_12")


def require_under_task(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(TASK_ROOT)
    except ValueError as exc:
        raise ValueError(f"Run output must stay under {TASK_ROOT}; got {resolved}") from exc
    return resolved


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=TASK_ROOT.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def choose_device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value)


def global_grad_norm(model: torch.nn.Module) -> float:
    squares = []
    for parameter in model.parameters():
        if parameter.grad is not None:
            squares.append(parameter.grad.detach().float().norm(2).square())
    if not squares:
        return 0.0
    return float(torch.stack(squares).sum().sqrt().cpu().item())


def append_csv(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def branch_generator(device: torch.device, *, seed: int, epoch: int, batch_index: int, branch: int) -> torch.Generator:
    # Branch-local random starts keep the primary adversarial example paired
    # across methods even when PDA additionally creates an L2 example.
    value = (seed + 1) * 1_000_003 + epoch * 10_007 + batch_index * 101 + branch
    generator = torch.Generator(device=device)
    generator.manual_seed(value)
    return generator


def validate_args(args: argparse.Namespace) -> None:
    if args.run_kind in {"manual_baseline", "formal_12"}:
        if args.epochs != 40 or args.batch_size != 64 or args.use_fake_data or args.max_train_batches:
            raise ValueError("Formal paper runs require epochs=40, batch_size=64, real CIFAR-10, and a full epoch.")
    if args.run_kind == "validation_only":
        if args.epochs != 1 or args.use_fake_data or args.max_train_batches:
            raise ValueError("The real validation path is exactly one complete CIFAR-10 epoch.")
    if args.run_kind == "smoke" and not args.use_fake_data:
        raise ValueError("Smoke runs must use FakeData.")
    if args.stop_after_epoch and args.run_kind != "smoke":
        raise ValueError("--stop-after-epoch is reserved for the lightweight FakeData resume test.")
    if args.max_train_batches and args.run_kind != "smoke":
        raise ValueError("Partial epochs are allowed only for FakeData smoke tests.")
    if not 0 <= args.seed:
        raise ValueError("seed must be non-negative")
    if args.experiment not in METHODS:
        raise ValueError(f"Unknown experiment: {args.experiment}")
    checkpoint = Path(args.pretrained_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing pretrained checkpoint: {checkpoint}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the isolated official DeiT-Tiny CIFAR-10 paper recipe.")
    parser.add_argument("--experiment", choices=METHODS, required=True)
    parser.add_argument("--run-kind", choices=RUN_KINDS, required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", default="")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--use-fake-data", action="store_true")
    parser.add_argument("--fake-train-size", type=int, default=8)
    parser.add_argument("--fake-test-size", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--stop-after-epoch", type=int, default=0)
    return parser


def method_fields(experiment: str) -> dict[str, Any]:
    spec = METHOD_SPECS[experiment]
    return {
        "method": spec["method"],
        "train_attack": spec.get("train_attack", ""),
        "attack_domains": spec.get("attack_domains", ["pgd_linf_10"]),
        "parsed_domain_weights": spec.get("parsed_domain_weights", {"pgd_linf_10": 1.0}),
        "normalize_domain_weights": spec.get("normalize_domain_weights", False),
        "primary_domain": spec.get("primary_domain", "pgd_linf_10"),
        "aux_domain": spec.get("aux_domain", "pgd_l2_5"),
        "aux_weight": spec.get("aux_weight", 0.0),
    }


def create_mixup() -> Any:
    try:
        import timm
        from timm.data import Mixup
    except ImportError as exc:
        raise RuntimeError(f"Paper recipe requires project-local timm=={PINNED_TIMM_VERSION}") from exc
    if timm.__version__ != PINNED_TIMM_VERSION:
        raise RuntimeError(f"Expected timm=={PINNED_TIMM_VERSION}, found {timm.__version__}")
    return Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        cutmix_minmax=None,
        prob=0.3,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=0.0,
        num_classes=10,
    )


def validate_resume_config(current: dict[str, Any], saved: dict[str, Any]) -> None:
    checks = (
        "paper_recipe_hash",
        "experiment",
        "seed",
        "epochs",
        "batch_size",
        "pretrained_checkpoint_sha256",
        "initial_trainable_parameters_hash",
        "implementation_hash",
    )
    mismatches = {key: (saved.get(key), current.get(key)) for key in checks if saved.get(key) != current.get(key)}
    if mismatches:
        raise ValueError(f"Resume checkpoint conflicts with current resolved config: {mismatches}")


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    mixup: Any,
    experiment: str,
    seed: int,
    epoch: int,
    device: torch.device,
    max_batches: int,
    run_dir: Path,
) -> dict[str, Any]:
    model.train()
    total_loss = total_primary = total_aux = total_correct = 0.0
    samples_seen = batches_seen = clipped_batches = 0
    pre_norms: list[float] = []
    post_norms: list[float] = []
    cosines: list[float] = []
    projections: list[bool] = []
    first_hashes: dict[str, str] = {}
    batch_fields = [
        "epoch", "batch", "samples", "primary_loss", "auxiliary_loss", "combined_loss",
        "clip_pre_global_norm", "clip_post_global_norm", "clip_max_norm", "clipped",
        "primary_aux_cosine", "primary_aux_projected", "primary_aux_projection_norm",
    ]

    for batch_index, (x, hard_target) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        hard_target = hard_target.to(device, non_blocking=True)
        x_mixed, target = mixup(x, hard_target)
        if target.ndim != 2 or target.shape[1] != 10:
            raise RuntimeError(f"Mixup/CutMix did not produce [N,10] soft targets: {tuple(target.shape)}")
        if not torch.allclose(target.sum(dim=1), torch.ones_like(target[:, 0]), atol=1e-6, rtol=0):
            raise RuntimeError("Mixup/CutMix target rows do not sum to one.")

        primary_adv = pgd_linf_soft_target(
            model,
            x_mixed,
            target,
            generator=branch_generator(device, seed=seed, epoch=epoch, batch_index=batch_index, branch=1),
        )
        model.train()
        primary_logits = model(primary_adv)
        primary_loss = soft_target_cross_entropy(primary_logits, target)
        auxiliary_loss: torch.Tensor | None = None
        if experiment != "pgd_at_linf10":
            auxiliary_adv = pgd_l2_soft_target(
                model,
                x_mixed,
                target,
                generator=branch_generator(device, seed=seed, epoch=epoch, batch_index=batch_index, branch=2),
            )
            model.train()
            auxiliary_loss = soft_target_cross_entropy(model(auxiliary_adv), target)

        if not first_hashes:
            first_hashes = {
                "first_mixed_batch_sha256": tensor_sha256(x_mixed),
                "first_soft_target_sha256": tensor_sha256(target),
                "first_primary_adversarial_batch_sha256": tensor_sha256(primary_adv),
            }

        optimizer.zero_grad(set_to_none=True)
        info: dict[str, Any] = {}
        if experiment == "pgd_at_linf10":
            combined_loss = primary_loss
            combined_loss.backward()
        elif experiment in {"pda_no_proj_w0p1", "pda_no_proj_w0p2"}:
            assert auxiliary_loss is not None
            weight = 0.1 if experiment.endswith("w0p1") else 0.2
            combined_loss = primary_loss + weight * auxiliary_loss
            combined_loss.backward()
        else:
            assert auxiliary_loss is not None
            info = primary_projected_aux_backward(
                {"pgd_linf_10": primary_loss, "pgd_l2_5": auxiliary_loss},
                model,
                primary_domain="pgd_linf_10",
                aux_domain="pgd_l2_5",
                aux_weight=0.2,
            )
            combined_loss = primary_loss.detach() + 0.2 * auxiliary_loss.detach()

        pre_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0).detach().cpu().item())
        post_norm = global_grad_norm(model)
        if not math.isfinite(pre_norm) or not math.isfinite(post_norm):
            raise FloatingPointError(f"Non-finite gradient norm: pre={pre_norm}, post={post_norm}")
        if post_norm > 1.00001:
            raise RuntimeError(f"Gradient clipping failed: post norm {post_norm} > 1.0")
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Non-finite gradient in {name}")
        optimizer.step()

        batch_size = hard_target.numel()
        batches_seen += 1
        samples_seen += batch_size
        clipped = pre_norm > 1.0
        clipped_batches += int(clipped)
        pre_norms.append(pre_norm)
        post_norms.append(post_norm)
        total_primary += float(primary_loss.detach().cpu().item()) * batch_size
        total_aux += float(auxiliary_loss.detach().cpu().item()) * batch_size if auxiliary_loss is not None else 0.0
        total_loss += float(combined_loss.detach().cpu().item()) * batch_size
        total_correct += float((primary_logits.argmax(1) == target.argmax(1)).sum().cpu().item())
        if "primary_aux_cosine" in info:
            cosines.append(float(info["primary_aux_cosine"]))
            projections.append(bool(info["primary_aux_projected"]))
        append_csv(
            run_dir / "batch_diagnostics.csv",
            batch_fields,
            {
                "epoch": epoch,
                "batch": batch_index,
                "samples": batch_size,
                "primary_loss": float(primary_loss.detach().cpu().item()),
                "auxiliary_loss": float(auxiliary_loss.detach().cpu().item()) if auxiliary_loss is not None else "",
                "combined_loss": float(combined_loss.detach().cpu().item()),
                "clip_pre_global_norm": pre_norm,
                "clip_post_global_norm": post_norm,
                "clip_max_norm": 1.0,
                "clipped": clipped,
                "primary_aux_cosine": info.get("primary_aux_cosine", ""),
                "primary_aux_projected": info.get("primary_aux_projected", ""),
                "primary_aux_projection_norm": info.get("primary_aux_projection_norm", ""),
            },
        )

    if not max_batches:
        if batches_seen != len(loader) or samples_seen != len(loader.dataset):
            raise RuntimeError(
                f"Incomplete epoch: batches {batches_seen}/{len(loader)}, samples {samples_seen}/{len(loader.dataset)}"
            )
    if not samples_seen:
        raise RuntimeError("No training samples were processed.")
    return {
        "train_loss": total_loss / samples_seen,
        "train_primary_loss": total_primary / samples_seen,
        "train_auxiliary_loss": total_aux / samples_seen if experiment != "pgd_at_linf10" else "",
        "train_primary_acc": total_correct / samples_seen,
        "train_batches_seen": batches_seen,
        "train_samples_seen": samples_seen,
        "clip_pre_global_norm_mean": sum(pre_norms) / len(pre_norms),
        "clip_pre_global_norm_max": max(pre_norms),
        "clip_post_global_norm_mean": sum(post_norms) / len(post_norms),
        "clip_threshold": 1.0,
        "clip_fraction": clipped_batches / batches_seen,
        "primary_aux_cosine": sum(cosines) / len(cosines) if cosines else "",
        "primary_aux_projected": sum(projections) / len(projections) if projections else "",
        **first_hashes,
    }


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = build_parser().parse_args(argv)
    validate_args(args)
    run_dir = require_under_task(args.save_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    if config_path.exists() and not args.resume:
        raise SystemExit(f"Refusing to overwrite an existing run config: {config_path}")

    set_seed(args.seed)
    device = choose_device(args.device)
    dataloader_generator = torch.Generator().manual_seed(args.seed)
    train_loader, test_loader, data_metadata = build_paper_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        generator=dataloader_generator,
        use_fake_data=args.use_fake_data,
        fake_train_size=args.fake_train_size,
        fake_test_size=args.fake_test_size,
        download=not args.no_download,
    )
    atexit.register(shutdown_dataloader_workers, train_loader, test_loader)
    try:
        model = build_paper_deit_tiny_patch4_32(
            num_classes=10, pretrained=False, pretrained_checkpoint=args.pretrained_checkpoint
        ).to(device)
        init_hash = initial_trainable_parameters_hash(model)
        provenance = model_provenance(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        # Smoke/validation may stop early, but they exercise the beginning of
        # the fixed 40-epoch paper schedule rather than compressing milestones.
        scheduler = PaperEpochLRScheduler(optimizer, epochs=40, lr_max=0.1)
        mixup = create_mixup()

        plan = resolved_config(
            experiment=args.experiment,
            seed=args.seed,
            pretrained_checkpoint=args.pretrained_checkpoint,
            num_workers=args.num_workers,
            run_kind=args.run_kind,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        config: dict[str, Any] = {
            **method_fields(args.experiment),
            "recipe": "paper",
            "paper_recipe_version": plan["paper_recipe"]["recipe_version"],
            "paper_recipe_hash": plan["paper_recipe_hash"],
            "implementation_hash": implementation_hash(),
            "paper_recipe": plan["paper_recipe"],
            "method_config_hash": plan["method_config_hash"],
            "experiment": args.experiment,
            "run_kind": args.run_kind,
            "dataset": "cifar10",
            "model": PAPER_MODEL_ALIAS,
            "model_alias": PAPER_MODEL_ALIAS,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "data_dir": str(Path(args.data_dir)),
            "use_fake_data": args.use_fake_data,
            "max_train_batches": args.max_train_batches,
            "optimizer": "sgd",
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "scheduler": "official_fractional_epoch_step",
            "gradient_clip_max_norm": 1.0,
            "gradient_clip_norm_type": 2.0,
            "gradient_clip_stage": "final_combined_parameter_gradient",
            "eps_linf": 8 / 255,
            "alpha_linf": 2 / 255,
            "eps_l2": 0.5,
            "alpha_l2": 0.1,
            "random_start": True,
            "pretrained_checkpoint": str(Path(args.pretrained_checkpoint).expanduser().resolve()),
            "pretrained_checkpoint_sha256": file_sha256(args.pretrained_checkpoint),
            "pretrained_loaded": True,
            "pretrained_source": provenance.get("pretrained_source"),
            "model_provenance": provenance,
            "normalization_mode": provenance.get("normalization_mode"),
            "normalization_mean": provenance.get("normalization_mean"),
            "normalization_std": provenance.get("normalization_std"),
            "initial_trainable_parameters_hash": init_hash,
            "dataloader": data_metadata,
            "library_versions": {
                "python": sys.version,
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "timm": provenance.get("timm_version"),
            },
            "project_git_commit": git_commit(),
            "resolved_command": [sys.executable, "-u", "-m", "vit_12_runs_task.paper_recipe.train", *sys.argv[1:]],
            "resume_semantics": "epoch-boundary; stateless seed/epoch/index augmentation and sampler; model/optimizer/scheduler/best metric/Python/NumPy/Torch/CUDA/DataLoader generator restored",
        }

        start_epoch = 1
        best_metric = -1.0
        if args.resume:
            resume_path = require_under_task(args.resume)
            payload = load_checkpoint(resume_path, map_location=device)
            validate_resume_config(config, payload["config"])
            model.load_state_dict(payload["model_state_dict"], strict=True)
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            scheduler.load_state_dict(payload["scheduler_state_dict"])
            best_metric = float(payload["best_metric"])
            start_epoch = int(payload["epoch"]) + 1
            restore_rng_state(payload["rng_state"], dataloader_generator)
            saved_config = json.loads(config_path.read_text(encoding="utf-8"))
            validate_resume_config(config, saved_config)
            # Preserve reproducibility hashes and any provenance already
            # written by the completed epochs.
            config = saved_config
            config.setdefault("resume_history", []).append(
                {
                    "resumed_at": datetime.now(timezone.utc).isoformat(),
                    "checkpoint": str(resume_path),
                    "next_epoch": start_epoch,
                    "command": [sys.executable, "-u", "-m", "vit_12_runs_task.paper_recipe.train", *sys.argv[1:]],
                }
            )
            print(f"strict_resume={resume_path} next_epoch={start_epoch}", flush=True)
        else:
            atomic_json(config_path, config)

        train_fields = [
            "epoch", "lr", "train_loss", "train_primary_loss", "train_auxiliary_loss", "train_primary_acc",
            "train_batches_seen", "train_samples_seen", "clip_pre_global_norm_mean", "clip_pre_global_norm_max",
            "clip_post_global_norm_mean", "clip_threshold", "clip_fraction", "primary_aux_cosine",
            "primary_aux_projected", "time_sec", "train_pgd_linf_10_loss", "train_pgd_l2_5_loss",
        ]
        if start_epoch > args.epochs:
            print(f"checkpoint already completed epoch {start_epoch - 1}/{args.epochs}", flush=True)
        for epoch in range(start_epoch, args.epochs + 1):
            lr = scheduler.set_epoch(epoch)
            set_paper_loader_epoch(train_loader, epoch)
            started = time.time()
            stats = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                mixup=mixup,
                experiment=args.experiment,
                seed=args.seed,
                epoch=epoch,
                device=device,
                max_batches=args.max_train_batches,
                run_dir=run_dir,
            )
            elapsed = time.time() - started
            append_csv(
                run_dir / "train_log.csv",
                train_fields,
                {
                    "epoch": epoch,
                    "lr": lr,
                    **stats,
                    "time_sec": elapsed,
                    "train_pgd_linf_10_loss": stats["train_primary_loss"],
                    "train_pgd_l2_5_loss": stats["train_auxiliary_loss"],
                },
            )
            for key in (
                "first_mixed_batch_sha256",
                "first_soft_target_sha256",
                "first_primary_adversarial_batch_sha256",
            ):
                config.setdefault("per_epoch_reproducibility", {}).setdefault(str(epoch), {})[key] = stats[key]
            atomic_json(config_path, config)
            best_metric = float(epoch)  # final-epoch selection, deliberately not a noisy stochastic PGD maximum
            for filename in ("last.pt", "best.pt"):
                save_training_checkpoint(
                    run_dir / filename,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_metric=best_metric,
                    config=config,
                    dataloader_generator=dataloader_generator,
                )
            print(
                f"epoch_complete={epoch} lr={lr:.6g} batches={stats['train_batches_seen']} "
                f"samples={stats['train_samples_seen']} seconds={elapsed:.1f}",
                flush=True,
            )
            if args.stop_after_epoch and epoch >= args.stop_after_epoch and epoch < args.epochs:
                print("intentional_smoke_stop_without_completion_marker", flush=True)
                return

        marker = {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "run_kind": args.run_kind,
            "epochs": args.epochs,
            "paper_recipe_hash": paper_recipe_hash(),
            "implementation_hash": config["implementation_hash"],
            "last_checkpoint_sha256": file_sha256(run_dir / "last.pt"),
            "note": "training complete only; baseline approval is a separate explicit user action",
        }
        atomic_json(run_dir / "training_complete.json", marker)
        if args.run_kind == "validation_only":
            (run_dir / "VALIDATION_ONLY_NOT_A_BASELINE").write_text(
                "This one-epoch run validates the pipeline only and must never enter formal summaries.\n",
                encoding="utf-8",
            )
        print(f"training_complete={run_dir}", flush=True)
    finally:
        shutdown_dataloader_workers(train_loader, test_loader)
        try:
            atexit.unregister(shutdown_dataloader_workers)
        except Exception:
            pass


if __name__ == "__main__":
    main()
