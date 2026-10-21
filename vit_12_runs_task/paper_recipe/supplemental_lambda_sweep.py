from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from . import config as paper_config
from . import train as base_train
from .integrity import canonical_json, implementation_manifest


TASK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TASK_ROOT.parent
SUPPLEMENT_VERSION = "paper-recipe-pda-lambda-sweep-supplement-v1"
SUPPLEMENT_FILES = (
    "vit_12_runs_task/paper_recipe/supplemental_lambda_sweep.py",
    "vit_12_runs_task/scripts/run_paper_recipe_pda_lambda_sweep.sh",
    "vit_12_runs_task/scripts/summarize_vit_33_runs.py",
)
WEIGHTS: dict[str, float] = {"0p05": 0.05, "0p15": 0.15, "0p3": 0.3}
EXPERIMENTS = tuple(
    [f"pda_no_proj_w{label}" for label in WEIGHTS]
    + [f"pda_proj_w{label}" for label in WEIGHTS]
)
METHOD_SPECS: dict[str, dict[str, Any]] = {}
for label, weight in WEIGHTS.items():
    METHOD_SPECS[f"pda_no_proj_w{label}"] = {
        "method": "avg_attack_dg",
        "attack_domains": ["pgd_linf_10", "pgd_l2_5"],
        "parsed_domain_weights": {"pgd_linf_10": 1.0, "pgd_l2_5": weight},
        "normalize_domain_weights": False,
        "objective": f"L_primary + {weight:g} * L_aux",
    }
    METHOD_SPECS[f"pda_proj_w{label}"] = {
        "method": "primary_projected_aux_l2",
        "attack_domains": ["pgd_linf_10", "pgd_l2_5"],
        "primary_domain": "pgd_linf_10",
        "aux_domain": "pgd_l2_5",
        "aux_weight": weight,
        "objective": f"g_primary + {weight:g} * projected(g_aux | g_primary)",
    }


def require_under_task(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(TASK_ROOT)
    except ValueError as exc:
        raise ValueError(f"Supplement path must remain under {TASK_ROOT}; got {resolved}") from exc
    return resolved


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def supplemental_implementation_manifest() -> dict[str, Any]:
    base = implementation_manifest()
    files: dict[str, str] = {}
    for relative in SUPPLEMENT_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Lambda-sweep implementation file is missing: {path}")
        files[relative] = file_sha256(path)
    payload: dict[str, Any] = {
        "manifest_version": SUPPLEMENT_VERSION,
        "base_implementation_hash": base["implementation_hash"],
        "supplemental_files": files,
    }
    payload["supplemental_implementation_hash"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def supplemental_implementation_hash() -> str:
    return str(supplemental_implementation_manifest()["supplemental_implementation_hash"])


def write_supplement_lock(path: str | Path) -> dict[str, Any]:
    target = require_under_task(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite lambda-sweep lock: {target}")
    payload = supplemental_implementation_manifest()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def verify_supplement_lock(path: str | Path) -> dict[str, Any]:
    target = require_under_task(path)
    try:
        saved = json.loads(target.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing lambda-sweep lock: {target}") from exc
    current = supplemental_implementation_manifest()
    if saved != current:
        raise ValueError(
            "PDA lambda-sweep supplement changed after lock creation: "
            f"saved={saved.get('supplemental_implementation_hash')}, "
            f"current={current.get('supplemental_implementation_hash')}"
        )
    return saved


def experiment_weight(experiment: str) -> float:
    if experiment not in METHOD_SPECS:
        raise ValueError(f"Unknown lambda-sweep experiment: {experiment}")
    spec = METHOD_SPECS[experiment]
    if experiment.startswith("pda_proj_"):
        return float(spec["aux_weight"])
    return float(spec["parsed_domain_weights"]["pgd_l2_5"])


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
    if experiment not in METHOD_SPECS:
        raise ValueError(f"The lambda-sweep trainer cannot run {experiment!r}")
    weight = experiment_weight(experiment)
    projected_method = experiment.startswith("pda_proj_")
    model.train()
    total_loss = total_primary = total_aux = total_correct = 0.0
    samples_seen = batches_seen = clipped_batches = 0
    pre_norms: list[float] = []
    post_norms: list[float] = []
    cosines: list[float] = []
    projections: list[bool] = []
    first_hashes: dict[str, str] = {}
    batch_fields = [
        "epoch",
        "batch",
        "samples",
        "primary_loss",
        "auxiliary_loss",
        "combined_loss",
        "clip_pre_global_norm",
        "clip_post_global_norm",
        "clip_max_norm",
        "clipped",
        "primary_aux_cosine",
        "primary_aux_projected",
        "primary_aux_projection_norm",
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

        primary_adv = base_train.pgd_linf_soft_target(
            model,
            x_mixed,
            target,
            generator=base_train.branch_generator(
                device, seed=seed, epoch=epoch, batch_index=batch_index, branch=1
            ),
        )
        model.train()
        primary_logits = model(primary_adv)
        primary_loss = base_train.soft_target_cross_entropy(primary_logits, target)
        auxiliary_adv = base_train.pgd_l2_soft_target(
            model,
            x_mixed,
            target,
            generator=base_train.branch_generator(
                device, seed=seed, epoch=epoch, batch_index=batch_index, branch=2
            ),
        )
        model.train()
        auxiliary_loss = base_train.soft_target_cross_entropy(model(auxiliary_adv), target)

        if not first_hashes:
            first_hashes = {
                "first_mixed_batch_sha256": base_train.tensor_sha256(x_mixed),
                "first_soft_target_sha256": base_train.tensor_sha256(target),
                "first_primary_adversarial_batch_sha256": base_train.tensor_sha256(primary_adv),
            }

        optimizer.zero_grad(set_to_none=True)
        info: dict[str, Any] = {}
        if projected_method:
            info = base_train.primary_projected_aux_backward(
                {"pgd_linf_10": primary_loss, "pgd_l2_5": auxiliary_loss},
                model,
                primary_domain="pgd_linf_10",
                aux_domain="pgd_l2_5",
                aux_weight=weight,
            )
            combined_loss = primary_loss.detach() + weight * auxiliary_loss.detach()
        else:
            combined_loss = primary_loss + weight * auxiliary_loss
            combined_loss.backward()

        pre_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)
            .detach()
            .cpu()
            .item()
        )
        post_norm = base_train.global_grad_norm(model)
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
        total_aux += float(auxiliary_loss.detach().cpu().item()) * batch_size
        total_loss += float(combined_loss.detach().cpu().item()) * batch_size
        total_correct += float((primary_logits.argmax(1) == target.argmax(1)).sum().cpu().item())
        if "primary_aux_cosine" in info:
            cosines.append(float(info["primary_aux_cosine"]))
            projections.append(bool(info["primary_aux_projected"]))
        base_train.append_csv(
            run_dir / "batch_diagnostics.csv",
            batch_fields,
            {
                "epoch": epoch,
                "batch": batch_index,
                "samples": batch_size,
                "primary_loss": float(primary_loss.detach().cpu().item()),
                "auxiliary_loss": float(auxiliary_loss.detach().cpu().item()),
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
                f"Incomplete epoch: batches {batches_seen}/{len(loader)}, "
                f"samples {samples_seen}/{len(loader.dataset)}"
            )
    if not samples_seen:
        raise RuntimeError("No training samples were processed.")
    return {
        "train_loss": total_loss / samples_seen,
        "train_primary_loss": total_primary / samples_seen,
        "train_auxiliary_loss": total_aux / samples_seen,
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


def _install_training_extension() -> None:
    paper_config.METHOD_SPECS.update({name: dict(spec) for name, spec in METHOD_SPECS.items()})
    base_train.METHODS = (*paper_config.METHODS, *EXPERIMENTS)
    base_train.train_one_epoch = train_one_epoch
    original_method_fields = base_train.method_fields
    extension_manifest = supplemental_implementation_manifest()

    def method_fields(experiment: str) -> dict[str, Any]:
        fields = original_method_fields(experiment)
        if experiment in EXPERIMENTS:
            fields.update(
                {
                    "supplement_version": SUPPLEMENT_VERSION,
                    "supplemental_base_implementation_hash": extension_manifest["base_implementation_hash"],
                    "supplemental_implementation_hash": extension_manifest["supplemental_implementation_hash"],
                    "supplemental_implementation_files": extension_manifest["supplemental_files"],
                    "formal_suite_target": "33_runs",
                }
            )
        return fields

    base_train.method_fields = method_fields


def _write_plan(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Write a formal PDA lambda-sweep plan.")
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    _install_training_extension()
    payload = paper_config.resolved_config(
        experiment=args.experiment,
        seed=args.seed,
        pretrained_checkpoint=args.pretrained_checkpoint,
        num_workers=args.num_workers,
        run_kind="formal_12",
        epochs=40,
        batch_size=64,
    )
    extension = supplemental_implementation_manifest()
    payload.update(
        {
            "supplement_version": SUPPLEMENT_VERSION,
            "supplemental_base_implementation_hash": extension["base_implementation_hash"],
            "supplemental_implementation_hash": extension["supplemental_implementation_hash"],
            "supplemental_implementation_files": extension["supplemental_files"],
            "formal_suite_target": "33_runs",
        }
    )
    output = require_under_task(args.output)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing planned config: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _augment_training_completion(run_dir: Path) -> None:
    marker_path = run_dir / "training_complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8-sig"))
    extension = supplemental_implementation_manifest()
    marker.update(
        {
            "supplement_version": SUPPLEMENT_VERSION,
            "supplemental_base_implementation_hash": extension["base_implementation_hash"],
            "supplemental_implementation_hash": extension["supplemental_implementation_hash"],
            "formal_suite_target": "33_runs",
            "supplement_metadata_updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    base_train.atomic_json(marker_path, marker)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--manifest"]:
        print(json.dumps(supplemental_implementation_manifest(), indent=2, sort_keys=True))
        return
    if args[:1] == ["--write-lock"]:
        if len(args) != 2:
            raise SystemExit("usage: supplemental_lambda_sweep --write-lock PATH")
        print(json.dumps(write_supplement_lock(args[1]), indent=2, sort_keys=True))
        return
    if args[:1] == ["--verify-lock"]:
        if len(args) != 2:
            raise SystemExit("usage: supplemental_lambda_sweep --verify-lock PATH")
        print(json.dumps(verify_supplement_lock(args[1]), indent=2, sort_keys=True))
        return
    if args[:1] == ["--write-plan"]:
        _write_plan(args[1:])
        return

    experiment_values = [args[index + 1] for index, value in enumerate(args[:-1]) if value == "--experiment"]
    if len(experiment_values) != 1 or experiment_values[0] not in EXPERIMENTS:
        raise SystemExit(f"This entry point only trains one of: {', '.join(EXPERIMENTS)}")
    save_dirs = [args[index + 1] for index, value in enumerate(args[:-1]) if value == "--save-dir"]
    if len(save_dirs) != 1:
        raise SystemExit("Exactly one --save-dir is required.")
    _install_training_extension()
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *args]
        base_train.main()
    finally:
        sys.argv = original_argv
    _augment_training_completion(require_under_task(save_dirs[0]))


if __name__ == "__main__":
    main()


__all__ = [
    "EXPERIMENTS",
    "METHOD_SPECS",
    "SUPPLEMENT_FILES",
    "SUPPLEMENT_VERSION",
    "WEIGHTS",
    "experiment_weight",
    "supplemental_implementation_hash",
    "supplemental_implementation_manifest",
    "verify_supplement_lock",
    "write_supplement_lock",
]
