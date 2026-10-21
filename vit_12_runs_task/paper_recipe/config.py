from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PAPER_RECIPE_VERSION = "vit-paper-recipe-v2-official-init"
PAPER_MODEL_ALIAS = "deit_tiny_patch4_32_paper"
OFFICIAL_REPOSITORY = "https://github.com/mo666666/When-Adversarial-Training-Meets-Vision-Transformers"
OFFICIAL_COMMIT = "b826030ad5c2dce94156d37ebe7f5bb3e4d67d83"
OFFICIAL_FILES = {
    name: f"{OFFICIAL_REPOSITORY}/blob/{OFFICIAL_COMMIT}/{name}"
    for name in ("parser_cifar.py", "train_cifar.py", "utils.py", "pgd.py", "model_for_cifar/deit.py")
}
PRETRAINED_MODEL_ID = "deit_tiny_patch16_224.fb_in1k"
PRETRAINED_PUBLIC_NAME = "deit_tiny_patch16_224"
PINNED_TIMM_VERSION = "1.0.15"
DEFAULT_PRETRAINED_SHA256 = "a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97"
PAPER_PATCH_EMBED_ADAPTATION = "official_block_sum_16x16_to_4x4"
PAPER_POS_EMBED_ADAPTATION = "official_bilinear_14x14_to_8x8_align_corners_false_no_antialias"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
METHODS = ("pgd_at_linf10", "pda_no_proj_w0p1", "pda_no_proj_w0p2", "pda_proj_w0p2")
METHOD_SPECS: dict[str, dict[str, Any]] = {
    "pgd_at_linf10": {
        "method": "pgd_at",
        "train_attack": "pgd_linf_10",
        "objective": "L_primary",
    },
    "pda_no_proj_w0p1": {
        "method": "avg_attack_dg",
        "attack_domains": ["pgd_linf_10", "pgd_l2_5"],
        "parsed_domain_weights": {"pgd_linf_10": 1.0, "pgd_l2_5": 0.1},
        "normalize_domain_weights": False,
        "objective": "L_primary + 0.1 * L_aux",
    },
    "pda_no_proj_w0p2": {
        "method": "avg_attack_dg",
        "attack_domains": ["pgd_linf_10", "pgd_l2_5"],
        "parsed_domain_weights": {"pgd_linf_10": 1.0, "pgd_l2_5": 0.2},
        "normalize_domain_weights": False,
        "objective": "L_primary + 0.2 * L_aux",
    },
    "pda_proj_w0p2": {
        "method": "primary_projected_aux_l2",
        "attack_domains": ["pgd_linf_10", "pgd_l2_5"],
        "primary_domain": "pgd_linf_10",
        "aux_domain": "pgd_l2_5",
        "aux_weight": 0.2,
        "objective": "g_primary + 0.2 * projected(g_aux | g_primary)",
    },
}


def lr_for_epoch(epoch: int, *, epochs: int = 40, lr_max: float = 0.1) -> float:
    """Return the LR used by optimizer steps in a one-indexed epoch.

    The official trainer updates LR after every optimizer step using
    ``epoch - 1 + (step + 1) / steps_per_epoch``. Consequently, for 40 epochs
    every step in epochs 1..35 uses 0.1, epochs 36..38 use 0.01, and epochs
    39..40 use 0.001. This function expresses that same step-wise behavior at
    epoch boundaries and avoids an off-by-one from a generic MultiStepLR.
    """

    if not 1 <= epoch <= epochs:
        raise ValueError(f"epoch must be in [1, {epochs}], got {epoch}")
    if epoch <= epochs - 5:
        return float(lr_max)
    if epoch <= epochs - 2:
        return float(lr_max / 10)
    return float(lr_max / 100)


def lr_sequence(*, epochs: int = 40, lr_max: float = 0.1) -> list[float]:
    return [lr_for_epoch(epoch, epochs=epochs, lr_max=lr_max) for epoch in range(1, epochs + 1)]


def scientific_recipe() -> dict[str, Any]:
    """Fields that must be identical across methods and seeds."""

    return {
        "recipe_version": PAPER_RECIPE_VERSION,
        "official_source": {
            "repository": OFFICIAL_REPOSITORY,
            "commit": OFFICIAL_COMMIT,
            "files": OFFICIAL_FILES,
        },
        "model": {
            "alias": PAPER_MODEL_ALIAS,
            "pretrained_model_id": PRETRAINED_MODEL_ID,
            "pretrained_public_name": PRETRAINED_PUBLIC_NAME,
            "pretrained_dataset": "ImageNet-1K",
            "input_size": [32, 32],
            "patch_size": [4, 4],
            "num_classes": 10,
            "drop_rate": 0.0,
            "drop_path_rate": 0.0,
            "timm_version": PINNED_TIMM_VERSION,
            "pretrained_adaptation": {
                "source_file": OFFICIAL_FILES["model_for_cifar/deit.py"],
                "patch_embed": PAPER_PATCH_EMBED_ADAPTATION,
                "position_embedding": PAPER_POS_EMBED_ADAPTATION,
                "generic_timm_checkpoint_filter": False,
            },
        },
        "data": {
            "dataset": "cifar10",
            "dataset_length": {"train": 50000, "test": 10000},
            "train_transform": ["Resize(32,32) [identity]", "RandomCrop(32,padding=4)", "RandomHorizontalFlip", "ToTensor"],
            "test_transform": ["Resize(32,32) [identity]", "ToTensor"],
            "normalization_mode": "once_inside_model_from_pixel_space",
            "normalization_mean": list(CIFAR10_MEAN),
            "normalization_std": list(CIFAR10_STD),
            "drop_last": False,
            "shuffle_train": True,
            "official_num_workers": 16,
            "runtime_num_workers_default": 2,
            "persistent_workers_when_positive": True,
        },
        "mixup_cutmix": {
            "enabled": True,
            "mixup_alpha": 0.8,
            "cutmix_alpha": 1.0,
            "cutmix_minmax": None,
            "prob": 0.3,
            "switch_prob": 0.5,
            "mode": "batch",
            "label_smoothing": 0.0,
            "num_classes": 10,
            "order": "transform -> mixup/cutmix -> soft-target PGD -> soft-target training loss",
        },
        "optimizer": {
            "name": "SGD",
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "gradient_clip": {"norm_type": 2.0, "max_norm": 1.0, "stage": "final_combined_parameter_gradient"},
        },
        "schedule": {
            "epochs": 40,
            "official_fractional_rule": "0.1 if t < 35; 0.01 if 35 <= t < 38; 0.001 otherwise",
            "optimizer_step_epoch_ranges": {"1-35": 0.1, "36-38": 0.01, "39-40": 0.001},
            "lr_by_epoch": lr_sequence(),
        },
        "batch_size": 64,
        "amp": False,
        "primary_attack": {"name": "PGD-Linf", "steps": 10, "eps": 8 / 255, "alpha": 2 / 255, "random_start": True},
        "auxiliary_attack": {"name": "PGD-L2", "steps": 5, "eps": 0.5, "alpha": 0.1, "random_start": True},
        "attack_model_mode": "eval_during_inner_attack_then_restore_train",
        "checkpoint_selection": "final_epoch (best.pt and last.pt intentionally coincide at epoch 40)",
        "evaluation": {
            "official_linf_pgd_cw": "correct-only updates, one random restart, per-sample maximum loss",
            "cw_margin_floor": -10.0,
            "autoattack_linf": "standard",
            "extended_project_matrix": "legacy FGSM/L2 attacks and epsilon sweeps retained as additional reports",
        },
    }


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def paper_recipe_hash() -> str:
    return hashlib.sha256(canonical_json(scientific_recipe()).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_config(
    *,
    experiment: str,
    seed: int,
    pretrained_checkpoint: str | Path,
    num_workers: int,
    run_kind: str,
    epochs: int = 40,
    batch_size: int = 64,
) -> dict[str, Any]:
    if experiment not in METHOD_SPECS:
        raise ValueError(f"unknown experiment {experiment!r}; expected one of {METHODS}")
    checkpoint = Path(pretrained_checkpoint).expanduser().resolve()
    payload = {
        "paper_recipe": scientific_recipe(),
        "paper_recipe_hash": paper_recipe_hash(),
        "experiment": experiment,
        "seed": int(seed),
        "method_spec": METHOD_SPECS[experiment],
        "runtime": {
            "run_kind": run_kind,
            "epochs_requested": int(epochs),
            "batch_size_actual": int(batch_size),
            "num_workers": int(num_workers),
            "persistent_workers": bool(num_workers > 0),
            "pretrained_checkpoint": str(checkpoint),
            "pretrained_checkpoint_sha256": file_sha256(checkpoint) if checkpoint.is_file() else None,
        },
    }
    payload["method_config_hash"] = hashlib.sha256(canonical_json({"recipe": scientific_recipe(), "method": METHOD_SPECS[experiment]}).encode("utf-8")).hexdigest()
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print/save the fully resolved paper recipe without starting training.")
    parser.add_argument("--experiment", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--run-kind", default="manual_baseline")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    payload = resolved_config(
        experiment=args.experiment,
        seed=args.seed,
        pretrained_checkpoint=args.pretrained_checkpoint,
        num_workers=args.num_workers,
        run_kind=args.run_kind,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise SystemExit(f"Refusing to overwrite existing planned config: {output}")
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
