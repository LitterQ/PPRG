from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from pareto_attack_dg.optim.mgda import mgda_backward
from pareto_attack_dg.optim.pcgrad import pcgrad_backward

from . import config as paper_config
from . import train as base_train
from .integrity import canonical_json, implementation_manifest, verify_lock


TASK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TASK_ROOT.parent
SUPPLEMENT_VERSION = "deit-table-s4-seed0-symmetric-aggregation-v1"
TABLE_S4_LOCK_NAME = "table_s4_implementation_lock.json"
SUPPLEMENT_FILES = (
    "vit_12_runs_task/paper_recipe/supplemental_table_s4.py",
    "vit_12_runs_task/scripts/smoke_test_deit_table_s4.py",
    "vit_12_runs_task/scripts/run_paper_recipe_table_s4_seed0.sh",
    "vit_12_runs_task/scripts/summarize_deit_table_s4_seed0.py",
    "pareto_attack_dg/optim/pcgrad.py",
    "pareto_attack_dg/optim/mgda.py",
    "pareto_attack_dg/optim/grad_utils.py",
)

FOUR_SOURCES = ("clean", "fgsm_linf", "pgd_linf_5", "pgd_l2_5")
FIVE_SOURCES = ("clean", "fgsm_linf", "pgd_linf_5", "pgd_linf_10", "pgd_l2_5")
MATCHED_TWO_SOURCES = ("pgd_linf_10", "pgd_l2_5")

# These IDs are domain identities, not list positions.  Shared domains therefore
# receive paired random starts in four- and five-source experiments.
SOURCE_BRANCH_IDS = {
    "clean": 10,
    "fgsm_linf": 20,
    "pgd_linf_5": 105,
    "pgd_linf_10": 110,
    "pgd_l2_5": 205,
}
SOURCE_ATTACK_SPECS: dict[str, dict[str, Any]] = {
    "clean": {"norm": None, "eps": None, "steps": 0, "alpha": None, "random_start": False},
    "fgsm_linf": {
        "norm": "linf",
        "eps": 8 / 255,
        "steps": 1,
        "alpha": 8 / 255,
        "random_start": False,
    },
    "pgd_linf_5": {
        "norm": "linf",
        "eps": 8 / 255,
        "steps": 5,
        "alpha": 2 / 255,
        "random_start": True,
    },
    "pgd_linf_10": {
        "norm": "linf",
        "eps": 8 / 255,
        "steps": 10,
        "alpha": 2 / 255,
        "random_start": True,
    },
    "pgd_l2_5": {
        "norm": "l2",
        "eps": 0.5,
        "steps": 5,
        "alpha": 0.1,
        "random_start": True,
    },
}

METHOD_SPECS: dict[str, dict[str, Any]] = {
    "s4_pgd_at_linf5": {
        "method": "pgd_at",
        "aggregation": "single",
        "train_attack": "pgd_linf_5",
        "attack_domains": ["pgd_linf_5"],
        "objective": "L_pgd_linf_5",
    },
    "s4_avg_four_source": {
        "method": "avg_attack_dg",
        "aggregation": "avg",
        "attack_domains": list(FOUR_SOURCES),
        "objective": "equal arithmetic mean of four source losses",
    },
    "s4_avg_five_source": {
        "method": "avg_attack_dg",
        "aggregation": "avg",
        "attack_domains": list(FIVE_SOURCES),
        "objective": "equal arithmetic mean of five source losses",
    },
    "s4_pcgrad_four_source": {
        "method": "pcgrad_attack_dg",
        "aggregation": "pcgrad",
        "attack_domains": list(FOUR_SOURCES),
        "objective": "standard symmetric PCGrad followed by source-gradient mean",
    },
    "s4_pcgrad_five_source": {
        "method": "pcgrad_attack_dg",
        "aggregation": "pcgrad",
        "attack_domains": list(FIVE_SOURCES),
        "objective": "standard symmetric PCGrad followed by source-gradient mean",
    },
    "s4_pcgrad_matched_two_source": {
        "method": "pcgrad_attack_dg",
        "aggregation": "pcgrad",
        "attack_domains": list(MATCHED_TWO_SOURCES),
        "objective": "standard symmetric PCGrad followed by source-gradient mean",
    },
    "s4_mgda_four_source": {
        "method": "mgda_attack_dg",
        "aggregation": "mgda",
        "attack_domains": list(FOUR_SOURCES),
        "objective": "minimum-norm combination on the standard simplex",
    },
    "s4_mgda_five_source": {
        "method": "mgda_attack_dg",
        "aggregation": "mgda",
        "attack_domains": list(FIVE_SOURCES),
        "objective": "minimum-norm combination on the standard simplex",
    },
    "s4_mgda_matched_two_source": {
        "method": "mgda_attack_dg",
        "aggregation": "mgda",
        "attack_domains": list(MATCHED_TWO_SOURCES),
        "objective": "minimum-norm combination on the standard simplex",
    },
}
EXPERIMENTS = tuple(METHOD_SPECS)


def require_under_task(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(TASK_ROOT)
    except ValueError as exc:
        raise ValueError(f"Table S4 path must remain under {TASK_ROOT}; got {resolved}") from exc
    return resolved


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def table_s4_implementation_manifest() -> dict[str, Any]:
    base_manifest = implementation_manifest()
    supplemental_files: dict[str, str] = {}
    for relative in SUPPLEMENT_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Table S4 implementation file is missing: {path}")
        supplemental_files[relative] = file_sha256(path)
    payload: dict[str, Any] = {
        "manifest_version": SUPPLEMENT_VERSION,
        # Store the complete base manifest, not only its digest.  This makes the
        # supplemental lock independently auditable while preserving old locks.
        "base_implementation_manifest": base_manifest,
        "base_implementation_hash": base_manifest["implementation_hash"],
        "supplemental_files": supplemental_files,
    }
    payload["table_s4_implementation_hash"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def table_s4_implementation_hash() -> str:
    return str(table_s4_implementation_manifest()["table_s4_implementation_hash"])


def write_table_s4_lock(path: str | Path) -> dict[str, Any]:
    target = require_under_task(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite Table S4 lock: {target}")
    payload = table_s4_implementation_manifest()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def verify_table_s4_lock(path: str | Path) -> dict[str, Any]:
    target = require_under_task(path)
    try:
        saved = json.loads(target.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing Table S4 lock: {target}") from exc
    current = table_s4_implementation_manifest()
    if saved != current:
        raise ValueError(
            "Table S4 implementation changed after lock creation: "
            f"saved={saved.get('table_s4_implementation_hash')}, "
            f"current={current.get('table_s4_implementation_hash')}"
        )
    return saved


def validate_experiment_specs() -> None:
    expected = {
        "s4_pgd_at_linf5": ("single", ("pgd_linf_5",)),
        "s4_avg_four_source": ("avg", FOUR_SOURCES),
        "s4_avg_five_source": ("avg", FIVE_SOURCES),
        "s4_pcgrad_four_source": ("pcgrad", FOUR_SOURCES),
        "s4_pcgrad_five_source": ("pcgrad", FIVE_SOURCES),
        "s4_pcgrad_matched_two_source": ("pcgrad", MATCHED_TWO_SOURCES),
        "s4_mgda_four_source": ("mgda", FOUR_SOURCES),
        "s4_mgda_five_source": ("mgda", FIVE_SOURCES),
        "s4_mgda_matched_two_source": ("mgda", MATCHED_TWO_SOURCES),
    }
    if tuple(expected) != EXPERIMENTS:
        raise AssertionError(f"Table S4 experiment order changed: {EXPERIMENTS}")
    for experiment, (aggregation, sources) in expected.items():
        spec = METHOD_SPECS[experiment]
        if spec["aggregation"] != aggregation or tuple(spec["attack_domains"]) != tuple(sources):
            raise AssertionError(f"Invalid Table S4 method spec for {experiment}: {spec}")
    if set(SOURCE_BRANCH_IDS) != set(SOURCE_ATTACK_SPECS):
        raise AssertionError("Every Table S4 source must have exactly one stable branch ID.")
    if len(set(SOURCE_BRANCH_IDS.values())) != len(SOURCE_BRANCH_IDS):
        raise AssertionError("Table S4 branch IDs must be unique.")


def experiment_sources(experiment: str) -> tuple[str, ...]:
    if experiment not in METHOD_SPECS:
        raise ValueError(f"Unknown Table S4 experiment: {experiment}")
    return tuple(str(value) for value in METHOD_SPECS[experiment]["attack_domains"])


def _source_input(
    *,
    model: torch.nn.Module,
    x_mixed: torch.Tensor,
    target: torch.Tensor,
    source: str,
    seed: int,
    epoch: int,
    batch_index: int,
    device: torch.device,
) -> torch.Tensor:
    if source == "clean":
        return x_mixed.detach()
    spec = SOURCE_ATTACK_SPECS[source]
    generator = None
    if bool(spec["random_start"]):
        generator = base_train.branch_generator(
            device,
            seed=seed,
            epoch=epoch,
            batch_index=batch_index,
            branch=SOURCE_BRANCH_IDS[source],
        )
    if spec["norm"] == "linf":
        return base_train.pgd_linf_soft_target(
            model,
            x_mixed,
            target,
            eps=float(spec["eps"]),
            alpha=float(spec["alpha"]),
            steps=int(spec["steps"]),
            random_start=bool(spec["random_start"]),
            generator=generator,
        )
    if spec["norm"] == "l2":
        return base_train.pgd_l2_soft_target(
            model,
            x_mixed,
            target,
            eps=float(spec["eps"]),
            alpha=float(spec["alpha"]),
            steps=int(spec["steps"]),
            random_start=bool(spec["random_start"]),
            generator=generator,
        )
    raise ValueError(f"Unsupported Table S4 source: {source}")


def generate_source_losses(
    *,
    model: torch.nn.Module,
    x_mixed: torch.Tensor,
    target: torch.Tensor,
    sources: Iterable[str],
    seed: int,
    epoch: int,
    batch_index: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    losses: dict[str, torch.Tensor] = {}
    source_inputs: dict[str, torch.Tensor] = {}
    logits: dict[str, torch.Tensor] = {}
    for source in sources:
        if source not in SOURCE_ATTACK_SPECS:
            raise ValueError(f"Unknown Table S4 source: {source}")
        source_input = _source_input(
            model=model,
            x_mixed=x_mixed,
            target=target,
            source=source,
            seed=seed,
            epoch=epoch,
            batch_index=batch_index,
            device=device,
        ).detach()
        # The attack context restores the previous mode; make the outer loss's
        # training-mode semantics explicit for every source.
        model.train()
        source_logits = model(source_input)
        losses[source] = base_train.soft_target_cross_entropy(source_logits, target)
        source_inputs[source] = source_input
        logits[source] = source_logits
    return losses, source_inputs, logits


def aggregate_source_gradients(
    *,
    model: torch.nn.Module,
    losses: dict[str, torch.Tensor],
    aggregation: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not losses:
        raise ValueError("At least one Table S4 source loss is required.")
    names = list(losses)
    values = list(losses.values())
    mean_loss = torch.stack(values).mean()
    if aggregation == "single":
        if len(values) != 1:
            raise ValueError("Single-source aggregation received more than one objective.")
        values[0].backward()
        return values[0].detach(), {"objective_names": names, "aggregation": aggregation}
    if aggregation == "avg":
        mean_loss.backward()
        return mean_loss.detach(), {"objective_names": names, "aggregation": aggregation}
    if aggregation == "pcgrad":
        # pcgrad_backward uses Python's random module for symmetric objective
        # shuffles.  The base checkpoint captures/restores that exact RNG state.
        info = pcgrad_backward(values, model, objective_names=names, preference_weights=None)
        info["aggregation"] = aggregation
        return mean_loss.detach(), info
    if aggregation == "mgda":
        info = mgda_backward(values, model, objective_names=names, preference_weights=None)
        info["aggregation"] = aggregation
        return mean_loss.detach(), info
    raise ValueError(f"Unknown Table S4 aggregation: {aggregation}")


def clip_final_gradients(model: torch.nn.Module) -> tuple[float, float]:
    """Apply the paper's one and only clip to the final aggregated gradient."""

    pre_norm = float(
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)
        .detach()
        .cpu()
        .item()
    )
    post_norm = base_train.global_grad_norm(model)
    return pre_norm, post_norm


def _json_floats(values: dict[str, Any]) -> str:
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


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
        raise ValueError(f"The Table S4 trainer cannot run {experiment!r}")
    spec = METHOD_SPECS[experiment]
    sources = experiment_sources(experiment)
    aggregation = str(spec["aggregation"])
    model.train()
    total_loss = total_correct = 0.0
    totals = {source: 0.0 for source in sources}
    samples_seen = batches_seen = clipped_batches = 0
    pre_norms: list[float] = []
    post_norms: list[float] = []
    first_hashes: dict[str, str] = {}
    batch_fields = [
        "epoch",
        "batch",
        "samples",
        "aggregation",
        "sources",
        "primary_source",
        "primary_loss",
        "combined_loss",
        "source_losses_json",
        "source_input_hashes_json",
        "objective_grad_norms_json",
        "mgda_alpha_json",
        "clip_pre_global_norm",
        "clip_post_global_norm",
        "clip_max_norm",
        "clipped",
    ]

    for batch_index, (x, hard_target) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        hard_target = hard_target.to(device, non_blocking=True)
        x_mixed, target = mixup(x, hard_target)
        if target.ndim != 2 or target.shape != (hard_target.numel(), 10):
            raise RuntimeError(f"Mixup/CutMix did not produce [N,10] soft targets: {tuple(target.shape)}")
        if not torch.allclose(target.sum(dim=1), torch.ones_like(target[:, 0]), atol=1e-6, rtol=0):
            raise RuntimeError("Mixup/CutMix target rows do not sum to one.")

        optimizer.zero_grad(set_to_none=True)
        losses, source_inputs, logits = generate_source_losses(
            model=model,
            x_mixed=x_mixed,
            target=target,
            sources=sources,
            seed=seed,
            epoch=epoch,
            batch_index=batch_index,
            device=device,
        )
        combined_loss, info = aggregate_source_gradients(
            model=model,
            losses=losses,
            aggregation=aggregation,
        )
        pre_norm, post_norm = clip_final_gradients(model)
        if not math.isfinite(pre_norm) or not math.isfinite(post_norm):
            raise FloatingPointError(f"Non-finite gradient norm: pre={pre_norm}, post={post_norm}")
        if post_norm > 1.00001:
            raise RuntimeError(f"Gradient clipping failed: post norm {post_norm} > 1.0")
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Non-finite gradient in {name}")
        optimizer.step()

        primary_source = sources[0]
        batch_size = hard_target.numel()
        batches_seen += 1
        samples_seen += batch_size
        clipped = pre_norm > 1.0
        clipped_batches += int(clipped)
        pre_norms.append(pre_norm)
        post_norms.append(post_norm)
        total_loss += float(combined_loss.cpu().item()) * batch_size
        for source, loss in losses.items():
            totals[source] += float(loss.detach().cpu().item()) * batch_size
        total_correct += float(
            (logits[primary_source].argmax(1) == target.argmax(1)).sum().detach().cpu().item()
        )
        source_loss_values = {
            source: float(loss.detach().cpu().item()) for source, loss in losses.items()
        }
        source_hashes = {
            source: base_train.tensor_sha256(value) for source, value in source_inputs.items()
        }
        if not first_hashes:
            first_hashes = {
                "first_mixed_batch_sha256": base_train.tensor_sha256(x_mixed),
                "first_soft_target_sha256": base_train.tensor_sha256(target),
                "first_primary_adversarial_batch_sha256": source_hashes[primary_source],
                "first_source_input_hashes_json": _json_floats(source_hashes),
            }
        base_train.append_csv(
            run_dir / "batch_diagnostics.csv",
            batch_fields,
            {
                "epoch": epoch,
                "batch": batch_index,
                "samples": batch_size,
                "aggregation": aggregation,
                "sources": " ".join(sources),
                "primary_source": primary_source,
                "primary_loss": source_loss_values[primary_source],
                "combined_loss": float(combined_loss.cpu().item()),
                "source_losses_json": _json_floats(source_loss_values),
                "source_input_hashes_json": _json_floats(source_hashes),
                "objective_grad_norms_json": _json_floats(info.get("objective_grad_norms", {})),
                "mgda_alpha_json": _json_floats(info.get("mgda_alpha", {})),
                "clip_pre_global_norm": pre_norm,
                "clip_post_global_norm": post_norm,
                "clip_max_norm": 1.0,
                "clipped": clipped,
            },
        )

    if not max_batches and (batches_seen != len(loader) or samples_seen != len(loader.dataset)):
        raise RuntimeError(
            f"Incomplete epoch: batches {batches_seen}/{len(loader)}, "
            f"samples {samples_seen}/{len(loader.dataset)}"
        )
    if not samples_seen:
        raise RuntimeError("No training samples were processed.")
    domain_means = {source: value / samples_seen for source, value in totals.items()}
    stats: dict[str, Any] = {
        "train_loss": total_loss / samples_seen,
        "train_primary_loss": domain_means[sources[0]],
        "train_auxiliary_loss": (
            sum(domain_means[source] for source in sources[1:]) / (len(sources) - 1)
            if len(sources) > 1
            else ""
        ),
        "train_primary_acc": total_correct / samples_seen,
        "train_batches_seen": batches_seen,
        "train_samples_seen": samples_seen,
        "clip_pre_global_norm_mean": sum(pre_norms) / len(pre_norms),
        "clip_pre_global_norm_max": max(pre_norms),
        "clip_post_global_norm_mean": sum(post_norms) / len(post_norms),
        "clip_threshold": 1.0,
        "clip_fraction": clipped_batches / batches_seen,
        "primary_aux_cosine": "",
        "primary_aux_projected": "",
        "train_source_losses_json": _json_floats(domain_means),
        "aggregation": aggregation,
        **first_hashes,
    }
    for source in SOURCE_ATTACK_SPECS:
        stats[f"train_{source}_loss"] = domain_means.get(source, "")
    return stats


def _install_training_extension() -> None:
    validate_experiment_specs()
    paper_config.METHOD_SPECS.update({name: dict(spec) for name, spec in METHOD_SPECS.items()})
    base_train.METHODS = (*paper_config.METHODS, *EXPERIMENTS)
    base_train.train_one_epoch = train_one_epoch
    extension_manifest = table_s4_implementation_manifest()

    original_method_fields = base_train.method_fields

    def method_fields(experiment: str) -> dict[str, Any]:
        fields = original_method_fields(experiment)
        if experiment in EXPERIMENTS:
            spec = METHOD_SPECS[experiment]
            sources = list(spec["attack_domains"])
            fields.update(
                {
                    "method": spec["method"],
                    "train_attack": spec.get("train_attack", ""),
                    "attack_domains": sources,
                    "parsed_domain_weights": (
                        {source: 1.0 / len(sources) for source in sources}
                        if spec["aggregation"] == "avg"
                        else {}
                    ),
                    "normalize_domain_weights": spec["aggregation"] == "avg",
                    "aggregation": spec["aggregation"],
                    "symmetric_aggregation": spec["aggregation"] in {"avg", "pcgrad", "mgda"},
                    "preference_weights": None,
                    "source_attack_specs": {source: SOURCE_ATTACK_SPECS[source] for source in sources},
                    "source_branch_ids": {source: SOURCE_BRANCH_IDS[source] for source in sources},
                    "supplement_version": SUPPLEMENT_VERSION,
                    "supplemental_base_implementation_hash": extension_manifest[
                        "base_implementation_hash"
                    ],
                    "table_s4_implementation_hash": extension_manifest[
                        "table_s4_implementation_hash"
                    ],
                    "table_s4_implementation_files": extension_manifest["supplemental_files"],
                    "formal_suite_target": "deit_table_s4_seed0",
                }
            )
        return fields

    base_train.method_fields = method_fields

    original_validate_resume_config = base_train.validate_resume_config

    def validate_resume_config(current: dict[str, Any], saved: dict[str, Any]) -> None:
        original_validate_resume_config(current, saved)
        checks = (
            "supplement_version",
            "supplemental_base_implementation_hash",
            "table_s4_implementation_hash",
            "table_s4_implementation_files",
            "aggregation",
            "attack_domains",
            "source_branch_ids",
        )
        mismatches = {
            key: (saved.get(key), current.get(key))
            for key in checks
            if saved.get(key) != current.get(key)
        }
        if mismatches:
            raise ValueError(f"Table S4 resume checkpoint conflicts with current code/config: {mismatches}")

    base_train.validate_resume_config = validate_resume_config

    original_append_csv = base_train.append_csv

    def append_csv(path: Path, fields: list[str], row: dict[str, Any]) -> None:
        if path.name == "train_log.csv" and row.get("aggregation") is not None:
            extra_fields = [
                "aggregation",
                "train_source_losses_json",
                *[f"train_{source}_loss" for source in SOURCE_ATTACK_SPECS],
            ]
            fields = [*fields, *(field for field in extra_fields if field not in fields)]
        original_append_csv(path, fields, row)

    base_train.append_csv = append_csv


def _write_plan(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Write one formal DeiT Table S4 run plan.")
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
    extension = table_s4_implementation_manifest()
    payload.update(
        {
            "supplement_version": SUPPLEMENT_VERSION,
            "supplemental_base_implementation_hash": extension["base_implementation_hash"],
            "table_s4_implementation_hash": extension["table_s4_implementation_hash"],
            "table_s4_implementation_files": extension["supplemental_files"],
            "formal_suite_target": "deit_table_s4_seed0",
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
    if not marker_path.is_file():
        return
    marker = json.loads(marker_path.read_text(encoding="utf-8-sig"))
    extension = table_s4_implementation_manifest()
    marker.update(
        {
            "supplement_version": SUPPLEMENT_VERSION,
            "supplemental_base_implementation_hash": extension["base_implementation_hash"],
            "table_s4_implementation_hash": extension["table_s4_implementation_hash"],
            "formal_suite_target": "deit_table_s4_seed0",
            "supplement_metadata_updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    base_train.atomic_json(marker_path, marker)


def _option_values(args: list[str], option: str) -> list[str]:
    return [args[index + 1] for index, value in enumerate(args[:-1]) if value == option]


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--manifest"]:
        print(json.dumps(table_s4_implementation_manifest(), indent=2, sort_keys=True))
        return
    if args[:1] == ["--write-lock"]:
        if len(args) != 2:
            raise SystemExit("usage: supplemental_table_s4 --write-lock PATH")
        print(json.dumps(write_table_s4_lock(args[1]), indent=2, sort_keys=True))
        return
    if args[:1] == ["--verify-lock"]:
        if len(args) != 2:
            raise SystemExit("usage: supplemental_table_s4 --verify-lock PATH")
        print(json.dumps(verify_table_s4_lock(args[1]), indent=2, sort_keys=True))
        return
    if args[:1] == ["--validate-specs"]:
        validate_experiment_specs()
        print(json.dumps({"valid": True, "experiments": list(EXPERIMENTS)}, indent=2))
        return
    if args[:1] == ["--write-plan"]:
        _write_plan(args[1:])
        return

    experiment_values = _option_values(args, "--experiment")
    if len(experiment_values) != 1 or experiment_values[0] not in EXPERIMENTS:
        raise SystemExit(f"This entry point only trains one of: {', '.join(EXPERIMENTS)}")
    save_dirs = _option_values(args, "--save-dir")
    if len(save_dirs) != 1:
        raise SystemExit("Exactly one --save-dir is required.")
    run_kinds = _option_values(args, "--run-kind")
    if len(run_kinds) != 1:
        raise SystemExit("Exactly one --run-kind is required.")
    run_dir = require_under_task(save_dirs[0])
    if run_kinds[0] == "formal_12":
        verify_lock(run_dir / "implementation_lock.json")
        verify_table_s4_lock(run_dir / TABLE_S4_LOCK_NAME)
    elif (run_dir / TABLE_S4_LOCK_NAME).exists():
        verify_table_s4_lock(run_dir / TABLE_S4_LOCK_NAME)

    _install_training_extension()
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *args]
        base_train.main()
    finally:
        sys.argv = original_argv
    _augment_training_completion(run_dir)


validate_experiment_specs()


if __name__ == "__main__":
    main()


__all__ = [
    "EXPERIMENTS",
    "FIVE_SOURCES",
    "FOUR_SOURCES",
    "MATCHED_TWO_SOURCES",
    "METHOD_SPECS",
    "SOURCE_ATTACK_SPECS",
    "SOURCE_BRANCH_IDS",
    "SUPPLEMENT_FILES",
    "SUPPLEMENT_VERSION",
    "TABLE_S4_LOCK_NAME",
    "aggregate_source_gradients",
    "clip_final_gradients",
    "experiment_sources",
    "generate_source_losses",
    "table_s4_implementation_hash",
    "table_s4_implementation_manifest",
    "train_one_epoch",
    "validate_experiment_specs",
    "verify_table_s4_lock",
    "write_table_s4_lock",
]
