from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TIMM_WHEEL = TASK_ROOT / ".deps" / "timm-1.0.15-py3-none-any.whl"
if TIMM_WHEEL.is_file() and str(TIMM_WHEEL) not in sys.path:
    sys.path.insert(0, str(TIMM_WHEEL))

from vit_12_runs_task.paper_recipe.supplemental_proj_w0p1 import (
    EXPERIMENT as SUPPLEMENT_METHOD,
    METHOD_SPEC as SUPPLEMENT_SPEC,
    SUPPLEMENT_VERSION,
    supplemental_implementation_hash,
    verify_supplement_lock,
)
from vit_12_runs_task.scripts import summarize_vit_runs as base_summary


METHODS = [
    "pgd_at_linf10",
    "pda_no_proj_w0p1",
    "pda_no_proj_w0p2",
    "pda_proj_w0p1",
    "pda_proj_w0p2",
]
SEEDS = [0, 1, 2]
SUPPLEMENT_LOCK = "supplemental_implementation_lock.json"
BASE_KEYS = base_summary.BASE_KEYS
DERIVED_KEYS = base_summary.DERIVED_KEYS
DELTA_KEYS = base_summary.DELTA_KEYS


def require_under_task(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(TASK_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path must remain under {TASK_ROOT}; got {resolved}") from exc
    return resolved


def supplemental_expected_artifacts() -> list[str]:
    return [
        "planned_config.json",
        "config.json",
        "implementation_lock.json",
        SUPPLEMENT_LOCK,
        "training_complete.json",
        "train_log.csv",
        "batch_diagnostics.csv",
        "best.pt",
        "last.pt",
        "eval.json",
        *[f"epsilon_sweep_eval/linf_{label}.json" for label in base_summary.LINF_LABELS],
        *[f"epsilon_sweep_eval/l2_{label}.json" for label in base_summary.L2_LABELS],
    ]


def validate_supplement_metadata(run_dir: Path, config: dict[str, Any], seed: int) -> None:
    extension_hash = supplemental_implementation_hash()
    lock = verify_supplement_lock(run_dir / SUPPLEMENT_LOCK)
    planned = base_summary.read_json(run_dir / "planned_config.json")
    completion = base_summary.read_json(run_dir / "training_complete.json")
    if run_dir.name != f"{SUPPLEMENT_METHOD}_seed{seed}":
        raise ValueError(f"Unexpected supplemental run name: {run_dir.name}")
    if int(config.get("seed", -1)) != seed or config.get("experiment") != SUPPLEMENT_METHOD:
        raise ValueError(f"Supplement experiment/seed mismatch in {run_dir}")
    if config.get("method") != SUPPLEMENT_SPEC["method"]:
        raise ValueError(f"Supplement method mismatch in {run_dir}")
    if (
        config.get("primary_domain") != "pgd_linf_10"
        or config.get("aux_domain") != "pgd_l2_5"
        or float(config.get("aux_weight", -1.0)) != 0.1
    ):
        raise ValueError(f"Projected PDA 0.1 configuration mismatch in {run_dir}")
    if config.get("supplement_version") != SUPPLEMENT_VERSION:
        raise ValueError(f"Supplement version mismatch in {run_dir}")
    for label, payload in (("config", config), ("planned config", planned), ("training completion", completion)):
        if payload.get("supplemental_implementation_hash") != extension_hash:
            raise ValueError(f"{label} supplement hash mismatch in {run_dir}")
        if payload.get("supplemental_base_implementation_hash") != lock.get("base_implementation_hash"):
            raise ValueError(f"{label} base implementation hash mismatch in {run_dir}")
    if planned.get("formal_suite_target") != "15_runs" or completion.get("formal_suite_target") != "15_runs":
        raise ValueError(f"Supplement suite target mismatch in {run_dir}")


def supplemental_run_row(run_dir: Path) -> dict[str, Any]:
    run_dir = require_under_task(run_dir)
    try:
        seed = int(run_dir.name.removeprefix(f"{SUPPLEMENT_METHOD}_seed"))
    except ValueError as exc:
        raise ValueError(f"Unexpected supplemental run name: {run_dir.name}") from exc
    if seed not in SEEDS or run_dir.name != f"{SUPPLEMENT_METHOD}_seed{seed}":
        raise ValueError(f"Unexpected supplemental run name: {run_dir.name}")

    # Reuse the locked 12-run structural audit. The supplement has the same
    # 40-epoch/data/checkpoint/evaluation contract and an additional lock.
    config = base_summary.validate_static_artifacts(run_dir)
    validate_supplement_metadata(run_dir, config, seed)
    eval_path = run_dir / "eval.json"
    base_payload = base_summary.verify_evaluation_artifact(
        eval_path,
        checkpoint=run_dir / "best.pt",
        expected_seed=seed,
        required_attacks=[key.removesuffix("_acc") for key in BASE_KEYS],
        expected_eps_linf=8 / 255,
        expected_eps_l2=0.5,
    )
    checkpoint_config = base_payload.get("checkpoint_config", {})
    if (
        checkpoint_config.get("initial_trainable_parameters_hash")
        != config["initial_trainable_parameters_hash"]
        or checkpoint_config.get("experiment") != SUPPLEMENT_METHOD
        or float(checkpoint_config.get("aux_weight", -1.0)) != 0.1
        or checkpoint_config.get("supplemental_implementation_hash")
        != supplemental_implementation_hash()
    ):
        raise ValueError(f"Supplement evaluation/checkpoint metadata mismatch in {run_dir}")

    row: dict[str, Any] = {
        "method": SUPPLEMENT_METHOD,
        "seed": seed,
        "checkpoint": str(run_dir / "best.pt"),
        "initialization_hash": config["initial_trainable_parameters_hash"],
        "evaluation_environment_hash": base_payload["evaluation_environment"]["environment_hash"],
    }
    for key in BASE_KEYS:
        row[key] = base_summary.numeric_metric(base_payload, key, eval_path)
    row["mean_primary_linf_acc"] = base_summary.mean(row[key] for key in base_summary.LINF_KEYS)
    row["mean_auxiliary_l2_acc"] = base_summary.mean(row[key] for key in base_summary.L2_KEYS)
    row["mean_robust_acc"] = base_summary.mean(row[key] for key in base_summary.ROBUST_KEYS)
    row["worst_robust_acc"] = min(row[key] for key in base_summary.ROBUST_KEYS)

    sweep_dir = run_dir / "epsilon_sweep_eval"
    linf_budget_values: list[float] = []
    l2_budget_values: list[float] = []
    for label, eps in zip(base_summary.LINF_LABELS, base_summary.LINF_EPS):
        path = sweep_dir / f"linf_{label}.json"
        payload = base_summary.verify_evaluation_artifact(
            path,
            checkpoint=run_dir / "best.pt",
            expected_seed=seed,
            required_attacks=[key.removesuffix("_acc") for key in base_summary.LINF_SWEEP_KEYS],
            expected_eps_linf=eps,
            expected_eps_l2=0.5,
        )
        for key in base_summary.LINF_SWEEP_KEYS:
            value = base_summary.numeric_metric(payload, key, path)
            row[f"{key}_{label}"] = value
            linf_budget_values.append(value)
    for label, eps in zip(base_summary.L2_LABELS, base_summary.L2_EPS):
        path = sweep_dir / f"l2_{label}.json"
        payload = base_summary.verify_evaluation_artifact(
            path,
            checkpoint=run_dir / "best.pt",
            expected_seed=seed,
            required_attacks=[key.removesuffix("_acc") for key in base_summary.L2_SWEEP_KEYS],
            expected_eps_linf=8 / 255,
            expected_eps_l2=eps,
        )
        for key in base_summary.L2_SWEEP_KEYS:
            value = base_summary.numeric_metric(payload, key, path)
            row[f"{key}_{label}"] = value
            l2_budget_values.append(value)
    row["linf_budget_mean_acc"] = base_summary.mean(linf_budget_values)
    row["l2_budget_mean_acc"] = base_summary.mean(l2_budget_values)
    row["overall_budget_mean_acc"] = base_summary.mean([*linf_budget_values, *l2_budget_values])

    marker_path = run_dir / ".complete.json"
    if marker_path.exists():
        marker = base_summary.read_json(marker_path)
        artifacts = supplemental_expected_artifacts()
        hashes = marker.get("artifact_sha256", {})
        if (
            marker.get("validated") is not True
            or marker.get("run_kind") != "formal_12"
            or marker.get("formal_suite_target") != "15_runs"
            or marker.get("supplemental_implementation_hash") != supplemental_implementation_hash()
            or set(hashes) != set(artifacts)
            or any(hashes[name] != base_summary.file_sha256(run_dir / name) for name in artifacts)
        ):
            raise ValueError(f"Invalid or stale supplemental completion marker: {marker_path}")
    return row


def validate_run(run_dir: Path) -> dict[str, Any]:
    if run_dir.name.startswith(f"{SUPPLEMENT_METHOD}_seed"):
        return supplemental_run_row(run_dir)
    return base_summary.run_row(run_dir)


def write_completion_marker(run_dir: Path) -> None:
    run_dir = require_under_task(run_dir)
    if not run_dir.name.startswith(f"{SUPPLEMENT_METHOD}_seed"):
        raise ValueError("Only the projected-PDA 0.1 supplement uses this completion-marker writer.")
    marker = run_dir / ".complete.json"
    if marker.exists():
        supplemental_run_row(run_dir)
        print(f"Validated existing completion marker: {marker}")
        return
    supplemental_run_row(run_dir)
    config = base_summary.read_json(run_dir / "config.json")
    artifacts = supplemental_expected_artifacts()
    payload = {
        "validated": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "run_kind": "formal_12",
        "formal_suite_target": "15_runs",
        "paper_recipe_hash": config["paper_recipe_hash"],
        "implementation_hash": config["implementation_hash"],
        "supplemental_implementation_hash": supplemental_implementation_hash(),
        "artifact_sha256": {name: base_summary.file_sha256(run_dir / name) for name in artifacts},
    }
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    supplemental_run_row(run_dir)
    print(f"Wrote and validated completion marker: {marker}")


def aggregate_rows(rows: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    metric_keys = [*BASE_KEYS, *DERIVED_KEYS, *DELTA_KEYS]
    fields = ["method", "n_seeds"]
    for key in metric_keys:
        fields.extend([f"{key}_mean", f"{key}_sample_std"])
    output: list[dict[str, Any]] = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != 3:
            raise ValueError(f"Expected three seeds for {method}, found {len(selected)}")
        item: dict[str, Any] = {"method": method, "n_seeds": 3}
        for key in metric_keys:
            values = [float(row[key]) for row in selected]
            item[f"{key}_mean"] = statistics.mean(values)
            item[f"{key}_sample_std"] = statistics.stdev(values)
        output.append(item)
    return fields, output


def write_key_metrics(output_dir: Path, aggregate: list[dict[str, Any]]) -> None:
    metrics = [
        "clean_acc",
        "pgd_linf_20_acc",
        "autoattack_linf_acc",
        "pgd_l2_20_acc",
        "autoattack_l2_acc",
        "mean_primary_linf_acc",
        "mean_auxiliary_l2_acc",
        "mean_robust_acc",
        "worst_robust_acc",
        "overall_budget_mean_acc",
    ]
    fields = ["method", "n_seeds", *[suffix for metric in metrics for suffix in (f"{metric}_mean", f"{metric}_sample_std")]]
    base_summary.write_table(
        output_dir / "vit_15_runs_key_metrics.csv",
        output_dir / "vit_15_runs_key_metrics.md",
        fields,
        aggregate,
    )


def write_projection_matched_deltas(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    metrics = [
        "clean_acc",
        "autoattack_linf_acc",
        "autoattack_l2_acc",
        "mean_primary_linf_acc",
        "mean_auxiliary_l2_acc",
        "mean_robust_acc",
        "worst_robust_acc",
        "overall_budget_mean_acc",
    ]
    by_key = {(row["method"], int(row["seed"])): row for row in rows}
    delta_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        low = by_key[("pda_proj_w0p1", seed)]
        high = by_key[("pda_proj_w0p2", seed)]
        item: dict[str, Any] = {"comparison": "pda_proj_w0p1_minus_w0p2", "seed": seed}
        for metric in metrics:
            item[f"delta_{metric}"] = float(low[metric]) - float(high[metric])
        delta_rows.append(item)
    fields = ["comparison", "seed", *[f"delta_{metric}" for metric in metrics]]
    base_summary.write_table(
        output_dir / "vit_projection_w0p1_minus_w0p2_by_seed.csv",
        output_dir / "vit_projection_w0p1_minus_w0p2_by_seed.md",
        fields,
        delta_rows,
    )


def summarize(runs_dir: Path, output_dir: Path) -> None:
    runs_dir = require_under_task(runs_dir)
    output_dir = require_under_task(output_dir)
    expected = [runs_dir / f"{method}_seed{seed}" for method in METHODS for seed in SEEDS]
    expected_names = {path.name for path in expected}
    if runs_dir.is_dir():
        extra = sorted(path.name for path in runs_dir.iterdir() if path.is_dir() and path.name not in expected_names)
        if extra:
            raise ValueError(f"Unexpected extra run directories under {runs_dir}: {extra}")
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    for run_dir in expected:
        try:
            if not (run_dir / ".complete.json").is_file():
                raise ValueError(f"Missing completion marker: {run_dir / '.complete.json'}")
            rows.append(validate_run(run_dir))
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError("15-run audit failed:\n- " + "\n- ".join(errors))

    base_summary.add_matched_deltas(rows)
    base_summary.validate_seed_hashes(rows)
    base_summary.validate_evaluation_environments(rows)
    run_fields = [
        "method",
        "seed",
        "checkpoint",
        "initialization_hash",
        "evaluation_environment_hash",
        *BASE_KEYS,
        *DERIVED_KEYS,
        *DELTA_KEYS,
    ]
    base_summary.write_table(
        output_dir / "vit_15_runs.csv",
        output_dir / "vit_15_runs.md",
        run_fields,
        rows,
    )
    aggregate_fields, aggregate = aggregate_rows(rows)
    base_summary.write_table(
        output_dir / "vit_15_runs_aggregate.csv",
        output_dir / "vit_15_runs_aggregate.md",
        aggregate_fields,
        aggregate,
    )
    write_key_metrics(output_dir, aggregate)
    write_projection_matched_deltas(output_dir, rows)
    print(f"Validated and summarized all {len(rows)} runs in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize the original 12 ViT runs plus PDA-Proj 0.1.")
    parser.add_argument("--runs_dir", default="vit_12_runs_task/paper_recipe_runs")
    parser.add_argument("--output_dir", default="vit_12_runs_task/paper_recipe_summaries")
    parser.add_argument("--validate_run", default="")
    parser.add_argument("--mark_complete", default="")
    args = parser.parse_args()
    if args.validate_run:
        print(json.dumps(validate_run(require_under_task(args.validate_run)), indent=2, sort_keys=True))
        return
    if args.mark_complete:
        write_completion_marker(require_under_task(args.mark_complete))
        return
    summarize(require_under_task(args.runs_dir), require_under_task(args.output_dir))


if __name__ == "__main__":
    main()
