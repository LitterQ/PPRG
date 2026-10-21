from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

TASK_ROOT = Path(__file__).resolve().parents[1]
TIMM_WHEEL = TASK_ROOT / ".deps" / "timm-1.0.15-py3-none-any.whl"
if TIMM_WHEEL.is_file() and str(TIMM_WHEEL) not in sys.path:
    sys.path.insert(0, str(TIMM_WHEEL))

from vit_12_runs_task.paper_recipe.integrity import (
    file_sha256,
    implementation_hash,
    verify_evaluation_artifact,
    verify_lock,
)


METHODS = ["pgd_at_linf10", "pda_no_proj_w0p1", "pda_no_proj_w0p2", "pda_proj_w0p2"]
SEEDS = [0, 1, 2]
BASELINE = "pgd_at_linf10"
BASE_KEYS = [
    "clean_acc",
    "fgsm_linf_acc",
    "pgd_linf_20_acc",
    "pgd_linf_50_acc",
    "cw_linf_20_acc",
    "cw_linf_50_acc",
    "autoattack_linf_acc",
    "pgd_l2_20_acc",
    "pgd_l2_50_acc",
    "cw_l2_20_acc",
    "cw_l2_50_acc",
    "autoattack_l2_acc",
]
LINF_KEYS = BASE_KEYS[1:7]
L2_KEYS = BASE_KEYS[7:12]
ROBUST_KEYS = [*LINF_KEYS, *L2_KEYS]
LINF_SWEEP_KEYS = [
    "fgsm_linf_acc",
    "pgd_linf_20_acc",
    "pgd_linf_50_acc",
    "cw_linf_20_acc",
    "cw_linf_50_acc",
]
L2_SWEEP_KEYS = ["pgd_l2_20_acc", "pgd_l2_50_acc", "cw_l2_20_acc", "cw_l2_50_acc"]
LINF_LABELS = ["eps4", "eps8", "eps12", "eps16"]
L2_LABELS = ["eps0p25", "eps0p5", "eps0p75", "eps1p0"]
LINF_EPS = [4 / 255, 8 / 255, 12 / 255, 16 / 255]
L2_EPS = [0.25, 0.5, 0.75, 1.0]
DERIVED_KEYS = [
    "mean_primary_linf_acc",
    "mean_auxiliary_l2_acc",
    "mean_robust_acc",
    "worst_robust_acc",
    "linf_budget_mean_acc",
    "l2_budget_mean_acc",
    "overall_budget_mean_acc",
]
DELTA_KEYS = [
    "delta_autoattack_linf_acc",
    "delta_autoattack_l2_acc",
    "delta_mean_primary_linf_acc",
    "delta_mean_auxiliary_l2_acc",
    "delta_mean_robust_acc",
    "delta_worst_robust_acc",
    "delta_linf_budget_mean_acc",
    "delta_l2_budget_mean_acc",
    "delta_overall_budget_mean_acc",
]


def require_under_task(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(TASK_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path must remain under {TASK_ROOT}; got {resolved}") from exc
    return resolved


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def numeric_metric(payload: dict[str, Any], key: str, path: Path) -> float:
    value = payload.get("metrics", {}).get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Metric {key} in {path} is missing or non-finite: {value!r}")
    return float(value)


def mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise ValueError("Cannot compute the mean of an empty metric set.")
    return sum(items) / len(items)


def parse_run_name(name: str) -> tuple[str, int]:
    match = re.fullmatch(r"(.+)_seed([012])", name)
    if not match or match.group(1) not in METHODS:
        raise ValueError(f"Unexpected ViT run name: {name}")
    return match.group(1), int(match.group(2))


def validate_static_artifacts(run_dir: Path) -> dict[str, Any]:
    common_required = [
        "config.json",
        "console.log",
        "train_log.csv",
        "best.pt",
        "last.pt",
        "eval.json",
    ]
    config = read_json(run_dir / "config.json")
    if config.get("model_alias") == "deit_tiny_patch4_32_paper":
        required = [
            *common_required,
            "batch_diagnostics.csv",
            "training_complete.json",
            "implementation_lock.json",
            "planned_config.json",
        ]
    else:
        required = [*common_required, "gradient_cosine_epoch.csv", "gradient_conflict_summary.json"]
    for name in required:
        path = run_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty required artifact: {path}")
    if config.get("model_alias") not in {"deit_tiny_patch4_32", "deit_tiny_patch4_32_paper"}:
        raise ValueError(f"Wrong model alias in {run_dir / 'config.json'}")
    if config.get("model_alias") == "deit_tiny_patch4_32_paper":
        from vit_12_runs_task.paper_recipe.config import paper_recipe_hash

        verify_lock(run_dir / "implementation_lock.json")
        if config.get("run_kind") != "formal_12":
            raise ValueError(f"Non-formal paper run is excluded from summaries: {run_dir}")
        if config.get("paper_recipe_hash") != paper_recipe_hash():
            raise ValueError(f"Paper recipe hash mismatch in {run_dir}")
        if config.get("implementation_hash") != implementation_hash():
            raise ValueError(f"Paper implementation hash mismatch in {run_dir}")
        if (
            int(config.get("epochs", -1)) != 40
            or int(config.get("batch_size", -1)) != 64
            or config.get("use_fake_data") is not False
            or int(config.get("max_train_batches", -1)) != 0
        ):
            raise ValueError(f"Formal training bounds mismatch in {run_dir}")
        completion = read_json(run_dir / "training_complete.json")
        if (
            completion.get("run_kind") != "formal_12"
            or int(completion.get("epochs", -1)) != 40
            or completion.get("paper_recipe_hash") != paper_recipe_hash()
            or completion.get("implementation_hash") != implementation_hash()
            or completion.get("last_checkpoint_sha256") != file_sha256(run_dir / "last.pt")
        ):
            raise ValueError(f"Training completion marker mismatch in {run_dir}")
        planned = read_json(run_dir / "planned_config.json")
        runtime = planned.get("runtime", {})
        if (
            planned.get("paper_recipe_hash") != paper_recipe_hash()
            or planned.get("experiment") != config.get("experiment")
            or int(planned.get("seed", -1)) != int(config.get("seed", -2))
            or runtime.get("run_kind") != "formal_12"
            or int(runtime.get("epochs_requested", -1)) != 40
            or int(runtime.get("batch_size_actual", -1)) != 64
            or int(runtime.get("num_workers", -1)) != int(config.get("num_workers", -2))
            or runtime.get("pretrained_checkpoint_sha256") != config.get("pretrained_checkpoint_sha256")
        ):
            raise ValueError(f"Planned/resolved formal config mismatch in {run_dir}")
        with (run_dir / "train_log.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            train_rows = list(csv.DictReader(handle))
        if [int(row["epoch"]) for row in train_rows] != list(range(1, 41)):
            raise ValueError(f"Formal train_log does not contain exactly epochs 1..40 in {run_dir}")
        expected_lrs = [0.1] * 35 + [0.01] * 3 + [0.001] * 2
        if any(
            int(row["train_samples_seen"]) != 50000
            or int(row["train_batches_seen"]) != 782
            or abs(float(row["lr"]) - expected_lrs[index]) > 1e-12
            for index, row in enumerate(train_rows)
        ):
            raise ValueError(f"Formal train_log sample/batch/LR audit failed in {run_dir}")
        with (run_dir / "batch_diagnostics.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            diagnostic_rows = list(csv.DictReader(handle))
        if len(diagnostic_rows) != 40 * 782:
            raise ValueError(f"Expected 31,280 batch diagnostics rows in {run_dir}, found {len(diagnostic_rows)}")
        for row in diagnostic_rows:
            for key, value in row.items():
                if value not in ("", None):
                    try:
                        number = float(value)
                    except ValueError:
                        continue
                    if not math.isfinite(number):
                        raise ValueError(f"Non-finite diagnostic {key}={value!r} in {run_dir}")
            if float(row["clip_post_global_norm"]) > 1.00001:
                raise ValueError(f"Gradient clipping audit failed in {run_dir}")
        try:
            best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
            last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        except TypeError:
            best = torch.load(run_dir / "best.pt", map_location="cpu")
            last = torch.load(run_dir / "last.pt", map_location="cpu")
        if int(best.get("epoch", -1)) != 40 or int(last.get("epoch", -1)) != 40:
            raise ValueError(f"Final checkpoints are not epoch 40 in {run_dir}")
        if set(best["model_state_dict"]) != set(last["model_state_dict"]) or any(
            not torch.equal(best["model_state_dict"][key], last["model_state_dict"][key])
            for key in best["model_state_dict"]
        ):
            raise ValueError(f"best.pt and last.pt model tensors differ in {run_dir}")
    if not config.get("pretrained_loaded"):
        raise ValueError(f"Pretrained weights were not recorded as loaded in {run_dir}")
    init_hash = config.get("initial_trainable_parameters_hash")
    if not isinstance(init_hash, str) or len(init_hash) != 64:
        raise ValueError(f"Missing deterministic initialization hash in {run_dir}")
    return config


def run_row(run_dir: Path) -> dict[str, Any]:
    method, seed = parse_run_name(run_dir.name)
    config = validate_static_artifacts(run_dir)
    if int(config.get("seed", -1)) != seed:
        raise ValueError(f"Seed metadata mismatch in {run_dir}")
    expected_methods = {
        "pgd_at_linf10": "pgd_at",
        "pda_no_proj_w0p1": "avg_attack_dg",
        "pda_no_proj_w0p2": "avg_attack_dg",
        "pda_proj_w0p2": "primary_projected_aux_l2",
    }
    if config.get("method") != expected_methods[method]:
        raise ValueError(f"Method metadata mismatch in {run_dir}: {config.get('method')!r}")
    if method == "pgd_at_linf10" and config.get("train_attack") != "pgd_linf_10":
        raise ValueError(f"PGD-AT attack mismatch in {run_dir}")
    if method in {"pda_no_proj_w0p1", "pda_no_proj_w0p2"}:
        expected_aux = 0.1 if method.endswith("w0p1") else 0.2
        expected_weights = {"pgd_linf_10": 1.0, "pgd_l2_5": expected_aux}
        if config.get("normalize_domain_weights") or config.get("parsed_domain_weights") != expected_weights:
            raise ValueError(f"Exact no-projection PDA weights mismatch in {run_dir}")
    if method == "pda_proj_w0p2":
        if (
            config.get("primary_domain") != "pgd_linf_10"
            or config.get("aux_domain") != "pgd_l2_5"
            or float(config.get("aux_weight", -1.0)) != 0.2
        ):
            raise ValueError(f"Projected PDA configuration mismatch in {run_dir}")
    eval_path = run_dir / "eval.json"
    base_payload = verify_evaluation_artifact(
        eval_path,
        checkpoint=run_dir / "best.pt",
        expected_seed=seed,
        required_attacks=[key.removesuffix("_acc") for key in BASE_KEYS],
        expected_eps_linf=8 / 255,
        expected_eps_l2=0.5,
    )
    checkpoint_hash = base_payload.get("checkpoint_config", {}).get("initial_trainable_parameters_hash")
    if checkpoint_hash != config["initial_trainable_parameters_hash"]:
        raise ValueError(f"Evaluation/checkpoint initialization metadata mismatch in {run_dir}")
    if (
        base_payload.get("checkpoint_config", {}).get("experiment") != method
        or base_payload.get("checkpoint_config", {}).get("method") != config.get("method")
    ):
        raise ValueError(f"Evaluation/checkpoint method metadata mismatch in {run_dir}")
    row: dict[str, Any] = {
        "method": method,
        "seed": seed,
        "checkpoint": str(run_dir / "best.pt"),
        "initialization_hash": config["initial_trainable_parameters_hash"],
        "evaluation_environment_hash": base_payload["evaluation_environment"]["environment_hash"],
    }
    for key in BASE_KEYS:
        row[key] = numeric_metric(base_payload, key, eval_path)

    row["mean_primary_linf_acc"] = mean(row[key] for key in LINF_KEYS)
    row["mean_auxiliary_l2_acc"] = mean(row[key] for key in L2_KEYS)
    row["mean_robust_acc"] = mean(row[key] for key in ROBUST_KEYS)
    row["worst_robust_acc"] = min(row[key] for key in ROBUST_KEYS)

    sweep_dir = run_dir / "epsilon_sweep_eval"
    linf_budget_values: list[float] = []
    l2_budget_values: list[float] = []
    for label, eps in zip(LINF_LABELS, LINF_EPS):
        path = sweep_dir / f"linf_{label}.json"
        payload = verify_evaluation_artifact(
            path,
            checkpoint=run_dir / "best.pt",
            expected_seed=seed,
            required_attacks=[key.removesuffix("_acc") for key in LINF_SWEEP_KEYS],
            expected_eps_linf=eps,
            expected_eps_l2=0.5,
        )
        for key in LINF_SWEEP_KEYS:
            value = numeric_metric(payload, key, path)
            row[f"{key}_{label}"] = value
            linf_budget_values.append(value)
    for label, eps in zip(L2_LABELS, L2_EPS):
        path = sweep_dir / f"l2_{label}.json"
        payload = verify_evaluation_artifact(
            path,
            checkpoint=run_dir / "best.pt",
            expected_seed=seed,
            required_attacks=[key.removesuffix("_acc") for key in L2_SWEEP_KEYS],
            expected_eps_linf=8 / 255,
            expected_eps_l2=eps,
        )
        for key in L2_SWEEP_KEYS:
            value = numeric_metric(payload, key, path)
            row[f"{key}_{label}"] = value
            l2_budget_values.append(value)
    row["linf_budget_mean_acc"] = mean(linf_budget_values)
    row["l2_budget_mean_acc"] = mean(l2_budget_values)
    row["overall_budget_mean_acc"] = mean([*linf_budget_values, *l2_budget_values])
    marker = run_dir / ".complete.json"
    if marker.exists():
        completion = read_json(marker)
        expected_artifacts = [
            "planned_config.json",
            "config.json",
            "implementation_lock.json",
            "training_complete.json",
            "train_log.csv",
            "batch_diagnostics.csv",
            "best.pt",
            "last.pt",
            "eval.json",
            *[f"epsilon_sweep_eval/linf_{label}.json" for label in LINF_LABELS],
            *[f"epsilon_sweep_eval/l2_{label}.json" for label in L2_LABELS],
        ]
        hashes = completion.get("artifact_sha256", {})
        if (
            completion.get("validated") is not True
            or completion.get("run_kind") != "formal_12"
            or completion.get("paper_recipe_hash") != config.get("paper_recipe_hash")
            or completion.get("implementation_hash") != implementation_hash()
            or set(hashes) != set(expected_artifacts)
            or any(hashes[name] != file_sha256(run_dir / name) for name in expected_artifacts)
        ):
            raise ValueError(f"Invalid or stale completion marker: {marker}")
    return row


def add_matched_deltas(rows: list[dict[str, Any]]) -> None:
    baselines = {int(row["seed"]): row for row in rows if row["method"] == BASELINE}
    if set(baselines) != set(SEEDS):
        raise ValueError(f"Expected baseline seeds {SEEDS}, found {sorted(baselines)}")
    mapping = {
        "delta_autoattack_linf_acc": "autoattack_linf_acc",
        "delta_autoattack_l2_acc": "autoattack_l2_acc",
        "delta_mean_primary_linf_acc": "mean_primary_linf_acc",
        "delta_mean_auxiliary_l2_acc": "mean_auxiliary_l2_acc",
        "delta_mean_robust_acc": "mean_robust_acc",
        "delta_worst_robust_acc": "worst_robust_acc",
        "delta_linf_budget_mean_acc": "linf_budget_mean_acc",
        "delta_l2_budget_mean_acc": "l2_budget_mean_acc",
        "delta_overall_budget_mean_acc": "overall_budget_mean_acc",
    }
    for row in rows:
        baseline = baselines[int(row["seed"])]
        for delta_key, metric_key in mapping.items():
            row[delta_key] = float(row[metric_key]) - float(baseline[metric_key])


def expected_run_dirs(runs_dir: Path) -> list[Path]:
    return [runs_dir / f"{method}_seed{seed}" for method in METHODS for seed in SEEDS]


def format_value(value: Any) -> str:
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def write_table(csv_path: Path, md_path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(fields) + " |\n")
        handle.write("| " + " | ".join("---" for _ in fields) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(format_value(row.get(field, "")) for field in fields) + " |\n")


def aggregate_rows(rows: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    metric_keys = [*BASE_KEYS, *DERIVED_KEYS, *DELTA_KEYS]
    fields = ["method", "n_seeds"]
    for key in metric_keys:
        fields.extend([f"{key}_mean", f"{key}_sample_std"])
    aggregated: list[dict[str, Any]] = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        if len(selected) != 3:
            raise ValueError(f"Expected three seeds for {method}, found {len(selected)}")
        item: dict[str, Any] = {"method": method, "n_seeds": len(selected)}
        for key in metric_keys:
            values = [float(row[key]) for row in selected]
            item[f"{key}_mean"] = statistics.mean(values)
            item[f"{key}_sample_std"] = statistics.stdev(values)
        aggregated.append(item)
    return fields, aggregated


def validate_seed_hashes(rows: list[dict[str, Any]]) -> None:
    for seed in SEEDS:
        hashes = {row["initialization_hash"] for row in rows if int(row["seed"]) == seed}
        if len(hashes) != 1:
            raise ValueError(f"Initialization hashes differ across methods for seed {seed}: {sorted(hashes)}")


def validate_evaluation_environments(rows: list[dict[str, Any]]) -> None:
    hashes = {row["evaluation_environment_hash"] for row in rows}
    if len(hashes) != 1:
        raise ValueError(f"Base evaluation environments differ across formal runs: {sorted(hashes)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize all 12 DeiT runs.")
    parser.add_argument("--runs_dir", default="vit_12_runs_task/runs")
    parser.add_argument("--output_dir", default="vit_12_runs_task/summaries")
    parser.add_argument("--validate_run", default="", help="Validate one run without requiring its completion marker.")
    parser.add_argument("--allow_incomplete", action="store_true")
    args = parser.parse_args()

    if args.validate_run:
        row = run_row(require_under_task(Path(args.validate_run)))
        print(json.dumps(row, indent=2, sort_keys=True))
        return

    runs_dir = require_under_task(Path(args.runs_dir))
    expected_dirs = expected_run_dirs(runs_dir)
    expected_names = {path.name for path in expected_dirs}
    if runs_dir.is_dir() and not args.allow_incomplete:
        extra_names = sorted(path.name for path in runs_dir.iterdir() if path.is_dir() and path.name not in expected_names)
        if extra_names:
            raise SystemExit(f"Unexpected extra run directories under {runs_dir}: {extra_names}")
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for run_dir in expected_dirs:
        try:
            if not args.allow_incomplete and not (run_dir / ".complete.json").is_file():
                raise ValueError(f"Missing completion marker: {run_dir / '.complete.json'}")
            rows.append(run_row(run_dir))
        except ValueError as exc:
            errors.append(str(exc))
    if errors and not args.allow_incomplete:
        raise SystemExit("Run audit failed:\n- " + "\n- ".join(errors))
    if not rows:
        raise SystemExit("No complete ViT runs were found.")

    add_matched_deltas(rows)
    validate_seed_hashes(rows)
    validate_evaluation_environments(rows)
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
    output_dir = require_under_task(Path(args.output_dir))
    write_table(output_dir / "vit_12_runs.csv", output_dir / "vit_12_runs.md", run_fields, rows)
    aggregate_fields, aggregate = aggregate_rows(rows)
    write_table(
        output_dir / "vit_12_runs_aggregate.csv",
        output_dir / "vit_12_runs_aggregate.md",
        aggregate_fields,
        aggregate,
    )
    print(f"Validated and summarized {len(rows)} runs in {output_dir}")


if __name__ == "__main__":
    main()
