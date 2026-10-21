from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = Path(__file__).resolve().parents[1]
TIMM_WHEEL = TASK_ROOT / ".deps" / "timm-1.0.15-py3-none-any.whl"
if TIMM_WHEEL.is_file() and str(TIMM_WHEEL) not in sys.path:
    sys.path.insert(0, str(TIMM_WHEEL))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vit_12_runs_task.paper_recipe.config import paper_recipe_hash
from vit_12_runs_task.paper_recipe.integrity import (
    file_sha256,
    implementation_hash,
    verify_evaluation_artifact,
)
from vit_12_runs_task.paper_recipe.supplemental_table_s4 import (
    EXPERIMENTS,
    METHOD_SPECS,
    SUPPLEMENT_VERSION,
    TABLE_S4_LOCK_NAME,
    table_s4_implementation_hash,
    verify_table_s4_lock,
)
from vit_12_runs_task.scripts import summarize_vit_runs as base_summary


SEED = 0
EXPECTED_INITIALIZATION_HASH = "1ea7e99480dc31523e62aa8af5c496204931c15052e4fc660277e9acba45d022"
ALL_EVAL_ATTACKS = [
    "clean",
    "fgsm_linf",
    "pgd_linf_20",
    "pgd_linf_50",
    "autoattack_linf",
    "pgd_l2_20",
    "pgd_l2_50",
    "autoattack_l2",
]
S4_ATTACKS = ALL_EVAL_ATTACKS[1:]
S4_METRIC_KEYS = [f"{attack}_acc" for attack in S4_ATTACKS]
REUSED_EXPERIMENTS = ("pgd_at_linf10", "pda_no_proj_w0p1")

TABLE_ROWS: tuple[dict[str, str], ...] = (
    {"experiment": "s4_pgd_at_linf5", "method": "PGD-AT", "sources": "PGD-Linf-5"},
    {"experiment": "pgd_at_linf10", "method": "PGD-AT", "sources": "PGD-Linf-10"},
    {"experiment": "s4_avg_four_source", "method": "Avg", "sources": "four-source"},
    {"experiment": "s4_avg_five_source", "method": "Avg", "sources": "five-source"},
    {"experiment": "s4_pcgrad_four_source", "method": "PCGrad", "sources": "four-source"},
    {"experiment": "s4_pcgrad_five_source", "method": "PCGrad", "sources": "five-source"},
    {
        "experiment": "s4_pcgrad_matched_two_source",
        "method": "PCGrad",
        "sources": "matched two-source",
    },
    {"experiment": "s4_mgda_four_source", "method": "MGDA", "sources": "four-source"},
    {"experiment": "s4_mgda_five_source", "method": "MGDA", "sources": "five-source"},
    {
        "experiment": "s4_mgda_matched_two_source",
        "method": "MGDA",
        "sources": "matched two-source",
    },
    {
        "experiment": "pda_no_proj_w0p1",
        "method": "PDA (`lambda=.1`)",
        "sources": "matched two-source",
    },
)
ROW_BY_EXPERIMENT = {row["experiment"]: row for row in TABLE_ROWS}
NEW_ARTIFACTS = [
    "planned_config.json",
    "config.json",
    "implementation_lock.json",
    TABLE_S4_LOCK_NAME,
    "training_complete.json",
    "train_log.csv",
    "batch_diagnostics.csv",
    "best.pt",
    "last.pt",
    "eval.json",
]


def require_under_task(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(TASK_ROOT)
    except ValueError as exc:
        raise ValueError(f"Table S4 path must remain under {TASK_ROOT}: {resolved}") from exc
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
        raise ValueError("Cannot average an empty Table S4 attack set.")
    return sum(items) / len(items)


def parse_run_name(run_dir: Path) -> str:
    suffix = f"_seed{SEED}"
    if not run_dir.name.endswith(suffix):
        raise ValueError(f"Table S4 only accepts seed-0 run directories: {run_dir.name}")
    experiment = run_dir.name[: -len(suffix)]
    if experiment not in ROW_BY_EXPERIMENT:
        raise ValueError(f"Unexpected Table S4 run directory: {run_dir.name}")
    return experiment


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint payload: {path}")
    return payload


def validate_checkpoint_pair(run_dir: Path, expected_config: dict[str, Any]) -> None:
    best = load_checkpoint(run_dir / "best.pt")
    last = load_checkpoint(run_dir / "last.pt")
    if int(best.get("epoch", -1)) != 40 or int(last.get("epoch", -1)) != 40:
        raise ValueError(f"Table S4 checkpoints are not final epoch 40: {run_dir}")
    if set(best.get("model_state_dict", {})) != set(last.get("model_state_dict", {})):
        raise ValueError(f"best.pt/last.pt model keys differ: {run_dir}")
    if any(
        not torch.equal(best["model_state_dict"][key], last["model_state_dict"][key])
        for key in best["model_state_dict"]
    ):
        raise ValueError(f"best.pt/last.pt model tensors differ: {run_dir}")
    checkpoint_config = best.get("config", {})
    checks = (
        "experiment",
        "seed",
        "method",
        "aggregation",
        "attack_domains",
        "initial_trainable_parameters_hash",
        "implementation_hash",
        "table_s4_implementation_hash",
        "table_s4_implementation_files",
    )
    mismatches = {
        key: (checkpoint_config.get(key), expected_config.get(key))
        for key in checks
        if checkpoint_config.get(key) != expected_config.get(key)
    }
    if mismatches:
        raise ValueError(f"Checkpoint/config metadata mismatch in {run_dir}: {mismatches}")


def validate_new_completion_marker(
    run_dir: Path,
    config: dict[str, Any],
    evaluation_environment_hash: str,
) -> None:
    marker_path = run_dir / ".complete.json"
    if not marker_path.exists():
        return
    marker = read_json(marker_path)
    hashes = marker.get("artifact_sha256", {})
    if (
        marker.get("validated") is not True
        or marker.get("run_kind") != "formal_table_s4_seed0"
        or marker.get("paper_recipe_hash") != paper_recipe_hash()
        or marker.get("implementation_hash") != implementation_hash()
        or marker.get("table_s4_implementation_hash") != table_s4_implementation_hash()
        or set(hashes) != set(NEW_ARTIFACTS)
        or any(hashes[name] != file_sha256(run_dir / name) for name in NEW_ARTIFACTS)
        or marker.get("checkpoint_sha256") != file_sha256(run_dir / "best.pt")
        or marker.get("evaluation_environment_hash") != evaluation_environment_hash
        or marker.get("initial_trainable_parameters_hash")
        != config.get("initial_trainable_parameters_hash")
    ):
        raise ValueError(f"Invalid or stale Table S4 completion marker: {marker_path}")


def validate_new_run(
    run_dir: Path,
    *,
    require_completion: bool,
    expected_environment_hash: str | None,
) -> dict[str, Any]:
    experiment = parse_run_name(run_dir)
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Not a supplemental Table S4 run: {run_dir}")
    if require_completion and not (run_dir / ".complete.json").is_file():
        raise ValueError(f"Missing Table S4 completion marker: {run_dir / '.complete.json'}")

    # Reuse the strict 40-epoch paper-recipe audit (recipe, base lock, exact LR,
    # 50k samples/epoch, diagnostics, and final checkpoint equality).
    config = base_summary.validate_static_artifacts(run_dir)
    spec = METHOD_SPECS[experiment]
    expected_sources = list(spec["attack_domains"])
    table_lock = verify_table_s4_lock(run_dir / TABLE_S4_LOCK_NAME)
    current_table_hash = table_s4_implementation_hash()
    if (
        config.get("experiment") != experiment
        or int(config.get("seed", -1)) != SEED
        or config.get("method") != spec["method"]
        or config.get("aggregation") != spec["aggregation"]
        or config.get("attack_domains") != expected_sources
        or config.get("preference_weights") is not None
        or config.get("table_s4_implementation_hash") != current_table_hash
        or config.get("supplemental_base_implementation_hash") != implementation_hash()
        or config.get("initial_trainable_parameters_hash") != EXPECTED_INITIALIZATION_HASH
    ):
        raise ValueError(f"Table S4 method/config mismatch in {run_dir}")
    if table_lock.get("base_implementation_hash") != implementation_hash():
        raise ValueError(f"Table S4 lock/base implementation mismatch in {run_dir}")
    if spec["aggregation"] == "avg":
        expected_weights = {source: 1.0 / len(expected_sources) for source in expected_sources}
        if not config.get("normalize_domain_weights") or config.get("parsed_domain_weights") != expected_weights:
            raise ValueError(f"Avg is not an equal arithmetic mean in {run_dir}")
    elif config.get("parsed_domain_weights") != {} or config.get("normalize_domain_weights"):
        raise ValueError(f"{spec['aggregation']} unexpectedly recorded preference weights in {run_dir}")

    planned = read_json(run_dir / "planned_config.json")
    planned_runtime = planned.get("runtime", {})
    if (
        planned.get("experiment") != experiment
        or int(planned.get("seed", -1)) != SEED
        or planned.get("method_spec") != spec
        or planned.get("table_s4_implementation_hash") != current_table_hash
        or planned.get("supplement_version") != SUPPLEMENT_VERSION
        or planned_runtime.get("run_kind") != "formal_12"
        or int(planned_runtime.get("epochs_requested", -1)) != 40
        or int(planned_runtime.get("batch_size_actual", -1)) != 64
    ):
        raise ValueError(f"Invalid Table S4 planned config in {run_dir}")
    completion = read_json(run_dir / "training_complete.json")
    if (
        completion.get("run_kind") != "formal_12"
        or int(completion.get("epochs", -1)) != 40
        or completion.get("table_s4_implementation_hash") != current_table_hash
        or completion.get("supplement_version") != SUPPLEMENT_VERSION
        or completion.get("last_checkpoint_sha256") != file_sha256(run_dir / "last.pt")
    ):
        raise ValueError(f"Invalid Table S4 training completion metadata in {run_dir}")
    validate_checkpoint_pair(run_dir, config)

    eval_path = run_dir / "eval.json"
    evaluation = verify_evaluation_artifact(
        eval_path,
        checkpoint=run_dir / "best.pt",
        expected_seed=SEED,
        required_attacks=ALL_EVAL_ATTACKS,
        expected_eps_linf=8 / 255,
        expected_eps_l2=0.5,
        expected_environment_hash=expected_environment_hash,
    )
    checkpoint_config = evaluation.get("checkpoint_config", {})
    if (
        checkpoint_config.get("experiment") != experiment
        or checkpoint_config.get("method") != spec["method"]
        or checkpoint_config.get("aggregation") != spec["aggregation"]
        or checkpoint_config.get("attack_domains") != expected_sources
        or checkpoint_config.get("table_s4_implementation_hash") != current_table_hash
        or checkpoint_config.get("initial_trainable_parameters_hash") != EXPECTED_INITIALIZATION_HASH
    ):
        raise ValueError(f"Evaluation/checkpoint Table S4 metadata mismatch in {run_dir}")
    evaluation_environment_hash = str(
        evaluation.get("evaluation_environment", {}).get("environment_hash", "")
    )
    if not evaluation_environment_hash:
        raise ValueError(f"Missing Table S4 evaluation environment hash in {eval_path}")
    validate_new_completion_marker(run_dir, config, evaluation_environment_hash)
    return build_result_row(
        experiment=experiment,
        run_dir=run_dir,
        evaluation=evaluation,
        training_implementation_hash=current_table_hash,
        base_implementation_hash=implementation_hash(),
    )


def validate_reused_run(
    run_dir: Path,
    *,
    require_completion: bool,
    expected_environment_hash: str | None,
) -> dict[str, Any]:
    experiment = parse_run_name(run_dir)
    if experiment not in REUSED_EXPERIMENTS:
        raise ValueError(f"Not a reused Table S4 run: {run_dir}")
    if require_completion and not (run_dir / ".complete.json").is_file():
        raise ValueError(f"Missing reused-run completion marker: {run_dir / '.complete.json'}")
    # This verifies the original lock, 40-epoch artifacts, checkpoint, complete
    # evaluation, epsilon artifacts, and old completion marker without changing
    # any old run.  Table S4 metrics below are still recomputed from eval.json.
    base_row = base_summary.run_row(run_dir)
    config = base_summary.read_json(run_dir / "config.json")
    if (
        int(config.get("seed", -1)) != SEED
        or config.get("initial_trainable_parameters_hash") != EXPECTED_INITIALIZATION_HASH
    ):
        raise ValueError(f"Reused Table S4 initialization mismatch in {run_dir}")
    environment_hash = str(base_row["evaluation_environment_hash"])
    if expected_environment_hash is not None and environment_hash != expected_environment_hash:
        raise ValueError(
            f"Reused evaluation environment differs from approval: {run_dir}; "
            f"saved={environment_hash}, expected={expected_environment_hash}"
        )
    evaluation = read_json(run_dir / "eval.json")
    return build_result_row(
        experiment=experiment,
        run_dir=run_dir,
        evaluation=evaluation,
        training_implementation_hash=str(config["implementation_hash"]),
        base_implementation_hash=str(config["implementation_hash"]),
    )


def build_result_row(
    *,
    experiment: str,
    run_dir: Path,
    evaluation: dict[str, Any],
    training_implementation_hash: str,
    base_implementation_hash: str,
) -> dict[str, Any]:
    metadata = ROW_BY_EXPERIMENT[experiment]
    eval_path = run_dir / "eval.json"
    values = {key: numeric_metric(evaluation, key, eval_path) for key in S4_METRIC_KEYS}
    robust_values = [values[key] for key in S4_METRIC_KEYS]
    return {
        "Method": metadata["method"],
        "Sources": metadata["sources"],
        "Clean": numeric_metric(evaluation, "clean_acc", eval_path),
        "AA-Linf": values["autoattack_linf_acc"],
        "AA-L2": values["autoattack_l2_acc"],
        "Mean": mean(robust_values),
        "Worst": min(robust_values),
        "Checkpoint Path": str((run_dir / "best.pt").resolve()),
        "Checkpoint SHA256": file_sha256(run_dir / "best.pt"),
        "Training Implementation Hash": training_implementation_hash,
        "Base Implementation Hash": base_implementation_hash,
        "Evaluation Environment Hash": str(
            evaluation.get("evaluation_environment", {}).get("environment_hash", "")
        ),
        "Experiment": experiment,
        **{key: values[key] for key in S4_METRIC_KEYS},
    }


def validate_run(
    run_dir: Path,
    *,
    require_completion: bool,
    expected_environment_hash: str | None,
) -> dict[str, Any]:
    experiment = parse_run_name(run_dir)
    if experiment in EXPERIMENTS:
        return validate_new_run(
            run_dir,
            require_completion=require_completion,
            expected_environment_hash=expected_environment_hash,
        )
    return validate_reused_run(
        run_dir,
        require_completion=require_completion,
        expected_environment_hash=expected_environment_hash,
    )


def mark_complete(run_dir: Path, expected_environment_hash: str | None) -> dict[str, Any]:
    experiment = parse_run_name(run_dir)
    if experiment not in EXPERIMENTS:
        raise ValueError("Only new supplemental Table S4 runs receive this completion marker.")
    marker_path = run_dir / ".complete.json"
    if marker_path.exists():
        row = validate_new_run(
            run_dir,
            require_completion=True,
            expected_environment_hash=expected_environment_hash,
        )
        print(f"Validated existing Table S4 completion marker: {marker_path}")
        return row
    row = validate_new_run(
        run_dir,
        require_completion=False,
        expected_environment_hash=expected_environment_hash,
    )
    marker = {
        "validated": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "run_kind": "formal_table_s4_seed0",
        "experiment": experiment,
        "seed": SEED,
        "paper_recipe_hash": paper_recipe_hash(),
        "implementation_hash": implementation_hash(),
        "table_s4_implementation_hash": table_s4_implementation_hash(),
        "initial_trainable_parameters_hash": EXPECTED_INITIALIZATION_HASH,
        "checkpoint_sha256": row["Checkpoint SHA256"],
        "evaluation_environment_hash": row["Evaluation Environment Hash"],
        "artifact_sha256": {name: file_sha256(run_dir / name) for name in NEW_ARTIFACTS},
    }
    temporary = marker_path.with_suffix(marker_path.suffix + ".tmp")
    temporary.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(marker_path)
    validate_new_run(
        run_dir,
        require_completion=True,
        expected_environment_hash=expected_environment_hash,
    )
    print(f"Wrote and validated Table S4 completion marker: {marker_path}")
    return row


CSV_FIELDS = [
    "Method",
    "Sources",
    "Clean",
    "AA-Linf",
    "AA-L2",
    "Mean",
    "Worst",
    "Checkpoint Path",
    "Checkpoint SHA256",
    "Training Implementation Hash",
    "Base Implementation Hash",
    "Evaluation Environment Hash",
    "Experiment",
    *S4_METRIC_KEYS,
]


def write_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "deit_table_s4_seed0.csv"
    md_path = output_dir / "deit_table_s4_seed0.md"
    csv_tmp = csv_path.with_suffix(".csv.tmp")
    md_tmp = md_path.with_suffix(".md.tmp")
    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    visible = ["Method", "Sources", "Clean", "AA-Linf", "AA-L2", "Mean", "Worst"]
    with md_tmp.open("w", encoding="utf-8") as handle:
        handle.write("# CIFAR-10/DeiT-Tiny, seed 0 — Table S4\n\n")
        handle.write("Mean averages seven attacks: FGSM-Linf, PGD-Linf-20/50, AutoAttack-Linf, ")
        handle.write("PGD-L2-20/50, and AutoAttack-L2. Worst is their minimum.\n\n")
        handle.write("| " + " | ".join(visible) + " |\n")
        handle.write("| " + " | ".join("---" for _ in visible) + " |\n")
        for row in rows:
            cells = [str(row["Method"]), str(row["Sources"])]
            cells.extend(f"{100.0 * float(row[field]):.2f}" for field in visible[2:])
            handle.write("| " + " | ".join(cells) + " |\n")
        handle.write("\n## Provenance\n\n")
        provenance_fields = [
            "Experiment",
            "Checkpoint Path",
            "Checkpoint SHA256",
            "Training Implementation Hash",
            "Evaluation Environment Hash",
        ]
        handle.write("| " + " | ".join(provenance_fields) + " |\n")
        handle.write("| " + " | ".join("---" for _ in provenance_fields) + " |\n")
        for row in rows:
            cells = [str(row[field]).replace("|", "\\|") for field in provenance_fields]
            handle.write("| " + " | ".join(cells) + " |\n")
    csv_tmp.replace(csv_path)
    md_tmp.replace(md_path)
    return csv_path, md_path


def expected_run_dirs(runs_dir: Path) -> list[Path]:
    return [runs_dir / f"{row['experiment']}_seed{SEED}" for row in TABLE_ROWS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Strictly validate and build DeiT-Tiny Table S4 seed 0.")
    parser.add_argument("--runs-dir", default="vit_12_runs_task/paper_recipe_runs")
    parser.add_argument("--output-dir", default="vit_12_runs_task/paper_recipe_summaries")
    parser.add_argument("--validate-run", default="")
    parser.add_argument("--mark-complete", default="")
    parser.add_argument("--expected-evaluation-environment-hash", default="")
    args = parser.parse_args()
    expected_environment_hash = args.expected_evaluation_environment_hash or None

    if args.validate_run and args.mark_complete:
        raise SystemExit("Choose only one of --validate-run and --mark-complete.")
    if args.validate_run:
        row = validate_run(
            require_under_task(args.validate_run),
            require_completion=True,
            expected_environment_hash=expected_environment_hash,
        )
        print(json.dumps(row, indent=2, sort_keys=True))
        return
    if args.mark_complete:
        row = mark_complete(require_under_task(args.mark_complete), expected_environment_hash)
        print(json.dumps(row, indent=2, sort_keys=True))
        return

    runs_dir = require_under_task(args.runs_dir)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for run_dir in expected_run_dirs(runs_dir):
        try:
            rows.append(
                validate_run(
                    run_dir,
                    require_completion=True,
                    expected_environment_hash=expected_environment_hash,
                )
            )
        except (ValueError, OSError, KeyError) as exc:
            errors.append(str(exc))
    if errors:
        raise SystemExit("Table S4 audit failed; no summary was written:\n- " + "\n- ".join(errors))
    if [row["Experiment"] for row in rows] != [row["experiment"] for row in TABLE_ROWS]:
        raise SystemExit("Internal Table S4 row order mismatch; no summary was written.")
    if any(row["Evaluation Environment Hash"] == "" for row in rows):
        raise SystemExit("Missing evaluation environment hash; no summary was written.")
    environment_hashes = {row["Evaluation Environment Hash"] for row in rows}
    if len(environment_hashes) != 1:
        raise SystemExit(f"Evaluation environments differ across Table S4 rows: {sorted(environment_hashes)}")
    output_dir = require_under_task(args.output_dir)
    csv_path, md_path = write_outputs(output_dir, rows)
    print(f"Validated 11 Table S4 rows and wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
