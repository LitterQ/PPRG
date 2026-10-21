from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

FIELDNAMES = [
    "method",
    "seed",
    "clean_acc",
    "fgsm_linf_acc",
    "pgd_linf_20_acc",
    "pgd_linf_50_acc",
    "autoattack_linf_acc",
    "pgd_l2_20_acc",
    "pgd_l2_50_acc",
    "autoattack_l2_acc",
    "mean_linf_acc",
    "mean_l2_acc",
    "mean_robust_acc",
    "worst_robust_acc",
    "linf_l2_gap",
    "checkpoint",
]


def sort_value(value: Any) -> float:
    if value is None or value == "":
        return float("-inf")
    return float(value)


def row_from_eval(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    config = payload.get("checkpoint_config", {})
    metrics = payload.get("metrics", {})
    derived = payload.get("derived_metrics", {})
    run_dir = path.parent.name
    return {
        "method": config.get("method", run_dir),
        "seed": config.get("seed", ""),
        "clean_acc": metrics.get("clean_acc"),
        "fgsm_linf_acc": metrics.get("fgsm_linf_acc"),
        "pgd_linf_20_acc": metrics.get("pgd_linf_20_acc"),
        "pgd_linf_50_acc": metrics.get("pgd_linf_50_acc"),
        "autoattack_linf_acc": metrics.get("autoattack_linf_acc"),
        "pgd_l2_20_acc": metrics.get("pgd_l2_20_acc"),
        "pgd_l2_50_acc": metrics.get("pgd_l2_50_acc"),
        "autoattack_l2_acc": metrics.get("autoattack_l2_acc"),
        "mean_linf_acc": derived.get("mean_linf_acc"),
        "mean_l2_acc": derived.get("mean_l2_acc"),
        "mean_robust_acc": derived.get("mean_robust_acc"),
        "worst_robust_acc": derived.get("worst_robust_acc"),
        "linf_l2_gap": derived.get("linf_l2_gap"),
        "checkpoint": payload.get("checkpoint", ""),
    }


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(FIELDNAMES) + " |\n")
        f.write("| " + " | ".join(["---"] * len(FIELDNAMES)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(format_cell(row.get(field)) for field in FIELDNAMES) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate evaluation JSON files from one run directory.")
    parser.add_argument("--runs_dir", default="runs")
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir)
    rows = [row_from_eval(path) for path in runs_dir.glob("*/eval.json")]
    rows.sort(
        key=lambda row: (
            sort_value(row.get("worst_robust_acc")),
            sort_value(row.get("mean_robust_acc")),
            sort_value(row.get("autoattack_linf_acc")),
        ),
        reverse=True,
    )

    runs_dir.mkdir(parents=True, exist_ok=True)
    with (runs_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(runs_dir / "summary.md", rows)
    print(f"wrote {len(rows)} rows to {runs_dir / 'summary.csv'} and {runs_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
