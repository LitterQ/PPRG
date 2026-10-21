from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


RUN_ORDER = [
    "pgd_at_linf10",
    "primary_aux_l2_no_proj_w0p1",
    "primary_aux_l2_no_proj_w0p2",
    "primary_projected_aux_l2_w0p2",
    "pcgrad_stronger_linf",
]
BASE_METRIC_KEYS = [
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
DERIVED_METRIC_KEYS = [
    "mean_linf_acc",
    "mean_l2_acc",
    "mean_robust_acc",
    "worst_robust_acc",
    "linf_l2_gap",
]
LINF_KEYS = [
    "fgsm_linf_acc",
    "pgd_linf_20_acc",
    "pgd_linf_50_acc",
    "cw_linf_20_acc",
    "cw_linf_50_acc",
    "autoattack_linf_acc",
]
L2_KEYS = [
    "pgd_l2_20_acc",
    "pgd_l2_50_acc",
    "cw_l2_20_acc",
    "cw_l2_50_acc",
    "autoattack_l2_acc",
]


def parse_list_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def eps_label(value: str, norm: str) -> str:
    text = value.strip()
    if norm == "linf" and "/" in text:
        return f"eps{text.split('/', 1)[0]}"
    return "eps" + re.sub(r"[^0-9A-Za-z]+", "p", text).strip("p")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_metric(path: Path, key: str) -> float | None:
    payload = read_json(path)
    value = payload.get("metrics", {}).get(key)
    return None if value is None else float(value)


def mean_available(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def method_label(run_dir: Path) -> str:
    return re.sub(r"_seed\d+$", "", run_dir.name)


def seed_label(run_dir: Path, payload: dict[str, Any]) -> str:
    match = re.search(r"_seed(\d+)$", run_dir.name)
    if match:
        return match.group(1)
    return str(payload.get("checkpoint_config", {}).get("seed", ""))


def sort_key(run_dir: Path) -> tuple[int, int, str]:
    label = method_label(run_dir)
    try:
        method_idx = RUN_ORDER.index(label)
    except ValueError:
        method_idx = len(RUN_ORDER)
    match = re.search(r"_seed(\d+)$", run_dir.name)
    seed_idx = int(match.group(1)) if match else 0
    return (method_idx, seed_idx, run_dir.name)


def run_dirs_with_eval(runs_dir: Path) -> list[Path]:
    run_dirs = [path.parent for path in runs_dir.glob("*/eval.json")]
    return sorted(run_dirs, key=sort_key)


def base_row(run_dir: Path) -> dict[str, Any]:
    eval_path = run_dir / "eval.json"
    payload = read_json(eval_path)
    metrics = payload.get("metrics", {})
    row: dict[str, Any] = {
        "method": method_label(run_dir),
        "seed": seed_label(run_dir, payload),
        "checkpoint": payload.get("checkpoint", str(run_dir / "best.pt")),
    }
    for key in BASE_METRIC_KEYS:
        row[key] = metrics.get(key)

    linf_values = [row.get(key) for key in LINF_KEYS]
    l2_values = [row.get(key) for key in L2_KEYS]
    robust_values = [row.get(key) for key in BASE_METRIC_KEYS if key != "clean_acc"]
    row["mean_linf_acc"] = mean_available(linf_values)
    row["mean_l2_acc"] = mean_available(l2_values)
    row["mean_robust_acc"] = mean_available(robust_values)
    present_robust = [float(value) for value in robust_values if value is not None]
    row["worst_robust_acc"] = min(present_robust) if present_robust else None
    row["linf_l2_gap"] = (
        row["mean_l2_acc"] - row["mean_linf_acc"]
        if row["mean_linf_acc"] is not None and row["mean_l2_acc"] is not None
        else None
    )
    return row


def write_table(csv_path: Path, md_path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(format_cell(row.get(field)) for field in fields) + " |\n")


def build_summary(run_dirs: list[Path]) -> tuple[list[str], list[dict[str, Any]]]:
    fields = [
        "method",
        "seed",
        "checkpoint",
        *BASE_METRIC_KEYS,
        *DERIVED_METRIC_KEYS,
    ]
    rows = [base_row(run_dir) for run_dir in run_dirs]
    return fields, rows


def build_epsilon_summary(
    run_dirs: list[Path],
    runs_dir: Path,
    linf_eps: list[str],
    l2_eps: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    fields = ["method", "seed", "checkpoint", *BASE_METRIC_KEYS, *DERIVED_METRIC_KEYS]
    for eps in linf_eps:
        label = eps_label(eps, "linf")
        fields.extend(
            [
                f"fgsm_linf_{label}",
                f"pgd_linf_20_{label}",
                f"pgd_linf_50_{label}",
                f"cw_linf_20_{label}",
                f"cw_linf_50_{label}",
                f"mean_linf_{label}",
            ]
        )
    for eps in l2_eps:
        label = eps_label(eps, "l2")
        fields.extend(
            [
                f"pgd_l2_20_{label}",
                f"pgd_l2_50_{label}",
                f"cw_l2_20_{label}",
                f"cw_l2_50_{label}",
                f"mean_l2_{label}",
            ]
        )

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        row = base_row(run_dir)
        sweep_dir = runs_dir / "epsilon_sweep_eval" / run_dir.name

        for eps in linf_eps:
            label = eps_label(eps, "linf")
            path = sweep_dir / f"linf_{label}.json"
            values = [
                read_metric(path, "fgsm_linf_acc"),
                read_metric(path, "pgd_linf_20_acc"),
                read_metric(path, "pgd_linf_50_acc"),
                read_metric(path, "cw_linf_20_acc"),
                read_metric(path, "cw_linf_50_acc"),
            ]
            row[f"fgsm_linf_{label}"] = values[0]
            row[f"pgd_linf_20_{label}"] = values[1]
            row[f"pgd_linf_50_{label}"] = values[2]
            row[f"cw_linf_20_{label}"] = values[3]
            row[f"cw_linf_50_{label}"] = values[4]
            row[f"mean_linf_{label}"] = mean_available(values)

        for eps in l2_eps:
            label = eps_label(eps, "l2")
            path = sweep_dir / f"l2_{label}.json"
            values = [
                read_metric(path, "pgd_l2_20_acc"),
                read_metric(path, "pgd_l2_50_acc"),
                read_metric(path, "cw_l2_20_acc"),
                read_metric(path, "cw_l2_50_acc"),
            ]
            row[f"pgd_l2_20_{label}"] = values[0]
            row[f"pgd_l2_50_{label}"] = values[1]
            row[f"cw_l2_20_{label}"] = values[2]
            row[f"cw_l2_50_{label}"] = values[3]
            row[f"mean_l2_{label}"] = mean_available(values)

        rows.append(row)

    return fields, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CIFAR-10 WRN train/eval runs.")
    parser.add_argument("--runs_dir", default="runs_cifar10_WRN")
    parser.add_argument("--linf_eps", default="4/255,8/255,12/255,16/255")
    parser.add_argument("--l2_eps", default="0.25,0.5,0.75,1.0")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    run_dirs = run_dirs_with_eval(runs_dir)
    linf_eps = parse_list_arg(args.linf_eps)
    l2_eps = parse_list_arg(args.l2_eps)

    fields, rows = build_summary(run_dirs)
    write_table(runs_dir / "summary.csv", runs_dir / "summary.md", fields, rows)

    fields, rows = build_epsilon_summary(run_dirs, runs_dir, linf_eps, l2_eps)
    write_table(runs_dir / "epsilon_sweep_summary.csv", runs_dir / "epsilon_sweep_summary.md", fields, rows)

    print(
        f"wrote {len(run_dirs)} rows to {runs_dir / 'summary.md'} and "
        f"{runs_dir / 'epsilon_sweep_summary.md'}"
    )


if __name__ == "__main__":
    main()
