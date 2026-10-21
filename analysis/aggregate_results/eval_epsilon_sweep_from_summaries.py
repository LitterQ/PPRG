from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ATTACKS_LINF = ["fgsm_linf", "pgd_linf_20", "pgd_linf_50", "cw_linf_20", "cw_linf_50"]
DEFAULT_ATTACKS_L2 = ["pgd_l2_20", "pgd_l2_50", "cw_l2_20", "cw_l2_50"]
BASE_COLUMNS = [
    "method",
    "seed",
    "checkpoint",
    "clean_acc",
    "fgsm_linf_acc",
    "pgd_linf_20_acc",
    "pgd_linf_50_acc",
    "pgd_l2_20_acc",
    "pgd_l2_50_acc",
]


@dataclass(frozen=True)
class SummaryRow:
    summary_path: Path
    root_dir: Path
    row: dict[str, str]

    @property
    def checkpoint(self) -> str:
        return self.row["checkpoint"]


def parse_markdown_table(path: Path) -> list[dict[str, str]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip().startswith("|")]
    if len(lines) < 2:
        return []
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells)))
    return [row for row in rows if row.get("checkpoint")]


def eps_label(value: str, norm: str) -> str:
    text = value.strip()
    if norm == "linf" and "/" in text:
        numerator = text.split("/", 1)[0]
        return f"eps{numerator}"
    return "eps" + re.sub(r"[^0-9A-Za-z]+", "p", text).strip("p")


def slugify(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("_")


def parse_list_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def float_from_eps(value: str) -> float:
    if "/" in value:
        left, right = value.split("/", 1)
        return float(left) / float(right)
    return float(value)


def linf_alpha(eps: str) -> str:
    if "/" in eps:
        numerator, denominator = eps.split("/", 1)
        return f"{float(numerator) / 4:g}/{denominator}"
    return f"{float(eps) / 4:g}"


def l2_alpha(eps: str) -> str:
    return f"{float(eps) / 5:g}"


def run_eval(
    item: SummaryRow,
    gpu: str,
    args: argparse.Namespace,
    norm: str,
    eps: str,
    output_path: Path,
) -> None:
    attacks = DEFAULT_ATTACKS_LINF if norm == "linf" else DEFAULT_ATTACKS_L2
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    output_path.parent.mkdir(parents=True, exist_ok=True)
    console_path = output_path.with_suffix(".log")

    cmd = [
        args.python,
        "eval.py",
        "--checkpoint",
        item.checkpoint,
        "--dataset",
        args.dataset,
        "--model",
        args.model,
        "--eval_attacks",
        *attacks,
        "--eps_linf",
        eps if norm == "linf" else args.default_eps_linf,
        "--eps_l2",
        eps if norm == "l2" else args.default_eps_l2,
        "--alpha_linf",
        linf_alpha(eps) if norm == "linf" else args.default_alpha_linf,
        "--alpha_l2",
        l2_alpha(eps) if norm == "l2" else args.default_alpha_l2,
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--data_dir",
        args.data_dir,
        "--save_json",
        str(output_path),
    ]
    if args.max_eval_batches:
        cmd.extend(["--max_eval_batches", str(args.max_eval_batches)])
    if args.no_download:
        cmd.append("--no_download")

    with console_path.open("w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=args.workdir, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def expected_metric_keys(norm: str) -> list[str]:
    attacks = DEFAULT_ATTACKS_LINF if norm == "linf" else DEFAULT_ATTACKS_L2
    return [f"{attack}_acc" for attack in attacks]


def output_has_expected_metrics(path: Path, norm: str) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    metrics = payload.get("metrics", {})
    return all(key in metrics for key in expected_metric_keys(norm))


def eval_outputs_for_item(item: SummaryRow, args: argparse.Namespace) -> list[tuple[str, str, Path]]:
    slug = slugify(item.checkpoint)
    base = item.root_dir / args.output_dirname / slug
    outputs: list[tuple[str, str, Path]] = []
    for eps in args.linf_eps:
        outputs.append(("linf", eps, base / f"linf_{eps_label(eps, 'linf')}.json"))
    for eps in args.l2_eps:
        outputs.append(("l2", eps, base / f"l2_{eps_label(eps, 'l2')}.json"))
    return outputs


def worker(gpu: str, jobs: queue.Queue[SummaryRow], args: argparse.Namespace, failures: list[str]) -> None:
    while True:
        try:
            item = jobs.get_nowait()
        except queue.Empty:
            return
        try:
            checkpoint_path = Path(args.workdir) / item.checkpoint
            if not checkpoint_path.exists():
                message = f"missing checkpoint: {item.checkpoint}"
                if args.skip_missing:
                    print(f"[gpu {gpu}] skip {message}", flush=True)
                    continue
                raise FileNotFoundError(message)

            for norm, eps, output_path in eval_outputs_for_item(item, args):
                if args.skip_existing and output_has_expected_metrics(output_path, norm):
                    print(f"[gpu {gpu}] skip existing {output_path}", flush=True)
                    continue
                if output_path.exists() and args.skip_existing:
                    print(f"[gpu {gpu}] rerun incomplete {output_path}", flush=True)
                print(f"[gpu {gpu}] eval {item.checkpoint} {norm} eps={eps}", flush=True)
                run_eval(item, gpu, args, norm, eps, output_path)
        except Exception as exc:
            failures.append(f"{item.checkpoint}: {exc}")
        finally:
            jobs.task_done()


def read_metric(path: Path, key: str) -> float | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
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


def build_summary_rows(items: list[SummaryRow], args: argparse.Namespace) -> tuple[list[str], list[dict[str, Any]]]:
    fields = list(BASE_COLUMNS)
    for eps in args.linf_eps:
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
    for eps in args.l2_eps:
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
    for item in items:
        row: dict[str, Any] = {field: item.row.get(field, "") for field in BASE_COLUMNS}
        outputs = {(norm, eps): path for norm, eps, path in eval_outputs_for_item(item, args)}

        for eps in args.linf_eps:
            label = eps_label(eps, "linf")
            path = outputs[("linf", eps)]
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

        for eps in args.l2_eps:
            label = eps_label(eps, "l2")
            path = outputs[("l2", eps)]
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


def write_summary(root_dir: Path, items: list[SummaryRow], args: argparse.Namespace) -> None:
    fields, rows = build_summary_rows(items, args)
    csv_path = root_dir / args.summary_csv_name
    md_path = root_dir / args.summary_md_name

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("| " + " | ".join(["---"] * len(fields)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(format_cell(row.get(field)) for field in fields) + " |\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate epsilon sweeps for checkpoints listed in runs/summary.md and runs_new/summary.md."
    )
    parser.add_argument("--summaries", nargs="+", default=["runs/summary.md", "runs_new/summary.md"])
    parser.add_argument("--gpus", nargs="+", default=[str(i) for i in range(8)])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    parser.add_argument("--no_download", action="store_true")
    parser.add_argument("--linf_eps", default="4/255,8/255,12/255,16/255")
    parser.add_argument("--l2_eps", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--default_eps_linf", default="8/255")
    parser.add_argument("--default_eps_l2", default="0.5")
    parser.add_argument("--default_alpha_linf", default="2/255")
    parser.add_argument("--default_alpha_l2", default="0.1")
    parser.add_argument("--output_dirname", default="epsilon_sweep_eval")
    parser.add_argument("--summary_md_name", default="epsilon_sweep_summary.md")
    parser.add_argument("--summary_csv_name", default="epsilon_sweep_summary.csv")
    parser.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.workdir = str(Path(args.workdir).resolve())
    args.linf_eps = parse_list_arg(args.linf_eps)
    args.l2_eps = parse_list_arg(args.l2_eps)

    items_by_root: dict[Path, list[SummaryRow]] = {}
    unique_jobs: dict[tuple[Path, str], SummaryRow] = {}
    all_items: list[SummaryRow] = []
    for summary in args.summaries:
        summary_path = Path(args.workdir) / summary
        if not summary_path.exists():
            print(f"warning: missing summary {summary_path}", flush=True)
            continue
        root_dir = summary_path.parent
        rows = [SummaryRow(summary_path=summary_path, root_dir=root_dir, row=row) for row in parse_markdown_table(summary_path)]
        items_by_root[root_dir] = rows
        all_items.extend(rows)
        for row in rows:
            unique_jobs.setdefault((root_dir, row.checkpoint), row)

    print(
        f"loaded {len(all_items)} rows / {len(unique_jobs)} unique checkpoint jobs from {len(items_by_root)} summaries; "
        f"linf_eps={args.linf_eps}; l2_eps={args.l2_eps}; gpus={args.gpus}",
        flush=True,
    )

    if args.dry_run:
        for item in unique_jobs.values():
            print(item.checkpoint)
        return

    jobs: queue.Queue[SummaryRow] = queue.Queue()
    for item in unique_jobs.values():
        jobs.put(item)

    failures: list[str] = []
    threads = [threading.Thread(target=worker, args=(gpu, jobs, args, failures), daemon=True) for gpu in args.gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    for root_dir, items in items_by_root.items():
        write_summary(root_dir, items, args)
        print(f"wrote {root_dir / args.summary_md_name}", flush=True)

    if failures:
        print("failures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
