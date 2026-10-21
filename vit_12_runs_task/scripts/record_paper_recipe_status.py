from __future__ import annotations

import json
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

from vit_12_runs_task.paper_recipe.config import PAPER_RECIPE_VERSION, paper_recipe_hash
from vit_12_runs_task.paper_recipe.integrity import (
    file_sha256,
    implementation_manifest,
    verify_evaluation_artifact,
)


CURRENT_RECIPE_HASH = paper_recipe_hash()
CURRENT_IMPLEMENTATION = implementation_manifest()
CURRENT_IMPLEMENTATION_HASH = str(CURRENT_IMPLEMENTATION["implementation_hash"])


def valid_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else None
    except (OSError, json.JSONDecodeError):
        return None


def completed_paths(root: Path, marker: str) -> list[Path]:
    if not root.is_dir():
        return []
    return [path for path in sorted(root.iterdir()) if path.is_dir() and (path / marker).is_file()]


def artifact_recipe_hash(path: Path) -> str | None:
    config = valid_json(path / "config.json") or valid_json(path / "planned_config.json") or {}
    value = config.get("paper_recipe_hash")
    return str(value) if value else None


def split_current(paths: list[Path]) -> tuple[list[str], list[str]]:
    current: list[str] = []
    stale: list[str] = []
    for path in paths:
        (current if artifact_recipe_hash(path) == CURRENT_RECIPE_HASH else stale).append(str(path))
    return current, stale


def status_with_stale(current: list[str], stale: list[str], complete: str = "COMPLETE") -> str:
    if current:
        return complete
    return "STALE_RECIPE" if stale else "NOT_RUN"


def main() -> None:
    implementation_files = [
        TASK_ROOT / "paper_recipe" / "train.py",
        TASK_ROOT / "paper_recipe" / "config.py",
        TASK_ROOT / "paper_recipe" / "model.py",
        TASK_ROOT / "scripts" / "run_paper_recipe_baseline.sh",
        TASK_ROOT / "scripts" / "evaluate_paper_recipe_baseline.sh",
        TASK_ROOT / "scripts" / "run_paper_recipe_12_experiments.sh",
        TASK_ROOT / "PAPER_RECIPE_MANUAL_RUNBOOK_CN.md",
        TASK_ROOT / "configs" / "paper_recipe.json",
    ]

    smoke_results: list[str] = []
    stale_smoke_results: list[str] = []
    for smoke_dir in sorted(path for path in TASK_ROOT.glob("paper_recipe_smoke*") if path.is_dir()):
        result_path = smoke_dir / "smoke_result.json"
        smoke_result = valid_json(result_path) or {}
        if smoke_result.get("status") != "PASSED":
            continue
        target = smoke_results if (
            smoke_result.get("paper_recipe_hash") == CURRENT_RECIPE_HASH
            and smoke_result.get("implementation_hash") == CURRENT_IMPLEMENTATION_HASH
        ) else stale_smoke_results
        target.append(str(result_path))
    smoke_current = bool(smoke_results)
    smoke_status = "PASSED" if smoke_current else "STALE_RECIPE" if stale_smoke_results else "NOT_RUN"

    validation_paths: list[Path] = []
    validation_root = TASK_ROOT / "paper_recipe_validation"
    if validation_root.is_dir():
        validation_paths = [
            path
            for path in sorted(validation_root.iterdir())
            if path.is_dir()
            and (path / "VALIDATION_ONLY_NOT_A_BASELINE").is_file()
            and (path / "training_complete.json").is_file()
            and (path / "validation_eval.json").is_file()
        ]
    validations, stale_validations = split_current(validation_paths)

    baseline_paths = completed_paths(TASK_ROOT / "paper_recipe_baseline", "training_complete.json")
    baselines, stale_baselines = split_current(baseline_paths)
    official_evaluation_paths: list[Path] = []
    invalid_evaluation_paths: list[str] = []
    for path in baseline_paths:
        eval_path = path / "baseline_eval_official.json"
        try:
            config = valid_json(path / "config.json") or {}
            verify_evaluation_artifact(
                eval_path,
                checkpoint=path / "last.pt",
                expected_seed=int(config.get("seed", -1)),
                required_attacks=["clean", "cw_linf_20", "pgd_linf_20", "pgd_linf_100", "autoattack_linf"],
                expected_eps_linf=8 / 255,
                expected_eps_l2=0.5,
            )
            valid_official = True
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            valid_official = False
        if valid_official:
            official_evaluation_paths.append(path)
        elif (path / "baseline_eval.json").is_file() or (path / "baseline_eval_official.json").is_file():
            invalid_evaluation_paths.append(str(path))
    evaluated, stale_evaluated = split_current(official_evaluation_paths)

    approvals: list[str] = []
    stale_approvals: list[str] = []
    approval_root = TASK_ROOT / "paper_recipe_approvals"
    if approval_root.is_dir():
        for path in sorted(approval_root.glob("*.json")):
            record = valid_json(path) or {}
            if record.get("approved") is not True:
                continue
            checkpoint = Path(str(record.get("baseline_checkpoint", "")))
            evaluation = Path(str(record.get("evaluation", "")))
            current = (
                record.get("paper_recipe_hash") == CURRENT_RECIPE_HASH
                and record.get("implementation_hash") == CURRENT_IMPLEMENTATION_HASH
                and record.get("implementation_files") == CURRENT_IMPLEMENTATION["files"]
                and checkpoint.is_file()
                and evaluation.is_file()
                and record.get("baseline_checkpoint_sha256") == file_sha256(checkpoint)
                and record.get("evaluation_sha256") == file_sha256(evaluation)
            )
            target = approvals if current else stale_approvals
            target.append(str(path))

    formal_paths = completed_paths(TASK_ROOT / "paper_recipe_runs", ".complete.json")
    formal_runs: list[str] = []
    stale_formal_runs: list[str] = []
    for path in formal_paths:
        marker = valid_json(path / ".complete.json") or {}
        lock = valid_json(path / "implementation_lock.json") or {}
        current = (
            artifact_recipe_hash(path) == CURRENT_RECIPE_HASH
            and marker.get("implementation_hash") == CURRENT_IMPLEMENTATION_HASH
            and lock.get("implementation_hash") == CURRENT_IMPLEMENTATION_HASH
            and lock.get("files") == CURRENT_IMPLEMENTATION["files"]
        )
        (formal_runs if current else stale_formal_runs).append(str(path))
    if len(formal_runs) == 12:
        formal_status = "COMPLETE"
    elif formal_runs:
        formal_status = "IN_PROGRESS"
    elif stale_formal_runs:
        formal_status = "STALE_RECIPE"
    else:
        formal_status = "NOT_RUN"

    phases = {
        "IMPLEMENTATION": {"status": "COMPLETE" if all(path.is_file() for path in implementation_files) else "IN_PROGRESS"},
        "SMOKE_TEST": {
            "status": smoke_status,
            "results": smoke_results,
            "stale_results": stale_smoke_results,
            "reason": None if smoke_current else "No passing FakeData smoke result is bound to the current recipe hash.",
        },
        "ONE_EPOCH_VALIDATION": {
            "status": status_with_stale(validations, stale_validations, "PASSED"),
            "runs": validations,
            "stale_runs": stale_validations,
            "reason": None if validations else "No real-CIFAR validation is bound to the current recipe hash; do not repeat it if the one-run allowance was already consumed.",
        },
        "MANUAL_BASELINE_TRAINING": {
            "status": status_with_stale(baselines, stale_baselines),
            "runs": baselines,
            "stale_runs": stale_baselines,
        },
        "MANUAL_BASELINE_EVALUATION": {
            "status": "COMPLETE" if evaluated else "NEEDS_OFFICIAL_RERUN" if invalid_evaluation_paths else status_with_stale(evaluated, stale_evaluated),
            "runs": evaluated,
            "stale_runs": stale_evaluated,
            "invalid_or_legacy_runs": invalid_evaluation_paths,
        },
        "BASELINE_APPROVAL": {
            "status": status_with_stale(approvals, stale_approvals, "APPROVED"),
            "records": approvals,
            "stale_records": stale_approvals,
        },
        "MANUAL_12_RUNS": {
            "status": formal_status,
            "completed": len(formal_runs),
            "runs": formal_runs,
            "stale_runs": stale_formal_runs,
        },
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_recipe_version": PAPER_RECIPE_VERSION,
        "current_recipe_hash": CURRENT_RECIPE_HASH,
        "current_implementation_hash": CURRENT_IMPLEMENTATION_HASH,
        "authority_note": "Formal/evaluation artifacts must be bound to both the current recipe and current implementation hashes.",
        "legacy_pilot_recipe": "legacy_simplified_recipe",
        "phases": phases,
    }
    (TASK_ROOT / "PAPER_RECIPE_STATUS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Paper recipe 状态",
        "",
        f"生成时间（UTC）：`{payload['generated_at']}`",
        "",
        f"当前版本：`{PAPER_RECIPE_VERSION}`",
        "",
        f"当前 recipe hash：`{CURRENT_RECIPE_HASH}`",
        "",
        f"当前 implementation hash：`{CURRENT_IMPLEMENTATION_HASH}`",
        "",
        "| 阶段 | 状态 |",
        "|---|---|",
    ]
    for name, info in phases.items():
        suffix = f" ({info['completed']}/12)" if name == "MANUAL_12_RUNS" else ""
        lines.append(f"| `{name}` | **{info['status']}**{suffix} |")
    lines.extend(
        [
            "",
            "> `STALE_RECIPE` 表示目录存在，但其 recipe 或 implementation 绑定已过期；它不会被计入批准或正式汇总。",
            "",
            "> 脚本已生成不等于实验已完成；旧 pilot 统一标记为 `legacy_simplified_recipe`。",
            "",
        ]
    )
    (TASK_ROOT / "PAPER_RECIPE_STATUS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({name: info["status"] for name, info in phases.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
