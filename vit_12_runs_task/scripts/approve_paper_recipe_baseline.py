from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_ROOT = Path(__file__).resolve().parents[1]
TIMM_WHEEL = TASK_ROOT / ".deps" / "timm-1.0.15-py3-none-any.whl"
if TIMM_WHEEL.is_file() and str(TIMM_WHEEL) not in sys.path:
    sys.path.insert(0, str(TIMM_WHEEL))
REFERENCE = {
    "clean_acc": 0.7546,
    "cw_linf_20_acc": 0.4540,
    "pgd_linf_20_acc": 0.4810,
    "pgd_linf_100_acc": 0.4796,
    "autoattack_linf_acc": 0.4362,
}


def require_task(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(TASK_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path must remain under {TASK_ROOT}: {resolved}") from exc
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_approval(path: Path) -> dict[str, Any]:
    from vit_12_runs_task.paper_recipe.config import paper_recipe_hash
    from vit_12_runs_task.paper_recipe.integrity import implementation_manifest, verify_evaluation_artifact

    approval = read_json(path)
    if approval.get("approved") is not True:
        raise ValueError("Approval record does not contain approved=true.")
    if approval.get("paper_recipe_hash") != paper_recipe_hash():
        raise ValueError("Approval recipe hash does not match the current implementation.")
    manifest = implementation_manifest()
    if (
        approval.get("implementation_hash") != manifest["implementation_hash"]
        or approval.get("implementation_files") != manifest["files"]
    ):
        raise ValueError("Approval implementation manifest does not match the current source tree.")
    checkpoint = require_task(approval["baseline_checkpoint"])
    if not checkpoint.is_file() or sha256(checkpoint) != approval.get("baseline_checkpoint_sha256"):
        raise ValueError("Approved baseline checkpoint is missing or its SHA256 changed.")
    evaluation = require_task(approval["evaluation"])
    if not evaluation.is_file() or sha256(evaluation) != approval.get("evaluation_sha256"):
        raise ValueError("Approved baseline evaluation is missing or its SHA256 changed.")
    evaluation_payload = verify_evaluation_artifact(
        evaluation,
        checkpoint=checkpoint,
        expected_seed=int(approval.get("baseline_seed", -1)),
        required_attacks=["clean", "cw_linf_20", "pgd_linf_20", "pgd_linf_100", "autoattack_linf"],
        expected_eps_linf=8 / 255,
        expected_eps_l2=0.5,
    )
    if (
        evaluation_payload.get("evaluation_environment", {}).get("environment_hash")
        != approval.get("evaluation_environment_hash")
    ):
        raise ValueError("Approved evaluation environment hash changed.")
    return approval


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicitly approve (or verify) a paper-recipe baseline gate.")
    parser.add_argument("--baseline-dir", default="")
    parser.add_argument("--output", default="vit_12_runs_task/paper_recipe_approvals/baseline_approval.json")
    parser.add_argument("--evaluation", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--approve", action="store_true", help="Required explicit confirmation for creating approval.")
    parser.add_argument("--allow-outside-gate", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--approval", default="")
    args = parser.parse_args()

    if args.verify_only:
        approval_path = require_task(args.approval or args.output)
        approval = verify_approval(approval_path)
        print(json.dumps({"verified": True, "approval": str(approval_path), "baseline_checkpoint": approval["baseline_checkpoint"]}, indent=2))
        return
    if not args.approve:
        raise SystemExit("Refusing to approve implicitly. Re-run with --approve after reviewing the metrics.")
    if not args.baseline_dir:
        raise SystemExit("--baseline-dir is required when creating approval.")

    from vit_12_runs_task.paper_recipe.config import paper_recipe_hash
    from vit_12_runs_task.paper_recipe.integrity import implementation_manifest, verify_evaluation_artifact

    baseline = require_task(args.baseline_dir)
    if baseline.parent != TASK_ROOT / "paper_recipe_baseline":
        raise SystemExit(f"Not a paper_recipe_baseline run: {baseline}")
    checkpoint = baseline / "last.pt"
    config_path = baseline / "config.json"
    eval_path = require_task(args.evaluation) if args.evaluation else baseline / "baseline_eval_official.json"
    complete_path = baseline / "training_complete.json"
    for path in (checkpoint, config_path, eval_path, complete_path):
        if not path.is_file():
            raise SystemExit(f"Missing required baseline artifact: {path}")
    config = read_json(config_path)
    completion = read_json(complete_path)
    checkpoint_sha = sha256(checkpoint)
    if (
        completion.get("last_checkpoint_sha256") != checkpoint_sha
        or completion.get("run_kind") != "manual_baseline"
        or int(completion.get("epochs", -1)) != 40
        or completion.get("paper_recipe_hash") != paper_recipe_hash()
    ):
        raise SystemExit("training_complete.json does not match the required 40-epoch baseline or current last.pt.")
    if (
        config.get("paper_recipe_hash") != paper_recipe_hash()
        or config.get("run_kind") != "manual_baseline"
        or config.get("experiment") != "pgd_at_linf10"
        or int(config.get("seed", -1)) != 0
        or int(config.get("epochs", -1)) != 40
        or int(config.get("batch_size", -1)) != 64
        or config.get("use_fake_data") is not False
        or int(config.get("max_train_batches", -1)) != 0
    ):
        raise SystemExit("Baseline config does not match the current manual paper recipe.")
    init_hash = config.get("initial_trainable_parameters_hash")
    if not isinstance(init_hash, str) or len(init_hash) != 64:
        raise SystemExit("Baseline config is missing a valid initialization hash.")
    manifest = implementation_manifest()
    required_attacks = ["clean", "cw_linf_20", "pgd_linf_20", "pgd_linf_100", "autoattack_linf"]
    try:
        evaluation = verify_evaluation_artifact(
            eval_path,
            checkpoint=checkpoint,
            expected_seed=int(config.get("seed", -1)),
            required_attacks=required_attacks,
            expected_eps_linf=8 / 255,
            expected_eps_l2=0.5,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Official evaluation integrity check failed: {exc}") from exc
    if evaluation.get("checkpoint_config", {}).get("initial_trainable_parameters_hash") != init_hash:
        raise SystemExit("Official evaluation/checkpoint initialization metadata does not match config.json.")
    metrics: dict[str, float] = {}
    deltas: dict[str, float] = {}
    outside: list[str] = []
    for key, reference in REFERENCE.items():
        value = evaluation.get("metrics", {}).get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise SystemExit(f"Missing/non-finite approval metric {key}: {value!r}")
        metrics[key] = float(value)
        deltas[key] = float(value) - reference
        if abs(deltas[key]) > 0.02:
            outside.append(key)
    if outside and not args.allow_outside_gate:
        raise SystemExit(
            "Baseline is outside the default ±2 percentage-point engineering gate for: "
            + ", ".join(outside)
            + ". Investigate first, or explicitly use --allow-outside-gate with a note."
        )

    output = require_task(args.output)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite approval record: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "approved": True,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "approved_by": "explicit_user_command",
        "note": args.note,
        "allow_outside_gate": bool(args.allow_outside_gate),
        "outside_default_gate": outside,
        "paper_recipe_hash": paper_recipe_hash(),
        "implementation_hash": manifest["implementation_hash"],
        "implementation_files": manifest["files"],
        "implementation_binding_scope": "formal_12_source_tree_at_approval; not retroactive proof of baseline training source",
        "baseline_run": str(baseline),
        "baseline_seed": int(config["seed"]),
        "baseline_checkpoint": str(checkpoint),
        "baseline_checkpoint_sha256": checkpoint_sha,
        "baseline_initialization_hash": init_hash,
        "baseline_training_project_git_commit": config.get("project_git_commit"),
        "baseline_training_source_binding": "recipe/checkpoint/artifact audit; formal implementation manifest is a forward gate",
        "evaluation": str(eval_path),
        "evaluation_sha256": sha256(eval_path),
        "evaluation_environment_hash": evaluation["evaluation_environment"]["environment_hash"],
        "metrics": metrics,
        "paper_reference": REFERENCE,
        "metric_deltas": deltas,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    subprocess.run(
        [sys.executable, "-u", str(TASK_ROOT / "scripts" / "record_paper_recipe_status.py")],
        check=True,
        cwd=TASK_ROOT.parent,
    )


if __name__ == "__main__":
    main()
