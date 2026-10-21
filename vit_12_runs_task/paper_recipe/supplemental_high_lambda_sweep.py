from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config as paper_config
from . import supplemental_lambda_sweep as shared_sweep
from . import train as base_train
from .integrity import canonical_json, implementation_manifest


TASK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TASK_ROOT.parent
SUPPLEMENT_VERSION = "paper-recipe-pda-high-lambda-sweep-supplement-v1"
SUPPLEMENT_FILES = (
    "vit_12_runs_task/paper_recipe/supplemental_lambda_sweep.py",
    "vit_12_runs_task/paper_recipe/supplemental_high_lambda_sweep.py",
    "vit_12_runs_task/scripts/run_paper_recipe_pda_high_lambda_sweep.sh",
    "vit_12_runs_task/scripts/summarize_vit_69_runs.py",
)
WEIGHTS: dict[str, float] = {
    "0p4": 0.4,
    "0p5": 0.5,
    "0p6": 0.6,
    "0p7": 0.7,
    "0p8": 0.8,
    "0p9": 0.9,
}
EXPERIMENTS = tuple(
    [f"pda_no_proj_w{label}" for label in WEIGHTS]
    + [f"pda_proj_w{label}" for label in WEIGHTS]
)
METHOD_SPECS: dict[str, dict[str, Any]] = {}
for label, weight in WEIGHTS.items():
    METHOD_SPECS[f"pda_no_proj_w{label}"] = {
        "method": "avg_attack_dg",
        "attack_domains": ["pgd_linf_10", "pgd_l2_5"],
        "parsed_domain_weights": {"pgd_linf_10": 1.0, "pgd_l2_5": weight},
        "normalize_domain_weights": False,
        "objective": f"L_primary + {weight:g} * L_aux",
    }
    METHOD_SPECS[f"pda_proj_w{label}"] = {
        "method": "primary_projected_aux_l2",
        "attack_domains": ["pgd_linf_10", "pgd_l2_5"],
        "primary_domain": "pgd_linf_10",
        "aux_domain": "pgd_l2_5",
        "aux_weight": weight,
        "objective": f"g_primary + {weight:g} * projected(g_aux | g_primary)",
    }


def require_under_task(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(TASK_ROOT)
    except ValueError as exc:
        raise ValueError(f"Supplement path must remain under {TASK_ROOT}; got {resolved}") from exc
    return resolved


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def supplemental_implementation_manifest() -> dict[str, Any]:
    base = implementation_manifest()
    files: dict[str, str] = {}
    for relative in SUPPLEMENT_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"High-lambda implementation file is missing: {path}")
        files[relative] = file_sha256(path)
    payload: dict[str, Any] = {
        "manifest_version": SUPPLEMENT_VERSION,
        "base_implementation_hash": base["implementation_hash"],
        "supplemental_files": files,
    }
    payload["supplemental_implementation_hash"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def supplemental_implementation_hash() -> str:
    return str(supplemental_implementation_manifest()["supplemental_implementation_hash"])


def write_supplement_lock(path: str | Path) -> dict[str, Any]:
    target = require_under_task(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite high-lambda lock: {target}")
    payload = supplemental_implementation_manifest()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def verify_supplement_lock(path: str | Path) -> dict[str, Any]:
    target = require_under_task(path)
    try:
        saved = json.loads(target.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing high-lambda lock: {target}") from exc
    current = supplemental_implementation_manifest()
    if saved != current:
        raise ValueError(
            "PDA high-lambda supplement changed after lock creation: "
            f"saved={saved.get('supplemental_implementation_hash')}, "
            f"current={current.get('supplemental_implementation_hash')}"
        )
    return saved


def experiment_weight(experiment: str) -> float:
    if experiment not in METHOD_SPECS:
        raise ValueError(f"Unknown high-lambda experiment: {experiment}")
    spec = METHOD_SPECS[experiment]
    if experiment.startswith("pda_proj_"):
        return float(spec["aux_weight"])
    return float(spec["parsed_domain_weights"]["pgd_l2_5"])


def _install_training_extension() -> None:
    # Reuse the already-audited generic lambda train_one_epoch implementation,
    # while binding this new range to a separate immutable supplement manifest.
    shared_sweep.METHOD_SPECS.update({name: dict(spec) for name, spec in METHOD_SPECS.items()})
    paper_config.METHOD_SPECS.update({name: dict(spec) for name, spec in METHOD_SPECS.items()})
    base_train.METHODS = (*paper_config.METHODS, *EXPERIMENTS)
    base_train.train_one_epoch = shared_sweep.train_one_epoch
    original_method_fields = base_train.method_fields
    extension_manifest = supplemental_implementation_manifest()

    def method_fields(experiment: str) -> dict[str, Any]:
        fields = original_method_fields(experiment)
        if experiment in EXPERIMENTS:
            fields.update(
                {
                    "supplement_version": SUPPLEMENT_VERSION,
                    "supplemental_base_implementation_hash": extension_manifest["base_implementation_hash"],
                    "supplemental_implementation_hash": extension_manifest["supplemental_implementation_hash"],
                    "supplemental_implementation_files": extension_manifest["supplemental_files"],
                    "formal_suite_target": "69_runs",
                }
            )
        return fields

    base_train.method_fields = method_fields


def _write_plan(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Write a formal PDA high-lambda plan.")
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
    extension = supplemental_implementation_manifest()
    payload.update(
        {
            "supplement_version": SUPPLEMENT_VERSION,
            "supplemental_base_implementation_hash": extension["base_implementation_hash"],
            "supplemental_implementation_hash": extension["supplemental_implementation_hash"],
            "supplemental_implementation_files": extension["supplemental_files"],
            "formal_suite_target": "69_runs",
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
    marker = json.loads(marker_path.read_text(encoding="utf-8-sig"))
    extension = supplemental_implementation_manifest()
    marker.update(
        {
            "supplement_version": SUPPLEMENT_VERSION,
            "supplemental_base_implementation_hash": extension["base_implementation_hash"],
            "supplemental_implementation_hash": extension["supplemental_implementation_hash"],
            "formal_suite_target": "69_runs",
            "supplement_metadata_updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    base_train.atomic_json(marker_path, marker)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--manifest"]:
        print(json.dumps(supplemental_implementation_manifest(), indent=2, sort_keys=True))
        return
    if args[:1] == ["--write-lock"]:
        if len(args) != 2:
            raise SystemExit("usage: supplemental_high_lambda_sweep --write-lock PATH")
        print(json.dumps(write_supplement_lock(args[1]), indent=2, sort_keys=True))
        return
    if args[:1] == ["--verify-lock"]:
        if len(args) != 2:
            raise SystemExit("usage: supplemental_high_lambda_sweep --verify-lock PATH")
        print(json.dumps(verify_supplement_lock(args[1]), indent=2, sort_keys=True))
        return
    if args[:1] == ["--write-plan"]:
        _write_plan(args[1:])
        return

    experiment_values = [args[index + 1] for index, value in enumerate(args[:-1]) if value == "--experiment"]
    if len(experiment_values) != 1 or experiment_values[0] not in EXPERIMENTS:
        raise SystemExit(f"This entry point only trains one of: {', '.join(EXPERIMENTS)}")
    save_dirs = [args[index + 1] for index, value in enumerate(args[:-1]) if value == "--save-dir"]
    if len(save_dirs) != 1:
        raise SystemExit("Exactly one --save-dir is required.")
    _install_training_extension()
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *args]
        base_train.main()
    finally:
        sys.argv = original_argv
    _augment_training_completion(require_under_task(save_dirs[0]))


if __name__ == "__main__":
    main()


__all__ = [
    "EXPERIMENTS",
    "METHOD_SPECS",
    "SUPPLEMENT_FILES",
    "SUPPLEMENT_VERSION",
    "WEIGHTS",
    "experiment_weight",
    "supplemental_implementation_hash",
    "supplemental_implementation_manifest",
    "verify_supplement_lock",
    "write_supplement_lock",
]
