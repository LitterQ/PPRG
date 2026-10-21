from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config as paper_config
from . import train as base_train
from .integrity import canonical_json, implementation_manifest


TASK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TASK_ROOT.parent
EXPERIMENT = "pda_proj_w0p1"
SUPPLEMENT_VERSION = "paper-recipe-projected-pda-w0p1-supplement-v1"
SUPPLEMENT_FILES = (
    "vit_12_runs_task/paper_recipe/supplemental_proj_w0p1.py",
    "vit_12_runs_task/scripts/run_paper_recipe_pda_proj_w0p1.sh",
    "vit_12_runs_task/scripts/summarize_vit_15_runs.py",
)
METHOD_SPEC: dict[str, Any] = {
    "method": "primary_projected_aux_l2",
    "attack_domains": ["pgd_linf_10", "pgd_l2_5"],
    "primary_domain": "pgd_linf_10",
    "aux_domain": "pgd_l2_5",
    "aux_weight": 0.1,
    "objective": "g_primary + 0.1 * projected(g_aux | g_primary)",
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
            raise FileNotFoundError(f"Supplement implementation file is missing: {path}")
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
        raise FileExistsError(f"Refusing to overwrite supplement lock: {target}")
    payload = supplemental_implementation_manifest()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def verify_supplement_lock(path: str | Path) -> dict[str, Any]:
    target = require_under_task(path)
    try:
        saved = json.loads(target.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing supplement lock: {target}") from exc
    current = supplemental_implementation_manifest()
    if saved != current:
        raise ValueError(
            "Projected-PDA 0.1 supplement changed after lock creation: "
            f"saved={saved.get('supplemental_implementation_hash')}, "
            f"current={current.get('supplemental_implementation_hash')}"
        )
    return saved


def _install_training_extension() -> None:
    """Add one method without modifying any file protected by the original 12-run lock."""

    paper_config.METHOD_SPECS[EXPERIMENT] = dict(METHOD_SPEC)
    base_train.METHODS = (*paper_config.METHODS, EXPERIMENT)

    original_method_fields = base_train.method_fields
    extension_manifest = supplemental_implementation_manifest()

    def method_fields(experiment: str) -> dict[str, Any]:
        fields = original_method_fields(experiment)
        if experiment == EXPERIMENT:
            fields.update(
                {
                    "supplement_version": SUPPLEMENT_VERSION,
                    "supplemental_base_implementation_hash": extension_manifest["base_implementation_hash"],
                    "supplemental_implementation_hash": extension_manifest["supplemental_implementation_hash"],
                    "supplemental_implementation_files": extension_manifest["supplemental_files"],
                }
            )
        return fields

    base_train.method_fields = method_fields

    original_backward = base_train.primary_projected_aux_backward

    def projected_backward_w0p1(
        losses: dict[str, Any],
        model: Any,
        primary_domain: str = "pgd_linf_10",
        aux_domain: str = "pgd_l2_5",
        aux_weight: float = 0.2,
        eps: float = 1e-12,
    ) -> dict[str, Any]:
        if primary_domain != "pgd_linf_10" or aux_domain != "pgd_l2_5":
            raise ValueError("The supplement only supports PGD-Linf-10 primary and PGD-L2-5 auxiliary.")
        return original_backward(
            losses,
            model,
            primary_domain=primary_domain,
            aux_domain=aux_domain,
            aux_weight=0.1,
            eps=eps,
        )

    base_train.primary_projected_aux_backward = projected_backward_w0p1

    # The locked trainer uses a 0.2 scalar only for reporting after the projected
    # gradient is written. Correct both batch- and epoch-level proxy losses; the
    # actual gradient above already uses the requested 0.1 coefficient.
    original_append_csv = base_train.append_csv

    def append_csv(path: Path, fields: list[str], row: dict[str, Any]) -> None:
        if path.name == "batch_diagnostics.csv" and row.get("auxiliary_loss") not in ("", None):
            row = dict(row)
            row["combined_loss"] = float(row["primary_loss"]) + 0.1 * float(row["auxiliary_loss"])
        original_append_csv(path, fields, row)

    base_train.append_csv = append_csv
    original_train_one_epoch = base_train.train_one_epoch

    def train_one_epoch(**kwargs: Any) -> dict[str, Any]:
        stats = original_train_one_epoch(**kwargs)
        if kwargs.get("experiment") == EXPERIMENT:
            stats["train_loss"] = float(stats["train_primary_loss"]) + 0.1 * float(
                stats["train_auxiliary_loss"]
            )
        return stats

    base_train.train_one_epoch = train_one_epoch


def _write_plan(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Write a projected-PDA 0.1 formal-run plan.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    _install_training_extension()
    payload = paper_config.resolved_config(
        experiment=EXPERIMENT,
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
            "formal_suite_target": "15_runs",
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
            "formal_suite_target": "15_runs",
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
            raise SystemExit("usage: supplemental_proj_w0p1 --write-lock PATH")
        print(json.dumps(write_supplement_lock(args[1]), indent=2, sort_keys=True))
        return
    if args[:1] == ["--verify-lock"]:
        if len(args) != 2:
            raise SystemExit("usage: supplemental_proj_w0p1 --verify-lock PATH")
        print(json.dumps(verify_supplement_lock(args[1]), indent=2, sort_keys=True))
        return
    if args[:1] == ["--write-plan"]:
        _write_plan(args[1:])
        return

    experiment_values = [args[index + 1] for index, value in enumerate(args[:-1]) if value == "--experiment"]
    if experiment_values != [EXPERIMENT]:
        raise SystemExit(f"This entry point only trains --experiment {EXPERIMENT}.")
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
    "EXPERIMENT",
    "METHOD_SPEC",
    "SUPPLEMENT_FILES",
    "SUPPLEMENT_VERSION",
    "supplemental_implementation_hash",
    "supplemental_implementation_manifest",
    "verify_supplement_lock",
    "write_supplement_lock",
]
