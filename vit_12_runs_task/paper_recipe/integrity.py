from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_VERSION = "paper-recipe-implementation-manifest-v1"
IMPLEMENTATION_FILES = (
    "eval.py",
    "pareto_attack_dg/attacks/__init__.py",
    "pareto_attack_dg/attacks/cw.py",
    "pareto_attack_dg/attacks/fgsm.py",
    "pareto_attack_dg/attacks/pgd.py",
    "pareto_attack_dg/attacks/projection.py",
    "pareto_attack_dg/eval.py",
    "pareto_attack_dg/models/__init__.py",
    "pareto_attack_dg/models/resnet.py",
    "pareto_attack_dg/optim/projected_aux.py",
    "pareto_attack_dg/utils/checkpoint.py",
    "pareto_attack_dg/utils/config.py",
    "pareto_attack_dg/utils/data.py",
    "pareto_attack_dg/utils/domains.py",
    "pareto_attack_dg/utils/logger.py",
    "pareto_attack_dg/utils/metrics.py",
    "vit_12_runs_task/model/deit_cifar.py",
    "vit_12_runs_task/paper_recipe/checkpoint.py",
    "vit_12_runs_task/paper_recipe/config.py",
    "vit_12_runs_task/paper_recipe/data.py",
    "vit_12_runs_task/paper_recipe/eval_attacks.py",
    "vit_12_runs_task/paper_recipe/integrity.py",
    "vit_12_runs_task/paper_recipe/losses.py",
    "vit_12_runs_task/paper_recipe/model.py",
    "vit_12_runs_task/paper_recipe/scheduler.py",
    "vit_12_runs_task/paper_recipe/train.py",
    "vit_12_runs_task/scripts/approve_paper_recipe_baseline.py",
    "vit_12_runs_task/scripts/evaluate_paper_recipe_baseline.sh",
    "vit_12_runs_task/scripts/paper_recipe_common.sh",
    "vit_12_runs_task/scripts/run_paper_recipe_12_experiments.sh",
    "vit_12_runs_task/scripts/summarize_vit_runs.py",
    "vit_12_runs_task/scripts/verify_paper_initialization.py",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_manifest() -> dict[str, Any]:
    files: dict[str, str] = {}
    for relative in IMPLEMENTATION_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Implementation manifest file is missing: {path}")
        files[relative] = file_sha256(path)
    payload: dict[str, Any] = {"manifest_version": MANIFEST_VERSION, "files": files}
    payload["implementation_hash"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return payload


def implementation_hash() -> str:
    return str(implementation_manifest()["implementation_hash"])


def require_under_task(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(TASK_ROOT)
    except ValueError as exc:
        raise ValueError(f"Integrity lock must remain under {TASK_ROOT}: {resolved}") from exc
    return resolved


def write_lock(path: str | Path) -> dict[str, Any]:
    target = require_under_task(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite implementation lock: {target}")
    payload = implementation_manifest()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def verify_lock(path: str | Path) -> dict[str, Any]:
    target = require_under_task(path)
    try:
        saved = json.loads(target.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing implementation lock: {target}") from exc
    current = implementation_manifest()
    if saved.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(f"Implementation lock version mismatch: {target}")
    if saved.get("implementation_hash") != current["implementation_hash"] or saved.get("files") != current["files"]:
        raise ValueError(
            f"Implementation changed after lock creation: saved={saved.get('implementation_hash')}, "
            f"current={current['implementation_hash']}"
        )
    return saved


def parse_number(value: str) -> float:
    return float(Fraction(value)) if "/" in value else float(value)


def verify_evaluation_environment(payload: dict[str, Any], *, require_autoattack: bool) -> dict[str, Any]:
    environment = payload.get("evaluation_environment")
    if not isinstance(environment, dict):
        raise ValueError("Evaluation environment metadata is missing.")
    recorded_hash = environment.get("environment_hash")
    hash_payload = dict(environment)
    hash_payload.pop("environment_hash", None)
    actual_hash = hashlib.sha256(canonical_json(hash_payload).encode("utf-8")).hexdigest()
    if recorded_hash != actual_hash:
        raise ValueError("Evaluation environment metadata hash mismatch.")

    import timm
    import torch
    import torchvision

    current_versions = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
    }
    for key, value in current_versions.items():
        if environment.get(key) != value:
            raise ValueError(f"Evaluation dependency changed: {key}={environment.get(key)!r}, current={value!r}")
    autoattack = environment.get("autoattack")
    if require_autoattack:
        if not isinstance(autoattack, dict) or not isinstance(autoattack.get("files"), dict):
            raise ValueError("AutoAttack source manifest is missing from evaluation metadata.")
        spec = importlib.util.find_spec("autoattack")
        if spec is None or spec.origin is None:
            raise ValueError("AutoAttack is no longer importable in the evaluation environment.")
        package_root = Path(spec.origin).resolve().parent
        current_files = {
            str(path.relative_to(package_root).as_posix()): file_sha256(path)
            for path in sorted(package_root.rglob("*.py"))
        }
        if str(package_root) != autoattack.get("package_root") or current_files != autoattack.get("files"):
            raise ValueError("AutoAttack package path or source hashes changed after evaluation.")
    elif autoattack is not None:
        raise ValueError("Evaluation unexpectedly recorded AutoAttack for an attack set that did not request it.")
    return environment


def verify_evaluation_artifact(
    path: str | Path,
    *,
    checkpoint: str | Path,
    expected_seed: int,
    required_attacks: list[str],
    expected_eps_linf: float | None = None,
    expected_eps_l2: float | None = None,
    expected_environment_hash: str | None = None,
) -> dict[str, Any]:
    evaluation_path = require_under_task(path)
    checkpoint_path = require_under_task(checkpoint)
    payload = json.loads(evaluation_path.read_text(encoding="utf-8-sig"))
    manifest = implementation_manifest()
    if Path(str(payload.get("checkpoint", ""))).expanduser().resolve() != checkpoint_path:
        raise ValueError(f"Evaluation checkpoint path mismatch: {evaluation_path}")
    if payload.get("checkpoint_sha256") != file_sha256(checkpoint_path):
        raise ValueError(f"Evaluation checkpoint SHA256 mismatch: {evaluation_path}")
    if payload.get("evaluation_implementation_hash") != manifest["implementation_hash"]:
        raise ValueError(f"Evaluation implementation hash mismatch: {evaluation_path}")
    if payload.get("evaluation_implementation_files") != manifest["files"]:
        raise ValueError(f"Evaluation implementation file manifest mismatch: {evaluation_path}")
    if payload.get("dataset") != "cifar10":
        raise ValueError(f"Evaluation dataset mismatch: {evaluation_path}")
    from vit_12_runs_task.paper_recipe.config import OFFICIAL_COMMIT, paper_recipe_hash

    checkpoint_config = payload.get("checkpoint_config", {})
    if checkpoint_config.get("paper_recipe_hash") != paper_recipe_hash():
        raise ValueError(f"Evaluation checkpoint recipe hash mismatch: {evaluation_path}")
    if int(checkpoint_config.get("seed", -1)) != int(expected_seed):
        raise ValueError(f"Evaluation checkpoint seed mismatch: {evaluation_path}")
    if int(payload.get("eval_seed", -1)) != int(expected_seed):
        raise ValueError(f"Evaluation seed mismatch: {evaluation_path}")
    semantics = payload.get("evaluation_semantics", {})
    if semantics.get("paper_official_linf") is not True:
        raise ValueError(f"Evaluation did not use official paper Linf semantics: {evaluation_path}")
    if semantics.get("official_source_commit") != OFFICIAL_COMMIT:
        raise ValueError(f"Evaluation official-source commit mismatch: {evaluation_path}")
    if any(attack.startswith("autoattack_") for attack in required_attacks) and (
        semantics.get("autoattack_invocation")
        != "single_complete_evaluated_set_call_with_internal_batching"
    ):
        raise ValueError(f"Evaluation did not use canonical complete-set AutoAttack invocation: {evaluation_path}")
    if payload.get("attacks_requested", []) != required_attacks:
        raise ValueError(f"Evaluation attack set mismatch: {evaluation_path}")
    expected_attack_seeds = {
        attack: int.from_bytes(
            hashlib.sha256(f"paper-eval-v1:{expected_seed}:{attack}".encode("utf-8")).digest()[:8],
            "little",
        )
        % (2**31)
        for attack in required_attacks
    }
    if payload.get("attack_seeds") != expected_attack_seeds:
        raise ValueError(f"Evaluation attack seeds mismatch: {evaluation_path}")
    environment = verify_evaluation_environment(
        payload,
        require_autoattack=any(
            attack in {"autoattack_linf", "autoattack_l2", "square"} for attack in required_attacks
        ),
    )
    if expected_environment_hash is not None and environment.get("environment_hash") != expected_environment_hash:
        raise ValueError(f"Evaluation environment differs from the approved baseline: {evaluation_path}")
    checkpoint_versions = checkpoint_config.get("library_versions", {})
    for checkpoint_key, environment_key in (
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("timm", "timm"),
    ):
        if checkpoint_versions.get(checkpoint_key) != environment.get(environment_key):
            raise ValueError(
                f"Evaluation {environment_key} version differs from checkpoint training metadata: {evaluation_path}"
            )
    dataloader = payload.get("dataloader", {})
    if (
        dataloader.get("max_eval_batches") != 0
        or dataloader.get("expected_samples_per_attack") != 10000
        or dataloader.get("use_fake_data") is not False
        or dataloader.get("download_requested") is not False
    ):
        raise ValueError(f"Evaluation was batch-limited: {evaluation_path}")
    actual = dataloader.get("actual_samples_per_attack", {})
    if set(actual) != set(required_attacks) or any(actual.get(attack) != 10000 for attack in required_attacks):
        raise ValueError(f"Evaluation did not process 10,000 samples per attack: {evaluation_path}")
    try:
        eps_linf = float(payload.get("eps_linf"))
        eps_l2 = float(payload.get("eps_l2"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Evaluation epsilon metadata is missing or invalid: {evaluation_path}") from exc
    if expected_eps_linf is not None and abs(eps_linf - expected_eps_linf) > 1e-12:
        raise ValueError(f"Evaluation Linf epsilon mismatch: {evaluation_path}")
    if expected_eps_l2 is not None and abs(eps_l2 - expected_eps_l2) > 1e-12:
        raise ValueError(f"Evaluation L2 epsilon mismatch: {evaluation_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Print, create, or verify the paper-recipe source manifest.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write-lock", default="")
    group.add_argument("--verify-lock", default="")
    group.add_argument("--verify-evaluation", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--expected-seed", type=int, default=-1)
    parser.add_argument("--required-attacks", nargs="*", default=[])
    parser.add_argument("--expected-eps-linf", default="")
    parser.add_argument("--expected-eps-l2", default="")
    parser.add_argument("--expected-environment-hash", default="")
    args = parser.parse_args()
    if args.write_lock:
        payload = write_lock(args.write_lock)
    elif args.verify_lock:
        payload = verify_lock(args.verify_lock)
    elif args.verify_evaluation:
        if not args.checkpoint or args.expected_seed < 0 or not args.required_attacks:
            raise SystemExit("--verify-evaluation requires --checkpoint, --expected-seed, and --required-attacks")
        payload = verify_evaluation_artifact(
            args.verify_evaluation,
            checkpoint=args.checkpoint,
            expected_seed=args.expected_seed,
            required_attacks=args.required_attacks,
            expected_eps_linf=parse_number(args.expected_eps_linf) if args.expected_eps_linf else None,
            expected_eps_l2=parse_number(args.expected_eps_l2) if args.expected_eps_l2 else None,
            expected_environment_hash=args.expected_environment_hash or None,
        )
    else:
        payload = implementation_manifest()
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "IMPLEMENTATION_FILES",
    "MANIFEST_VERSION",
    "file_sha256",
    "implementation_hash",
    "implementation_manifest",
    "verify_evaluation_environment",
    "verify_evaluation_artifact",
    "verify_lock",
    "write_lock",
]
