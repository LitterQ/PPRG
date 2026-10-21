from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import queue as queue_module
import re
import sys
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from pareto_attack_dg.attacks import generate_attack_domain
from pareto_attack_dg.models import build_model


SUPPORTED_MODEL_ALIASES = {"deit_tiny_patch4_32", "deit_tiny_patch4_32_paper"}
DEFAULT_PRIMARY_ATTACK = "pgd_linf_10"
DEFAULT_AUXILIARY_ATTACK = "pgd_l2_5"
DEFAULT_UNSEEN_ATTACKS = "pgd_linf_20,pgd_l2_20,cw_linf_20"
DEFAULT_PLOT_ATTACKS = "pgd_linf_10,cw_linf_20,pgd_l2_20"
DIRECTION_NORM_EPS = 1e-12
DELTA_CENTER_TOL = 1e-10
PARAMETER_IDENTITY_TOL = 1e-7
PROFILE_MATCH_RTOL = 1e-10
PROFILE_MATCH_ATOL = 1e-12


def checkpoint_specs_for_seed(seed: int) -> list[tuple[str, str]]:
    """Return the matched four-checkpoint comparison for one training seed."""
    return [
        ("PGD-AT", f"pgd_at_linf10_seed{seed}/best.pt"),
        ("PDA 0.1", f"pda_no_proj_w0p1_seed{seed}/best.pt"),
        ("PDA 0.2", f"pda_no_proj_w0p2_seed{seed}/best.pt"),
        ("PDA-Proj 0.2", f"pda_proj_w0p2_seed{seed}/best.pt"),
    ]


# -----------------------------------------------------------------------------
# Provenance / checkpoint validation
# -----------------------------------------------------------------------------


def require_task_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(TASK_ROOT)
    except ValueError as exc:
        raise ValueError(f"Output must remain under {TASK_ROOT}; got {resolved}") from exc
    return resolved


def require_project_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path must remain under {PROJECT_ROOT}; got {resolved}") from exc
    return resolved


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Checkpoint lacks model_state_dict: {path}")
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in state.items():
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_checkpoints(
    paths: list[Path], payloads: list[dict[str, Any]]
) -> tuple[list[str], list[tuple[int, ...]], list[dict[str, torch.Tensor]]]:
    """Preserve the original strict architecture/provenance/state validation."""
    configs = [payload.get("config", {}) for payload in payloads]
    model_aliases = {config.get("model_alias") for config in configs}
    if len(model_aliases) != 1:
        raise ValueError(f"Checkpoints disagree on model_alias: {model_aliases}")
    model_alias = next(iter(model_aliases))
    if model_alias not in SUPPORTED_MODEL_ALIASES:
        raise ValueError(f"Unsupported checkpoint architecture: {model_alias!r}")
    for path, config in zip(paths, configs):
        if not config.get("pretrained_loaded"):
            raise ValueError(f"Checkpoint does not record pretrained initialization: {path}")

    provenance_keys = ["pretrained_source", "initial_trainable_parameters_hash"]
    if model_alias == "deit_tiny_patch4_32":
        provenance_keys.append("upstream_timm_model_identifier")
    else:
        provenance_keys.extend(
            [
                "paper_recipe_hash",
                "implementation_hash",
                "pretrained_checkpoint_sha256",
                "normalization_mode",
                "normalization_mean",
                "normalization_std",
            ]
        )
    for key in provenance_keys:
        serialized = {json.dumps(config.get(key), sort_keys=True) for config in configs}
        if len(serialized) != 1:
            raise ValueError(f"Checkpoints disagree on {key}: {serialized}")

    states = [payload["model_state_dict"] for payload in payloads]
    state_names = list(states[0].keys())
    state_shapes = [tuple(states[0][name].shape) for name in state_names]
    for path, state in zip(paths, states):
        if list(state.keys()) != state_names:
            raise ValueError(f"State ordering/keys differ in {path}")
        for name, shape in zip(state_names, state_shapes):
            value = state[name]
            if tuple(value.shape) != shape:
                raise ValueError(f"Shape mismatch for {name} in {path}")
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite tensor {name} in {path}")
    return state_names, state_shapes, states


# -----------------------------------------------------------------------------
# Local parameter plane: shared raw directions + checkpoint-relative norm
# -----------------------------------------------------------------------------


def get_trainable_parameter_spec(model: torch.nn.Module) -> tuple[list[str], list[tuple[int, ...]]]:
    named_parameters = dict(model.named_parameters())
    names = [name for name, parameter in named_parameters.items() if parameter.requires_grad]
    shapes = [tuple(named_parameters[name].shape) for name in names]
    return names, shapes


def state_trainable_parameters(
    state: dict[str, torch.Tensor], names: list[str]
) -> dict[str, np.ndarray]:
    return {
        name: state[name].detach().cpu().float().numpy().copy()
        for name in names
    }


def arrays_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def make_shared_raw_directions(
    names: list[str], shapes: list[tuple[int, ...]], direction_seed: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Generate exactly one pair of Gaussian raw directions shared by all methods."""
    rng = np.random.default_rng(direction_seed)
    directions: list[dict[str, np.ndarray]] = []
    for _ in range(2):
        direction: dict[str, np.ndarray] = {}
        for name, shape in zip(names, shapes):
            direction[name] = rng.standard_normal(shape).astype(np.float32)
        directions.append(direction)
    return directions[0], directions[1]


def normalize_direction_for_checkpoint(
    raw_direction: dict[str, np.ndarray],
    base_parameters: dict[str, np.ndarray],
    eps: float = DIRECTION_NORM_EPS,
) -> dict[str, np.ndarray]:
    """Filter-wise relative normalization for 2-D+ tensors and tensor-wise for 1-D.

    For a zero-norm parameter tensor/filter, the corresponding direction is set
    to zero. This stable fallback avoids division by zero and, importantly, does
    not introduce an arbitrary perturbation scale for parameters whose reference
    magnitude is zero.
    """
    normalized: dict[str, np.ndarray] = {}
    for name, raw in raw_direction.items():
        theta = base_parameters[name].astype(np.float32, copy=False)
        raw = raw.astype(np.float32, copy=False)
        if theta.ndim >= 2:
            theta_flat = theta.reshape(theta.shape[0], -1)
            raw_flat = raw.reshape(raw.shape[0], -1)
            theta_norm = np.linalg.norm(theta_flat.astype(np.float64), axis=1)
            raw_norm = np.linalg.norm(raw_flat.astype(np.float64), axis=1)
            scale = np.zeros_like(theta_norm, dtype=np.float64)
            valid = (theta_norm > eps) & (raw_norm > eps)
            scale[valid] = theta_norm[valid] / raw_norm[valid]
            shaped_scale = scale.reshape((theta.shape[0],) + (1,) * (theta.ndim - 1))
            value = raw.astype(np.float64) * shaped_scale
        else:
            theta_norm = float(np.linalg.norm(theta.astype(np.float64).reshape(-1)))
            raw_norm = float(np.linalg.norm(raw.astype(np.float64).reshape(-1)))
            if theta_norm <= eps or raw_norm <= eps:
                value = np.zeros_like(raw, dtype=np.float64)
            else:
                value = raw.astype(np.float64) * (theta_norm / raw_norm)
        normalized[name] = value.astype(np.float32, copy=False)
    return normalized


def build_local_directions(
    base_parameters: dict[str, np.ndarray],
    raw_direction_1: dict[str, np.ndarray],
    raw_direction_2: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    return (
        normalize_direction_for_checkpoint(raw_direction_1, base_parameters),
        normalize_direction_for_checkpoint(raw_direction_2, base_parameters),
    )


def assign_local_perturbation(
    model: torch.nn.Module,
    names: list[str],
    base_parameters: dict[str, np.ndarray],
    direction_1: dict[str, np.ndarray],
    direction_2: dict[str, np.ndarray],
    alpha: float,
    beta: float,
) -> None:
    """Assign theta + alpha*d1 + beta*d2 directly; never accumulate grid updates."""
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name in names:
            parameter = parameters[name]
            value = (
                base_parameters[name].astype(np.float64)
                + float(alpha) * direction_1[name].astype(np.float64)
                + float(beta) * direction_2[name].astype(np.float64)
            )
            parameter.copy_(
                torch.from_numpy(value).to(device=parameter.device, dtype=parameter.dtype)
            )


def center_parameter_identity_error(
    model: torch.nn.Module,
    names: list[str],
    base_parameters: dict[str, np.ndarray],
) -> float:
    parameters = dict(model.named_parameters())
    max_error = 0.0
    for name in names:
        actual = parameters[name].detach().cpu().float().numpy()
        expected = base_parameters[name]
        if actual.size:
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
    return max_error


def direction_l2_norm(direction: dict[str, np.ndarray]) -> float:
    total = 0.0
    for value in direction.values():
        flat = value.astype(np.float64).reshape(-1)
        total += float(np.dot(flat, flat))
    return math.sqrt(total)


# -----------------------------------------------------------------------------
# Attack resolution / deterministic evaluation
# -----------------------------------------------------------------------------


def parse_csv_list(value: str) -> list[str]:
    if not value.strip():
        return []
    result: list[str] = []
    for token in value.split(","):
        token = token.strip()
        if token and token not in result:
            result.append(token)
    return result


def parse_gpu_ids(value: str) -> list[int]:
    if not value.strip():
        return []
    gpu_ids: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            gpu_id = int(token)
        except ValueError as exc:
            raise SystemExit(f"Invalid GPU id {token!r} in --gpus") from exc
        if gpu_id < 0:
            raise SystemExit("GPU ids in --gpus must be non-negative")
        if gpu_id in gpu_ids:
            raise SystemExit(f"Duplicate GPU id {gpu_id} in --gpus")
        gpu_ids.append(gpu_id)
    return gpu_ids


def stable_attack_seed(name: str, seed_base: int) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int(seed_base + int.from_bytes(digest[:4], "little") % 1_000_000)


def infer_attack_configuration(
    name: str,
    role: str,
    seed: int,
    eps_linf: float,
    alpha_linf: float,
    eps_l2: float,
    alpha_l2: float,
) -> dict[str, Any]:
    """Describe configuration passed to the existing project attack generator.

    The script deliberately does not create a new attack registry. The attack
    name is forwarded unchanged to generate_attack_domain(...). For conventional
    project names like pgd_linf_20 / cw_l2_50, algorithm, norm, and step count
    are parsed only for metadata. Unknown names remain valid CLI inputs and are
    resolved by the project's own generate_attack_domain implementation.
    """
    lowered = name.lower()
    norm: str | None = None
    if "linf" in lowered:
        norm = "linf"
    elif re.search(r"(^|_)l2($|_)", lowered):
        norm = "l2"

    algorithm: str | None = None
    if lowered.startswith("pgd_"):
        algorithm = "pgd"
    elif lowered.startswith("cw_"):
        algorithm = "cw"
    elif "autoattack" in lowered or lowered.startswith("aa_"):
        algorithm = "autoattack"
    elif lowered.startswith("fgsm_"):
        algorithm = "fgsm"

    steps: int | None = None
    match = re.match(r"^(?:pgd|cw)_(?:linf|l2)_(\d+)$", lowered)
    if match:
        steps = int(match.group(1))
    elif algorithm == "fgsm":
        steps = 1

    if norm == "linf":
        epsilon: float | None = eps_linf
        step_size: float | None = alpha_linf
    elif norm == "l2":
        epsilon = eps_l2
        step_size = alpha_l2
    else:
        epsilon = None
        step_size = None

    return {
        "name": name,
        "role": role,
        "algorithm": algorithm,
        "norm": norm,
        "epsilon": epsilon,
        "step_size": step_size,
        "steps": steps,
        "random_start": True,
        "seed": seed,
        "attack_eval_mode": False,
        "resolution": "existing_generate_attack_domain",
        "generator_kwargs": {
            "eps_linf": eps_linf,
            "alpha_linf": alpha_linf,
            "eps_l2": eps_l2,
            "alpha_l2": alpha_l2,
            "random_start": True,
            "attack_eval_mode": False,
        },
    }


def resolve_attack_specs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    unseen = parse_csv_list(args.unseen_attacks)
    if args.primary_attack == args.auxiliary_attack:
        raise SystemExit("--primary_attack and --auxiliary_attack must be different")
    if args.primary_attack in unseen or args.auxiliary_attack in unseen:
        raise SystemExit("Source attacks must not be repeated in --unseen_attacks")

    seed_map = {
        args.primary_attack: args.primary_attack_seed,
        args.auxiliary_attack: args.auxiliary_attack_seed,
    }
    for attack_name in unseen:
        seed_map[attack_name] = stable_attack_seed(attack_name, args.unseen_attack_seed_base)

    roles = {
        args.primary_attack: "primary",
        args.auxiliary_attack: "auxiliary",
        **{attack_name: "unseen" for attack_name in unseen},
    }
    names = [args.primary_attack, args.auxiliary_attack, *unseen]
    specs = [
        infer_attack_configuration(
            name=name,
            role=roles[name],
            seed=seed_map[name],
            eps_linf=args.eps_linf,
            alpha_linf=args.alpha_linf,
            eps_l2=args.eps_l2,
            alpha_l2=args.alpha_l2,
        )
        for name in names
    ]

    raw_plot_attacks = parse_csv_list(args.plot_attacks)
    plot_attacks: list[str] = []
    for token in raw_plot_attacks:
        resolved = token
        if token == "primary":
            resolved = args.primary_attack
        elif token == "auxiliary":
            resolved = args.auxiliary_attack
        if resolved not in names:
            raise SystemExit(
                f"--plot_attacks contains {token!r} -> {resolved!r}, which is not among evaluated attacks {names}"
            )
        if resolved not in plot_attacks:
            plot_attacks.append(resolved)
    return specs, plot_attacks


def set_attack_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def clean_loss(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            total_loss += float(F.cross_entropy(model(x), y, reduction="sum").item())
            total += y.numel()
    return total_loss / max(1, total)


def adversarial_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    domain: str,
    attack_seed: int,
    *,
    eps_linf: float,
    alpha_linf: float,
    eps_l2: float,
    alpha_l2: float,
) -> float:
    """Re-attack the currently perturbed model for every invocation."""
    set_attack_seed(attack_seed, device)
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_adv = generate_attack_domain(
            model,
            x,
            y,
            domain,
            eps_linf=eps_linf,
            alpha_linf=alpha_linf,
            eps_l2=eps_l2,
            alpha_l2=alpha_l2,
            random_start=True,
            attack_eval_mode=False,
        )
        with torch.no_grad():
            total_loss += float(F.cross_entropy(model(x_adv), y, reduction="sum").item())
        total += y.numel()
    return total_loss / max(1, total)


def evaluate_attack_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    attack_spec: dict[str, Any],
) -> float:
    kwargs = attack_spec["generator_kwargs"]
    return adversarial_loss(
        model,
        loader,
        device,
        attack_spec["name"],
        int(attack_spec["seed"]),
        eps_linf=float(kwargs["eps_linf"]),
        alpha_linf=float(kwargs["alpha_linf"]),
        eps_l2=float(kwargs["eps_l2"]),
        alpha_l2=float(kwargs["alpha_l2"]),
    )


def evaluate_local_landscapes(
    model: torch.nn.Module,
    states: list[dict[str, torch.Tensor]],
    labels: list[str],
    parameter_names: list[str],
    raw_direction_1: dict[str, np.ndarray],
    raw_direction_2: dict[str, np.ndarray],
    attack_specs: list[dict[str, Any]],
    alpha_axis: np.ndarray,
    beta_axis: np.ndarray,
    loader: DataLoader,
    device: torch.device,
    log_prefix: str = "",
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    num_methods = len(labels)
    num_attacks = len(attack_specs)
    grid_size = len(alpha_axis)
    raw_loss = np.empty((num_methods, num_attacks, grid_size, grid_size), dtype=np.float64)
    direction_metadata: list[dict[str, Any]] = []

    total_evaluations = num_methods * num_attacks * grid_size * grid_size
    completed = 0
    print(
        f"{log_prefix}Total attack evaluations: {total_evaluations} = {num_methods} methods x "
        f"{grid_size ** 2} grid points x {num_attacks} attacks",
        flush=True,
    )

    for method_idx, (label, state) in enumerate(zip(labels, states)):
        # Required: each method starts from its own complete checkpoint state.
        model.load_state_dict(state, strict=True)
        model.eval()
        base_parameters = state_trainable_parameters(state, parameter_names)
        direction_1, direction_2 = build_local_directions(
            base_parameters, raw_direction_1, raw_direction_2
        )

        # The same d1/d2 are reused for every attack of this method.
        assign_local_perturbation(
            model, parameter_names, base_parameters, direction_1, direction_2, 0.0, 0.0
        )
        identity_error = center_parameter_identity_error(model, parameter_names, base_parameters)
        if identity_error > PARAMETER_IDENTITY_TOL:
            raise RuntimeError(
                f"Center parameter identity check failed for {label}: max_abs_error={identity_error:.3e}"
            )
        direction_metadata.append(
            {
                "label": label,
                "center_parameter_identity_max_abs_error": identity_error,
                "direction_1_sha256": arrays_sha256(direction_1),
                "direction_2_sha256": arrays_sha256(direction_2),
                "direction_1_l2_norm": direction_l2_norm(direction_1),
                "direction_2_l2_norm": direction_l2_norm(direction_2),
            }
        )

        grid_counter = 0
        for row, beta in enumerate(beta_axis):
            for col, alpha in enumerate(alpha_axis):
                grid_counter += 1
                # Direct construction from checkpoint center; no accumulated perturbation.
                assign_local_perturbation(
                    model,
                    parameter_names,
                    base_parameters,
                    direction_1,
                    direction_2,
                    float(alpha),
                    float(beta),
                )
                model.eval()
                for attack_idx, attack_spec in enumerate(attack_specs):
                    # Fresh adversarial examples are generated inside every call, after
                    # resetting the RNG to this attack's deterministic seed.
                    loss = evaluate_attack_loss(model, loader, device, attack_spec)
                    if not math.isfinite(loss):
                        raise RuntimeError(
                            f"Non-finite loss: method={label}, attack={attack_spec['name']}, "
                            f"alpha={alpha}, beta={beta}"
                        )
                    raw_loss[method_idx, attack_idx, row, col] = loss
                    completed += 1
                    print(
                        f"{log_prefix}method={method_idx + 1}/{num_methods} label={label!r} "
                        f"grid={grid_counter}/{grid_size ** 2} "
                        f"attack={attack_idx + 1}/{num_attacks} "
                        f"attack_name={attack_spec['name']} alpha={alpha:.5f} beta={beta:.5f} "
                        f"loss={loss:.6f} total={completed}/{total_evaluations}",
                        flush=True,
                    )
    return raw_loss, direction_metadata


def _gpu_landscape_worker(
    method_idx: int,
    label: str,
    checkpoint_path: str,
    model_alias: str,
    expected_parameter_names: list[str],
    expected_parameter_shapes: list[tuple[int, ...]],
    expected_raw_direction_hashes: dict[str, str],
    direction_seed: int,
    attack_specs: list[dict[str, Any]],
    alpha_axis: np.ndarray,
    beta_axis: np.ndarray,
    data_dir: str,
    subset_indices: list[int],
    batch_size: int,
    num_workers: int,
    gpu_id: int,
    result_queue: Any,
) -> None:
    """Evaluate one checkpoint on one GPU and return its local landscapes."""
    prefix = f"[GPU {gpu_id} | {label}] "
    try:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        payload = load_checkpoint(Path(checkpoint_path))
        state = payload["model_state_dict"]

        model = build_model(model_alias, num_classes=10, pretrained=False)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        parameter_names, parameter_shapes = get_trainable_parameter_spec(model)
        if parameter_names != expected_parameter_names or parameter_shapes != expected_parameter_shapes:
            raise RuntimeError(
                f"{prefix}Trainable parameter specification differs from the validated parent model"
            )

        raw_direction_1, raw_direction_2 = make_shared_raw_directions(
            parameter_names, parameter_shapes, direction_seed
        )
        worker_raw_hashes = {
            "direction_1_sha256": arrays_sha256(raw_direction_1),
            "direction_2_sha256": arrays_sha256(raw_direction_2),
        }
        if worker_raw_hashes != expected_raw_direction_hashes:
            raise RuntimeError(
                f"{prefix}Shared raw direction hashes differ from the parent process"
            )

        dataset = datasets.CIFAR10(
            root=Path(data_dir),
            train=False,
            download=False,
            transform=transforms.ToTensor(),
        )
        loader = DataLoader(
            Subset(dataset, subset_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        method_loss, direction_metadata = evaluate_local_landscapes(
            model=model,
            states=[state],
            labels=[label],
            parameter_names=parameter_names,
            raw_direction_1=raw_direction_1,
            raw_direction_2=raw_direction_2,
            attack_specs=attack_specs,
            alpha_axis=alpha_axis,
            beta_axis=beta_axis,
            loader=loader,
            device=device,
            log_prefix=prefix,
        )
        worker_metadata = direction_metadata[0]
        worker_metadata["gpu_id"] = int(gpu_id)
        worker_metadata["device"] = str(device)
        result_queue.put(
            {
                "status": "ok",
                "method_idx": int(method_idx),
                "raw_loss": method_loss[0],
                "direction_metadata": worker_metadata,
            }
        )
        print(f"{prefix}checkpoint evaluation complete", flush=True)
    except BaseException:
        result_queue.put(
            {
                "status": "error",
                "method_idx": int(method_idx),
                "label": label,
                "gpu_id": int(gpu_id),
                "traceback": traceback.format_exc(),
            }
        )
        raise


def evaluate_local_landscapes_multi_gpu(
    paths: list[Path],
    labels: list[str],
    model_alias: str,
    parameter_names: list[str],
    parameter_shapes: list[tuple[int, ...]],
    raw_direction_hashes: dict[str, str],
    direction_seed: int,
    attack_specs: list[dict[str, Any]],
    alpha_axis: np.ndarray,
    beta_axis: np.ndarray,
    data_dir: Path,
    subset_indices: list[int],
    batch_size: int,
    num_workers: int,
    gpu_ids: list[int],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Run one checkpoint per GPU, then aggregate in the parent process."""
    if len(gpu_ids) != len(labels):
        raise RuntimeError(
            f"Multi-GPU evaluation requires one GPU per checkpoint: "
            f"got {len(gpu_ids)} GPUs for {len(labels)} checkpoints"
        )

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes: list[mp.Process] = []
    for method_idx, (label, path, gpu_id) in enumerate(zip(labels, paths, gpu_ids)):
        print(f"Assigning {label!r} to cuda:{gpu_id} ({path})", flush=True)
        process = context.Process(
            target=_gpu_landscape_worker,
            kwargs={
                "method_idx": method_idx,
                "label": label,
                "checkpoint_path": str(path),
                "model_alias": model_alias,
                "expected_parameter_names": parameter_names,
                "expected_parameter_shapes": parameter_shapes,
                "expected_raw_direction_hashes": raw_direction_hashes,
                "direction_seed": direction_seed,
                "attack_specs": attack_specs,
                "alpha_axis": alpha_axis,
                "beta_axis": beta_axis,
                "data_dir": str(data_dir),
                "subset_indices": subset_indices,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "gpu_id": gpu_id,
                "result_queue": result_queue,
            },
            name=f"loss-landscape-{safe_filename_token(label)}-gpu{gpu_id}",
        )
        process.start()
        processes.append(process)

    results: dict[int, dict[str, Any]] = {}
    try:
        while len(results) < len(processes):
            try:
                message = result_queue.get(timeout=5.0)
            except queue_module.Empty:
                failed = [
                    process
                    for process in processes
                    if process.exitcode is not None and process.exitcode != 0
                ]
                if failed:
                    details = ", ".join(
                        f"{process.name}:exitcode={process.exitcode}" for process in failed
                    )
                    raise RuntimeError(
                        "A GPU worker exited before returning a result: " + details
                    )
                if not any(process.is_alive() for process in processes):
                    raise RuntimeError(
                        "All GPU workers exited before every checkpoint result was collected"
                    )
                continue

            if message.get("status") != "ok":
                raise RuntimeError(
                    "GPU checkpoint evaluation failed:\n" + str(message.get("traceback", message))
                )
            method_idx = int(message["method_idx"])
            if method_idx in results:
                raise RuntimeError(f"Received duplicate GPU result for method index {method_idx}")
            results[method_idx] = message
    except BaseException:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        raise
    finally:
        result_queue.close()
        result_queue.join_thread()

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(
                f"GPU worker {process.name} exited with code {process.exitcode}"
            )

    expected_shape = (len(attack_specs), len(beta_axis), len(alpha_axis))
    ordered_losses: list[np.ndarray] = []
    ordered_direction_metadata: list[dict[str, Any]] = []
    for method_idx, label in enumerate(labels):
        result = results[method_idx]
        method_loss = np.asarray(result["raw_loss"], dtype=np.float64)
        if method_loss.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected landscape shape from {label}: "
                f"got {method_loss.shape}, expected {expected_shape}"
            )
        ordered_losses.append(method_loss)
        ordered_direction_metadata.append(result["direction_metadata"])

    raw_loss = np.stack(ordered_losses, axis=0)
    print("All GPU workers completed; aggregating metrics and figures in the parent process", flush=True)
    return raw_loss, ordered_direction_metadata


# -----------------------------------------------------------------------------
# Post-processing: delta loss, local flatness, transfer, consistency
# -----------------------------------------------------------------------------


def compute_delta_losses(raw_loss: np.ndarray, center_idx: int) -> np.ndarray:
    center = raw_loss[:, :, center_idx, center_idx][:, :, None, None]
    delta = raw_loss - center
    center_delta = np.abs(delta[:, :, center_idx, center_idx])
    if np.max(center_delta) > DELTA_CENTER_TOL:
        raise RuntimeError(
            f"Delta-center sanity check failed: max |Delta L(0,0)| = {np.max(center_delta):.3e}"
        )
    return delta


def circular_grid_mask(alpha_axis: np.ndarray, beta_axis: np.ndarray, radius: float) -> np.ndarray:
    grid_alpha, grid_beta = np.meshgrid(alpha_axis, beta_axis)
    grid_radius = np.sqrt(
        grid_alpha.astype(np.float64) ** 2 + grid_beta.astype(np.float64) ** 2
    )
    # Axes are stored as float32, so the nominal +/-radius endpoints can be a
    # few ulps outside their decimal CLI value. Include those boundary points.
    radius_tolerance = max(1e-12, abs(float(radius)) * 1e-7)
    return grid_radius <= float(radius) + radius_tolerance


def compute_flatness_metrics(
    raw_loss: np.ndarray,
    delta_loss: np.ndarray,
    labels: list[str],
    paths: list[Path],
    attack_specs: list[dict[str, Any]],
    mask: np.ndarray,
    center_idx: int,
    radius: float,
    direction_seed: int,
    grid_size: int,
    num_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_idx, label in enumerate(labels):
        for attack_idx, attack_spec in enumerate(attack_specs):
            raw_values = raw_loss[method_idx, attack_idx][mask]
            delta_values = delta_loss[method_idx, attack_idx][mask]
            rows.append(
                {
                    "seed": int(seed),
                    "label": label,
                    "checkpoint": str(paths[method_idx]),
                    "attack": attack_spec["name"],
                    "attack_role": attack_spec["role"],
                    "center_loss": float(raw_loss[method_idx, attack_idx, center_idx, center_idx]),
                    "radius": float(radius),
                    "mean_raw_loss_in_radius": float(np.mean(raw_values)),
                    "median_raw_loss_in_radius": float(np.median(raw_values)),
                    "max_raw_loss_in_radius": float(np.max(raw_values)),
                    "min_raw_loss_in_radius": float(np.min(raw_values)),
                    "max_delta_loss": float(np.max(delta_values)),
                    "mean_delta_loss": float(np.mean(delta_values)),
                    "mean_positive_delta_loss": float(
                        np.mean(np.maximum(delta_values, 0.0))
                    ),
                    "min_delta_loss": float(np.min(delta_values)),
                    "direction_seed": int(direction_seed),
                    "grid_size": int(grid_size),
                    "num_samples": int(num_samples),
                    "num_grid_points_in_radius": int(mask.sum()),
                }
            )
    return rows


def compute_flatness_transfer(
    flatness_rows: list[dict[str, Any]],
    labels: list[str],
    attack_specs: list[dict[str, Any]],
    ratio_epsilon: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_names = ["mean_positive_delta_loss", "max_delta_loss", "mean_delta_loss"]
    by_key = {(row["label"], row["attack"]): row for row in flatness_rows}
    reference_label = "PGD-AT"
    if reference_label not in labels:
        raise RuntimeError("PGD-AT checkpoint is required as the flatness-transfer reference")

    transfer_rows: list[dict[str, Any]] = []
    for metric_name in metric_names:
        for label in labels:
            for attack_spec in attack_specs:
                attack = attack_spec["name"]
                value = float(by_key[(label, attack)][metric_name])
                reference = float(by_key[(reference_label, attack)][metric_name])
                ratio = (value + ratio_epsilon) / (reference + ratio_epsilon)
                transfer_rows.append(
                    {
                        "seed": int(seed),
                        "method": label,
                        "attack": attack,
                        "attack_role": attack_spec["role"],
                        "flatness_metric": metric_name,
                        "flatness_value": value,
                        "pgd_at_reference_value": reference,
                        "relative_flatness_ratio": float(ratio),
                    }
                )

    primary_attack = next(spec["name"] for spec in attack_specs if spec["role"] == "primary")
    auxiliary_attack = next(spec["name"] for spec in attack_specs if spec["role"] == "auxiliary")
    unseen_attacks = [spec["name"] for spec in attack_specs if spec["role"] == "unseen"]
    main_metric = "mean_positive_delta_loss"
    ratio_map = {
        (row["method"], row["attack"]): float(row["relative_flatness_ratio"])
        for row in transfer_rows
        if row["flatness_metric"] == main_metric
    }
    summary_rows: list[dict[str, Any]] = []
    for label in labels:
        unseen_values = [ratio_map[(label, attack)] for attack in unseen_attacks]
        summary_rows.append(
            {
                "seed": int(seed),
                "method": label,
                "flatness_metric": main_metric,
                "primary_ratio": ratio_map[(label, primary_attack)],
                "auxiliary_ratio": ratio_map[(label, auxiliary_attack)],
                "mean_unseen_ratio": float(np.mean(unseen_values)) if unseen_values else float("nan"),
                "median_unseen_ratio": float(np.median(unseen_values)) if unseen_values else float("nan"),
                "num_unseen_attacks": len(unseen_values),
            }
        )
    return transfer_rows, summary_rows


def compute_robust_loss_region(
    flatness_rows: list[dict[str, Any]],
    labels: list[str],
    attack_specs: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    """Compare absolute center and neighborhood robust loss against PGD-AT."""
    reference_label = "PGD-AT"
    if reference_label not in labels:
        raise RuntimeError("PGD-AT checkpoint is required as the robust-loss-region reference")
    by_key = {(row["label"], row["attack"]): row for row in flatness_rows}

    rows: list[dict[str, Any]] = []
    for label in labels:
        for attack_spec in attack_specs:
            attack = attack_spec["name"]
            current = by_key[(label, attack)]
            reference = by_key[(reference_label, attack)]
            center_loss = float(current["center_loss"])
            reference_center = float(reference["center_loss"])
            mean_region_loss = float(current["mean_raw_loss_in_radius"])
            reference_mean_region = float(reference["mean_raw_loss_in_radius"])
            center_ratio = (
                center_loss / reference_center
                if abs(reference_center) > np.finfo(np.float64).tiny
                else float("nan")
            )
            region_ratio = (
                mean_region_loss / reference_mean_region
                if abs(reference_mean_region) > np.finfo(np.float64).tiny
                else float("nan")
            )
            rows.append(
                {
                    "seed": int(seed),
                    "method": label,
                    "attack": attack,
                    "attack_role": attack_spec["role"],
                    "center_loss": center_loss,
                    "pgd_at_center_loss": reference_center,
                    "center_loss_difference_vs_pgd_at": center_loss - reference_center,
                    "center_loss_ratio_vs_pgd_at": float(center_ratio),
                    "mean_raw_loss_in_radius": mean_region_loss,
                    "pgd_at_mean_raw_loss_in_radius": reference_mean_region,
                    "mean_region_loss_difference_vs_pgd_at": (
                        mean_region_loss - reference_mean_region
                    ),
                    "mean_region_loss_ratio_vs_pgd_at": float(region_ratio),
                    "radius": float(current["radius"]),
                    "grid_size": int(current["grid_size"]),
                    "num_samples": int(current["num_samples"]),
                }
            )
    return rows


def compute_radius_profiles(
    raw_loss: np.ndarray,
    delta_loss: np.ndarray,
    labels: list[str],
    attack_specs: list[dict[str, Any]],
    alpha_axis: np.ndarray,
    beta_axis: np.ndarray,
    radius: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Post-process cumulative disk profiles without rerunning any attack."""
    grid_alpha, grid_beta = np.meshgrid(alpha_axis, beta_axis)
    grid_radius = np.sqrt(
        grid_alpha.astype(np.float64) ** 2 + grid_beta.astype(np.float64) ** 2
    )
    radius_tolerance = max(1e-12, abs(float(radius)) * 1e-7)
    full_mask = circular_grid_mask(alpha_axis, beta_axis, radius)
    radius_points = np.unique(grid_radius[(grid_radius > 0.0) & full_mask])
    radius_points.sort()
    if radius_points.size == 0:
        raise RuntimeError("No positive grid radii are available for radius profiles")

    rows: list[dict[str, Any]] = []
    for method_idx, label in enumerate(labels):
        for attack_idx, attack_spec in enumerate(attack_specs):
            for profile_radius in radius_points:
                profile_mask = full_mask & (
                    grid_radius <= float(profile_radius) + radius_tolerance
                )
                raw_values = raw_loss[method_idx, attack_idx][profile_mask]
                delta_values = delta_loss[method_idx, attack_idx][profile_mask]
                rows.append(
                    {
                        "seed": int(seed),
                        "method": label,
                        "attack": attack_spec["name"],
                        "attack_role": attack_spec["role"],
                        "radius": float(profile_radius),
                        "num_grid_points": int(profile_mask.sum()),
                        "mean_raw_loss": float(np.mean(raw_values)),
                        "median_raw_loss": float(np.median(raw_values)),
                        "mean_delta_loss": float(np.mean(delta_values)),
                        "mean_positive_delta_loss": float(
                            np.mean(np.maximum(delta_values, 0.0))
                        ),
                        "max_delta_loss": float(np.max(delta_values)),
                    }
                )
    return rows


def validate_absolute_center_losses(
    raw_loss: np.ndarray,
    flatness_rows: list[dict[str, Any]],
    labels: list[str],
    attack_specs: list[dict[str, Any]],
    center_idx: int,
) -> None:
    """Ensure every absolute landscape center is the center loss written to CSV."""
    by_key = {(row["label"], row["attack"]): row for row in flatness_rows}
    for method_idx, label in enumerate(labels):
        for attack_idx, attack_spec in enumerate(attack_specs):
            raw_center = float(raw_loss[method_idx, attack_idx, center_idx, center_idx])
            csv_center = float(by_key[(label, attack_spec["name"])]["center_loss"])
            if not np.isclose(
                raw_center,
                csv_center,
                rtol=PROFILE_MATCH_RTOL,
                atol=PROFILE_MATCH_ATOL,
            ):
                raise RuntimeError(
                    "Absolute-center sanity check failed for "
                    f"method={label}, attack={attack_spec['name']}: "
                    f"grid={raw_center:.12g}, metrics={csv_center:.12g}"
                )


def validate_radius_profiles(
    profile_rows: list[dict[str, Any]],
    flatness_rows: list[dict[str, Any]],
    labels: list[str],
    attack_specs: list[dict[str, Any]],
    radius: float,
) -> None:
    """Validate monotonic profile support and full-radius metric agreement."""
    flatness_by_key = {(row["label"], row["attack"]): row for row in flatness_rows}
    for label in labels:
        for attack_spec in attack_specs:
            attack = attack_spec["name"]
            selected = [
                row
                for row in profile_rows
                if row["method"] == label and row["attack"] == attack
            ]
            if not selected:
                raise RuntimeError(f"Missing radius profile for method={label}, attack={attack}")
            radii = np.asarray([row["radius"] for row in selected], dtype=np.float64)
            counts = np.asarray([row["num_grid_points"] for row in selected], dtype=np.int64)
            if np.any(np.diff(radii) <= 0.0):
                raise RuntimeError(
                    f"Radius profile is not strictly increasing for method={label}, attack={attack}"
                )
            if np.any(np.diff(counts) < 0):
                raise RuntimeError(
                    f"Radius-profile point count decreased for method={label}, attack={attack}"
                )
            if not np.isclose(radii[-1], radius, rtol=1e-6, atol=1e-10):
                raise RuntimeError(
                    f"Radius profile does not reach --radius for method={label}, attack={attack}: "
                    f"last={radii[-1]:.12g}, expected={radius:.12g}"
                )

            last = selected[-1]
            flatness = flatness_by_key[(label, attack)]
            comparisons = {
                "mean_raw_loss": "mean_raw_loss_in_radius",
                "mean_delta_loss": "mean_delta_loss",
                "mean_positive_delta_loss": "mean_positive_delta_loss",
                "max_delta_loss": "max_delta_loss",
            }
            for profile_key, flatness_key in comparisons.items():
                profile_value = float(last[profile_key])
                flatness_value = float(flatness[flatness_key])
                if not np.isclose(
                    profile_value,
                    flatness_value,
                    rtol=PROFILE_MATCH_RTOL,
                    atol=PROFILE_MATCH_ATOL,
                ):
                    raise RuntimeError(
                        "Full-radius profile sanity check failed for "
                        f"method={label}, attack={attack}, metric={profile_key}: "
                        f"profile={profile_value:.12g}, metrics={flatness_value:.12g}"
                    )


def rankdata_average(values: np.ndarray) -> np.ndarray:
    """Small NumPy-only equivalent of scipy.stats.rankdata(method='average')."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        return float("nan")
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = math.sqrt(float(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)))
    if denom <= 1e-30:
        return float("nan")
    return float(np.dot(x_centered, y_centered) / denom)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    return safe_pearson(rankdata_average(x), rankdata_average(y))


def compute_landscape_consistency(
    delta_loss: np.ndarray,
    labels: list[str],
    attack_specs: list[dict[str, Any]],
    mask: np.ndarray,
    center_idx: int,
    radius: float,
    seed: int,
    exclude_center: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    primary_idx = next(idx for idx, spec in enumerate(attack_specs) if spec["role"] == "primary")
    primary_name = attack_specs[primary_idx]["name"]
    target_indices = [idx for idx, spec in enumerate(attack_specs) if idx != primary_idx]

    effective_mask = mask.copy()
    if exclude_center:
        effective_mask[center_idx, center_idx] = False
    num_points = int(effective_mask.sum())
    if num_points < 2:
        raise RuntimeError("Too few legal grid points for cross-attack correlation")

    rows: list[dict[str, Any]] = []
    pair_rows_by_method: dict[str, list[dict[str, Any]]] = {}
    for method_idx, label in enumerate(labels):
        reference = delta_loss[method_idx, primary_idx][effective_mask]
        for attack_idx in target_indices:
            target = delta_loss[method_idx, attack_idx][effective_mask]
            spec = attack_specs[attack_idx]
            rows.append(
                {
                    "seed": int(seed),
                    "method": label,
                    "reference_attack": primary_name,
                    "target_attack": spec["name"],
                    "target_role": spec["role"],
                    "pearson": safe_pearson(reference, target),
                    "spearman": safe_spearman(reference, target),
                    "num_grid_points": num_points,
                    "radius": float(radius),
                    "exclude_center": bool(exclude_center),
                }
            )

        # Optional enhancement: complete attack-by-attack consistency matrix.
        method_pairs: list[dict[str, Any]] = []
        for attack_a in range(len(attack_specs)):
            values_a = delta_loss[method_idx, attack_a][effective_mask]
            for attack_b in range(len(attack_specs)):
                values_b = delta_loss[method_idx, attack_b][effective_mask]
                method_pairs.append(
                    {
                        "seed": int(seed),
                        "method": label,
                        "attack_a": attack_specs[attack_a]["name"],
                        "attack_b": attack_specs[attack_b]["name"],
                        "pearson": safe_pearson(values_a, values_b),
                        "spearman": safe_spearman(values_a, values_b),
                        "num_grid_points": num_points,
                        "radius": float(radius),
                        "exclude_center": bool(exclude_center),
                    }
                )
        pair_rows_by_method[label] = method_pairs
    return rows, pair_rows_by_method


# -----------------------------------------------------------------------------
# CSV / figure rendering
# -----------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_filename_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return token.strip("_") or "attack"


def attack_contour_filename(attack_name: str, primary_attack: str, auxiliary_attack: str) -> str:
    if attack_name == primary_attack:
        return "primary_delta_loss_contours.png"
    if attack_name == auxiliary_attack:
        return "auxiliary_delta_loss_contours.png"
    return f"{safe_filename_token(attack_name)}_delta_loss_contours.png"


def render_absolute_landscape_grid(
    output_dir: Path,
    raw_loss: np.ndarray,
    labels: list[str],
    attack_specs: list[dict[str, Any]],
    plot_attacks: list[str],
    alpha_axis: np.ndarray,
    beta_axis: np.ndarray,
) -> None:
    """Render one raw-loss row per attack with a scale shared across methods."""
    grid_alpha, grid_beta = np.meshgrid(alpha_axis, beta_axis)
    attack_index = {spec["name"]: idx for idx, spec in enumerate(attack_specs)}
    role_by_name = {spec["name"]: spec["role"] for spec in attack_specs}
    num_rows = len(plot_attacks)
    num_cols = len(labels)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(3.25 * num_cols + 1.3, 3.0 * num_rows),
        constrained_layout=True,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row_idx, attack_name in enumerate(plot_attacks):
        attack_idx = attack_index[attack_name]
        row_values = raw_loss[:, attack_idx]
        raw_vmin = float(np.min(row_values))
        raw_vmax = float(np.max(row_values))
        if not (math.isfinite(raw_vmin) and math.isfinite(raw_vmax)):
            raise RuntimeError(f"Non-finite absolute contour range for attack={attack_name}")
        if raw_vmax <= raw_vmin:
            padding = max(abs(raw_vmin) * 1e-6, 1e-9)
            plot_vmin, plot_vmax = raw_vmin - padding, raw_vmax + padding
        else:
            plot_vmin, plot_vmax = raw_vmin, raw_vmax
        # The same levels and Normalize object are deliberately reused by all
        # four method panels in this attack row.
        levels = np.linspace(plot_vmin, plot_vmax, 25)
        norm = Normalize(vmin=plot_vmin, vmax=plot_vmax)
        row_contour = None
        for method_idx, label in enumerate(labels):
            ax = axes[row_idx, method_idx]
            row_contour = ax.contourf(
                grid_alpha,
                grid_beta,
                row_values[method_idx],
                levels=levels,
                norm=norm,
                cmap="viridis",
                extend="neither",
            )
            ax.scatter(
                [0.0],
                [0.0],
                s=42,
                facecolors="white",
                edgecolors="black",
                linewidths=1.1,
                zorder=5,
            )
            if row_idx == 0:
                ax.set_title(label)
            if method_idx == 0:
                ax.set_ylabel(f"{attack_name} ({role_by_name[attack_name]})\n$\\beta$")
            if row_idx == num_rows - 1:
                ax.set_xlabel(r"$\alpha$")
            ax.set_aspect("equal", adjustable="box")
        assert row_contour is not None
        fig.colorbar(
            row_contour,
            ax=axes[row_idx, :].ravel().tolist(),
            label="Robust cross-entropy",
            shrink=0.92,
        )

    fig.suptitle("Absolute Robust-Loss Landscapes", fontsize=14)
    fig.savefig(output_dir / "absolute_robust_loss_landscapes.png", dpi=220)
    plt.close(fig)


def render_delta_loss_contours(
    output_dir: Path,
    delta_loss: np.ndarray,
    labels: list[str],
    attack_specs: list[dict[str, Any]],
    plot_attacks: list[str],
    alpha_axis: np.ndarray,
    beta_axis: np.ndarray,
    primary_attack: str,
    auxiliary_attack: str,
) -> None:
    grid_alpha, grid_beta = np.meshgrid(alpha_axis, beta_axis)
    attack_index = {spec["name"]: idx for idx, spec in enumerate(attack_specs)}
    role_by_name = {spec["name"]: spec["role"] for spec in attack_specs}

    for attack_name in plot_attacks:
        idx = attack_index[attack_name]
        values = delta_loss[:, idx]
        magnitude = max(float(np.max(np.abs(values))), 1e-12)
        norm = TwoSlopeNorm(vmin=-magnitude, vcenter=0.0, vmax=magnitude)
        levels = np.linspace(-magnitude, magnitude, 25)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True, sharex=True, sharey=True)
        contour = None
        for method_idx, (ax, label) in enumerate(zip(axes.flat, labels)):
            contour = ax.contourf(
                grid_alpha,
                grid_beta,
                values[method_idx],
                levels=levels,
                norm=norm,
                cmap="coolwarm",
                extend="both",
            )
            ax.scatter([0.0], [0.0], s=45, facecolors="white", edgecolors="black", linewidths=1.0, zorder=5)
            ax.set_title(label)
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$\beta$")
            ax.set_aspect("equal", adjustable="box")
        assert contour is not None
        fig.colorbar(contour, ax=axes.ravel().tolist(), label=r"$\Delta$ cross-entropy")
        fig.suptitle(f"{attack_name} ({role_by_name[attack_name]}) local robust-loss landscape", fontsize=13)
        fig.savefig(
            output_dir / attack_contour_filename(attack_name, primary_attack, auxiliary_attack),
            dpi=220,
        )
        plt.close(fig)


def render_robust_loss_region_heatmap(
    output_dir: Path,
    region_rows: list[dict[str, Any]],
    labels: list[str],
    attack_specs: list[dict[str, Any]],
) -> None:
    attacks = [spec["name"] for spec in attack_specs]
    lookup = {
        (row["method"], row["attack"]): float(row["mean_region_loss_ratio_vs_pgd_at"])
        for row in region_rows
    }
    matrix = np.asarray(
        [[lookup[(label, attack)] for attack in attacks] for label in labels],
        dtype=np.float64,
    )
    finite = matrix[np.isfinite(matrix)]
    max_dev = max(float(np.max(np.abs(finite - 1.0))) if finite.size else 0.0, 0.05)
    norm = TwoSlopeNorm(vmin=1.0 - max_dev, vcenter=1.0, vmax=1.0 + max_dev)

    fig_width = max(8.0, 1.45 * len(attacks) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", norm=norm)
    ax.set_xticks(np.arange(len(attacks)), attacks, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Evaluated attack domain")
    ax.set_ylabel("Method")
    ax.set_title("Cross-Attack Robust-Loss Region (mean loss within radius)")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text_value = "nan" if not np.isfinite(value) else f"{value:.3f}"
            ax.text(col, row, text_value, ha="center", va="center", fontsize=8)
    fig.colorbar(
        image,
        ax=ax,
        label="Mean robust-loss ratio vs PGD-AT (<1 lower; >1 higher)",
    )
    fig.savefig(output_dir / "cross_attack_robust_loss_region_heatmap.png", dpi=220)
    plt.close(fig)


def _profile_rows_for_plot(
    profile_rows: list[dict[str, Any]], method: str, attack: str
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in profile_rows
        if row["method"] == method and row["attack"] == attack
    ]
    stable_rows = [row for row in rows if int(row["num_grid_points"]) >= 5]
    return stable_rows if stable_rows else rows


def render_mean_absolute_loss_profiles(
    output_dir: Path,
    profile_rows: list[dict[str, Any]],
    labels: list[str],
    attack_specs: list[dict[str, Any]],
    plot_attacks: list[str],
) -> None:
    role_by_name = {spec["name"]: spec["role"] for spec in attack_specs}
    fig, axes = plt.subplots(
        1,
        len(plot_attacks),
        figsize=(4.6 * len(plot_attacks), 4.2),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
    )
    for attack_idx, attack in enumerate(plot_attacks):
        ax = axes[0, attack_idx]
        for label in labels:
            rows = _profile_rows_for_plot(profile_rows, label, attack)
            ax.plot(
                [float(row["radius"]) for row in rows],
                [float(row["mean_raw_loss"]) for row in rows],
                marker="o",
                markersize=3.0,
                linewidth=1.6,
                label=label,
            )
        ax.set_title(f"{attack}\n({role_by_name[attack]})")
        ax.set_xlabel("Normalized parameter perturbation radius")
        ax.grid(alpha=0.25, linewidth=0.7)
        if attack_idx == 0:
            ax.set_ylabel("Mean robust cross-entropy within radius")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle("Mean Absolute Robust Loss vs Radius", fontsize=13)
    fig.savefig(output_dir / "mean_absolute_robust_loss_vs_radius.png", dpi=220)
    plt.close(fig)


def render_local_flatness_profiles(
    output_dir: Path,
    profile_rows: list[dict[str, Any]],
    labels: list[str],
    attack_specs: list[dict[str, Any]],
    plot_attacks: list[str],
) -> None:
    role_by_name = {spec["name"]: spec["role"] for spec in attack_specs}
    fig, axes = plt.subplots(
        1,
        len(plot_attacks),
        figsize=(4.6 * len(plot_attacks), 4.2),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
    )
    for attack_idx, attack in enumerate(plot_attacks):
        ax = axes[0, attack_idx]
        for label in labels:
            rows = _profile_rows_for_plot(profile_rows, label, attack)
            ax.plot(
                [float(row["radius"]) for row in rows],
                [float(row["mean_positive_delta_loss"]) for row in rows],
                marker="o",
                markersize=3.0,
                linewidth=1.6,
                label=label,
            )
        ax.set_title(f"{attack}\n({role_by_name[attack]})")
        ax.set_xlabel("Normalized parameter perturbation radius")
        ax.grid(alpha=0.25, linewidth=0.7)
        if attack_idx == 0:
            ax.set_ylabel(r"Mean positive $\Delta$ cross-entropy within radius")
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle("Local Sensitivity / Flatness vs Radius", fontsize=13)
    fig.savefig(output_dir / "local_flatness_vs_radius.png", dpi=220)
    plt.close(fig)


def render_flatness_transfer_heatmap(
    output_dir: Path,
    transfer_rows: list[dict[str, Any]],
    labels: list[str],
    attack_specs: list[dict[str, Any]],
) -> None:
    main_metric = "mean_positive_delta_loss"
    filtered = [row for row in transfer_rows if row["flatness_metric"] == main_metric]
    lookup = {(row["method"], row["attack"]): float(row["relative_flatness_ratio"]) for row in filtered}
    attacks = [spec["name"] for spec in attack_specs]
    matrix = np.array([[lookup[(label, attack)] for attack in attacks] for label in labels], dtype=np.float64)

    finite = matrix[np.isfinite(matrix)]
    max_dev = max(float(np.max(np.abs(finite - 1.0))) if finite.size else 0.0, 0.05)
    vmin, vmax = 1.0 - max_dev, 1.0 + max_dev
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    fig_width = max(8.0, 1.45 * len(attacks) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", norm=norm)
    ax.set_xticks(np.arange(len(attacks)), attacks, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Attack domain (Primary | Auxiliary | Unseen)")
    ax.set_ylabel("Method")
    ax.set_title("Cross-Attack Flatness Transfer (mean positive delta loss)")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text = "nan" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(col, row, text, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Relative flatness ratio R (<1 flatter; >1 sharper vs PGD-AT)")
    fig.savefig(output_dir / "cross_attack_flatness_transfer_heatmap.png", dpi=220)
    plt.close(fig)


def render_landscape_consistency_heatmap(
    output_dir: Path,
    consistency_rows: list[dict[str, Any]],
    labels: list[str],
    attack_specs: list[dict[str, Any]],
) -> None:
    target_attacks = [spec["name"] for spec in attack_specs if spec["role"] != "primary"]
    lookup = {(row["method"], row["target_attack"]): float(row["pearson"]) for row in consistency_rows}
    matrix = np.array([[lookup[(label, attack)] for attack in target_attacks] for label in labels], dtype=np.float64)

    fig_width = max(7.0, 1.45 * len(target_attacks) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(target_attacks)), target_attacks, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Auxiliary / unseen target attack")
    ax.set_ylabel("Method")
    ax.set_title("Primary-vs-Cross-Attack Landscape Consistency (Pearson)")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text = "nan" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(col, row, text, ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.savefig(output_dir / "cross_attack_landscape_consistency_heatmap.png", dpi=220)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate four matched-seed DeiT checkpoints on checkpoint-centered normalized "
            "local parameter planes, optionally assigning one checkpoint to each GPU, then "
            "aggregate absolute robust-loss basin quality and secondary diagnostics."
        )
    )
    parser.add_argument("--runs_dir", default="vit_12_runs_task/runs")
    parser.add_argument("--output_dir", default="vit_12_runs_task/landscape")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid_size", type=int, default=15)
    parser.add_argument("--radius", type=float, default=0.2)
    parser.add_argument("--direction_seed", type=int, default=20260818)
    parser.add_argument("--num_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--subset_seed", type=int, default=20250816)
    parser.add_argument("--primary_attack", default=DEFAULT_PRIMARY_ATTACK)
    parser.add_argument("--auxiliary_attack", default=DEFAULT_AUXILIARY_ATTACK)
    parser.add_argument("--unseen_attacks", default=DEFAULT_UNSEEN_ATTACKS)
    parser.add_argument("--plot_attacks", default=DEFAULT_PLOT_ATTACKS)
    parser.add_argument("--render_delta_contours", action="store_true")
    parser.add_argument("--render_flatness_transfer_heatmap", action="store_true")
    parser.add_argument("--render_consistency_heatmap", action="store_true")
    # Keep the original source attack seed CLI for backward compatibility.
    parser.add_argument("--primary_attack_seed", type=int, default=7301)
    parser.add_argument("--auxiliary_attack_seed", type=int, default=7302)
    parser.add_argument("--unseen_attack_seed_base", type=int, default=8300)
    parser.add_argument("--flatness_ratio_epsilon", type=float, default=1e-12)
    # Keep source attack strengths unchanged; these kwargs are also forwarded to
    # existing project attack-domain implementations selected by --unseen_attacks.
    parser.add_argument("--eps_linf", type=float, default=8 / 255)
    parser.add_argument("--alpha_linf", type=float, default=2 / 255)
    parser.add_argument("--eps_l2", type=float, default=0.5)
    parser.add_argument("--alpha_l2", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--gpu_ids",
        "--gpus",
        dest="gpu_ids",
        default="",
        help=(
            "Comma-separated logical CUDA device ids for checkpoint-parallel evaluation. "
            "Provide exactly one id per checkpoint, e.g. --gpus 0,1,2,3."
        ),
    )
    args = parser.parse_args()
    gpu_ids = parse_gpu_ids(args.gpu_ids)

    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    if args.grid_size < 3:
        raise SystemExit("--grid_size must be at least 3")
    if args.grid_size % 2 != 1:
        raise SystemExit("--grid_size must be odd so that (alpha,beta)=(0,0) is an actual grid point")
    if args.radius <= 0:
        raise SystemExit("--radius must be positive")
    if args.num_samples < 1:
        raise SystemExit("--num_samples must be positive")
    if args.flatness_ratio_epsilon <= 0:
        raise SystemExit("--flatness_ratio_epsilon must be positive")
    if gpu_ids and args.device != "auto":
        raise SystemExit("--device cannot be combined with --gpus/--gpu_ids")

    runs_dir = require_task_path(Path(args.runs_dir))
    output_dir = require_task_path(Path(args.output_dir))
    data_dir = require_project_path(Path(args.data_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty landscape directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_specs = checkpoint_specs_for_seed(args.seed)
    labels = [label for label, _ in checkpoint_specs]
    paths = [runs_dir / relative for _, relative in checkpoint_specs]
    if gpu_ids and len(gpu_ids) != len(labels):
        raise SystemExit(
            f"--gpus must provide exactly {len(labels)} ids so each checkpoint has one GPU; "
            f"got {len(gpu_ids)}"
        )
    if gpu_ids:
        if not torch.cuda.is_available():
            raise SystemExit("--gpus was specified, but CUDA is not available")
        visible_gpu_count = torch.cuda.device_count()
        unavailable = [gpu_id for gpu_id in gpu_ids if gpu_id >= visible_gpu_count]
        if unavailable:
            raise SystemExit(
                f"Requested unavailable logical GPU ids {unavailable}; "
                f"torch.cuda.device_count()={visible_gpu_count}"
            )
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"Missing seed-{args.seed} checkpoint: {path}")
        if not (path.parent / ".complete.json").is_file():
            raise SystemExit(f"Run is not validated complete: {path.parent}")

    payloads = [load_checkpoint(path) for path in paths]
    _, _, states = validate_checkpoints(paths, payloads)
    model_alias = str(payloads[0]["config"]["model_alias"])

    # Build the validation model on CPU. In multi-GPU mode the parent must not
    # own a CUDA model; each spawned worker creates its own CUDA context.
    model = build_model(model_alias, num_classes=10, pretrained=False)
    model.load_state_dict(states[0], strict=True)
    model.eval()

    parameter_names, parameter_shapes = get_trainable_parameter_spec(model)
    raw_direction_1, raw_direction_2 = make_shared_raw_directions(
        parameter_names, parameter_shapes, args.direction_seed
    )
    raw_direction_hashes = {
        "direction_1_sha256": arrays_sha256(raw_direction_1),
        "direction_2_sha256": arrays_sha256(raw_direction_2),
    }

    attack_specs, plot_attacks = resolve_attack_specs(args)
    if not plot_attacks:
        raise SystemExit("--plot_attacks must select at least one evaluated attack")
    attack_names = [spec["name"] for spec in attack_specs]
    attack_roles = [spec["role"] for spec in attack_specs]

    axis = np.linspace(-args.radius, args.radius, args.grid_size, dtype=np.float32)
    alpha_axis = axis.copy()
    beta_axis = axis.copy()
    center_idx = args.grid_size // 2
    if not (abs(float(alpha_axis[center_idx])) <= 1e-12 and abs(float(beta_axis[center_idx])) <= 1e-12):
        raise RuntimeError("Center grid point is not exactly (0,0) within tolerance")

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=not args.no_download,
        transform=transforms.ToTensor(),
    )
    if args.num_samples > len(dataset):
        raise SystemExit(f"Requested {args.num_samples} samples from a dataset of size {len(dataset)}")
    rng = np.random.default_rng(args.subset_seed)
    subset_indices = np.sort(rng.choice(len(dataset), size=args.num_samples, replace=False)).tolist()
    if gpu_ids:
        raw_loss, direction_metadata = evaluate_local_landscapes_multi_gpu(
            paths=paths,
            labels=labels,
            model_alias=model_alias,
            parameter_names=parameter_names,
            parameter_shapes=parameter_shapes,
            raw_direction_hashes=raw_direction_hashes,
            direction_seed=args.direction_seed,
            attack_specs=attack_specs,
            alpha_axis=alpha_axis,
            beta_axis=beta_axis,
            data_dir=data_dir,
            subset_indices=subset_indices,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            gpu_ids=gpu_ids,
        )
        method_to_device = {
            label: f"cuda:{gpu_id}" for label, gpu_id in zip(labels, gpu_ids)
        }
        device_description = ",".join(method_to_device.values())
        execution_mode = "checkpoint_parallel_multi_gpu"
    else:
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        model.to(device).eval()
        loader = DataLoader(
            Subset(dataset, subset_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        raw_loss, direction_metadata = evaluate_local_landscapes(
            model=model,
            states=states,
            labels=labels,
            parameter_names=parameter_names,
            raw_direction_1=raw_direction_1,
            raw_direction_2=raw_direction_2,
            attack_specs=attack_specs,
            alpha_axis=alpha_axis,
            beta_axis=beta_axis,
            loader=loader,
            device=device,
        )
        method_to_device = {label: str(device) for label in labels}
        device_description = str(device)
        execution_mode = "sequential_single_device"
    delta_loss = compute_delta_losses(raw_loss, center_idx)
    mask = circular_grid_mask(alpha_axis, beta_axis, args.radius)

    flatness_rows = compute_flatness_metrics(
        raw_loss=raw_loss,
        delta_loss=delta_loss,
        labels=labels,
        paths=paths,
        attack_specs=attack_specs,
        mask=mask,
        center_idx=center_idx,
        radius=args.radius,
        direction_seed=args.direction_seed,
        grid_size=args.grid_size,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    transfer_rows, transfer_summary_rows = compute_flatness_transfer(
        flatness_rows=flatness_rows,
        labels=labels,
        attack_specs=attack_specs,
        ratio_epsilon=args.flatness_ratio_epsilon,
        seed=args.seed,
    )
    robust_loss_region_rows = compute_robust_loss_region(
        flatness_rows=flatness_rows,
        labels=labels,
        attack_specs=attack_specs,
        seed=args.seed,
    )
    radius_profile_rows = compute_radius_profiles(
        raw_loss=raw_loss,
        delta_loss=delta_loss,
        labels=labels,
        attack_specs=attack_specs,
        alpha_axis=alpha_axis,
        beta_axis=beta_axis,
        radius=args.radius,
        seed=args.seed,
    )
    consistency_rows, pair_rows_by_method = compute_landscape_consistency(
        delta_loss=delta_loss,
        labels=labels,
        attack_specs=attack_specs,
        mask=mask,
        center_idx=center_idx,
        radius=args.radius,
        seed=args.seed,
        exclude_center=True,
    )
    validate_absolute_center_losses(
        raw_loss=raw_loss,
        flatness_rows=flatness_rows,
        labels=labels,
        attack_specs=attack_specs,
        center_idx=center_idx,
    )
    validate_radius_profiles(
        profile_rows=radius_profile_rows,
        flatness_rows=flatness_rows,
        labels=labels,
        attack_specs=attack_specs,
        radius=args.radius,
    )

    np.savez_compressed(
        output_dir / "loss_landscape_local.npz",
        labels=np.asarray(labels, dtype=str),
        attack_names=np.asarray(attack_names, dtype=str),
        attack_roles=np.asarray(attack_roles, dtype=str),
        alpha_axis=alpha_axis,
        beta_axis=beta_axis,
        raw_loss=raw_loss,
        delta_loss=delta_loss,
        seed=np.asarray(args.seed, dtype=np.int64),
    )

    write_csv(
        output_dir / "loss_landscape_flatness_metrics.csv",
        flatness_rows,
        [
            "seed",
            "label",
            "checkpoint",
            "attack",
            "attack_role",
            "center_loss",
            "radius",
            "mean_raw_loss_in_radius",
            "median_raw_loss_in_radius",
            "max_raw_loss_in_radius",
            "min_raw_loss_in_radius",
            "max_delta_loss",
            "mean_delta_loss",
            "mean_positive_delta_loss",
            "min_delta_loss",
            "direction_seed",
            "grid_size",
            "num_samples",
            "num_grid_points_in_radius",
        ],
    )
    write_csv(
        output_dir / "cross_attack_flatness_transfer.csv",
        transfer_rows,
        [
            "seed",
            "method",
            "attack",
            "attack_role",
            "flatness_metric",
            "flatness_value",
            "pgd_at_reference_value",
            "relative_flatness_ratio",
        ],
    )
    write_csv(
        output_dir / "cross_attack_flatness_transfer_summary.csv",
        transfer_summary_rows,
        [
            "seed",
            "method",
            "flatness_metric",
            "primary_ratio",
            "auxiliary_ratio",
            "mean_unseen_ratio",
            "median_unseen_ratio",
            "num_unseen_attacks",
        ],
    )
    write_csv(
        output_dir / "cross_attack_robust_loss_region.csv",
        robust_loss_region_rows,
        [
            "seed",
            "method",
            "attack",
            "attack_role",
            "center_loss",
            "pgd_at_center_loss",
            "center_loss_difference_vs_pgd_at",
            "center_loss_ratio_vs_pgd_at",
            "mean_raw_loss_in_radius",
            "pgd_at_mean_raw_loss_in_radius",
            "mean_region_loss_difference_vs_pgd_at",
            "mean_region_loss_ratio_vs_pgd_at",
            "radius",
            "grid_size",
            "num_samples",
        ],
    )
    write_csv(
        output_dir / "loss_landscape_radius_profiles.csv",
        radius_profile_rows,
        [
            "seed",
            "method",
            "attack",
            "attack_role",
            "radius",
            "num_grid_points",
            "mean_raw_loss",
            "median_raw_loss",
            "mean_delta_loss",
            "mean_positive_delta_loss",
            "max_delta_loss",
        ],
    )
    write_csv(
        output_dir / "cross_attack_landscape_consistency.csv",
        consistency_rows,
        [
            "seed",
            "method",
            "reference_attack",
            "target_attack",
            "target_role",
            "pearson",
            "spearman",
            "num_grid_points",
            "radius",
            "exclude_center",
        ],
    )
    for label, rows in pair_rows_by_method.items():
        write_csv(
            output_dir / f"attack_pair_landscape_consistency_{safe_filename_token(label)}.csv",
            rows,
            [
                "seed",
                "method",
                "attack_a",
                "attack_b",
                "pearson",
                "spearman",
                "num_grid_points",
                "radius",
                "exclude_center",
            ],
        )

    render_absolute_landscape_grid(
        output_dir=output_dir,
        raw_loss=raw_loss,
        labels=labels,
        attack_specs=attack_specs,
        plot_attacks=plot_attacks,
        alpha_axis=alpha_axis,
        beta_axis=beta_axis,
    )
    render_robust_loss_region_heatmap(
        output_dir, robust_loss_region_rows, labels, attack_specs
    )
    render_mean_absolute_loss_profiles(
        output_dir, radius_profile_rows, labels, attack_specs, plot_attacks
    )
    render_local_flatness_profiles(
        output_dir, radius_profile_rows, labels, attack_specs, plot_attacks
    )
    if args.render_delta_contours:
        render_delta_loss_contours(
            output_dir=output_dir,
            delta_loss=delta_loss,
            labels=labels,
            attack_specs=attack_specs,
            plot_attacks=plot_attacks,
            alpha_axis=alpha_axis,
            beta_axis=beta_axis,
            primary_attack=args.primary_attack,
            auxiliary_attack=args.auxiliary_attack,
        )
    if args.render_flatness_transfer_heatmap:
        render_flatness_transfer_heatmap(output_dir, transfer_rows, labels, attack_specs)
    if args.render_consistency_heatmap:
        render_landscape_consistency_heatmap(
            output_dir, consistency_rows, labels, attack_specs
        )

    try:
        import timm

        timm_version = timm.__version__
    except ImportError:
        timm_version = None

    metadata = {
        "analysis_scope_note": (
            "The reported landscapes and flatness metrics characterize normalized "
            "two-dimensional local parameter-space slices and should not be interpreted as "
            "complete high-dimensional sharpness measures. Lower absolute robust loss and "
            "lower local sharpness are treated as distinct properties. The analysis does not "
            "assume that improved unseen-attack robustness must correspond to a flatter "
            "parameter-space minimum."
        ),
        "seed": args.seed,
        "analysis_priority": {
            "primary": "absolute robust-loss geometry / basin quality",
            "secondary": "local parameter sensitivity / flatness",
            "diagnostic_only": "whole-grid cross-attack consistency",
        },
        "landscape_type": "local_parameter_space",
        "center_definition": "each checkpoint's own trainable parameter vector at alpha=beta=0",
        "direction_method": "shared Gaussian raw directions with checkpoint-relative normalization",
        "direction_seed": args.direction_seed,
        "direction_normalization": {
            "name": "filter-wise relative normalization",
            "ndim_ge_2": "row/output-unit norm of direction matched to row/output-unit norm of checkpoint parameter",
            "ndim_le_1": "whole-tensor direction norm matched to whole-tensor checkpoint parameter norm",
            "zero_norm_fallback": "set corresponding normalized direction tensor/filter to zero",
            "epsilon": DIRECTION_NORM_EPS,
        },
        "shared_raw_direction_realization_across_methods": True,
        "shared_parameter_plane_within_method": True,
        "re_attack_every_grid_point": True,
        "raw_direction_hashes": raw_direction_hashes,
        "method_direction_metadata": direction_metadata,
        "radius": args.radius,
        "alpha_range": [float(alpha_axis[0]), float(alpha_axis[-1])],
        "beta_range": [float(beta_axis[0]), float(beta_axis[-1])],
        "grid_size": args.grid_size,
        "center_grid_index": center_idx,
        "center_grid_value": [float(alpha_axis[center_idx]), float(beta_axis[center_idx])],
        "circular_flatness_mask_num_points": int(mask.sum()),
        "checkpoints": [
            {
                "label": label,
                "path": str(path),
                "file_sha256": file_sha256(path),
                "state_sha256": state_sha256(state),
                "initialization_hash": payload.get("config", {}).get("initial_trainable_parameters_hash"),
            }
            for label, path, state, payload in zip(labels, paths, states, payloads)
        ],
        "dataset": "CIFAR-10 test",
        "model_alias": model_alias,
        "dataset_subset_indices": subset_indices,
        "subset_seed": args.subset_seed,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "batch_order": "fixed; shuffle=False",
        "primary_attack": args.primary_attack,
        "auxiliary_attack": args.auxiliary_attack,
        "unseen_attacks": [spec["name"] for spec in attack_specs if spec["role"] == "unseen"],
        "plot_attacks": plot_attacks,
        "attack_roles": {spec["name"]: spec["role"] for spec in attack_specs},
        "attack_seeds": {spec["name"]: int(spec["seed"]) for spec in attack_specs},
        "attack_specs": attack_specs,
        "attack_rng_reset_at_every_method_grid_attack_evaluation": True,
        "execution": {
            "mode": execution_mode,
            "gpu_ids": gpu_ids,
            "method_to_device": method_to_device,
            "one_checkpoint_per_gpu": bool(gpu_ids),
            "aggregation": "parent process after all checkpoint evaluations complete",
        },
        "flatness": {
            "region": "alpha^2 + beta^2 <= radius^2",
            "status": "secondary diagnostic",
            "main_metric": "mean_positive_delta_loss",
            "additional_metrics": ["max_delta_loss", "mean_delta_loss", "min_delta_loss"],
            "negative_delta_loss_clipped_in_saved_landscape": False,
        },
        "robust_loss_region": {
            "status": "primary analysis",
            "reference_method": "PGD-AT",
            "main_metric": "mean_region_loss_ratio_vs_pgd_at",
            "region": "alpha^2 + beta^2 <= radius^2",
        },
        "radius_profiles": {
            "computed_from_saved_grid_without_attack_reruns": True,
            "radii": "sorted unique positive grid radii up to --radius",
            "minimum_points_shown_in_figures_when_available": 5,
        },
        "flatness_transfer": {
            "reference_method": "PGD-AT",
            "ratio_epsilon": args.flatness_ratio_epsilon,
            "main_summary_metric": "mean_positive_delta_loss",
        },
        "landscape_consistency": {
            "reference_attack": args.primary_attack,
            "metrics": ["pearson", "spearman"],
            "region": "same circular local grid used for flatness",
            "exclude_center": True,
            "legality": "same method, same alpha/beta grid, same checkpoint-local d1/d2 plane",
        },
        "sanity_checks": {
            "odd_grid_required": True,
            "center_exists": True,
            "center_parameter_identity_tolerance": PARAMETER_IDENTITY_TOL,
            "delta_center_tolerance": DELTA_CENTER_TOL,
            "same_grid_across_methods": True,
            "same_plane_across_attacks_within_method": True,
            "shared_raw_directions_across_methods": True,
            "fresh_attack_generation": True,
            "absolute_center_matches_flatness_metrics": True,
            "full_radius_profile_matches_flatness_metrics": True,
            "shared_absolute_color_scale_within_attack_row": True,
            "different_absolute_color_ranges_allowed_across_attacks": True,
        },
        "plotting": {
            "main_contour_quantity": "raw_loss",
            "main_contour_normalization": (
                "sequential normalization shared across four methods within each attack row"
            ),
            "main_contour_colormap": "viridis",
            "robust_loss_region_heatmap_metric": "mean loss within radius, ratio to PGD-AT",
            "delta_contours_default": False,
            "flatness_transfer_heatmap_default": False,
            "consistency_heatmap_default": False,
            "render_delta_contours": bool(args.render_delta_contours),
            "render_flatness_transfer_heatmap": bool(
                args.render_flatness_transfer_heatmap
            ),
            "render_consistency_heatmap": bool(args.render_consistency_heatmap),
            "raw_losses_preserved": True,
        },
        "library_versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "timm": timm_version,
            "numpy": np.__version__,
        },
        "device": device_description,
        "model_mode": "eval",
        "mixed_precision": False,
    }
    (output_dir / "loss_landscape_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote local loss-landscape artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
