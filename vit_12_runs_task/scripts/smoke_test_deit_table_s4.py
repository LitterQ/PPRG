from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pareto_attack_dg.optim.mgda import mgda_backward
from pareto_attack_dg.optim.pcgrad import pcgrad_backward
from vit_12_runs_task.paper_recipe import supplemental_table_s4 as s4
from vit_12_runs_task.paper_recipe.checkpoint import (
    load_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
)
from vit_12_runs_task.paper_recipe.data import (
    build_paper_dataloaders,
    set_paper_loader_epoch,
    shutdown_dataloader_workers,
)
from vit_12_runs_task.paper_recipe.integrity import verify_lock
from vit_12_runs_task.paper_recipe.train import set_seed


SMOKE_EXPERIMENTS = (
    "s4_pgd_at_linf5",
    "s4_avg_five_source",
    "s4_pcgrad_five_source",
    "s4_mgda_five_source",
    "s4_pcgrad_matched_two_source",
    "s4_mgda_matched_two_source",
)


def require_under_task(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(TASK_ROOT)
    except ValueError as exc:
        raise ValueError(f"Smoke output must stay under {TASK_ROOT}: {resolved}") from exc
    return resolved


class TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.head = torch.nn.Linear(6, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


class StochasticSoftMixup:
    """Small RNG-consuming stand-in used only by the bounded FakeData smoke."""

    def __call__(self, x: torch.Tensor, hard_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target = torch.nn.functional.one_hot(hard_target, 10).to(dtype=x.dtype)
        if x.shape[0] < 2:
            return x, target
        torch_component = float(torch.rand((), device=x.device).cpu().item())
        numpy_component = float(np.random.random())
        lam = 0.55 + 0.1 * torch_component + 0.05 * numpy_component
        if random.random() < 0.5:
            permutation = torch.arange(x.shape[0] - 1, -1, -1, device=x.device)
        else:
            permutation = torch.roll(torch.arange(x.shape[0], device=x.device), shifts=1)
        return (
            lam * x + (1.0 - lam) * x[permutation],
            lam * target + (1.0 - lam) * target[permutation],
        )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def make_fake_loader(seed: int, *, size: int = 2) -> tuple[Any, Any, torch.Generator]:
    generator = torch.Generator().manual_seed(seed)
    train_loader, test_loader, _ = build_paper_dataloaders(
        data_dir=TASK_ROOT / "fake_data_unused",
        batch_size=2,
        num_workers=0,
        generator=generator,
        use_fake_data=True,
        fake_train_size=size,
        fake_test_size=2,
        download=False,
    )
    return train_loader, test_loader, generator


def spec_and_attack_checks(device: torch.device) -> dict[str, Any]:
    s4.validate_experiment_specs()
    expected_sources = {
        "s4_pgd_at_linf5": ("pgd_linf_5",),
        "s4_avg_four_source": s4.FOUR_SOURCES,
        "s4_avg_five_source": s4.FIVE_SOURCES,
        "s4_pcgrad_four_source": s4.FOUR_SOURCES,
        "s4_pcgrad_five_source": s4.FIVE_SOURCES,
        "s4_pcgrad_matched_two_source": s4.MATCHED_TWO_SOURCES,
        "s4_mgda_four_source": s4.FOUR_SOURCES,
        "s4_mgda_five_source": s4.FIVE_SOURCES,
        "s4_mgda_matched_two_source": s4.MATCHED_TWO_SOURCES,
    }
    for experiment, sources in expected_sources.items():
        if s4.experiment_sources(experiment) != tuple(sources):
            raise AssertionError(f"Source order mismatch for {experiment}")

    torch.manual_seed(100)
    model = TinyClassifier().to(device).train()
    x = torch.rand(2, 3, 32, 32, device=device)
    hard = torch.tensor([1, 7], device=device)
    target = torch.nn.functional.one_hot(hard, 10).to(dtype=x.dtype)
    recorded: list[tuple[str, dict[str, Any]]] = []

    def fake_linf(
        _: torch.nn.Module, images: torch.Tensor, __: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        recorded.append(("linf", kwargs))
        return images.detach()

    def fake_l2(
        _: torch.nn.Module, images: torch.Tensor, __: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        recorded.append(("l2", kwargs))
        return images.detach()

    with mock.patch.object(s4.base_train, "pgd_linf_soft_target", side_effect=fake_linf), mock.patch.object(
        s4.base_train, "pgd_l2_soft_target", side_effect=fake_l2
    ):
        s4.generate_source_losses(
            model=model,
            x_mixed=x,
            target=target,
            sources=s4.FIVE_SOURCES,
            seed=0,
            epoch=1,
            batch_index=0,
            device=device,
        )
    compact = [
        (
            norm,
            float(kwargs["eps"]),
            int(kwargs["steps"]),
            float(kwargs["alpha"]),
            bool(kwargs["random_start"]),
            kwargs["generator"] is not None,
        )
        for norm, kwargs in recorded
    ]
    expected_calls = [
        ("linf", 8 / 255, 1, 8 / 255, False, False),
        ("linf", 8 / 255, 5, 2 / 255, True, True),
        ("linf", 8 / 255, 10, 2 / 255, True, True),
        ("l2", 0.5, 5, 0.1, True, True),
    ]
    if compact != expected_calls:
        raise AssertionError(f"Attack call mismatch: {compact}")

    losses, source_inputs, _ = s4.generate_source_losses(
        model=model,
        x_mixed=x,
        target=target,
        sources=s4.FIVE_SOURCES,
        seed=0,
        epoch=1,
        batch_index=0,
        device=device,
    )
    if list(losses) != list(s4.FIVE_SOURCES) or not model.training:
        raise AssertionError("Source loss generation changed source order or failed to restore train mode.")
    for source in ("fgsm_linf", "pgd_linf_5", "pgd_linf_10"):
        delta = source_inputs[source] - x
        if float(delta.abs().max().cpu()) > 8 / 255 + 1e-6:
            raise AssertionError(f"{source} exceeded the 8/255 Linf budget")
    l2_norm = (source_inputs["pgd_l2_5"] - x).flatten(1).norm(p=2, dim=1)
    if float(l2_norm.max().cpu()) > 0.5 + 1e-5:
        raise AssertionError("pgd_l2_5 exceeded the L2=0.5 budget")
    for source, value in source_inputs.items():
        if float(value.min().cpu()) < 0.0 or float(value.max().cpu()) > 1.0:
            raise AssertionError(f"{source} escaped [0,1]")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("Inner attacks populated parameter .grad tensors.")
    return {"attack_calls": compact, "source_order": list(s4.FIVE_SOURCES)}


def _objective_losses(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    x = torch.tensor([[0.2, -0.4, 0.7], [0.8, 0.1, -0.5]], dtype=torch.float64)
    output = model(x).flatten()
    return {
        "one": (output - torch.tensor([0.1, 0.9], dtype=output.dtype)).square().mean(),
        "two": (output - torch.tensor([0.7, -0.2], dtype=output.dtype)).square().mean(),
        "three": (output - torch.tensor([-0.4, 0.4], dtype=output.dtype)).square().mean(),
    }


def aggregation_consistency_checks() -> dict[str, Any]:
    initial = torch.tensor([[0.3, -0.2, 0.5]], dtype=torch.float64)

    avg_model = torch.nn.Linear(3, 1, bias=False, dtype=torch.float64)
    avg_model.weight.data.copy_(initial)
    avg_losses = _objective_losses(avg_model)
    individual = torch.stack(
        [
            torch.autograd.grad(loss, avg_model.weight, retain_graph=True)[0]
            for loss in avg_losses.values()
        ]
    ).mean(dim=0)
    avg_model.zero_grad(set_to_none=True)
    s4.aggregate_source_gradients(model=avg_model, losses=avg_losses, aggregation="avg")
    torch.testing.assert_close(avg_model.weight.grad, individual, rtol=0, atol=1e-12)

    def fresh() -> torch.nn.Linear:
        model = torch.nn.Linear(3, 1, bias=False, dtype=torch.float64)
        model.weight.data.copy_(initial)
        return model

    pc_a = fresh()
    random.seed(12345)
    state = random.getstate()
    s4.aggregate_source_gradients(model=pc_a, losses=_objective_losses(pc_a), aggregation="pcgrad")
    grad_a = pc_a.weight.grad.detach().clone()
    pc_b = fresh()
    random.setstate(state)
    pcgrad_backward(list(_objective_losses(pc_b).values()), pc_b, objective_names=["one", "two", "three"])
    torch.testing.assert_close(grad_a, pc_b.weight.grad, rtol=0, atol=1e-12)

    mgda_a = fresh()
    _, info_a = s4.aggregate_source_gradients(
        model=mgda_a, losses=_objective_losses(mgda_a), aggregation="mgda"
    )
    mgda_b = fresh()
    info_b = mgda_backward(
        list(_objective_losses(mgda_b).values()),
        mgda_b,
        objective_names=["one", "two", "three"],
    )
    torch.testing.assert_close(mgda_a.weight.grad, mgda_b.weight.grad, rtol=0, atol=1e-12)
    if info_a.get("mgda_alpha") != info_b.get("mgda_alpha"):
        raise AssertionError("MGDA simplex weights differ from the shared optimizer implementation.")
    return {
        "avg_gradient": avg_model.weight.grad.detach().tolist(),
        "pcgrad_gradient": grad_a.tolist(),
        "mgda_alpha": info_a["mgda_alpha"],
    }


def method_smokes(output: Path, device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, experiment in enumerate(SMOKE_EXPERIMENTS):
        set_seed(900 + index)
        loader, test_loader, _ = make_fake_loader(900 + index)
        run_dir = output / experiment
        run_dir.mkdir(parents=True, exist_ok=False)
        try:
            set_paper_loader_epoch(loader, 1)
            model = TinyClassifier().to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            with mock.patch.object(
                s4, "clip_final_gradients", wraps=s4.clip_final_gradients
            ) as clip_mock:
                stats = s4.train_one_epoch(
                    model=model,
                    loader=loader,
                    optimizer=optimizer,
                    mixup=StochasticSoftMixup(),
                    experiment=experiment,
                    seed=0,
                    epoch=1,
                    device=device,
                    max_batches=0,
                    run_dir=run_dir,
                )
            if clip_mock.call_count != int(stats["train_batches_seen"]):
                raise AssertionError(f"Final gradient clipping was not called exactly once for {experiment}")
            rows = read_csv(run_dir / "batch_diagnostics.csv")
            if len(rows) != 1 or float(rows[0]["clip_post_global_norm"]) > 1.00001:
                raise AssertionError(f"Invalid clipping diagnostics for {experiment}")
            if rows[0]["sources"].split() != list(s4.experiment_sources(experiment)):
                raise AssertionError(f"Logged source order mismatch for {experiment}")
            result[experiment] = {
                "sources": list(s4.experiment_sources(experiment)),
                "aggregation": s4.METHOD_SPECS[experiment]["aggregation"],
                "train_loss": stats["train_loss"],
                "clip_calls": clip_mock.call_count,
            }
        finally:
            shutdown_dataloader_workers(loader, test_loader)
    return result


def resume_rng_check(output: Path, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(333)
    prototype = TinyClassifier().to(device)
    initial_state = copy.deepcopy(prototype.state_dict())
    model_a = TinyClassifier().to(device)
    model_b = TinyClassifier().to(device)
    model_a.load_state_dict(initial_state)
    model_b.load_state_dict(initial_state)
    optimizer_a = torch.optim.SGD(model_a.parameters(), lr=0.01, momentum=0.9)
    optimizer_b = torch.optim.SGD(model_b.parameters(), lr=0.01, momentum=0.9)
    scheduler_a = torch.optim.lr_scheduler.LambdaLR(optimizer_a, lr_lambda=lambda _: 1.0)
    scheduler_b = torch.optim.lr_scheduler.LambdaLR(optimizer_b, lr_lambda=lambda _: 1.0)

    set_seed(20260825)
    loader_a, test_a, generator_a = make_fake_loader(20260825)
    # A real resume reconstructs the loader with the same CLI seed before the
    # checkpoint restores its mutable generator state.  The dataset/sampler's
    # stateless base seed must therefore match as well.
    loader_b, test_b, generator_b = make_fake_loader(20260825)
    try:
        set_paper_loader_epoch(loader_a, 1)
        for _ in loader_a:
            pass

        run_a = output / "resume_uninterrupted_epoch2"
        run_b = output / "resume_restored_epoch2"
        run_a.mkdir(parents=True, exist_ok=False)
        run_b.mkdir(parents=True, exist_ok=False)
        boundary_checkpoint = output / "resume_boundary.pt"
        save_training_checkpoint(
            boundary_checkpoint,
            model=model_a,
            optimizer=optimizer_a,
            scheduler=scheduler_a,
            epoch=1,
            best_metric=1.0,
            config={"smoke": "table_s4_epoch_boundary"},
            dataloader_generator=generator_a,
        )
        payload = load_checkpoint(boundary_checkpoint, map_location=device)
        model_b.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer_b.load_state_dict(payload["optimizer_state_dict"])
        scheduler_b.load_state_dict(payload["scheduler_state_dict"])

        set_paper_loader_epoch(loader_a, 2)
        stats_a = s4.train_one_epoch(
            model=model_a,
            loader=loader_a,
            optimizer=optimizer_a,
            mixup=StochasticSoftMixup(),
            experiment="s4_pcgrad_five_source",
            seed=0,
            epoch=2,
            device=device,
            max_batches=0,
            run_dir=run_a,
        )
        python_after_a = random.getstate()
        numpy_after_a = np.random.get_state()
        torch_after_a = torch.get_rng_state().clone()
        loader_after_a = generator_a.get_state().clone()

        restore_rng_state(payload["rng_state"], generator_b)
        set_paper_loader_epoch(loader_b, 2)
        stats_b = s4.train_one_epoch(
            model=model_b,
            loader=loader_b,
            optimizer=optimizer_b,
            mixup=StochasticSoftMixup(),
            experiment="s4_pcgrad_five_source",
            seed=0,
            epoch=2,
            device=device,
            max_batches=0,
            run_dir=run_b,
        )
        if python_after_a != random.getstate():
            raise AssertionError("Python/PCGrad RNG diverged after resume.")
        numpy_after_b = np.random.get_state()
        if numpy_after_a[0] != numpy_after_b[0] or not np.array_equal(numpy_after_a[1], numpy_after_b[1]):
            raise AssertionError("NumPy/Mixup RNG diverged after resume.")
        if not torch.equal(torch_after_a, torch.get_rng_state()):
            raise AssertionError("Torch/Mixup RNG diverged after resume.")
        if not torch.equal(loader_after_a, generator_b.get_state()):
            raise AssertionError("DataLoader RNG diverged after resume.")
        for key, value in model_a.state_dict().items():
            torch.testing.assert_close(value, model_b.state_dict()[key], rtol=0, atol=0)
        if read_csv(run_a / "batch_diagnostics.csv") != read_csv(run_b / "batch_diagnostics.csv"):
            raise AssertionError("Mixup/attack/aggregation diagnostics diverged after resume.")
        if stats_a["first_source_input_hashes_json"] != stats_b["first_source_input_hashes_json"]:
            raise AssertionError("Attack branch random starts diverged after resume.")
        return {
            "experiment": "s4_pcgrad_five_source",
            "epoch": 2,
            "first_source_input_hashes_json": stats_a["first_source_input_hashes_json"],
            "model_state_identical": True,
        }
    finally:
        shutdown_dataloader_workers(loader_a, test_a)
        shutdown_dataloader_workers(loader_b, test_b)


def verify_old_locks_if_present() -> dict[str, str]:
    results: dict[str, str] = {}
    runs_root = TASK_ROOT / "paper_recipe_runs"
    for name in ("pgd_at_linf10_seed0", "pda_no_proj_w0p1_seed0"):
        run_dir = runs_root / name
        if not run_dir.exists():
            results[name] = "not_present_locally; formal launcher will require it"
            continue
        verify_lock(run_dir / "implementation_lock.json")
        results[name] = "verified"
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded DeiT Table S4 FakeData and optimizer smoke tests.")
    parser.add_argument("--output-dir", default="vit_12_runs_task/table_s4_smoke")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = require_under_task(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty smoke directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    spec_result = spec_and_attack_checks(device)
    aggregation_result = aggregation_consistency_checks()
    methods_result = method_smokes(output, device)
    resume_result = resume_rng_check(output, device)
    old_locks = verify_old_locks_if_present()
    lock_path = output / s4.TABLE_S4_LOCK_NAME
    s4.write_table_s4_lock(lock_path)
    s4.verify_table_s4_lock(lock_path)

    result = {
        "status": "PASSED",
        "real_cifar10_used": False,
        "formal_training_started": False,
        "device": str(device),
        "experiments": list(SMOKE_EXPERIMENTS),
        "spec_and_attack_checks": spec_result,
        "aggregation_consistency": aggregation_result,
        "method_smokes": methods_result,
        "resume_rng_check": resume_result,
        "old_base_locks": old_locks,
        "table_s4_implementation_hash": s4.table_s4_implementation_hash(),
    }
    (output / "smoke_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
