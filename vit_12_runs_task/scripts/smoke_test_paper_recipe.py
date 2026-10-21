from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pareto_attack_dg.models import build_model
from vit_12_runs_task.model.deit_cifar import _extract_state_dict
from vit_12_runs_task.paper_recipe.checkpoint import load_checkpoint
from vit_12_runs_task.paper_recipe.config import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    DEFAULT_PRETRAINED_SHA256,
    METHODS,
    OFFICIAL_COMMIT,
    PAPER_PATCH_EMBED_ADAPTATION,
    PAPER_POS_EMBED_ADAPTATION,
    lr_sequence,
    paper_recipe_hash,
)
from vit_12_runs_task.paper_recipe.losses import pgd_l2_soft_target, pgd_linf_soft_target, soft_target_cross_entropy
from vit_12_runs_task.paper_recipe.integrity import implementation_hash
from vit_12_runs_task.paper_recipe.data import (
    build_paper_dataloaders,
    set_paper_loader_epoch,
    shutdown_dataloader_workers,
)
from vit_12_runs_task.paper_recipe.model import (
    _official_block_sum_patch_embed,
    _official_resize_pos_embed,
    build_paper_deit_tiny_patch4_32,
    initial_trainable_parameters_hash,
    model_provenance,
)


def require_under_task(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(TASK_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path must stay under {TASK_ROOT}; got {resolved}") from exc
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def persistent_worker_resume_check() -> str:
    """Prove epoch-2 FakeData bytes survive an epoch-boundary worker restart."""

    def digest_epoch(loader: torch.utils.data.DataLoader, epoch: int) -> str:
        set_paper_loader_epoch(loader, epoch)
        digest = hashlib.sha256()
        for images, targets in loader:
            digest.update(images.contiguous().numpy().tobytes())
            digest.update(targets.contiguous().numpy().tobytes())
        return digest.hexdigest()

    generator_a = torch.Generator().manual_seed(20260817)
    loader_a, test_a, _ = build_paper_dataloaders(
        data_dir=TASK_ROOT / "fake_data_unused",
        batch_size=2,
        num_workers=2,
        generator=generator_a,
        use_fake_data=True,
        fake_train_size=8,
        fake_test_size=2,
        download=False,
    )
    try:
        digest_epoch(loader_a, 1)
        saved_generator_state = generator_a.get_state()
        uninterrupted = digest_epoch(loader_a, 2)
    finally:
        shutdown_dataloader_workers(loader_a, test_a)

    generator_b = torch.Generator().manual_seed(20260817)
    loader_b, test_b, _ = build_paper_dataloaders(
        data_dir=TASK_ROOT / "fake_data_unused",
        batch_size=2,
        num_workers=2,
        generator=generator_b,
        use_fake_data=True,
        fake_train_size=8,
        fake_test_size=2,
        download=False,
    )
    generator_b.set_state(saved_generator_state)
    try:
        resumed = digest_epoch(loader_b, 2)
    finally:
        shutdown_dataloader_workers(loader_b, test_b)
    if uninterrupted != resumed:
        raise AssertionError(
            f"Persistent-worker epoch-boundary resume mismatch: {uninterrupted} != {resumed}"
        )
    return uninterrupted


def run_logged(command: list[str], log: Path, env: dict[str, str], *, append: bool = False) -> None:
    with log.open("a" if append else "w", encoding="utf-8") as handle:
        handle.write("resolved_command: " + subprocess.list2cmdline(command) + "\n")
        handle.flush()
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def unit_checks(checkpoint: Path, output: Path, device: torch.device) -> None:
    expected = [0.1] * 35 + [0.01] * 3 + [0.001] * 2
    if lr_sequence() != expected:
        raise AssertionError(f"Official LR sequence mismatch: {lr_sequence()}")

    logits = torch.tensor([[1.2, -0.4, 0.3], [-0.2, 0.1, 0.9]], dtype=torch.float64)
    target = torch.tensor([[0.25, 0.50, 0.25], [0.0, 0.30, 0.70]], dtype=torch.float64)
    manual = (-(target * torch.log_softmax(logits, dim=1)).sum(dim=1)).mean()
    torch.testing.assert_close(soft_target_cross_entropy(logits, target), manual, rtol=0, atol=1e-12)

    torch.manual_seed(0)
    model = build_paper_deit_tiny_patch4_32(
        num_classes=10, pretrained=False, pretrained_checkpoint=checkpoint
    ).to(device)
    provenance = model_provenance(model)
    if provenance.get("normalization_mean") != list(CIFAR10_MEAN):
        raise AssertionError("Paper model does not use the official CIFAR-10 mean.")
    if provenance.get("normalization_std") != list(CIFAR10_STD):
        raise AssertionError("Paper model does not use the official CIFAR-10 std.")
    if provenance.get("pretrained_source", {}).get("sha256", "").lower() != DEFAULT_PRETRAINED_SHA256:
        raise AssertionError("Pretrained checkpoint hash was not recorded correctly.")
    adapter = provenance.get("pretrained_source", {}).get("adapter", {})
    if adapter.get("patch_embed") != PAPER_PATCH_EMBED_ADAPTATION:
        raise AssertionError("Paper model did not record the official block-sum patch adapter.")
    if adapter.get("position_embedding") != PAPER_POS_EMBED_ADAPTATION:
        raise AssertionError("Paper model did not record the official bilinear position adapter.")

    try:
        raw_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        raw_payload = torch.load(checkpoint, map_location="cpu")
    raw_state = _extract_state_dict(raw_payload)
    actual_patch = model.backbone.patch_embed.proj.weight.detach().cpu()
    expected_patch = _official_block_sum_patch_embed(
        raw_state["patch_embed.proj.weight"], actual_patch.shape
    )
    torch.testing.assert_close(actual_patch, expected_patch, rtol=0, atol=0)
    actual_pos = model.backbone.pos_embed.detach().cpu()
    expected_pos = _official_resize_pos_embed(
        raw_state["pos_embed"],
        actual_pos,
        num_tokens=int(getattr(model.backbone, "num_prefix_tokens", 1)),
        grid_size=tuple(int(value) for value in model.backbone.patch_embed.grid_size),
    )
    torch.testing.assert_close(actual_pos, expected_pos, rtol=0, atol=0)
    del raw_payload, raw_state, expected_patch, expected_pos

    x = torch.rand(2, 3, 32, 32, device=device, requires_grad=True)
    hard = torch.tensor([1, 4], device=device)
    soft = torch.nn.functional.one_hot(hard, 10).to(dtype=x.dtype)
    model.eval()
    output_logits = model(x)
    if tuple(output_logits.shape) != (2, 10):
        raise AssertionError(f"Wrong paper model output shape: {tuple(output_logits.shape)}")
    mean = torch.tensor(CIFAR10_MEAN, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    with torch.no_grad():
        direct = model.backbone((x.detach() - mean) / std)
    torch.testing.assert_close(output_logits.detach(), direct, rtol=0, atol=1e-6)
    soft_target_cross_entropy(output_logits, soft).backward()
    if x.grad is None or not torch.isfinite(x.grad).all():
        raise AssertionError("Input gradients do not pass through paper normalization.")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
            raise AssertionError(f"Missing/non-finite parameter gradient: {name}")

    model.zero_grad(set_to_none=True)
    x_clean = x.detach()
    attack_generator = torch.Generator(device=device).manual_seed(99)
    x_adv = pgd_linf_soft_target(model, x_clean, soft, steps=2, generator=attack_generator)
    if float((x_adv - x_clean).abs().max().cpu()) > 8 / 255 + 1e-6:
        raise AssertionError("Pixel-space Linf budget exceeded 8/255.")
    if float(x_adv.min().cpu()) < 0 or float(x_adv.max().cpu()) > 1:
        raise AssertionError("Pixel-space clamp escaped [0,1].")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("Inner PGD left parameter gradients behind.")
    l2_generator = torch.Generator(device=device).manual_seed(100)
    x_adv_l2 = pgd_l2_soft_target(model, x_clean, soft, steps=2, generator=l2_generator)
    l2_norms = (x_adv_l2 - x_clean).flatten(1).norm(p=2, dim=1)
    if float(l2_norms.max().cpu()) > 0.5 + 1e-5:
        raise AssertionError("Pixel-space L2 budget exceeded 0.5.")
    if float(x_adv_l2.min().cpu()) < 0 or float(x_adv_l2.max().cpu()) > 1:
        raise AssertionError("L2 attack clamp escaped [0,1].")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("Inner PGD-L2 left parameter gradients behind.")

    payload_path = output / "unit_checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": {"model_alias": "deit_tiny_patch4_32_paper"}}, payload_path)
    restored = build_model("deit_tiny_patch4_32_paper", num_classes=10, pretrained=False).to(device)
    try:
        payload = torch.load(payload_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(payload_path, map_location=device)
    restored.load_state_dict(payload["model_state_dict"], strict=True)
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(x_clean), model(x_clean), rtol=0, atol=1e-6)


def method_smoke(
    python: str,
    checkpoint: Path,
    output: Path,
    device: torch.device,
    env: dict[str, str],
    *,
    resume_incomplete: bool,
) -> None:
    configs: dict[str, dict[str, object]] = {}
    for experiment in METHODS:
        run_dir = output / experiment
        if run_dir.exists() and not resume_incomplete:
            raise AssertionError(f"Unexpected existing smoke method directory: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        common = [
            python,
            "-u",
            "-m",
            "vit_12_runs_task.paper_recipe.train",
            "--experiment",
            experiment,
            "--run-kind",
            "smoke",
            "--save-dir",
            str(run_dir),
            "--pretrained-checkpoint",
            str(checkpoint),
            "--data-dir",
            str(TASK_ROOT / "fake_data_unused"),
            "--batch-size",
            "2",
            "--seed",
            "0",
            "--num-workers",
            "0",
            "--device",
            str(device),
            "--no-download",
            "--use-fake-data",
            "--fake-train-size",
            "4",
            "--fake-test-size",
            "2",
        ]
        if not (run_dir / "training_complete.json").is_file():
            if experiment == "pgd_at_linf10":
                if not (run_dir / "last.pt").is_file():
                    first = [*common, "--epochs", "2", "--max-train-batches", "1", "--stop-after-epoch", "1"]
                    run_logged(first, run_dir / "console.log", env)
                    if (run_dir / "training_complete.json").exists():
                        raise AssertionError("Intentional resume stop incorrectly created a completion marker.")
                else:
                    prior_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
                    prior_checkpoint = load_checkpoint(run_dir / "last.pt", map_location="cpu")
                    if prior_config.get("paper_recipe_hash") != paper_recipe_hash() or prior_checkpoint.get("epoch") != 1:
                        raise AssertionError("Refusing to resume a smoke checkpoint not bound to current recipe epoch 1.")
                second = [
                    *common,
                    "--epochs",
                    "2",
                    "--max-train-batches",
                    "1",
                    "--resume",
                    str(run_dir / "last.pt"),
                ]
                run_logged(second, run_dir / "console.log", env, append=True)
            else:
                existing = [path for path in run_dir.iterdir()]
                if existing:
                    raise AssertionError(f"Non-resumable incomplete smoke output for {experiment}: {existing}")
                run_logged([*common, "--epochs", "1", "--max-train-batches", "2"], run_dir / "console.log", env)

        eval_command = [
            python,
            "-u",
            "eval.py",
            "--checkpoint",
            str(run_dir / "best.pt"),
            "--dataset",
            "cifar10",
            "--eval_attacks",
            "clean",
            "pgd_linf_2",
            "--batch_size",
            "2",
            "--max_eval_batches",
            "1",
            "--use_fake_data",
            "--fake_test_size",
            "2",
            "--num_workers",
            "0",
            "--device",
            str(device),
            "--save_json",
            str(run_dir / "smoke_eval.json"),
        ]
        if not (run_dir / "smoke_eval.json").is_file():
            run_logged(eval_command, run_dir / "eval_console.log", env)

        required = (
            "config.json", "train_log.csv", "batch_diagnostics.csv", "best.pt", "last.pt",
            "training_complete.json", "smoke_eval.json",
        )
        for name in required:
            path = run_dir / name
            if not path.is_file() or path.stat().st_size == 0:
                raise AssertionError(f"Missing smoke artifact: {path}")
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        configs[experiment] = config
        rows = read_rows(run_dir / "train_log.csv")
        expected_rows = 2 if experiment == "pgd_at_linf10" else 1
        if len(rows) != expected_rows:
            raise AssertionError(f"Unexpected epoch rows for {experiment}: {len(rows)}")
        if sum(int(row["train_batches_seen"]) for row in rows) > 2:
            raise AssertionError(f"FakeData batch limit exceeded for {experiment}")
        for row in rows:
            for value in row.values():
                if value.lower() in {"nan", "inf", "-inf"}:
                    raise AssertionError(f"Non-finite CSV value for {experiment}: {row}")
            if float(row["clip_post_global_norm_mean"]) > 1.00001:
                raise AssertionError(f"Final gradient was not clipped for {experiment}")
        checkpoint_payload = load_checkpoint(run_dir / "last.pt", map_location=device)
        if checkpoint_payload["epoch"] != expected_rows:
            raise AssertionError(f"Checkpoint epoch mismatch for {experiment}")

    init_hashes = {str(config["initial_trainable_parameters_hash"]) for config in configs.values()}
    if len(init_hashes) != 1:
        raise AssertionError(f"Matched-seed initializations differ: {sorted(init_hashes)}")
    for key in ("first_mixed_batch_sha256", "first_soft_target_sha256", "first_primary_adversarial_batch_sha256"):
        hashes = {str(config["per_epoch_reproducibility"]["1"][key]) for config in configs.values()}
        if len(hashes) != 1:
            raise AssertionError(f"Shared batch/primary semantics differ for {key}: {sorted(hashes)}")
    for experiment, expected in (("pda_no_proj_w0p1", 0.1), ("pda_no_proj_w0p2", 0.2)):
        weights = configs[experiment]["parsed_domain_weights"]
        if weights != {"pgd_linf_10": 1.0, "pgd_l2_5": expected} or configs[experiment]["normalize_domain_weights"]:
            raise AssertionError(f"No-projection PDA silently changed exact lambda for {experiment}")
    projected_rows = read_rows(output / "pda_proj_w0p2" / "batch_diagnostics.csv")
    if any(row["primary_aux_cosine"] == "" or row["primary_aux_projected"] == "" for row in projected_rows):
        raise AssertionError("Projected PDA did not record cosine/projection diagnostics.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded unit and four-method FakeData paper-recipe smoke tests.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--pretrained-checkpoint",
        default="vit_12_runs_task/pretrained/deit_tiny_patch16_224_fb_in1k.pt",
    )
    parser.add_argument("--output-dir", default="vit_12_runs_task/paper_recipe_smoke")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--resume-incomplete-smoke",
        action="store_true",
        help="Resume only a validated current-recipe epoch-1 smoke checkpoint; never use for formal runs.",
    )
    args = parser.parse_args()
    checkpoint = require_under_task(args.pretrained_checkpoint)
    output = require_under_task(args.output_dir)
    if not checkpoint.is_file():
        raise SystemExit(f"Missing pretrained checkpoint: {checkpoint}")
    if sha256(checkpoint).lower() != DEFAULT_PRETRAINED_SHA256:
        raise SystemExit(f"Unexpected default pretrained checkpoint SHA256: {sha256(checkpoint)}")
    if output.exists() and any(output.iterdir()) and not args.resume_incomplete_smoke:
        raise SystemExit(f"Refusing to overwrite non-empty smoke directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONPYCACHEPREFIX", str(TASK_ROOT / ".cache" / "pycache"))
    if not (args.resume_incomplete_smoke and (output / "unit_checkpoint.pt").is_file()):
        unit_checks(checkpoint, output, device)
    resume_epoch2_digest = persistent_worker_resume_check()
    method_smoke(
        args.python,
        checkpoint,
        output,
        device,
        env,
        resume_incomplete=args.resume_incomplete_smoke,
    )
    result = {
        "status": "PASSED",
        "paper_recipe_hash": paper_recipe_hash(),
        "implementation_hash": implementation_hash(),
        "official_commit": OFFICIAL_COMMIT,
        "pretrained_checkpoint": str(checkpoint),
        "pretrained_sha256": sha256(checkpoint),
        "methods": list(METHODS),
        "max_train_batches_per_method": 2,
        "persistent_worker_resume_epoch2_sha256": resume_epoch2_digest,
        "real_cifar10_used": False,
    }
    (output / "smoke_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
