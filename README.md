# Primary-Preserving Robust Generalization Across Attack Domains

This directory is the supplementary code release for **When Robust Objectives Are Not Peers: Primary-Preserving Robust Generalization Across Attack Domains**.

## Overview

The training problem is deliberately asymmetric. The primary source is PGD-Linf-10 at `epsilon=8/255`; the auxiliary source is PGD-L2-5 at `epsilon=0.5`. PDA optimizes

```text
L_PDA = L_primary + lambda * L_auxiliary
```

The main setting is `lambda=0.1`, and the main comparison also reports `lambda=0.2`. Each source adversarial input is generated separately. AutoAttack, CW, PGD-20/50, and unseen perturbation radii are evaluation domains and are not training sources.

PDA-Proj is a projection ablation: it preserves the PDA primary/auxiliary roles and applies the repository's auxiliary-gradient projection rule. It is not PCGrad.

## Repository structure

- `pareto_attack_dg/`: shared attacks, CIFAR loaders, ResNet-18, WideResNet-28-10, training methods, gradient aggregation, evaluation, and utilities.
- `vit_12_runs_task/`: the final DeiT-Tiny paper recipe, source-set diagnostic, lambda/projection supplements, result summarizers, and loss-landscape analysis.
- `configs/`: reviewer-facing manifests for main training, the eleven-attack matrix, radius sweeps, source sets, lambda sweeps, and geometry analysis.
- `scripts/train/`: parameterized Table 1 training entry points.
- `scripts/eval/`: the main held-out evaluation and perturbation-radius sweep.
- `scripts/sweeps/`: ResNet-18 and DeiT-Tiny PDA/PDA-Proj lambda sweeps.
- `scripts/diagnostics/`: the DeiT source-set diagnostic and final local robust-loss geometry analysis.
- `analysis/aggregate_results/`: compact result aggregation scripts; no raw run folders are bundled.
- `train.py` and `eval.py`: stable top-level entry points for the shared implementation.

See `PAPER_TO_CODE.md` for the paper-item mapping and `SOURCE_MANIFEST.md` for the server-source mapping.

## Environment

The release was assembled without executing Python locally. Dependencies were recovered from the source and repository metadata:

```bash
python -m pip install -r requirements.txt
```

The DeiT recipe pins `timm==1.0.15`. The repository does not contain a sufficiently precise AutoAttack version/commit pin; install the official implementation so that `from autoattack import AutoAttack` succeeds, then record the resolved source before final evaluation. See `REQUIRES_AUTHOR_CHECK.md`.

Exact PyTorch, torchvision, NumPy, matplotlib, and AutoAttack versions are intentionally not invented here. Validate the uploaded package in the intended server environment using `SERVER_VALIDATION.md`.

## Data

CIFAR-10 and CIFAR-100 use torchvision datasets and are not bundled. CNN training uses `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`, and `ToTensor`; testing uses `ToTensor`.

The final DeiT-Tiny recipe uses CIFAR-10, the same crop/flip policy, CIFAR-10 normalization inside the model, and the recorded Mixup/CutMix settings. Its complete recipe is in `vit_12_runs_task/configs/paper_recipe.json`.

Pass `DATA_DIR` to all wrappers. By default, wrappers allow torchvision to download a missing dataset into that directory. Set `NO_DOWNLOAD=1` explicitly when running offline or when the dataset must already exist.

## DeiT-Tiny initialization

The final recipe adapts the ImageNet-1K `deit_tiny_patch16_224.fb_in1k` initialization to 32x32 inputs with 4x4 patches.

Place that file at `vit_12_runs_task/pretrained/deit_tiny_patch16_224_fb_in1k.pt`, or set `PRETRAINED_CHECKPOINT` to another verified copy with `export PRETRAINED_CHECKPOINT="/path/deit_tiny_patch16_224_fb_in1k.pt"`

## Main training commands

Run commands from the release root. Every command below runs seeds `0 1 2`. Set `DEVICE=cuda:0` or another torch device as needed.

### CIFAR-10 / ResNet-18

```bash
DATASET=cifar10 MODEL=resnet18 EXPERIMENT=pgd_at SEEDS="0 1 2" DATA_DIR=data OUTPUT_ROOT=runs/c10_r18 bash scripts/train/run_cnn_main.sh
DATASET=cifar10 MODEL=resnet18 EXPERIMENT=pda_0p1 SEEDS="0 1 2" DATA_DIR=data OUTPUT_ROOT=runs/c10_r18 bash scripts/train/run_cnn_main.sh
DATASET=cifar10 MODEL=resnet18 EXPERIMENT=pda_0p2 SEEDS="0 1 2" DATA_DIR=data OUTPUT_ROOT=runs/c10_r18 bash scripts/train/run_cnn_main.sh
```

### CIFAR-100 / ResNet-18

```bash
DATASET=cifar100 MODEL=resnet18 EXPERIMENT=pgd_at SEEDS="0 1 2" DATA_DIR=data OUTPUT_ROOT=runs/c100_r18 bash scripts/train/run_cnn_main.sh
DATASET=cifar100 MODEL=resnet18 EXPERIMENT=pda_0p1 SEEDS="0 1 2" DATA_DIR=data OUTPUT_ROOT=runs/c100_r18 bash scripts/train/run_cnn_main.sh
DATASET=cifar100 MODEL=resnet18 EXPERIMENT=pda_0p2 SEEDS="0 1 2" DATA_DIR=data OUTPUT_ROOT=runs/c100_r18 bash scripts/train/run_cnn_main.sh
```

### CIFAR-10 / WideResNet-28-10

```bash
DATASET=cifar10 MODEL=wrn_28_10 EXPERIMENT=pgd_at SEEDS="0 1 2" DATA_DIR=data OUTPUT_ROOT=runs/c10_wrn bash scripts/train/run_cnn_main.sh
DATASET=cifar10 MODEL=wrn_28_10 EXPERIMENT=pda_0p1 SEEDS="0 1 2" DATA_DIR=data OUTPUT_ROOT=runs/c10_wrn bash scripts/train/run_cnn_main.sh
DATASET=cifar10 MODEL=wrn_28_10 EXPERIMENT=pda_0p2 SEEDS="0 1 2" DATA_DIR=data OUTPUT_ROOT=runs/c10_wrn bash scripts/train/run_cnn_main.sh
```

### CIFAR-10 / DeiT-Tiny

```bash
export PRETRAINED_CHECKPOINT="$PWD/vit_12_runs_task/pretrained/deit_tiny_patch16_224_fb_in1k.pt"
EXPERIMENT=pgd_at_linf10 SEEDS="0 1 2" DATA_DIR=data bash scripts/train/run_deit_main.sh
EXPERIMENT=pda_no_proj_w0p1 SEEDS="0 1 2" DATA_DIR=data bash scripts/train/run_deit_main.sh
EXPERIMENT=pda_no_proj_w0p2 SEEDS="0 1 2" DATA_DIR=data bash scripts/train/run_deit_main.sh
```

CNN runs use 100 epochs, batch size 128, SGD with learning rate 0.1, momentum 0.9, weight decay `5e-4`, and cosine scheduling. `best.pt` is selected by PGD-Linf-20 test accuracy, matching the server implementation.

DeiT runs use the isolated 40-epoch paper recipe: batch size 64, SGD 0.1/0.9, weight decay `1e-4`, final global gradient clipping at 1.0, the recorded step schedule, official initialization adaptation, and Mixup/CutMix. For these runs `best.pt` and `last.pt` intentionally identify the final epoch.

The guarded formal-suite launchers used on the server are retained under `vit_12_runs_task/scripts/`. The compact wrappers above expose the same scientific training code without requiring historical approval artifacts.

## Evaluation

The principal standardized primary metric is `AA-Linf`. `AA-L2` is the corresponding auxiliary cross-norm metric. The eleven-attack mean is a breadth summary; it does not replace the principal primary metric.

The full held-out matrix is:

1. FGSM-Linf
2. PGD-Linf-20
3. PGD-Linf-50
4. CW-Linf-20
5. CW-Linf-50
6. AutoAttack-Linf
7. PGD-L2-20
8. PGD-L2-50
9. CW-L2-20
10. CW-L2-50
11. AutoAttack-L2

Evaluate a CNN checkpoint with:

```bash
CHECKPOINT=runs/c10_r18/pda_no_proj_w0p1_seed0/best.pt DATASET=cifar10 MODEL=resnet18 DATA_DIR=data bash scripts/eval/evaluate_checkpoint.sh
```

For DeiT-Tiny, omit `MODEL`; the architecture is recovered from checkpoint metadata and the evaluator will not fetch upstream pretrained weights:

```bash
CHECKPOINT=vit_12_runs_task/paper_recipe_runs/pda_no_proj_w0p1_seed0/best.pt DATASET=cifar10 BATCH_SIZE=64 DATA_DIR=data bash scripts/eval/evaluate_checkpoint.sh
```

Run the held-out perturbation-budget sweep with:

```bash
CHECKPOINT=runs/c10_r18/pda_no_proj_w0p1_seed0/best.pt DATASET=cifar10 MODEL=resnet18 DATA_DIR=data bash scripts/eval/radius_sweep.sh
```

The Linf radii are `4/255, 8/255, 12/255, 16/255`; the L2 radii are `0.25, 0.50, 0.75, 1.00`. The source radii are `8/255` and `0.5`; all other radii are held out.

## Diagnostics and ablations

### PDA-Proj

For the DeiT-Tiny projection ablation at `lambda=0.2`:

```bash
EXPERIMENT=pda_proj_w0p2 SEEDS="0 1 2" PRETRAINED_CHECKPOINT="$PWD/vit_12_runs_task/pretrained/deit_tiny_patch16_224_fb_in1k.pt" bash scripts/train/run_deit_main.sh
```

For ResNet-18 PDA-Proj, use the lambda-sweep wrapper below with `MODE=pda_proj`.

### Lambda sweeps

ResNet-18 values are `0.05, 0.10, 0.15, 0.20, 0.30`:

```bash
MODE=both SEEDS="0 1 2" DATA_DIR=data bash scripts/sweeps/resnet_lambda_sweep.sh
```

DeiT-Tiny values are `0.05` through `0.30` as above, followed by `0.40, ..., 0.90`:

```bash
MODE=both SEEDS="0 1 2" PRETRAINED_CHECKPOINT="$PWD/vit_12_runs_task/pretrained/deit_tiny_patch16_224_fb_in1k.pt" DATA_DIR=data bash scripts/sweeps/deit_lambda_sweep.sh
```

Both wrappers support `MODE=pda`, `MODE=pda_proj`, or `MODE=both`. Set `EVALUATE=0` to train without immediately running the full eleven-attack matrix.

### Source-set and symmetric-aggregation diagnostic

The controlled single-seed diagnostic uses the four-, five-, and matched two-source sets recorded in `configs/diagnostics/source_sets.json`, with Avg, PCGrad, and MGDA. It is not a three-seed main comparison.

First ensure the seed-0 main PGD-AT and PDA `lambda=0.1` reference runs exist, then run:

```bash
PRETRAINED_CHECKPOINT="$PWD/vit_12_runs_task/pretrained/deit_tiny_patch16_224_fb_in1k.pt" DATA_DIR=data bash scripts/diagnostics/run_deit_source_sets.sh
```

The training logic is `vit_12_runs_task/paper_recipe/supplemental_table_s4.py`; PCGrad and MGDA use the included repository implementations.

### Robust-loss geometry and local sensitivity

The final Figure 3 implementation is `vit_12_runs_task/analysis/loss_landscape_v3_multi.py`. It compares matched seed-0 DeiT-Tiny checkpoints in checkpoint-centered normalized two-dimensional neighborhoods, shares the same raw random directions across methods, regenerates adversarial examples at every point, records absolute robust cross-entropy, compares neighborhood mean loss to PGD-AT, and emits radius profiles for local sensitivity.

```bash
RUNS_DIR=vit_12_runs_task/paper_recipe_runs DATA_DIR=data GPUS=0,1,2,3 bash scripts/diagnostics/run_loss_landscape.sh
```

The fixed final settings are grid size 15, radius 0.2, 256 samples, seed 0, direction seed `20260818`, and subset seed `20250816`. This is a diagnostic two-dimensional slice, not a complete high-dimensional curvature estimate.

## Result aggregation

Examples:

```bash
python analysis/aggregate_results/aggregate_results.py --runs_dir runs/c10_r18
python analysis/aggregate_results/summarize_cifar100_evals.py --runs_dir runs/c100_r18
python analysis/aggregate_results/summarize_cifar10_wrn_evals.py --runs_dir runs/c10_wrn
python vit_12_runs_task/scripts/summarize_vit_69_runs.py --runs_dir vit_12_runs_task/paper_recipe_runs --output_dir vit_12_runs_task/paper_recipe_summaries
python vit_12_runs_task/scripts/summarize_deit_table_s4_seed0.py --runs-dir vit_12_runs_task/paper_recipe_runs --output-dir vit_12_runs_task/paper_recipe_summaries
```

These scripts aggregate experiment outputs; they do not contain paper results or checkpoints.

