# Paper-to-code mapping

All paths and commands below are relative to the release root. Main comparisons use seeds `0,1,2`; the source-set and geometry diagnostics are explicitly single-seed analyses.

| Paper item | Experiment | Release entry point | Key configuration | Seeds |
|---|---|---|---|---|
| Table 1 | C10 / ResNet-18 / PGD-AT | `DATASET=cifar10 MODEL=resnet18 EXPERIMENT=pgd_at OUTPUT_ROOT=runs/c10_r18 bash scripts/train/run_cnn_main.sh` | `configs/train/cnn_main.json`; PGD-Linf-10 | 0,1,2 |
| Table 1 | C10 / ResNet-18 / PDA 0.1 | `DATASET=cifar10 MODEL=resnet18 EXPERIMENT=pda_0p1 OUTPUT_ROOT=runs/c10_r18 bash scripts/train/run_cnn_main.sh` | `L_primary + 0.1 L_auxiliary` | 0,1,2 |
| Table 1 | C10 / ResNet-18 / PDA 0.2 | `DATASET=cifar10 MODEL=resnet18 EXPERIMENT=pda_0p2 OUTPUT_ROOT=runs/c10_r18 bash scripts/train/run_cnn_main.sh` | `L_primary + 0.2 L_auxiliary` | 0,1,2 |
| Table 1 | C100 / ResNet-18 / PGD-AT | `DATASET=cifar100 MODEL=resnet18 EXPERIMENT=pgd_at OUTPUT_ROOT=runs/c100_r18 bash scripts/train/run_cnn_main.sh` | `configs/train/cnn_main.json`; PGD-Linf-10 | 0,1,2 |
| Table 1 | C100 / ResNet-18 / PDA 0.1 | `DATASET=cifar100 MODEL=resnet18 EXPERIMENT=pda_0p1 OUTPUT_ROOT=runs/c100_r18 bash scripts/train/run_cnn_main.sh` | `L_primary + 0.1 L_auxiliary` | 0,1,2 |
| Table 1 | C100 / ResNet-18 / PDA 0.2 | `DATASET=cifar100 MODEL=resnet18 EXPERIMENT=pda_0p2 OUTPUT_ROOT=runs/c100_r18 bash scripts/train/run_cnn_main.sh` | `L_primary + 0.2 L_auxiliary` | 0,1,2 |
| Table 1 | C10 / WRN-28-10 / PGD-AT | `DATASET=cifar10 MODEL=wrn_28_10 EXPERIMENT=pgd_at OUTPUT_ROOT=runs/c10_wrn bash scripts/train/run_cnn_main.sh` | PGD-Linf-10 | 0,1,2 |
| Table 1 | C10 / WRN-28-10 / PDA 0.1 | `DATASET=cifar10 MODEL=wrn_28_10 EXPERIMENT=pda_0p1 OUTPUT_ROOT=runs/c10_wrn bash scripts/train/run_cnn_main.sh` | `L_primary + 0.1 L_auxiliary` | 0,1,2 |
| Table 1 | C10 / WRN-28-10 / PDA 0.2 | `DATASET=cifar10 MODEL=wrn_28_10 EXPERIMENT=pda_0p2 OUTPUT_ROOT=runs/c10_wrn bash scripts/train/run_cnn_main.sh` | `L_primary + 0.2 L_auxiliary` | 0,1,2 |
| Table 1 | C10 / DeiT-Tiny / PGD-AT | `EXPERIMENT=pgd_at_linf10 bash scripts/train/run_deit_main.sh` | `vit_12_runs_task/configs/paper_recipe.json` | 0,1,2 |
| Table 1 | C10 / DeiT-Tiny / PDA 0.1 | `EXPERIMENT=pda_no_proj_w0p1 bash scripts/train/run_deit_main.sh` | final 40-epoch paper recipe | 0,1,2 |
| Table 1 | C10 / DeiT-Tiny / PDA 0.2 | `EXPERIMENT=pda_no_proj_w0p2 bash scripts/train/run_deit_main.sh` | final 40-epoch paper recipe | 0,1,2 |
| Main held-out evaluation | Eleven attacks | `CHECKPOINT=runs/c10_r18/pda_no_proj_w0p1_seed0/best.pt bash scripts/eval/evaluate_checkpoint.sh` | `configs/eval/main_attack_matrix.json` | checkpoint seed |
| Table 2 | Perturbation-radius generalization | `CHECKPOINT=runs/c10_r18/pda_no_proj_w0p1_seed0/best.pt bash scripts/eval/radius_sweep.sh` | `configs/eval/radius_sweep.json` | 0,1,2 checkpoints |
| Figure 1 / source diagnostic | Avg, PCGrad, MGDA on 4/5/2 sources | `bash scripts/diagnostics/run_deit_source_sets.sh` | `configs/diagnostics/source_sets.json` | 0, diagnostic |
| Figure 1 reference | PGD-AT and PDA 0.1 | `EXPERIMENT=pgd_at_linf10 SEEDS=0 bash scripts/train/run_deit_main.sh`; `EXPERIMENT=pda_no_proj_w0p1 SEEDS=0 bash scripts/train/run_deit_main.sh` | primary-only and asymmetric two-source objectives | 0, diagnostic |
| Figure 2 | ResNet-18 PDA and PDA-Proj lambda sweep | `bash scripts/sweeps/resnet_lambda_sweep.sh` | `0.05,0.10,0.15,0.20,0.30` | 0,1,2 |
| Figure 2 | DeiT-Tiny PDA and PDA-Proj lambda sweep | `bash scripts/sweeps/deit_lambda_sweep.sh` | `0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90` | 0,1,2 |
| Projection ablation | PDA-Proj | `pareto_attack_dg/optim/projected_aux.py` and `vit_12_runs_task/paper_recipe/supplemental_lambda_sweep.py` | auxiliary projection relative to primary gradient | as reported |
| Figure 3 | Absolute robust-loss geometry | `bash scripts/diagnostics/run_loss_landscape.sh` | `configs/diagnostics/loss_landscape.json` | 0 |
| Figure 3 supplement | Local sensitivity / radius profile | `vit_12_runs_task/analysis/loss_landscape_v3_multi.py` | derived from the saved normalized local grid | 0 |
| Supplement | CNN aggregation | `analysis/aggregate_results/aggregate_results.py`, `analysis/aggregate_results/summarize_cifar100_evals.py`, `analysis/aggregate_results/summarize_cifar10_wrn_evals.py` | experiment outputs to summaries | 0,1,2 |
| Supplement | DeiT main/lambda aggregation | `vit_12_runs_task/scripts/summarize_vit_69_runs.py` | main + projection + lambda families | 0,1,2 |
| Supplement | DeiT source-set aggregation | `vit_12_runs_task/scripts/summarize_deit_table_s4_seed0.py` | 11-row controlled diagnostic | 0 |

## Implementation anchors

- PDA source inputs are generated independently by `pareto_attack_dg/train.py` and by the soft-target DeiT attack functions in `vit_12_runs_task/paper_recipe/losses.py`.
- The unnormalized PDA sum is selected in `pareto_attack_dg/train.py` with `avg_attack_dg`, explicit weights `1:lambda`, and `--no-normalize-domain-weights`.
- PDA-Proj calls `primary_projected_aux_backward` from `pareto_attack_dg/optim/projected_aux.py`.
- PCGrad and MGDA use `pareto_attack_dg/optim/pcgrad.py` and `pareto_attack_dg/optim/mgda.py`; they are not aliases for PDA-Proj.
- PGD, CW-margin, FGSM, and AutoAttack evaluation are routed through `pareto_attack_dg/eval.py` and `pareto_attack_dg/attacks/`.

The sole unresolved dependency pin is documented in `REQUIRES_AUTHOR_CHECK.md`.
