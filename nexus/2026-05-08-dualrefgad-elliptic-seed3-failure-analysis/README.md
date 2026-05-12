# DualRefGAD Elliptic Seed 3 Failure Analysis

**Date**: 2026-05-08

## Context

DualRefGAD `dual_margin_two_score` completed a protocol-clean no-val 5-seed Elliptic sweep on HCCS-88:

- Sweep: https://wandb.ai/HCCS/DualRefGAD/sweeps/0d0py9y1
- Code: HCCS-88 `/data/linziyao/DualRefGAD`, commit `f954cd8`
- Protocol: `train_rate=0.05`, `val_rate=0.0`, `num_epoch=50`, seeds `[0,1,2,3,4]`
- Primary metrics: `final_test_auc`, `final_test_ap`

## Observed result

| seed | final AUC | final AP |
|---:|---:|---:|
| 0 | 0.7518 | 0.2174 |
| 1 | 0.7601 | 0.2276 |
| 2 | 0.7564 | 0.2028 |
| 3 | 0.7131 | 0.1663 |
| 4 | 0.7460 | 0.1918 |
| **mean±std** | **0.7455±0.0188** | **0.2011±0.0238** |

Seed 3 is the clear low outlier.

## Core question

Why does seed 3 underperform relative to the other four seeds?

The goal is not to cherry-pick or remove seed 3. The goal is to determine whether the low performance is caused by:

1. random split quality,
2. reference construction quality,
3. score/margin collapse,
4. final epoch degradation,
5. or a deeper weakness of the DualRefGAD head/objective.

## Scope

In scope:

- compare seed 3 against seeds 0/1/2/4;
- inspect train/test split composition and structural position;
- inspect normal/deviation reference purity;
- inspect score, margin, `sn`, `sa` distribution statistics;
- inspect epoch-wise degradation if logs/history are available.

Out of scope for the first pass:

- changing model architecture;
- excluding seed 3 from reported results;
- starting a new sweep before root-cause analysis.

## Outputs

- `hypothesis.md`: root-cause hypotheses and tests
- `experiments/scripts/`: analysis scripts
- `experiments/outputs/`: tables / JSON summaries
- `experiments/plots/`: visualizations
- `insights.md`: final findings and next actions
