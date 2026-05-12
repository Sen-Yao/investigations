# DualRefGAD Training Degradation Analysis

**Date**: 2026-05-08

## Context

The preceding investigation `2026-05-08-dualrefgad-elliptic-seed3-failure-analysis` began as a seed-specific diagnosis, but the evidence showed a broader method-level issue.

In the protocol-clean no-val Elliptic sweep:

- Sweep: https://wandb.ai/HCCS/DualRefGAD/sweeps/0d0py9y1
- Code: HCCS-88 `/data/linziyao/DualRefGAD`, commit `f954cd8`
- Protocol: `train_rate=0.05`, `val_rate=0.0`, `num_epoch=50`, seeds `[0,1,2,3,4]`
- Primary metric: `final_test_auc/final_test_ap`

All seeds start with a strong epoch-0 dual-reference ranking signal, but training degrades that signal.

| seed | epoch0 / peak AUC | final AUC | drop |
|---:|---:|---:|---:|
| 0 | 0.8048 | 0.7518 | 0.0531 |
| 1 | 0.7997 | 0.7601 | 0.0396 |
| 2 | 0.7831 | 0.7564 | 0.0266 |
| 3 | 0.7826 | 0.7131 | 0.0695 |
| 4 | 0.7798 | 0.7460 | 0.0338 |

Seed 3 is the most visible symptom, not the root cause.

## Core question

Why does training degrade the strong epoch-0 dual-reference ranking signal?

More specifically:

```text
fixed dual-reference score is already informative;
current dual_margin_two_score training makes it worse.
```

The goal is to determine whether the degradation comes from:

1. objective/metric mismatch,
2. score sign or semantic mismatch,
3. over-flexible head destroying reference ranking,
4. loss gradients pushing normal/anomaly margins in the wrong direction,
5. or a general need for calibration-only / residual training.

## Working hypothesis

The current head/objective is not learning a useful anomaly ranking from dual-reference evidence. Instead, it may be learning transformations that reduce the ranking quality already present in the fixed reference signal.

## Relationship to seed3 investigation

`2026-05-08-dualrefgad-elliptic-seed3-failure-analysis` provides symptom evidence:

- seed 3 is not explained by reference purity;
- seed 3 is not a gross split imbalance;
- seed 3 has the largest final-epoch degradation;
- all seeds degrade, so the root issue is method-level.

This investigation takes over the method-level question.

## Scope

In scope:

- loss-vs-AUC trajectory audit;
- score sign audit;
- epoch0-vs-final ranking correlation;
- score/margin distribution drift;
- calibration-only and residual-training probe design;
- objective-preserving alternatives.

Out of scope for first pass:

- claiming epoch0/best-test as formal metric;
- cherry-picking checkpoints;
- large hyperparameter sweep before mechanism is understood.

## Expected outputs

- `hypothesis.md`: degradation hypotheses and verification plan
- `experiments/scripts/`: analysis/probe scripts
- `experiments/outputs/`: raw and processed diagnostics
- `insights.md`: conclusions and next method design actions
