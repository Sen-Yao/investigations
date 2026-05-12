# Phase 2: Trainable GT judgment with fixed dual-reference

## Status

Started after Mini Experiment 1 showed:

```text
fixed dual-reference evidence has signal, but frozen/random GT embedding is not a normal-calibrated anomaly judgment space.
```

Therefore the next step is not frozen prototype scoring. The next step is to train GT to interpret fixed dual-reference evidence.

## Core principle

```text
R_n(v), R_a(v) are fixed global precomputed evidence.
Only the GT judgment function is learned.
```

The method should not:

- learn/update dual-reference retrieval;
- treat `R_a` as anomaly labels;
- use real anomaly labels in training;
- rely on embedding-level pseudo anomaly displacement.

## Learning target

GT should learn:

```text
f_GT(v, R_n(v), R_a(v)) -> anomaly_degree(v)
```

with 5% labeled-normal nodes as normal calibration.

## Candidate mini objective: relation-consistency judgment

Use only labeled-normal nodes for supervised consistency construction.

For a labeled-normal node `v`:

```text
valid tuple:      (v, R_n(v), R_a(v))
corrupted tuple:  (v, R_n(u), R_a(u)) where u is another labeled-normal node
```

The binary task is:

```text
valid normal reference context vs mismatched normal reference context
```

Important distinction:

```text
This does not say R_a is anomaly.
It says the fixed dual-reference context has to be self-consistent for a normal target.
```

## Why this uses 5% labels

The only nodes used to construct valid/corrupted supervised pairs are labeled-normal nodes. Thus the loss calibrates what a normal target-reference relation should look like.

## Why this avoids bootstrap

- Reference retrieval is fixed before training.
- Corruption is constructed by permutation/mismatch, not by model prediction.
- The model does not update its own references.

## Why this is deeper than hand-crafted reference scoring

The score is produced by a trainable GT judgment function over target/reference tokens, not a fixed formula. The hand-crafted reference scores are only probes/baselines.

## Minimal metrics

During the mini run, log:

- `valid_corrupt_train_auc`: whether GT learns consistency on labeled normals;
- `test_auc`: anomaly AUC from the learned valid-context score on test nodes;
- `test_ap`;
- `normal_score_mean`;
- `test_score_mean`;
- `score_std`: detect all-low/all-constant collapse;
- `epoch_best_auc`, `epoch_best_ap`.

## Mini variants

Keep variants minimal:

| mode | description | hyperparameter burden |
|---|---|---|
| `head_only` | freeze/weakly use GT encoder, train judgment head | low |
| `train_gt` | train GT encoder + judgment head | medium |

Initial run should start with one seed and small epoch count.

## Immediate next implementation

Create a non-intrusive script:

```text
experiments/scripts/fixed_dual_reference_gt_consistency_train.py
```

It should reuse fixed reference construction from prior diagnostics and be launched via `experiment-runner mini`.
