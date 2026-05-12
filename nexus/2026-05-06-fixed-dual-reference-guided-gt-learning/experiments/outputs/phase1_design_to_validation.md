# Phase 1: Design-to-validation bridge

## Goal

Start from the clarified setting:

```text
fixed globally precomputed dual-reference + GT anomaly judgment learning
```

The first validation should not train a full new model. It should answer whether the existing fixed dual-reference GT embedding already contains a normal-calibrated anomaly signal that can be recovered from 5% labeled-normal nodes.

## Fixed assumptions

- `R_n(v)` and `R_a(v)` are fixed before learning.
- They are evidence, not labels.
- GT learns the judgment function, not the reference retrieval function.
- 5% labeled-normal nodes calibrate normal judgment.
- No anomaly labels are used for training or prototype construction.

## Mini Experiment 1

Name:

```text
frozen_fixed_dual_reference_normal_prototype_audit
```

Input:

```text
GT embedding h(v) built from fixed target + R_n/R_a tokens
```

Training-free calibration:

```text
use 5% labeled-normal h(v) to build normal prototypes
```

Scores:

1. `single_center_l2`: distance to the mean of labeled-normal embeddings.
2. `multi_proto_kmeans_l2`: distance to nearest labeled-normal prototype.
3. `local_knn_normal_l2`: distance to k nearest labeled-normal embeddings.
4. `rn_ra_profile_delta`: reference-profile distances as probe only.

Expected interpretation:

- If single-center works poorly but multi-prototype improves, RHO-style heterogeneous normality is relevant.
- If all prototype scores fail, GT input/embedding may not encode fixed dual-reference evidence in a normal-calibratable way.
- If prototype scores are promising, next step is a trainable GT judgment head with normal calibration.

## Anti-collapse note

This mini audit is frozen/training-free, so all-low score collapse cannot occur. It is a diagnostic gate before trainable objectives.

## Bootstrap note

Because `R_n/R_a` are fixed and prototypes are built only from labeled-normal nodes, the audit avoids online self-confirming reference updates.
