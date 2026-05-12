# Fixed Dual-Reference Guided GT Learning

**Date**: 2026-05-06

## Core question

How should GT learn to judge the anomaly degree of a target node using a **fixed, globally precomputed dual-reference structure**?

The desired object is not a learnable dual-reference module. Dual-reference retrieval is treated as fixed global information. The learning problem is instead:

```text
Given target node v and its fixed dual references R_n(v), R_a(v),
learn how to infer anomaly degree from this fixed relational evidence.
```

## Motivation

The previous investigation `2026-05-06-pseudo-anomaly-generation-quality` showed that converting dual-reference relations into pseudo anomaly samples is unreliable:

- local displacement does not move samples into the anomaly manifold;
- directly using `R_a` as positives does not define a clean positive class;
- `R_a/R_n` appears to be contextual evidence rather than anomaly labels.

This motivates a new direction:

```text
Do not learn or update dual-reference itself.
Do not treat R_a as pseudo anomaly labels.
Instead, learn a GT anomaly judgment rule conditioned on fixed dual-reference evidence.
```

## Scope

In scope:

- fixed dual-reference as global precomputed information;
- GT / Transformer learning over target + reference tokens;
- 5% labeled-normal as semi-supervised normal calibration;
- anomaly degree prediction based on fixed relation evidence;
- low-hyperparameter objectives where possible;
- comparison with GGAD / RHO / VecGAD label usage.

Out of scope:

- making dual-reference retrieval learnable;
- updating R_n/R_a online during training;
- using R_a as anomaly pseudo labels;
- embedding-level pseudo anomaly displacement as the main route;
- hand-crafted reference score as final method.

## Relation to previous investigations

- `2026-05-06-pseudo-anomaly-generation-quality`: failure mechanism diagnosis for pseudo anomaly construction.
- This investigation: method design after accepting fixed dual-reference as evidence and focusing on GT learning.
