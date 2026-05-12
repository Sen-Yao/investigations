# DualRefGAD Two-Sided Score Head

**Date**: 2026-05-06

## Context

This investigation follows `2026-05-06-fixed-dual-reference-guided-gt-learning`.

The previous investigation established that fixed dual-reference evidence is valid and learnable:

- `full` relation-consistency head reached `best_test_auc = 0.7384±0.0147` on Elliptic.
- Removing deviation-side reference (`no_ra`) dropped to `0.5513±0.0197`.
- Shuffling deviation-side reference (`shuffled_ra`) dropped to `0.5689±0.0309`.
- Replacing references with fixed labeled-normal anchors remained strong (`0.7219±0.0252`), indicating that normal-side calibration is powerful and the current diagnostic objective may not fully release `R_a`.

Therefore, the research center shifts from evidence validation to score-head design.

## Project / method naming

- Project codename remains: **VoxG**.
- Current method branch name: **DualRefGAD**.
- DualRefGAD is not yet an independent project; it remains under VoxG until the method matures.

## Core question

How should GT-interacted target, normal-side reference, and deviation-side reference tokens be read out and combined into a scalar anomaly score?

The central comparison is:

```text
VecGAD-style scalar head:
GT token exchange -> pooled embedding -> scalar MLP score
```

versus:

```text
DualRefGAD structured head:
GT token exchange -> structured readout (target / normal / deviation)
-> normal compatibility + deviation support + interaction
-> scalar anomaly score
```

## Theoretical semantics

`R_n(v)` answers:

```text
If v is normal, what normal-side references should it be compatible with?
```

`R_a(v)` answers:

```text
If v is not normal-like, does its deviation have a similar reference?
```

GT / Transformer should learn the relation between the target node and the two sides of evidence. We should not treat attention weights themselves as the explanation, and we should not treat downstream ablation as the theoretical core.

## Scope

In scope:

- structured score-head design;
- target / normal-side / deviation-side readout from GT tokens;
- normal compatibility and deviation support decomposition;
- interaction features between the two evidence sides;
- frozen-GT head ablations before GT training;
- more refined negative-scene construction after head semantics are defined.

Out of scope for the first phase:

- full GT fine-tuning;
- making `R_n/R_a` learnable or online-updated;
- treating `R_a` as pseudo anomaly labels;
- declaring context replacement as the final training objective.

## Working principle

Context replacement is a diagnostic probe that discovered a trainable signal. It is not considered the permanent DualRefGAD training strategy.

The next objective is to design a head whose score semantics match the two-sided evidence structure.
