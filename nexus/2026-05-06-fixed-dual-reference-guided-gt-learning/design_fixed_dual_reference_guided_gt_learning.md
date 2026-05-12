# Design: Fixed Dual-Reference Guided GT Learning

## 1. User clarification

The target is **not** a learnable dual-reference module.

Dual-reference is global precomputed information:

```text
R_n(v), R_a(v) are fixed before GT training.
```

The actual learning target is:

```text
How should GT learn to judge anomaly degree using fixed dual-reference evidence?
```

This reframes the method from:

```text
learn better references
```

to:

```text
learn better anomaly judgment conditioned on references
```

## 2. Why this differs from previous pseudo anomaly route

Previous route:

```text
fixed R_n/R_a -> generate pseudo anomaly -> BCE normal vs pseudo
```

Observed issue:

```text
pseudo positives are not aligned with real anomaly manifold.
```

New route:

```text
fixed R_n/R_a -> GT consumes target/reference tokens -> learns anomaly judgment rule
```

No assumption that `R_a` is a positive label.

## 3. Semi-supervised label usage reference

### GGAD / VecGAD style

Use labeled normals to generate outlier/negative samples and train discriminatively.

Lesson for us:

- labeled-normal should be central;
- but pseudo outlier quality is critical;
- our current reference-displacement pseudo outliers are unreliable.

### RHO style

Use labeled normals to learn robust/diverse normality patterns, especially heterogeneous homophily.

Lesson for us:

- labeled-normal nodes should calibrate what normal evidence patterns look like;
- avoid assuming a single normal center;
- fixed dual-reference profiles may be multi-modal.

## 4. Proposed framing

Let fixed evidence for node `v` be:

```text
E(v) = [x_v, tokens_v, R_n(v), R_a(v), relation_features(v)]
```

GT learns:

```text
s_v = f_GT(E(v))
```

where `s_v` is anomaly degree.

Training should use:

```text
labeled normals: enforce low anomaly score / normal-consistent judgment
unlabeled nodes: prevent trivial all-low collapse through distributional or ranking constraints
fixed reference evidence: provide context for judgment
```

## 5. Candidate objectives

### A. Normal-calibrated one-class GT head

Use GT representation of target + fixed dual-reference tokens. Train labeled-normal nodes to have low score.

Risk: all-low score collapse. Need minimal anti-collapse mechanism.

### B. Multi-prototype normal judgment

Learn several normal judgment prototypes from labeled-normal GT embeddings.

Score:

```text
anomaly_score(v) = distance from nearest normal judgment prototype
```

Rationale:

- follows RHO's heterogeneous normality motivation;
- avoids single-center normality.

### C. Relation-level corruption as auxiliary task

Construct corrupted evidence tuples for labeled normal nodes:

```text
valid:   (v, R_n(v), R_a(v))
invalid: (v, R_n(u), R_a(u)) or swapped/mismatched references
```

GT learns to detect whether the reference context is normal-consistent.

This is not treating `R_a` as anomaly; it is learning consistency of fixed evidence.

### D. Prior-preserving normal calibration

Use fixed reference-derived weak prior only as regularizer/probe, not final rule.

The learned GT score should preserve useful ordering without becoming a hand-crafted score.

## 6. Current priority

Recommended first implementation candidate:

```text
Multi-prototype normal judgment over fixed dual-reference GT embeddings
```

Reason:

- uses 5% labeled-normal directly;
- keeps dual-reference fixed;
- avoids pseudo anomaly labels;
- captures multi-modal normality inspired by RHO;
- remains a learnable deep model rather than a hand-crafted trick.

## 7. Minimal validation plan

1. Freeze/precompute dual-reference retrieval.
2. Build GT input from target + fixed R_n/R_a tokens.
3. Train only anomaly judgment head or lightweight GT variant on labeled-normal calibration.
4. Compare single-center normal head, multi-prototype normal head, relation-level corruption auxiliary, and previous pseudo anomaly BCE baseline.
5. Evaluate AUC/AP and check whether training improves beyond epoch-0/reference prior without collapse.
