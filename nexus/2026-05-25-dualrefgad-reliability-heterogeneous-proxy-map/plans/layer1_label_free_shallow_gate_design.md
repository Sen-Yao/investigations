# Layer 1 Label-free Shallow Gate Design

Status: design proposed after Layer 0 fixed-formula probe.
Date: 2026-05-26
Lineage:
- v2 report formalized `consensus_minus_fragmentation` and clarified that shallow logistic/isotonic gate must be label-free.
- Layer 0 fixed-formula gate probe found usable but weak proxy signal: best reliability-blend AUC +0.0038 vs `mat_mean`, AP -0.0095, Spearman vs `mat_mean` ≈ 0.876, decision `LAYER0_FIXED_GATE_HAS_USABLE_PROXY_SIGNAL_REVIEW_REQUIRED`.

## 1. Goal

Design a strictly label-free Layer 1 gate that learns a monotone reliability calibration over response-matrix-derived proxy features without using anomaly labels for fitting, thresholding, early stopping, or hyperparameter selection.

Layer 1 should answer:

> Can a small, constrained gate learn a better reliability map than the fixed Layer 0 formulas while controlling false-positive reintroduction?

It must not answer:

> Can label-tuned hyperparameters produce better AUC on Elliptic?

That is explicitly out of scope.

## 2. Inputs

For each node `v`, reuse the Layer 0 feature family:

- `mat_mean(v)`: mean over response matrix entries.
- `margin(v)`: original margin score.
- `cmf(v) = consensus_minus_fragmentation(v)`.
- `fragmentation(v)`: row/column range penalty.
- `joint_reliability(v)`: row/column effective-count reliability.
- Optional anti-shortcut features for audit only: degree, rejection, residual_norm.

Only unlabeled proxy features are allowed in the training objective.

## 3. Candidate models

### L1-A: monotone logistic gate

Small parametric gate:

```text
g_theta(v) = sigmoid(w_c * z(cmf(v)) + w_r * z(joint_reliability(v)) - w_f * z(fragmentation(v)) + b)
```

Constraints:

- `w_c >= 0`
- `w_r >= 0`
- `w_f >= 0`
- L2 regularization on all weights
- optional temperature cap to avoid hard threshold collapse

Score:

```text
S_theta(v) = g_theta(v) * z(mat_mean(v)) + (1 - g_theta(v)) * z(margin(v))
```

### L1-B: isotonic-like monotone calibrator

Fit a one-dimensional monotone map over `cmf - beta * fragmentation`, or two-stage binwise monotone table with very few bins.

Constraints:

- monotone non-decreasing in consensus signal
- monotone non-increasing in fragmentation penalty
- no anomaly labels
- bin count ≤ 5 unless later evidence justifies expansion

## 4. Label-free training signals

### 4.1 Pseudo-anchor construction

Define pseudo-positive reliability anchors from high-consensus / low-fragmentation nodes:

```text
P+ = top q% by cmf and bottom q_f% by fragmentation
```

Define pseudo-negative reliability anchors from low-consensus / high-fragmentation nodes:

```text
P- = bottom q% by cmf or top q_f% by fragmentation
```

These are reliability anchors, not anomaly labels. They mean “this node's response-matrix support looks coherent / incoherent”, not “this node is anomaly / normal”.

### 4.2 Pairwise ranking loss

Encourage reliability anchors to rank above unreliable anchors:

```text
L_rank = mean_{i in P+, j in P-} log(1 + exp(-(h(i)-h(j))))
```

where `h(v)` is the pre-sigmoid gate logit or calibrated reliability score.

### 4.3 Anchor calibration loss

```text
L_anchor = BCE(g(v), 1) for v in P+ + BCE(g(v), 0) for v in P-
```

This trains reliability confidence, not anomaly score.

### 4.4 Monotonicity regularization

Sample nearby pairs where `cmf_i >= cmf_j` and `fragmentation_i <= fragmentation_j`, then penalize violations:

```text
L_mono = mean relu(g(j) - g(i))
```

### 4.5 Stability / capacity penalty

```text
L = L_rank + alpha_anchor L_anchor + alpha_mono L_mono + lambda ||theta||_2^2
```

Keep parameter count tiny and fixed.

## 5. Validation protocol

Training is label-free. After fitting, labels may be used only for diagnostic reporting:

- AUC/AP mean ± std over seeds 0..4.
- ΔAUC/ΔAP vs `mat_mean` and vs best Layer 0 fixed gate.
- Spearman vs `mat_mean` and vs Layer 0 gate.
- Top-K autopsy:
  - recovered lost anomalies;
  - reintroduced removed false positives;
  - retained introduced false positives;
  - retained rescued anomalies.
- Anti-shortcut correlation:
  - abs Spearman with degree/rejection/residual_norm.

## 6. Continuation gate

Promote to next step only if all hold:

1. `AUC_mean >= mat_mean_AUC_mean + 0.003` OR `AP_mean >= mat_mean_AP_mean + 0.005`.
2. Does not worsen FP reintroduction relative to the best Layer 0 reliability gate by more than 5%.
3. Spearman vs `mat_mean` < 0.95, otherwise it is mostly a monotone rewrite.
4. abs shortcut correlation stays low and does not exceed Layer 0 by > 0.02.
5. Cross-seed standard deviation does not explode relative to `mat_mean`.

If it fails:

- If AP improves but FP reintro worsens: refine penalty / mask, not head capacity.
- If only AUC improves but AP drops: keep as ranking diagnostic, do not promote.
- If only works with label selection: reject as label leakage / deployability failure.

## 7. Recommended next implementation

Implement `cleg3_layer1_label_free_shallow_gate_probe.py` as a runner-registered pure probe reusing Layer 0 feature extraction, with sequential seeds and split fingerprints.

Start with L1-A only. Do not implement a generic MLP/head yet.

Default scan:

- pseudo quantiles: `q ∈ {0.05, 0.10}`
- fragmentation quantiles: `q_f ∈ {0.10, 0.20}`
- loss weights:
  - `alpha_anchor ∈ {0.5, 1.0}`
  - `alpha_mono ∈ {0.1, 0.5}`
  - `lambda_l2 ∈ {1e-3, 1e-2}`
- optimizer: LBFGS or Adam for ≤ 300 steps; no early stopping by labels.

Expected report: a new result HTML only after all seeds finish.
