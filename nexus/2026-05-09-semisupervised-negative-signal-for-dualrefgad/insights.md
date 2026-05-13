# Insights: Semi-Supervised Negative Signal for DualRefGAD

**Investigation**: `2026-05-09-semisupervised-negative-signal-for-dualrefgad`
**Phase**: Phase 1 → Phase 2 (transitioning)
**Last Updated**: 2026-05-12

---

## 🎯 Core Research Question

> In semi-supervised GAD with only normal training nodes, how should DualRefGAD construct negative supervision signals that align with anomaly ranking?

---

## 🔴 Critical Finding: Normal-Only Density Modeling Fails

### The Experiment

We tested whether learning `p(φ)` (relation feature density) on labeled normal nodes could replace closed-form margin `m = cos(u, d)`.

**Results** (Elliptic, 2026-05-12):

| Method | AUC | AP | vs Margin (0.7938) |
|--------|-----|-----|-------------------|
| diag Gaussian | 0.6674 | 0.1453 | **-0.1264** |
| RealNVP h256 l8 | 0.5850 | 0.1520 | **-0.2088** |
| RealNVP h128 l8 | 0.5223 | 0.1370 | **-0.2715** |

### The Diagnosis

**Why density modeling failed:**

The closed-form margin `m = cos(u, d)` succeeds because it **explicitly uses anomaly direction `d`**:
```
m_i = cos(u_i, d_i)
where u_i = h_i - r_n(i)  (normal offset)
      d_i = r_a(i) - r_n(i)  (anomaly direction)
```

The direction `d` is constructed from anomaly-side reference, carrying information about "where anomalies should deviate".

When we treat `d` as a latent variable and learn unconditional density `p(φ)`, we **discard DualRef's key prior**: anomaly direction is not arbitrary — it's pre-directed by anomaly reference set.

Normal-only training cannot infer "what direction anomalies should take" because it has no anomaly-side information.

### The Corrected Formulation

> DualRef's anomaly direction `d` cannot be absorbed by latent-space density modeling; it must enter scoring via **conditioning or explicit geometry**.

The problem formulation should change from:

❌ "Learn normal density, use `-log p(φ)` as anomaly score"

to:

✅ "Learn normal density **conditioned on anomaly direction**, use likelihood ratio under two conditions as anomaly score"

---

## ✅ Recommended Direction: Likelihood-Ratio with d-Conditioning

### Core Idea

```
s_i = log p(φ_i | c_anom) - log p(φ_i | c_normal)
```

Where conditioning contexts are constructed from anomaly direction `d`:
- `c_normal`: φ should be consistent with normal-side context
- `c_anom`: φ should be consistent with anomaly-side context

**Key advantages**:
1. No pseudo-anomaly generation → protocol-clean
2. `d` enters scoring via conditioning → geometry-aware
3. Still trains only on normal nodes → semi-supervised compliant
4. Likelihood-ratio is classic hypothesis testing framework

### First Experiment Recommendation

Use **scalar conditioning** `||d||` for minimal probe:

```python
# Input features (same as density probe)
φ_i = concat(
    normalize(u_i), normalize(d_i),
    normalize(u_i) * normalize(d_i),
    |normalize(u_i) - normalize(d_i)|
)

# Conditioning
c_i = ||d_i||  # scalar

# Model
p(φ | c)  # conditional RealNVP

# Anomaly score
s_i = log p(φ_i | ||d||_large) - log p(φ_i | ||d||_small)
```

Where `||d||_large` and `||d||_small` can be:
- Fixed thresholds (e.g., 75th percentile vs 25th percentile of `||d||` on labeled normal nodes)
- Or quantile-based

### Why Scalar First

- Simplest implementation
- Avoids high-dimensional conditioning complexity
- `||d||` has physical interpretation: distance from normal reference to anomaly reference
- After validation, can extend to vector conditioning `normalize(d)`

---

## 🔴 New Finding: Scalar `||d||` Likelihood-Ratio Fails

### Formal Evaluation

After smoke testing the implementation, we ran a formal 5-seed / 80-epoch evaluation on HCCS-88. The result is decisively negative:

| Method | AUC | AP | Top-1% Ratio | Top-5% Ratio | Spearman vs Margin |
|--------|-----|----|--------------|--------------|--------------------|
| Scalar `||d||` Likelihood-Ratio | `0.4272±0.0473` | `0.0821±0.0047` | `0.0471±0.0240` | `0.0661±0.0193` | `0.1091±0.0259` |
| Margin baseline | `0.7938` | `0.5510` | `0.9955` | `0.7680` | `1.0000` |

**Insight:** This is not a capacity or convergence issue in the usual sense; after 80 epochs and 5 seeds, the conditional flow score remains far below even the failed unconditional density probe. Scalar `||d||` conditioning destroys the useful part of DualRef's anomaly direction, because the successful margin depends on alignment between `u` and `d`, not merely on the length of `d`.

### Updated Diagnosis

The previous conclusion remains partly correct: `d` must enter scoring explicitly. However, this experiment refines it:

> `d` must enter as directional geometry, not as scalar magnitude.

The failure suggests that likelihood-ratio scoring with `||d||` cannot distinguish between nodes whose anomaly direction has similar length but very different orientation relative to the node offset `u`. Margin succeeds exactly because it computes this directional compatibility.

### Consequence

Scalar conditioning should not be pursued further. If likelihood-ratio is revisited, it must use richer conditioning such as `normalize(d)` or anomaly-side rejection score. The more conservative next path is margin + learned correction, where the closed-form margin remains the ranking backbone and the learnable module only performs local calibration.

---

## 🟡 New Finding: Vector-Conditioned Likelihood-Ratio Partially Recovers Signal

### Formal Evaluation

To test whether scalar conditioning failed only because it discarded direction, we replaced `||d||` with `normalize(d)` as the conditioning vector and ran the same 5-seed / 80-epoch protocol.

| Method | AUC | AP | Top-1% Ratio | Top-5% Ratio | Spearman vs Margin |
|--------|-----|----|--------------|--------------|--------------------|
| Scalar `||d||` Likelihood-Ratio | `0.4272±0.0473` | `0.0821±0.0047` | `0.0471±0.0240` | `0.0661±0.0193` | `0.1091±0.0259` |
| Vector `normalize(d)` Likelihood-Ratio | `0.6880±0.0335` | `0.2839±0.0644` | `0.3489±0.3098` | `0.5051±0.1284` | `0.2974±0.0543` |
| Margin baseline | `0.7938` | `0.5510` | `0.9955` | `0.7680` | `1.0000` |

**Insight:** Vector conditioning substantially improves over scalar conditioning, raising AUC by about `+0.2608` and AP by about `+0.2018`. This confirms that the discarded directional information is real and useful. However, the method still remains far below the explicit margin baseline, and its Spearman correlation with margin is only `0.2974±0.0543`, so density-ratio scoring is still not preserving the ranking geometry that matters most.

### Interpretation

The vector experiment splits the diagnosis into two parts. First, scalar `||d||` failed because magnitude alone is insufficient; direction carries essential anomaly-reference information. Second, even when direction is supplied to the conditional flow, likelihood-ratio remains a weak replacement for direct geometric scoring. The likely reason is that the margin score is a simple explicit interaction `cos(u,d)`, whereas conditional density estimates how plausible `φ` is under a context. Plausibility under context is not the same as anomaly ranking.

### Consequence

Vector-conditioned likelihood-ratio is scientifically informative but not a promising main path. It should be treated as evidence for the importance of direction, not as a final scoring method. The next rational step is to keep margin as the backbone and learn only residual correction/calibration, or test a complementary protocol-clean consistency score.

---

## 🔄 Alternative Paths (after Scalar Likelihood-Ratio Failure)

### Option A: Residual Probe Around Margin (Diagnostic Only)

```python
s_i = m_i + β · f_θ(φ_i)
```

This additive form is **not** considered elegant enough to be the final method narrative. Its role is diagnostic: test whether the strong closed-form margin still leaves stable residual signal under a protocol-clean setup.

The first probe should therefore be intentionally constrained:

- Keep closed-form margin as frozen ranking backbone
- Train correction head using labeled normal nodes only
- Bound correction magnitude with `tanh` and small `corr_scale`
- Select epoch by normal-only validation loss, not anomaly labels
- If improvement is unstable or tiny, discard the route

If the residual probe works, the next step is not to keep the additive formula, but to inspect what the residual learns and rewrite it as a unified geometry-aware scoring principle.

### Option B: Reference-View Consistency

- Bootstrap normal reference set → two views `r_n^(a)` and `r_n^(b)`
- Train VICReg/Barlow-style consistency on labeled normal nodes
- Anomaly score = cross-view inconsistency

- Pros: No pseudo-anomaly, no conditioning design
- Cons: Not directly optimized for anomaly ranking

---

## 📊 Summary Table

| Approach | Protocol Clean | d Geometry | Ranking Alignment | Recommendation |
|----------|---------------|------------|-------------------|----------------|
| Pseudo-anomaly shell | ❌ (generates fake anomalies) | ✅ (uses d) | ⚠️ (ranking surrogate) | Not preferred |
| Pure density p(φ) | ✅ (normal-only) | ❌ (d absorbed) | ❌ (AUC 0.67 < 0.79) | **FAILED** |
| **Scalar `||d||` Likelihood-Ratio** | ✅ (normal-only) | ⚠️ (length only) | ❌ (AUC 0.427±0.047) | **FAILED** |
| Vector-conditioned Likelihood-Ratio | ✅ (normal-only) | ✅ (direction retained) | ⚠️ (AUC 0.688±0.034 < margin) | Diagnostic only |
| Residual probe around margin | ✅ (normal-only correction) | ✅ (margin backbone) | ? | Diagnostic only; not final narrative |
| Reference-view consistency | ✅ (normal-only) | ⚠️ (bootstrap views) | ? (to test) | **SECONDARY** |

---

## 🔑 Key Lessons

1. **Proxy metrics ≠ real anomaly ranking** — cannot optimize intermediate metrics directly
2. **Geometry-aware design matters** — DualRef's `d` is a pre-computed anomaly prior, not a learnable latent
3. **MLP capacity is not the bottleneck** — normalized interaction features can recover margin (Spearman ≈ 0.99)
4. **Normal-only ≠ "ignore anomaly direction"** — `d` must enter somehow, either via conditioning or explicit formula

---

## 📋 Next Actions

| Priority | Action |
|----------|--------|
| 1 | Run bounded normal-only residual probe around margin |
| 2 | If residual signal exists, inspect learned correction and redesign as unified geometry-aware score |
| 3 | Consider reference-view consistency as protocol-clean alternative |
| 4 | Avoid scalar `||d||` likelihood-ratio and unbounded additive correction in future experiments |

---

## 📁 Evidence Files

| File | Content |
|------|---------|
| `experiments/outputs/relation_density_probe_summary_20260512.html` | Density failure diagnosis |
| `experiments/outputs/likelihood_ratio_d_conditioning_design_20260512.html` | Likelihood-Ratio design proposal |
| `experiments/scripts/stage2_likelihood_ratio_probe.py` | Scalar `||d||` likelihood-ratio implementation |
| `outputs/stage2_probe/stage2_lr_probe_s*_e80.json` | Formal 5-seed / 80-epoch results on HCCS-88 |
| `experiments/scripts/stage2_vector_likelihood_ratio_probe.py` | Vector-conditioned likelihood-ratio implementation |
| `outputs/stage2_probe/stage2_vector_lr_probe_s*_e80.json` | Formal vector-conditioned 5-seed / 80-epoch results on HCCS-88 |
| `experiments/scripts/stage3_margin_residual_normalonly_probe.py` | Bounded normal-only residual probe around margin |
| `outputs/stage3_probe/stage3_margin_residual_normalonly_s*_e80.json` | Formal residual-probe 5-seed / 80-epoch results on HCCS-88; launched via manual SSH, so execution process did not comply with the research-explorer sweep-monitor rule |
| `experiments/outputs/dualref_normal_only_objectives_no_pseudo_anomaly_20260511.html` | Normal-only objectives comparison |
| `RESEARCH_SURVEY.md` | GGAD/VecGAD/RHO mechanism audit |

---

_Insights consolidated by Nexus, 2026-05-12._