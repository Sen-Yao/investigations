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

## 🔄 Alternative Paths (if Likelihood-Ratio fails)

### Option A: Margin + Learned Correction (Calibration Mode)

```python
s_i = m_i + β · f_θ(φ_i)
```

- Keep closed-form margin as **global ranking backbone**
- Learnable head only does **local calibration**
- Risk: may not improve AUC if margin is already optimal

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
| **Likelihood-Ratio + conditioning** | ✅ (normal-only) | ✅ (d as conditioning) | ? (to test) | **PRIMARY** |
| Margin + correction | ✅ (normal-only) | ✅ (margin backbone) | ⚠️ (calibration only) | **BACKUP** |
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
| 1 | Implement Likelihood-Ratio minimal probe (scalar `||d||` conditioning) |
| 2 | Compare AUC vs margin baseline (target: ≥ 0.79) |
| 3 | If succeeds → extend to vector conditioning |
| 4 | If fails → try margin + correction (calibration mode) |

---

## 📁 Evidence Files

| File | Content |
|------|---------|
| `experiments/outputs/relation_density_probe_summary_20260512.html` | Density failure diagnosis |
| `experiments/outputs/likelihood_ratio_d_conditioning_design_20260512.html` | Likelihood-Ratio design proposal |
| `experiments/outputs/dualref_normal_only_objectives_no_pseudo_anomaly_20260511.html` | Normal-only objectives comparison |
| `RESEARCH_SURVEY.md` | GGAD/VecGAD/RHO mechanism audit |

---

_Insights consolidated by Nexus, 2026-05-12._