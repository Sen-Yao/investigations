# Dual Margin Two-Score Head: Breakthrough Result

**Date**: 2026-05-07
**Experiment**: exp_20260507_164210_dual_margin_only
**WandB Sweep**: fifycsn6
**Protocol**: Elliptic, 5/95, no validation, num_epoch=50 fixed

---

## Key Result

| head_mode | n | final_test_auc | final_test_ap |
|---|---:|---:|---:|
| **dual_margin_two_score** | 5 | **0.7429±0.0164** | **0.1983±0.0210** |
| contrastive_two_score | 5 | 0.7148±0.0210 | 0.1722±0.0224 |

**Improvement**:
- AUC: +0.0281 (relative +3.9%)
- AP: +0.0261 (relative +15.1%)
- Stability: std reduced (0.0164 vs 0.0210 for AUC)

---

## Head Definition

**Lambda-free normalized margin two-score head**:



**No auxiliary loss, no lambda hyperparameter, no validation-based selection**.

---

## Why This Works

1. **Direction matters**: VecGAD already proved deviation direction is informative
2. **Normalized margin**: m_norm ∈ [-1, 1] matches sn/s_a logit scale without tuning
3. **Dual-reference geometry explicitly read**: target position relative to R_n → R_a axis
4. **Lambda-free**: no hyperparameter to tune, deployable under no-validation constraint

---

## Comparison with All Previous Heads

| head | final_test_auc | note |
|---|---:|---|
| **dual_margin_two_score** | **0.7429±0.0164** | direction-aware margin ✅ |
| decomposition_no_sublogit_to_final | 0.7167±0.0257 | best decomposition variant |
| contrastive_two_score | 0.7148±0.0210 | plain two-score |
| decomposition_head | 0.7093±0.0340 | baseline decomposition |
| structured_readout | 0.7089±0.0280 | structured baseline |
| scalar_mlp_baseline | 0.6599±0.0379 | too weak |

**dual_margin_two_score is currently the best head under 5/95 no-validation protocol**.

---

## Implications

- R_a - R_n direction is **not noise**, has real discriminative signal
- Dual-reference geometry can be explicitly read by head
- Margin term should be **kept**, not deleted
- Future work: stronger margin forms, cross-dataset validation

---

## Training Tuple Framework (Unchanged)

| tuple | label |
|---|---:|
| (v, R_n(v), R_a(v)) | 1 |
| (v, R_n(c), R_a(c)) | 0 |

Loss: 

---

_This validates the theory-aligned direction-aware scoring path._
