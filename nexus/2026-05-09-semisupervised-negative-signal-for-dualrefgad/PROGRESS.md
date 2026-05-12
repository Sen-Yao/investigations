# Progress Tracking

**Investigation**: `2026-05-09-semisupervised-negative-signal-for-dualrefgad`
**Phase**: Phase 1 → Phase 2 (transitioning)
**Last Updated**: 2026-05-12

---

## Current Status

| Component | Status |
|-----------|--------|
| Literature Survey (RESEARCH_SURVEY.md) | ✅ Complete |
| Candidate Designs (hypothesis.md) | ✅ Complete |
| Density Probe Experiment | ✅ Complete (negative result) |
| Likelihood-Ratio Design | ✅ Proposed |
| Minimal Probe (Likelihood-Ratio) | 🔄 Pending |
| insights.md | ✅ Complete (2026-05-12) |

---

## Timeline

### Day 1: 2026-05-09

**Activity**: Investigation creation + Literature mechanism survey

- Created investigation structure
- Completed offline consistency diagnosis: proxy metrics vs real AUC/AP
- Started mechanism survey: GGAD / VecGAD / RHO
- Key finding: proxy AUC only shows compatibility between rule and score geometry, not validity for real anomaly detection

**Outputs**:
- `README.md` - Investigation scope and plan
- `hypothesis.md` - Candidate negative signal designs
- `RESEARCH_SURVEY.md` - GGAD/VecGAD/RHO mechanism comparison

---

### Day 2: 2026-05-10

**Activity**: Stage 1 results summary + planning

- Reviewed Stage 1 margin regression results
- Discussed next experimental directions
- Sent `stage1_results_and_next_plan_20260510.html`

**Key Discussion**:
- Margin-only baseline AUC 0.7938 remains strongest
- Need to design normal-only objectives that align with anomaly ranking

---

### Day 3: 2026-05-11 (Major Design Day)

**Activity**: Normal-only training objective design

#### Morning: Normal-only Objectives Discussion
- Sent `dualref_normal_only_objectives_no_pseudo_anomaly_20260511.html`
- Proposed three candidate approaches:
  1. **Conditional Normalizing Flow** (首推) - direct density modeling
  2. **Reference-view Consistency** (VICReg/Barlow style)
  3. **Energy/Denoising Score Matching** (理论优雅但难诊断)

**Key Judgment**: DualRefGAD learnable head should be defined as *normal relation distribution estimator*, not normal-vs-pseudo-anomaly classifier.

#### Afternoon: Objective Discussion (Margin → Ranking)
- Sent `objective_discussion_margin_to_ranking_20260511.html`
- Analyzed why margin regression ≠ anomaly ranking
- Proposed pairwise ranking with pseudo-anomaly candidates as alternative

#### Evening: RHO Analysis + MSE Probe Results
- Sent `rho_training_objective_analysis_20260511.html`
- Sent `stage1_margin_regression_mse_probe_normfix_report_20260511.html`
- Key finding: normalized interaction features (`ud_prod_absdiff_norm`) can recover margin ranking with Spearman≈0.99

---

### Day 4: 2026-05-12 (Failure Analysis + New Design)

**Activity**: Density Probe failure analysis + Likelihood-Ratio design

#### Morning: Relation Density Probe Summary
- Sent `relation_density_probe_summary_and_next_steps_20260512.html`
- **Critical Finding**: Normal-only density modeling **failed**
  - Diag Gaussian AUC: 0.6674 (far below margin 0.7938)
  - RealNVP worse than Gaussian
  - Flow approach does not help

**Root Cause Identified**:
> DualRef's anomaly direction `d=r_a-r_n` cannot be absorbed as a latent variable. It must enter scoring function via conditioning or explicit geometry.

#### Afternoon: Likelihood-Ratio with d-Conditioning Design
- Sent `likelihood_ratio_d_conditioning_design_20260512.html`
- Proposed new approach: `s_i = log p(φ_i | anom-context) - log p(φ_i | normal-context)`
- Uses `d` as conditioning variable, not pseudo-anomaly generation
- Recommended: `||d||` as scalar conditioning for first experiment

---

## Key Findings Summary

| Finding | Date | Implication |
|---------|------|-------------|
| Proxy AUC ≠ real anomaly ranking validity | 2026-05-09 | Cannot optimize proxy metrics directly |
| Normal-only density modeling fails | 2026-05-12 | `d` must be geometry-aware, not latent |
| Likelihood-Ratio with conditioning | 2026-05-12 | New direction: use `d` as hypothesis testing context |
| MLP can recover margin with normalized features | 2026-05-11 | Input design matters, not MLP capacity |

---

## Next Actions

| Priority | Action | Status |
|----------|--------|--------|
| 1 | Implement Likelihood-Ratio minimal probe (scalar `||d||` conditioning) | Pending |
| 2 | Compare Likelihood-Ratio AUC vs margin baseline | Pending |
| 3 | If Likelihood-Ratio succeeds → extend to vector conditioning | Planned |
| 4 | If fails → try margin + learned correction (calibration mode) | Backup |

---

## Open Questions

1. **Scalar vs vector conditioning**: Should first experiment use `||d||` or `normalize(d)`?
2. **Likelihood-Ratio calibration**: How to set `||d||_large` vs `||d||_small` thresholds?
3. **Flow model capacity**: RealNVP h256 l8 vs simpler models for first probe?

---

## File Inventory

| File | Type | Date Created |
|------|------|--------------|
| `README.md` | Investigation plan | 2026-05-09 |
| `hypothesis.md` | Candidate designs | 2026-05-09 |
| `RESEARCH_SURVEY.md` | Literature audit | 2026-05-09 |
| `experiments/outputs/*.html` | Email reports | 2026-05-10~12 |
| `insights.md` | Key conclusions | 2026-05-12 |
| `PROGRESS.md` | This file | 2026-05-12 |

---

_Investigation tracking started by Nexus, updated 2026-05-12._