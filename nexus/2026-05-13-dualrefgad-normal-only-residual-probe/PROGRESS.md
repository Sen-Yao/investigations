# PROGRESS — DualRefGAD Normal-only Residual Probe

## Day 1: 2026-05-13 — Investigation created

**Activity:** Created investigation archive and framed the route as a diagnostic probe, not a final method.

- Location: `~/investigations/nexus/2026-05-13-dualrefgad-normal-only-residual-probe/`
- Related prior investigation: `2026-05-09-semisupervised-negative-signal-for-dualrefgad`
- Current status: **ROUTE CLOSED** — residual probe showed no stable improvement over margin-only.

**Interpretation Rule:**

> If the residual probe does not produce stable improvement over margin-only, drop the route. If it does, inspect what the residual learns and redesign it as a unified scoring principle rather than keeping an additive patch.

## Pending

- [x] Run 5-seed diagnostic on HCCS-88 after confirmation.
- [x] Record per-seed AUC/AP deltas, correction statistics, Spearman, top-k overlap, and selected epochs.
- [x] Decide: close route, inspect unstable signal, or redesign mechanism.


## Day 1: 2026-05-13 14:40 — HCCS-88 5-seed diagnostic completed

**Activity:** The Stage-3 residual ABCD diagnostics sweep completed. WandB aggregation shows **no stable improvement**.

- Sweep ID: `vtkl5ykv`
- WandB: <https://wandb.ai/HCCS/DualRefGAD/sweeps/vtkl5ykv>
- Status: `finished`
- Progress: `5/5`

**5-Seed Aggregation Results (Elliptic):**

| Metric | Mean ± Std |
|--------|------------|
| Margin AUC | 0.7952 ± 0.0071 |
| Margin AP  | 0.5165 ± 0.0220 |
| Score AUC  | 0.7953 ± 0.0071 |
| Score AP   | 0.5161 ± 0.0221 |
| Δ AUC vs Margin | 0.000063 ± 0.000272 |
| Δ AP vs Margin  | -0.000446 ± 0.002498 |
| Corr Mean | -0.1767 ± 0.0037 |
| Corr Std  | 0.1073 ± 0.0015 |
| Spearman(score, margin) | 0.9985 ± 0.0002 |
| Top-5 Jaccard | 1.0000 |

**Per-Seed Details:**

| Seed | Margin AUC | Score AUC | Δ AUC | Margin AP | Score AP | Δ AP |
|------|------------|-----------|-------|-----------|----------|------|
| 0 | 0.7938 | 0.7938 | +4.5e-05 | 0.5510 | 0.5512 | +1.4e-04 |
| 1 | 0.7960 | 0.7963 | +2.5e-04 | 0.5192 | 0.5164 | -2.8e-03 |
| 2 | 0.7991 | 0.7987 | -4.0e-04 | 0.5112 | 0.5144 | +3.2e-03 |
| 3 | 0.7840 | 0.7843 | +2.1e-04 | 0.4904 | 0.4905 | +5.7e-05 |
| 4 | 0.8030 | 0.8032 | +2.1e-04 | 0.5108 | 0.5079 | -2.8e-03 |

**ABCD Diagnostic Flags (all seeds):**

- `flag_A_global_shift`: True — correction has non-zero mean (~-0.17)
- `flag_B_margin_linked`: True — corr tightly linked to margin (Spearman ≈ -0.995)
- `flag_C_too_weak_to_change_rank`: True — |corr| / margin_std ≈ 0.23
- `flag_D_no_anomaly_separation`: False — corr differs between anom/normal (Cohen's d ≈ -0.52)

**Key Observations:**

1. Δ AUC essentially zero: The residual correction adds negligible improvement (~10^-5).
2. Δ AP mixed: AP deltas are inconsistent across seeds, sometimes negative.
3. Spearman(score, margin) ≈ 0.998: Score is almost perfectly monotonic with margin.
4. Top-5 Jaccard = 1.0: Residual never changes top-5 ranking.
5. Correction is globally shifted and too weak to flip rankings.

**Decision: Close the normal-only residual route.**

Per the interpretation rule: "If the residual probe does not produce stable improvement over margin-only, drop the route."

The residual term adds no orthogonal information to the margin score. It is:
- Too weak relative to margin scale
- Globally shifted (not anomaly-specific)
- Tightly linked to margin (no new information)

This route is closed. The focus should shift to redesigning a unified scoring principle rather than additive patches.


## Day 1: 2026-05-13 14:29 — HCCS-88 5-seed diagnostic launched

**Activity:** Started the formal Stage-3 residual ABCD diagnostics sweep through `experiment-runner`, not by manually launching WandB agents.

- Job ID: `exp_20260513_142903_dualrefgad_stage3_residual_abcd_diagnost`
- Sweep ID: `vtkl5ykv`
- WandB: <https://wandb.ai/HCCS/DualRefGAD/sweeps/vtkl5ykv>
- Profile: `sweep`
- Dataset: `elliptic`
- Seeds: `[0, 1, 2, 3, 4]`
- GPU allocation: HCCS-88 GPUs `0,1,2,3` selected by `--gpu-mode auto`
- Agents started: `4`
- Monitor cron: `b452c4264bff`, every 10 minutes, deliver to origin
- Config archive copy: `experiments/configs/stage3_residual_abcd_diagnostics_sweep.yaml`
- Script archive copy: `experiments/scripts/stage3_residual_abcd_diagnostics.py`

**Launch notes:**

Initial `launch-sweep` failed at `remote_preflight` with `MISSING:experiments/scripts/stage3_residual_abcd_diagnostics.py`. Manual SSH check showed the script did exist under `/data/linziyao/DualRefGAD`; the failure was caused by quoted `~` not expanding in the runner's default `REMOTE_CWD=~/DualRefGAD`. The experiment-runner skill was patched with this gotcha, and the sweep was relaunched with absolute remote paths:

```bash
EXPERIMENT_RUNNER_REMOTE_CWD=/data/linziyao/DualRefGAD
EXPERIMENT_RUNNER_REMOTE_TMP=/data/linziyao/DualRefGAD/tmp
EXPERIMENT_RUNNER_REMOTE_LOG_DIR=/data/linziyao/DualRefGAD/logs
```

**Current monitor observation:**

- Status: `running`
- Progress: `0/5`
- Run count: `4`
- Running seeds observed: `0,1,2,3`

**Next:** Wait for periodic monitor or manually aggregate after all 5 runs finish. Record per-seed `margin_auc`, `score_auc`, `delta_auc_vs_margin`, `delta_ap_vs_margin`, `corr_mean/std/abs_mean`, `spearman_score_margin`, top-k overlap, and ABCD diagnosis summary.
