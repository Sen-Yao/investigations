# PROGRESS — DualRefGAD Normal-only Residual Probe

## Day 1: 2026-05-13 — Investigation created

**Activity:** Created investigation archive and framed the route as a diagnostic probe, not a final method.

- Location: `~/investigations/nexus/2026-05-13-dualrefgad-normal-only-residual-probe/`
- Related prior investigation: `2026-05-09-semisupervised-negative-signal-for-dualrefgad`
- Current status: waiting for user confirmation before HCCS-88 5-seed diagnostic run.

**Interpretation Rule:**

> If the residual probe does not produce stable improvement over margin-only, drop the route. If it does, inspect what the residual learns and redesign it as a unified scoring principle rather than keeping an additive patch.

## Pending

- [x] Run 5-seed diagnostic on HCCS-88 after confirmation.
- [ ] Record per-seed AUC/AP deltas, correction statistics, Spearman, top-k overlap, and selected epochs.
- [ ] Decide: close route, inspect unstable signal, or redesign mechanism.


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
