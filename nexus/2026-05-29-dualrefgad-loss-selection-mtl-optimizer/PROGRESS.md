# PROGRESS — DualRefGAD Loss Selection + MTL Optimizer

## Current status

- 2026-05-29 19:32 CST: Investigation created from report `dualrefgad-loss-selection-mtl-optimizer-2026-05-29.html`.
- Active objective: prepare and run runner-registered A0→A1→A2→A3 probe, then publish an HTML result report extending the design report.

## Execution plan

1. Create investigation archive and record research question/hypotheses.
2. Inspect local investigations repo and DualRefGAD project git state.
3. Locate current RIFT-GT / ROCC-MC scripts and implement minimal A0→A3 bundled probe if missing.
4. Validate config with `experiment-runner`, sync minimal scripts/configs to selected HCCS host, remote `py_compile`.
5. Register one `profile=probe` job, launch via Hermes-tracked SSH session, create pure-probe watchdog cron.
6. Pull terminal artifacts, update insights, publish HTML report.

## Notes

- Experiments must be runner-registered; any direct remote execution is only the bridge pattern documented by `experiment-runner/references/runner-registered-probe.md`.
- Tests/AUC/AP are diagnostic-only. Candidate selection should be argued through no-leakage monitors first.

## Result update — 2026-05-29 20:00 CST

- Runner job `exp_20260529_194143_dualrefgad_loss_portfolio_a0_a3` finished successfully: 20/20 tasks completed, no aggregate errors.
- Watchdog cron `dualrefgad_loss_portfolio_a0_a3_watchdog` was removed after terminal verification.
- Pulled artifacts:
  - `experiments/outputs/loss_portfolio_a0_a3.json`
  - `experiments/outputs/loss_portfolio_a0_a3.progress.json`
  - `experiments/logs/loss_portfolio_a0_a3.log`
- Result report published: https://report.senyao.org/reports/2026/05/29/dualrefgad-loss-portfolio-a0-a3-result-2026-05-29.html
- Decision: A1/A2/A3 are not promoted. Keep A0 as diagnostic baseline; next gate is gradient audit + score decomposition before any PCGrad/FAMO comparison.

### Aggregate snapshot

| variant | RIFT AUC | RIFT AP | mat_mean AUC | RIFT - mat | Spearman | decision |
|---|---:|---:|---:|---:|---:|---|
| A0_rocc_mc | 0.7223 ± 0.0106 | 0.4106 ± 0.0515 | 0.8104 ± 0.0068 | -0.0881 ± 0.0120 | 0.1082 ± 0.0686 | keep as diagnostic baseline |
| A1_hinge_rank_barrier | 0.6554 ± 0.0400 | 0.1821 ± 0.0516 | 0.8104 ± 0.0068 | -0.1549 ± 0.0453 | -0.1667 ± 0.0926 | not promoted |
| A2_view_consistency | 0.6554 ± 0.0400 | 0.1821 ± 0.0516 | 0.8104 ± 0.0068 | -0.1549 ± 0.0453 | -0.1667 ± 0.0926 | not promoted |
| A3_pair_reliability | 0.6656 ± 0.0328 | 0.2243 ± 0.0769 | 0.8104 ± 0.0068 | -0.1448 ± 0.0378 | -0.2001 ± 0.0532 | not promoted |

