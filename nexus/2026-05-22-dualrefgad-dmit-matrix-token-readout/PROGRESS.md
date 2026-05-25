# PROGRESS — D-MiT Matrix Token Readout

## 2026-05-22 — Investigation approved and created

Timestamp: 2026-05-22 19:19 CST

User approved:

1. D-MiT naming;
2. creation of `dualrefgad-dmit-matrix-token-readout` investigation;
3. execution of V0/V1/V2, starting with V0 on HCCS-25 and using GPU fully;
4. a sufficiently detailed report after completion.

### Execution boundary

This investigation will use `experiment-runner` / runner-registered probe pattern. HCCS-25 execution must be registered, tracked, and later written back to the runner job store. Direct manual `wandb agent` or unregistered remote experiments are not allowed.

### First executable step

Prepare and launch `v0_set_dmit_probe`:

- fixed C-LEG3 / old_exact reference regime;
- no position encoding;
- 64 response entries as set tokens;
- train only on known-normal nodes plus pseudo anomalies generated from known normals;
- true anomaly labels diagnostic-only;
- seeds `0,1,2,3,4`;
- HCCS-25 GPU parallel execution.

## 2026-05-22 — V0 Set-D-MiT launched

Timestamp: 2026-05-22 19:36 CST

### Local preparation

- Created `experiments/scripts/v0_set_dmit_probe.py`.
- Created `experiments/configs/v0_set_dmit_probe.yaml`.
- Local `python3 -m py_compile` passed.
- `experiment.py validate --profile probe` passed.
- Validator warning: reference/tokenization modes are implicit in the script through `old_exact_080_regime`; acceptable for this V0 gate, but final report must state the C-LEG3 mapping explicitly.

### HCCS-25 preflight

- Direct SSH succeeded.
- Remote project: `~/DualRefGAD`.
- Dataset: `dataset/elliptic.mat` present.
- Environment: conda env `DualRefGAD`, Python 3.8.19, torch 2.0.0+cu117, CUDA available, 8 devices.
- GPU state at launch: 8 × RTX 2080 Ti, all ~0 MiB used and 0% util.

### Runner registration and launch

- Runner job: `exp_20260522_193615_v0_set_dmit_probe`.
- Status at launch: `running`.
- Remote host label: `HCCS-25-direct`.
- Devices: `0,1,2,3,4,5,6,7`.
- Remote log: `/home/linziyao/DualRefGAD/logs/v0_set_dmit_probe.log`.
- Remote output: `/home/linziyao/DualRefGAD/experiments/outputs/v0_set_dmit_probe.json`.
- Remote progress: `/home/linziyao/DualRefGAD/experiments/outputs/v0_set_dmit_probe.progress.json`.
- Hermes background session: `proc_600a24ea5aec`.
- Monitor cron: `7833a0708256`, every 10m, max 48 runs.

Initial progress showed all five seeds started on GPUs 0–4. GPUs 5–7 remain available but the V0 task grid has only 5 seed tasks; the launch still exposes all 8 devices to the internal queue.

## 2026-05-22 — V0 Set-D-MiT completed

Timestamp: 2026-05-22 19:45 CST

### Execution summary

- Status: `finished`, 5/5 seeds completed.
- Elapsed: 275.86 seconds (~4.6 minutes).
- Errors: none.
- Output JSON: `experiments/outputs/v0_set_dmit_probe.json` (102KB).
- Progress JSON: `experiments/outputs/v0_set_dmit_probe.progress.json` (12KB).
- Log: `experiments/logs/v0_set_dmit_probe.log` (2KB).

### Key findings (variant: old_exact_080_regime)

| Metric | V0 Set-D-MiT | mat_mean | Delta |
|--------|--------------|----------|-------|
| AUC (mean ± std) | 0.5745 ± 0.146 | 0.8059 ± 0.010 | **-0.231** |
| AP (mean ± std) | 0.1350 ± 0.060 | 0.5449 ± 0.032 | **-0.410** |
| Spearman with mat_mean | 0.139 ± 0.365 | 1.0 ± 0 | — |

- Pseudo constraint satisfaction rate: 99.1% ± 0.6% → training objective satisfied.
- Pseudo vs source score gap: 7.46 ± 1.4 → strong score separation achieved.
- Top-5 Jaccard overlap (V0 vs mat_mean): 0.003 ± 0.004 → essentially no overlap.
- Best AUC votes: mat_mean (2 seeds), mean_top16_blend (3 seeds).

### Decision

**V0_SET_DMIT_UNDERPERFORMS_STRONG_SCALAR_BASELINE**

The pure Set-D-MiT diagnostic head trained on pseudo-normal pairs fails to recover anomaly signals. While the pseudo constraint is well satisfied (99.1% pair accuracy), the resulting score has near-zero correlation with the anomaly margin and virtually no top-k overlap with mat_mean.

This confirms that **set-based token readout without position encoding cannot capture meaningful anomaly structure**. The scalar mat_mean baseline (row-wise mean of reference matrix) is a robust ~0.80 AUC baseline that the D-MiT token approach must beat.

### Implications for V1/V2

- V0 hypothesis: "Set-D-MiT can diagnose anomalies by treating response entries as unordered tokens" → **REJECTED**.
- V1 must introduce position encoding (row/column/reference-indexed).
- V2 must test whether learned token embeddings + position encoding can beat mat_mean.
- The mat_mean scalar at ~0.806 AUC is the threshold; D-MiT must achieve ≥0.82+ to justify additional complexity.

### Next step

Await user decision to proceed with V1 (position-encoded D-MiT) or close investigation if V0 outcome already answers the core question.

## 2026-05-22 — V0 detailed report published

Timestamp: 2026-05-22 20:00 CST

- Monitor cron removed: `7833a0708256`.
- Runner state updated to `finished` for `exp_20260522_193615_v0_set_dmit_probe`.
- Detailed HTML report published and service-token verified:
  `https://report.senyao.org/reports/2026/05/22/dualrefgad-dmit-v0-set-dmit-report-2026-05-22.html`

### Final V0 decision

`V0_SET_DMIT_UNDERPERFORMS_STRONG_SCALAR_BASELINE`.

V0 satisfies the pseudo-normal pair training constraint but does not transfer to real anomaly ranking. Recommended next decision: only proceed to V1 if the goal is a controlled reference-identity diagnostic with permutation/order controls; do not promote V0 itself as a method.

