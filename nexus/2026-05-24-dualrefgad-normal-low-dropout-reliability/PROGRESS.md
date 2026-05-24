# Progress

## 2026-05-24

### Created investigation

Created new investigation directory:

`~/investigations/nexus/2026-05-24-dualrefgad-normal-low-dropout-reliability/`

### Decision source

This investigation follows the corrected research decision for DualRefGAD early validation:

- Previous five-loss early design is considered dangerous / potentially catastrophic for early validation.
- First step must use at most two losses.
- Chosen losses must be the most fundamental, first-principles-aligned, and interpretable.

Selected first-step losses:

1. `L_normal-low`: known normal nodes should have low anomaly scores.
2. `L_ref-drop`: reference dropout consistency / reference subset stability.

Explicitly excluded from first-step training loss:

- reference ranking
- entropy regularization
- anti-hub regularization
- residual-guided hard negatives

These can be diagnostics, report metrics, or stop gates only.

### Next experiment

Prepare and run a first-step probe through `experiment-runner` only. The probe should test whether normal-low + reference-dropout consistency can train a shallow reliability gate without degenerating into a margin proxy.

Expected next artifacts:

- runner-compatible script under `experiments/scripts/` or project codebase;
- concrete experiment profile/config derived from `experiments/configs/first_step_probe.yaml`;
- output JSON/logs under `experiments/outputs/`;
- report metrics covering normal score suppression, reference dropout sensitivity, and gate-vs-margin correlation.

### First-step probe launched and completed via experiment-runner

Ran the first-stage DualRefGAD normal-low + reference-dropout probe on HCCS-25 through the runner-registered pure-probe pattern.

Preflight evidence:

- SSH target: `ssh -p 22 -i ~/.ssh/config.d/id_rsa linziyao@81.70.88.243`
- Hostname: `VM-2-25-ubuntu`
- Project: `~/DualRefGAD` exists.
- Dataset: `~/DualRefGAD/dataset/elliptic.mat` exists.
- Conda env: `/home/linziyao/.conda/envs/DualRefGAD`, Python `/home/linziyao/.conda/envs/DualRefGAD/bin/python`.
- Runtime: torch `2.0.0+cu117`, CUDA available, 8 GPUs visible.
- GPU state before launch: all 8 × RTX 2080 Ti showed 0 MiB used / 0% util; launched on GPU 0.

Runner / process / watchdog:

- Runner job id: `exp_20260524_111411_dualrefgad_first_step_normal_low_refdrop`
- Profile/kind: `probe` / `probe`
- Hermes background session: `proc_9b405af9012c`
- Watchdog cron id: `3ca9410aa84b`
- Watchdog script: `~/.hermes/scripts/dualrefgad_first_step_normal_low_refdrop_probe_watchdog.py`
- Watchdog was created and verified; after terminal completion it marked the runner job `finished` and pulled artifacts locally. It was then paused to avoid repeated no-op checks after terminal state.

Remote artifacts:

- Log: `/home/linziyao/DualRefGAD/investigations/2026-05-24-dualrefgad-normal-low-dropout-reliability/experiments/logs/first_step_normal_low_refdrop_probe.log`
- Output: `/home/linziyao/DualRefGAD/investigations/2026-05-24-dualrefgad-normal-low-dropout-reliability/experiments/outputs/first_step_normal_low_refdrop_probe.json`
- Progress: `/home/linziyao/DualRefGAD/investigations/2026-05-24-dualrefgad-normal-low-dropout-reliability/experiments/outputs/first_step_normal_low_refdrop_probe.progress.json`

Local pulled artifacts:

- `experiments/scripts/first_step_normal_low_refdrop_probe.py`
- `experiments/configs/first_step_normal_low_refdrop_probe.yaml`
- `experiments/logs/first_step_normal_low_refdrop_probe.log`
- `experiments/outputs/first_step_normal_low_refdrop_probe.json`
- `experiments/outputs/first_step_normal_low_refdrop_probe.progress.json`

Protocol guardrails:

- Training losses were exactly `L_normal-low` and `L_ref-drop`.
- `uses_anomaly_labels_in_loss=false`.
- `uses_pseudo_anomalies=false`.
- Validation/test labels were used only for report-only diagnostics.
- Excluded from training objective: reference ranking, entropy, anti-hub, residual-guided hard negatives.

Terminal status evidence:

- Background SSH process exited 0 and printed `REMOTE_PROBE_FINISHED finished [46564, 9, 93] ...`.
- Progress JSON: `status=finished`, `done=41`, `total=41`, elapsed about 34.2s.
- Runner status after watchdog: `status=finished`, `progress=41/41`.
- Output summary: token shape `[46564, 9, 93]`, labeled-normal train nodes `2086`.
- Best report-only diagnostic checkpoint: epoch 5, val AUC/AP `0.5840/0.0997`, test AUC/AP `0.5981/0.1103`.
- Final epoch report-only diagnostics: val AUC/AP `0.5358/0.0883`, test AUC/AP `0.5413/0.0959`; final `normal_score_mean=-9.1177`, `dropout_score_mse_normal=0.000174`.

Caveat: this is a short first-step probe/smoke-style experiment for pipeline and objective sanity, not method-quality evidence.
