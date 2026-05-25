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

### Ablation + long-epoch pure probe completed and reported

Ran the ablation/long-epoch follow-up probe to answer whether the first-step objective needed more than the original short horizon.

Runner / process / watchdog:

- Runner job id: `exp_20260525_102405_dualrefgad_ablation_long_epoch`
- Profile/kind: `probe` / `probe`
- Hermes background session: `proc_fd4d42ff4856`
- Temporary watchdog cron id: `3c4ce93aac4a`
- Watchdog script: `~/.hermes/scripts/dualrefgad_ablation_long_epoch_watchdog.py`
- The watchdog was verified during the run, then disabled/removed after terminal completion so it would not keep polling a finished probe.

Protocol:

- Dataset: `elliptic`
- Devices: GPU `0,1,2,3` on HCCS-25
- Seed: `0`
- Variants: `R0_baseline_untrained`, `R1_normal_low_only`, `R2_ref_drop_only`, `R3_normal_low_ref_drop`
- Epoch horizon: `120`, evaluated every `5` epochs
- Descriptor/reference setup: `descriptor_mode=hop_attr_rwse`, `pn_estimator=diag_gaussian`, `normal_k=8`, `pp_k=8`, `GT_num_layers=1`, `embedding_dim=128`, `GT_ffn_dim=128`
- Training labels boundary: validation/test anomaly labels used only for report-only diagnostics; no anomaly labels/pseudo anomalies in the training loss.

Terminal status evidence:

- Background SSH process exited with code `0` and printed `REMOTE_FINAL_OK 4 results 76 rows`.
- Runner status was manually reconciled to `finished`, `progress=4/4` after pulling remote artifacts.
- Local artifacts:
  - `experiments/scripts/ablation_long_epoch_probe.py`
  - `experiments/configs/ablation_long_epoch_probe.yaml`
  - `experiments/logs/ablation_long_epoch_probe.log`
  - `experiments/outputs/ablation_long_epoch_probe.json`
  - `experiments/outputs/ablation_long_epoch_probe.progress.json`

Key report-only diagnostics:

| variant | best epoch | val AUC | test AUC | val AP | test AP |
|---|---:|---:|---:|---:|---:|
| `R2_ref_drop_only` | 90 | 0.6980 | 0.6955 | 0.2378 | 0.2596 |
| `R3_normal_low_ref_drop` | 50 | 0.5951 | 0.5912 | 0.1022 | 0.1082 |
| `R1_normal_low_only` | 5 | 0.5837 | 0.5949 | 0.0993 | 0.1093 |
| `R0_baseline_untrained` | 0 | 0.4486 | 0.4383 | 0.0757 | 0.0800 |

Interpretation:

- Increasing epoch horizon was justified: R3 continued improving past the original short window and peaked around epoch 50.
- Continuing to 120 epochs is not automatically beneficial: R3 later collapsed back toward weak AUC while driving `normal_score_mean` extremely negative and `L_normal_low` to zero.
- R2 produced the highest report-only AUC/AP, but with severe score-scale drift (`normal_score_mean` eventually exploding positive), so it should be treated as a diagnostic clue rather than a ready method variant.
- Next low-cost step should diagnose whether R2's peak survives score bounding/normalization and multi-seed replication.

Published HTML report:

- URL: https://report.senyao.org/reports/2026/05/25/dualrefgad-ablation-long-epoch-result-2026-05-25.html
- Senyao Reports validation passed; Cloudflare Access protected URL verified with service token (HTTP 200).

### R2-bounded + self-stop probe completed and reported

Ran the approved `R2-bounded + self-stop` follow-up probe to test whether the previous `R2_ref_drop_only` peak survives bounded score scale and label-free checkpoint selection.

Runner / process:

- Runner job id: `exp_20260525_111348_dualrefgad_r2_bounded_self_stop`
- Profile/kind: `probe` / `probe`
- Hermes background session: `proc_da80178ac79d`
- Watchdog plan: cancelled as unnecessary because the Hermes-tracked process finished before periodic watchdog creation; terminal reconciliation was performed immediately.

Protocol:

- Dataset: `elliptic`
- Devices: GPU `0,1,2,3` on HCCS-25
- Seeds: `0,1,2,3,4`
- Variant: `R2_bounded_refdrop_selfstop`
- Training loss: `L_ref-drop` only
- Bounded score: `score = 5 * tanh(raw / 5)`
- Self-stop: label-free health over reference-dropout stability, saturation fraction, normal-score drift, and score-collapse penalty
- Validation/test labels: report-only diagnostics; not used by loss or self-stop

Terminal status evidence:

- Background SSH process exited with code `0` and printed `REMOTE_FINAL_OK 5 results 65 rows`.
- Runner status was reconciled to `finished`, `progress=5/5` after pulling remote artifacts.
- Local artifacts:
  - `experiments/scripts/r2_bounded_self_stop_probe.py`
  - `experiments/configs/r2_bounded_self_stop_probe.yaml`
  - `experiments/logs/r2_bounded_self_stop_probe.log`
  - `experiments/outputs/r2_bounded_self_stop_probe.json`
  - `experiments/outputs/r2_bounded_self_stop_probe.progress.json`

Key report-only diagnostics:

| selection | test AUC | test AP | epoch |
|---|---:|---:|---:|
| label-free self-stop | 0.5261 ± 0.1064 | 0.1010 ± 0.0280 | 20.0 ± 0.0 |
| best-val oracle upper bound | 0.6491 ± 0.0269 | 0.1302 ± 0.0132 | 21.0 ± 8.0 |

Interpretation:

- The deployable label-free self-stop version does not preserve the previous unbounded R2 peak; it collapses to weak ranking quality.
- The best-val oracle upper bound shows R2 still contains some local ordering signal, but this is not a valid method-selection rule.
- The result supports treating R2 as a diagnostic clue/auxiliary rather than a ready method variant.

Published HTML report:

- URL: https://report.senyao.org/reports/2026/05/25/dualrefgad-r2-bounded-self-stop-result-2026-05-25.html
- Senyao Reports validation passed; Cloudflare Access protected URL verified with service token (HTTP 200).

### A/C/D reference-loss ablation completed and reported

- Time: 2026-05-25 12:32
- Runner job id: `exp_20260525_115424_dualrefgad_acd_reference_loss_ablation`
- Watchdog cron id: `6198084d3254`
- Output: `experiments/outputs/acd_reference_loss_ablation_probe.json`
- Best variant: `A_normal_low`
- Best test AUC/AP: 0.5544 ± 0.0508 / 0.1007 ± 0.0118
- Published HTML report: https://report.senyao.org/reports/2026/05/25/dualrefgad-acd-reference-loss-ablation-result-2026-05-25.html

## 2026-05-25 Learning Signal Discovery pivot

### Decision source

Published discussion report: https://report.senyao.org/reports/2026/05/25/dualrefgad-learning-signal-discovery-discussion-2026-05-25.html

Current route decision: pause further loss stacking and treat the next DualRefGAD work as **Learning Signal Discovery**. The target is not to claim that the present trainable reference/reliability losses are sufficient, but to locate a reliable, stable learning signal that can approach the strong C-LEG3 response-matrix positive control without using anomaly labels in the final method.

### Evidence chain recorded

1. **C-LEG3 positive control / teacher, not a solved trainable method.** In the fixed old-exact C-LEG3 decomposition gate, untrained `mat_mean` achieved AUC `0.8168 ± 0.0054`, while centroid `margin` achieved `0.7928 ± 0.0054`. This establishes that the reference relation can contain strong signal, but it does **not** prove that the current learnable reference losses can recover it.
2. **A/C/D loss ablation is negative.** The A/C/D reference-loss ablation under normal-only supervision did not produce a competitive learnable signal. Best variant was `A_normal_low` with report-only best-val selected test AUC/AP `0.5544 ± 0.0508 / 0.1007 ± 0.0118`; adding C reference ranking or D entropy/anti-hub did not rescue the objective.
3. **Main bottleneck shifted from readout capacity to learning-signal reliability.** Earlier robust/readout or reliability directions should now be treated as diagnostics unless they identify why the teacher signal is recoverable. The scientific question is: what label-free or normal-only signal can make learning move toward C-LEG3-like ordering rather than toward score collapse, margin proxying, or unstable reference artifacts?
4. **Exploration labels are diagnostic-only.** During oracle autopsy / Learning Signal Discovery exploration, anomaly labels may be used only to diagnose where signal exists and why losses fail. They must not enter the final training loss, early stopping, checkpoint selection, or method claim.

### Operational boundary

No experiment was launched for this pivot update. This entry only updates investigation state and interpretation. Next experimental work must be separately approved and framed as diagnostic Learning Signal Discovery, not as immediate method validation.
