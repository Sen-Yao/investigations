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

### Not run yet

No experiment has been run in this investigation at creation time. `first_step_probe.yaml` is a design placeholder, not evidence of completed results.
