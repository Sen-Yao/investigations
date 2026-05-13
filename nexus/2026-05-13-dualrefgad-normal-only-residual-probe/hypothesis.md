# Hypotheses — DualRefGAD Normal-only Residual Probe

## H1: Frozen margin leaves learnable residual signal

The frozen DualRefGAD margin may not exhaust all anomaly-relevant information. A small bounded residual probe trained only on labeled normal nodes can test whether there is remaining signal without turning the experiment into supervised anomaly tuning.

## H2: Calibration is not ranking discovery

If the learned correction has a negative mean but low variance, high Spearman correlation with baseline, and little top-k membership change, then the probe trained but mostly learned normal-score suppression or global calibration. This should not be interpreted as a new anomaly mechanism.

## H3: Stable residual ranking implies mechanism redesign, not additive patching

If the probe improves AUC/AP stably across seeds and changes ranking geometry, the right next step is to inspect what the residual captures and redesign the method as a unified normal-manifold or reference-inconsistency score. The additive head remains diagnostic-only.

## Stop-loss rule

No stable improvement over margin-only in 5-seed diagnostics means this route should be dropped. Small or unstable improvement may justify one analysis pass, but not open-ended tuning.
