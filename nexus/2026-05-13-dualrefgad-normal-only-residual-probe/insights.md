# Insights — DualRefGAD Normal-only Residual Probe

## Initial framing

This investigation exists because the earlier DualRefGAD evidence points in two directions at once. The fixed dual-reference margin contains real ranking signal, but learned heads can degrade it. That contradiction should not be resolved by adding another unconstrained head; it should be resolved by a diagnostic probe that asks whether the baseline leaves residual signal under a protocol-clean normal-only setup.

## Diagnostic-only status

The additive residual form is not considered elegant enough to be the final method narrative. Its role is diagnostic: test whether margin-only has a stable blind spot. If the probe works, the output should be inspected and reformulated; if it does not work, the negative result should stop this branch.

## Interpretation rule

A successful result requires stable improvement and evidence of ranking-geometry change. A correction that simply suppresses known-normal scores is valid training behavior, but it is not a new anomaly detection principle unless it changes anomaly ranking in a stable way.

## Relation to prior investigation

`2026-05-09-semisupervised-negative-signal-for-dualrefgad` established that pseudo-negative or contrastive signals need a theoretical basis. This investigation narrows that lesson into a concrete diagnostic route: residual signal must be grounded in normal-manifold deviation or reference inconsistency, not in arbitrary proxy compatibility.
