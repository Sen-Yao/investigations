# Hypotheses

## H1: Fixed dual-reference is evidence, not supervision

`R_n(v)` and `R_a(v)` should be interpreted as fixed contextual evidence for target node `v`, not as learnable retrieval and not as pseudo anomaly labels.

## H2: The key learning problem is GT-based anomaly judgment

The main question is how GT should learn the function:

```text
f_GT(v, R_n(v), R_a(v)) -> anomaly_degree(v)
```

under a semi-supervised setting with only labeled-normal nodes.

## H3: 5% labeled-normal should calibrate normal judgment, not generate unreliable anomaly labels

Labeled-normal nodes should define how normal evidence patterns look under fixed dual-reference, while unlabeled nodes provide the mixture distribution.

## H4: Anomalies are deviations from normal calibrated dual-reference judgment

Anomaly degree should emerge from deviation/inconsistency relative to normal-calibrated target-reference relations, rather than from a generated pseudo anomaly embedding.

## H5: The method should avoid bootstrap instability

Because dual-reference is fixed/precomputed, the model should not repeatedly update the reference graph based on its own predictions. This avoids confirmation bias from online pseudo-label or online retrieval updates.
