# Hypotheses

## H1: Objective / ranking metric mismatch

The training loss may decrease while AUC/AP decrease.

Tests:

- Pull epoch-wise loss, test_auc, test_ap from WandB.
- Compute correlation between loss and AUC/AP.
- Check whether lower loss corresponds to worse ranking.

## H2: Score sign / semantic mismatch

The metric may use `anomaly_score = -normality_logit`, but the objective may optimize the opposite or a partially inconsistent direction.

Tests:

- For saved epoch outputs, compute AUC for both score directions.
- Audit definitions of `normality_logit`, `sn`, `sa`, `margin`, and final score.
- Verify whether training increases normal/anomaly separation in the intended direction.

## H3: Full head training destroys a strong fixed-reference ranking

Epoch0 score already encodes useful dual-reference information. Full training may overfit or distort it.

Tests:

- Compare epoch0 and epoch50 score Spearman/Kendall correlation.
- Compare top-k anomaly overlap between epoch0 and epoch50.
- Compare per-node score drift for normal vs anomaly nodes.

## H4: Current loss pushes margin components in the wrong direction

The two-sided components may individually drift toward less discriminative values.

Tests:

- Track `sn`, `sa`, `margin`, and final score distributions by label across epochs.
- Check whether anomaly-normal score gap narrows during training.

## H5: Calibration-only or residual training can preserve the signal

Instead of learning a full score head, a constrained calibration or residual correction may preserve epoch0 ranking while improving stability.

Tests:

- no-training score baseline;
- calibration-only scalar/affine mapping;
- residual adapter with small regularization;
- full head training as control.

## H6: Epoch0 is not random; it is a hand-crafted reference baseline

The initial score is high because fixed dual-reference construction already encodes anomaly ranking.

Tests:

- Compare epoch0 score to explicit reference-only heuristics.
- Compare trained head against reference-only baseline.
