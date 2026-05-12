# Insights

_Status: margin-only baseline confirmed 2026-05-09._

## Core finding

The margin-only baseline (epoch0, no training) significantly outperforms the learned final score after 50 epochs.

| metric | margin-only (epoch0) | current final (epoch50) | difference |
|---|---:|---:|---:|
| AUC | **0.7952±0.0071** | 0.7455±0.0188 | **+0.0497** |
| AP  | **0.5166±0.0220** | 0.2011±0.0238 | **+0.3155** |

## Interpretation

The current `dual_margin_two_score` head training systematically degrades the strong geometric margin signal that already exists in the fixed dual-reference construction.

Training does not improve ranking; it actively harms it.

## Component analysis

The final anomaly score is:

```text
final_score = margin + sa - sn
```

Diagnostic audit showed:

- margin alone: AUC 0.7938-0.8030 per seed
- sa - sn: weak/noisy
- margin + sa - sn: worse than margin alone

Therefore:

```text
learned correction terms (sa - sn) are not aligned with anomaly ranking.
They dilute the strong margin signal.
```

## Seed 3 clarification

Seed 3 has the lowest margin AUC among seeds (0.7840), but it is still higher than seed 3's final score (0.7131).

Seed 3 is not "reference failure"; it is "training degradation most visible".

## Conclusion

DualRefGAD's effective anomaly signal comes from the fixed dual-reference geometry (margin), not from learned two-score head.

Current training objective/objective-head combination is counterproductive.

## Method recommendation

DualRefGAD should move from:

```text
current: margin + sa - sn (learned)
```

to:

```text
baseline: margin-only
```

as the default anomaly score.

Future work can explore constrained additions to margin:

- calibration-only scalar/affine
- small residual regularized toward zero
- ranking-preserving objectives

but full free-form sn/sa training should be avoided.

## Updated hypothesis status

| Hypothesis | Status | Notes |
|---|---|---|
| H1 objective/ranking mismatch | confirmed | training degrades ranking |
| H2 score sign mismatch | partial | final sign correct, but component semantics wrong |
| H3 full head destroys fixed-reference ranking | confirmed | margin-only > final |
| H4 margin component drift | reframed | margin is strong; learned terms are harmful |
| H5 calibration/residual can preserve signal | now actionable | next design phase |
| H6 epoch0 is reference baseline | confirmed | margin-only baseline established |

## Next step

Report margin-only baseline as the current best DualRefGAD result.

Optionally start a constrained correction probe (margin + small residual) if further improvement is desired.