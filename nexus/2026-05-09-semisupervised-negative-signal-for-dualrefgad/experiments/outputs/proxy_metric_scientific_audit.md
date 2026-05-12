# Audit: Proxy AP/AUC Scientific Validity

**Date**: 2026-05-09  
**Trigger**: User questioned why N1/N2 proxy methods show AP > 0.70.

---

## Verdict

The previously reported `Proxy AP` values for N1/N2/N3 are **not anomaly-detection AP** and should not be compared with Elliptic test AP.

They are pair-classification AP under an artificial balanced dataset:

```text
positive pairs: valid relation pairs
negative pairs: constructed mismatched/anti-direction pairs
class prior: approximately 50% positive, 50% negative
```

Therefore:

```text
Proxy AP > 0.70 is not surprising and is not evidence of high anomaly detection performance.
```

---

## What the metric actually measured

The script computed:

```python
s = concat([positive_pair_scores, negative_pair_scores])
y = concat([ones_like(positive_scores), zeros_like(negative_scores)])
average_precision_score(y, s)
```

This answers:

> Can the margin score separate constructed positive relation pairs from constructed negative relation pairs?

It does **not** answer:

> Can the method rank real anomalous nodes above normal nodes?

---

## Why AP is inflated relative to anomaly AP

Average Precision is sensitive to class prevalence.

In real Elliptic anomaly detection, anomaly prevalence is low. In the proxy pair task, the constructed dataset is balanced 1:1:

```text
#positive pairs = #negative pairs
baseline AP ≈ 0.50
```

So `Proxy AP = 0.73 / 0.79` means:

```text
better than random pair separability in a balanced artificial relation task
```

not:

```text
AP 0.73/0.79 on real anomaly detection
```

The correct real anomaly AP from the same reconstruction remains:

```text
real_test_margin_ap = 0.5510
```

---

## Which proxy metrics remain useful?

| Metric | Scientific status | Why |
|---|---|---|
| Proxy AUC | cautiously useful | threshold/class-prior insensitive; measures pair separability |
| Proxy AP | not comparable | strongly class-prior dependent; artificial 50/50 labels |
| Real test AUC/AP | valid benchmark | computed on true test anomaly labels |

Proxy AUC can still be used as a **diagnostic of relation-pair separability**, but not as final performance evidence.

---

## N1/N2/N3 audited interpretation

| ID | Proxy AUC | Proxy AP | Audited interpretation |
|---|---:|---:|---|
| N1 context mismatch | 0.6638 | 0.7326 | Pair separability exists, but AP inflated by balanced proxy labels |
| N2 directional mismatch | 0.7166 | 0.7899 | Better non-tautological pair separability than N1; AP not performance evidence |
| N3 anti-direction | 0.7797 | 0.7619 | Not independent evidence; negative score is `-positive`, so separation is partly tautological |
| N4 hard normal | N/A | N/A | Circular if labels are defined by margin itself |

---

## Corrected conclusion

The scientific conclusion should be weakened to:

```text
N2 has stronger non-tautological pair-separability than N1 under the margin score.
```

It should **not** be stated as:

```text
N2 achieves AP 0.7899 or is close to anomaly AP performance.
```

---

## Next audit requirement before training

Before using N2 as a training signal, we should compute at least one more diagnostic:

```text
Does a high N2 pair-separability objective correlate with real test anomaly AUC across seeds?
```

Recommended next step:

```text
Run real proxy computation for seeds [0,1,2,3,4]
Report Proxy AUC only as diagnostic
Compare with real margin-only test AUC/AP per seed
Do not report Proxy AP as performance evidence
```

---

_Conclusion: the user's concern is valid; Proxy AP was a misleading metric for this diagnostic._
