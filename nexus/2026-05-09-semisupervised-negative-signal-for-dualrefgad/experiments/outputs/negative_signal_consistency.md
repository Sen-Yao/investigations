# Negative Signal Consistency Diagnosis

**Status**: corrected with real offline computation  
**Timestamp**: 2026-05-09 15:49  
**Dataset**: Elliptic  
**Seed**: 0  
**Protocol**: train_rate=0.05, val_rate=0.0, train set normal-only

---

## Correction

The previous `Proxy AUC (estimate)` table was not an experiment result and is invalid as evidence.

This file has been corrected to use real offline computation from HCCS-88:

```text
/data/linziyao/DualRefGAD/outputs/negative_signal_diagnosis/real_proxy_auc_seed0.json
```

The computation reconstructs frozen GT embeddings and references, then computes pair-level proxy AUC/AP for candidate negative signals.

---

## Real Results

> **Audit note (2026-05-09):** Proxy AP below is pair-classification AP on an artificial 50/50 positive-vs-negative relation-pair task. It is **not anomaly-detection AP** and should not be compared with Elliptic test AP. Proxy AUC is the safer diagnostic; Proxy AP is kept only for reproducibility.

| ID | Negative signal | Proxy AUC | Proxy AP (not comparable) | Negative mean | Notes |
|---|---|---:|---:|---:|---|
| N1 | context mismatch: target_i with rn_c, ra_c | 0.6638 | 0.7326 | 0.0666 | current-style full tuple replacement |
| N2 | directional mismatch: target_i/rn_i with ra_c anchored at rn_i | 0.7166 | 0.7899 | -0.0474 | relation-level mismatch |
| N3 | anti-direction: score(u_i, -d_i) | 0.7797 | 0.7619 | -0.3044 | tautological separability because neg=-pos |
| N4 | hard normal | N/A | N/A | N/A | circular if converted to AUC by margin-defined labels |

Real margin-only test result for the same reconstruction:

```text
AUC = 0.7937893927
AP  = 0.5510314415
```

---

## Interpretation

**Important audit correction:** The Proxy AP values are inflated by construction because the proxy task is a balanced pair-classification task. They do not indicate anomaly-detection AP. See `proxy_metric_scientific_audit.md`.

N1 is not as hopeless as the earlier estimate implied, but it remains weaker than N2 under the margin score. N2 is the most meaningful non-tautological proxy because it changes only the deviation direction while preserving the target and normal anchor. This directly tests whether the target deviation direction prefers its own deviation reference over another node's deviation reference.

N3 has the highest proxy AUC, but it is partly tautological: if the negative score is exactly `-positive`, separability is mathematically induced. Therefore N3 should not be treated as an independent training-signal discovery without a less circular construction, such as sampled low-cosine directions or orthogonalized real directions.

N4 cannot be honestly converted into Proxy AUC without circular labels, because high/low hard-normal groups are defined by the same margin score being evaluated.

---

## Recommendation After Correction

The next minimal experiment should prioritize **N2 directional mismatch**, not N3 anti-direction.

Reason:

```text
N2 is real, non-tautological, relation-level, and stronger than full context mismatch.
```

N3 may still be useful as a sanity/control loss, but not as the main candidate.
