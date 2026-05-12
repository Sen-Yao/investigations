# Semi-Supervised Negative Signal for DualRefGAD

## Status

**Created**: 2026-05-09  
**Phase**: Phase 1 - Literature & Mechanism Survey + Offline Consistency Diagnosis

---

## Background

Previous investigation `2026-05-08-dualrefgad-training-degradation` revealed:

```text
margin-only AUC 0.7952±0.0071
current dual_margin_two_score AUC 0.7455±0.0188
```

The learned head degrades margin ranking.  
We traced the root cause to:

```text
Training objective misalignment:
current BCE trains tuple-context matching,
not anomaly ranking.
```

This investigation shifts focus from head structure to a more foundational question:

> **In semi-supervised GAD with only normal training nodes, how should DualRefGAD construct negative supervision signals that align with anomaly ranking?**

---

## Core Research Question

```text
Given:
- Training set contains only normal nodes (semi-supervised protocol)
- DualRefGAD uses fixed dual-reference geometry (margin)
- Goal: anomaly detection (rank anomalies higher than normal)

Question:
What negative signals can we construct from normal-only training nodes,
that will guide learned heads toward better anomaly ranking?
```

---

## Relation to Prior Work

| Investigation | Main Finding | Connection to This Investigation |
|---|---|---|
| `2026-05-08-dualrefgad-training-degradation` | margin-only > learned final; BCE misaligned | Problem identification |
| `2026-05-06-dualrefgad-two-sided-score-head` | dual-reference structure valid; head design matters | Structural context |
| `2026-05-06-fixed-dual-reference-guided-gt-learning` | fixed dual-reference evidence learnable | Precondition |

This investigation builds on the diagnosis and asks:

```text
If head structure alone is not the bottleneck,
what training signal should drive the head?
```

---

## Scope

In scope:

- Mechanism survey of semi-supervised GAD negative signals (GGAD / VecGAD / RHO / others)
- Score semantics: normality vs abnormality
- Candidate negative signal designs for DualRefGAD
- Offline consistency diagnosis: proxy metrics vs real anomaly ranking
- Minimal probe experiments (not full sweep)

Out of scope for Phase 1:

- Full 5-seed sweep with new training objectives
- End-to-end GT fine-tuning
- Changing reference construction logic
- Multi-dataset experiments

---

## Key Decisions to Make

| Decision | Options | Constraint |
|---|---|---|
| **Score semantics** | high = anomaly, or high = normality | Must match margin-only observation |
| **Negative signal type** | pseudo anomaly, relation mismatch, view contrastive, hard normal negative | Must be constructible from normal-only train |
| **Loss function** | BCE, InfoNCE, ranking loss, hybrid | Must align with anomaly ranking |
| **Negative pair definition** | node-level, relation-level, view-level | Must respect dualref semantics |

---

## Working Principle

This investigation follows Nexus rules:

- No cherry-picking
- No data leakage (train set contains only normal nodes)
- Literature mechanisms must be verified from original papers/code, not imagined
- Offline diagnosis before full experiments
- Minimal probes to validate assumptions before sweep

---

## Planned Outputs

| Output | Target |
|---|---|
| Literature mechanism comparison table | RESEARCH_SURVEY.md |
| Candidate negative signal designs | hypothesis.md |
| Offline consistency diagnosis script + results | experiments/scripts/ + outputs/ |
| Recommendation for next training objective | insights.md |

---

## Timeline

Phase 1 (today):

```text
1. Create investigation structure
2. Offline consistency diagnosis: proxy metrics vs real AUC/AP
3. Mechanism survey (partial)
```

Phase 2 (next):

```text
1. Minimal probe: one seed, one negative signal variant
2. Compare with margin-only baseline
3. Iterate or scale to 5-seed
```

---

_Investigation created by Nexus, 2026-05-09._