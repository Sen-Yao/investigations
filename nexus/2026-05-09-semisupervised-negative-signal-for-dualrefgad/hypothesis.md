# Hypotheses

**Investigation**: 2026-05-09-semisupervised-negative-signal-for-dualrefgad  
**Phase**: Phase 1 - Candidate Negative Signal Design

---

## H1: Current context-mismatch BCE misaligns with anomaly ranking

**Claim**: The current negative signal `(v, R_n(c), R_a(c))` trains reference identity matching, not anomaly ranking.

**Evidence**:
- margin-only AUC 0.7952 > current final AUC 0.7455
- learned residual `s_a - s_n` harms margin ranking
- BCE objective cares about tuple validity, not node-level anomaly ordering

**Implication**: Changing head structure alone won't fix the problem; training signal must be redesigned.

---

## H2: Pseudo anomaly generation is strong but risky

**Claim**: GGAD / VecGAD generate pseudo anomalies from normal nodes, providing direct normal vs pseudo-anomaly supervision. This is strong but sensitive to pseudo anomaly quality.

**Mechanism (to verify)**:
- GGAD: modifies local structure / features to create pseudo anomalies
- VecGAD: constructs deviation vectors from normal statistics

**Risk**:
- Pseudo anomalies may not match real anomaly distribution
- Model may learn to detect generation artifacts, not real anomalies
- Quality of pseudo anomalies determines upper bound

**Connection to DualRefGAD**: Could generate pseudo deviation directions / pseudo deviation references, but must ensure they align with dualref semantics.

---

## H3: RHO-style contrastive normal alignment provides alternative signal

**Claim**: RHO does not rely on pseudo anomaly nodes; instead, it uses view-level contrastive alignment (e.g., channel-wise vs cross-channel filtering, Graph Normal Alignment).

**Mechanism (to verify)**:
- RHO constructs two views of normal representations
- Positive: same node in two views aligned
- Negative: different nodes across views
- Score semantics: alignment = normality, or alignment = anomaly?

**Potential for DualRefGAD**:
- Could define views based on dual-reference relation
- View 1: target deviation direction
- View 2: reference deviation direction
- Positive: same node's (u, d) aligned
- Negative: mismatched (u_i, d_j)

**Critical question**: Does RHO's score represent normality or anomaly? Must verify from paper.

---

## H4: DualRefGAD's negative signal should be relation-level, not node-level

**Claim**: DualRefGAD's core is target deviation vs reference deviation relation. Therefore negative signals should break the relation, not just replace node contexts.

**Candidate relation-level negatives**:
- `directional mismatch`: replace deviation direction d, keep target deviation u
- `anti-direction negative`: construct d^- = -d or orthogonal direction
- `hard normal negative`: use high-margin normals as pseudo deviation sources

**Rationale**: Relation-level negative directly targets what margin measures; node-level context replacement is too coarse.

---

## H5: Score semantics must be clarified before designing loss

**Claim**: Any negative signal design must first clarify whether:

```text
score high → anomaly high (margin-like semantics)
score high → normality high (contrastive alignment semantics)
```

**Current observation**: margin-only high score correlates with anomaly high.

**Implication**:
- If we keep anomaly-high semantics, we cannot train normal nodes to have high scores.
- Training signal must either:
  1. suppress high-score normals (risk: may suppress margin signal)
  2. enhance deviation direction alignment for high-margin nodes
  3. use ranking-based loss that does not force absolute score values

**Alternative**: If we flip semantics to normality-high, then:
- anomaly_score = -score(u,d)
- Training normal nodes to have high alignment is correct
- But this reverses margin-only observation and must be re-validated

---

## H6: Offline proxy metrics can diagnose negative signal alignment

**Claim**: Before running full experiments, we can offline compute:

```text
For each candidate negative signal definition:
1. Compute proxy metric: AUC(positive vs negative scores)
2. Compare proxy metric with real test anomaly AUC/AP
3. Check if proxy metric ranking aligns with anomaly ranking
```

**Rationale**:
- If proxy metric (positive > negative) aligns with anomaly ranking, the negative signal is a good candidate.
- If proxy metric contradicts anomaly ranking, the negative signal is misaligned.

**Method**:
- Use frozen GT embeddings from previous experiment
- Compute scores for positive and negative pairs under each definition
- Compute proxy AUC/AP
- Correlate with real test AUC/AP

---

## H7: Minimal probe before full sweep

**Claim**: After offline diagnosis, we should run a minimal probe (1 seed, small epochs) to validate whether training with the candidate negative signal improves test AUC/AP.

**Design**:
- Fixed GT embeddings
- Elliptic
- One seed (e.g., seed 0)
- Candidate loss + candidate negative signal
- Compare with margin-only baseline and current BCE baseline

**Decision rule**:
- If probe improves over current BCE → scale to 5-seed
- If probe degrades → revisit negative signal design

---

## Hypothesis Status Table

| Hypothesis | Status | Verification Method |
|---|---|---|
| H1 context mismatch misaligned | confirmed | Previous experiment |
| H2 pseudo anomaly strong but risky | to verify | Literature survey |
| H3 RHO contrastive alignment | to verify | Literature survey + code |
| H4 relation-level negative | candidate | Offline diagnosis |
| H5 score semantics constraint | active | Must decide before loss design |
| H6 offline proxy metrics diagnostic | actionable | Run offline analysis |
| H7 minimal probe before sweep | actionable | After H6 validation |

---

_Updated by Nexus, 2026-05-09._