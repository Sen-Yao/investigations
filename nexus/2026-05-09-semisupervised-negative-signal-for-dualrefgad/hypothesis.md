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

## H8: Elliptic requires regime-conditioned orientation, not a global score sign

**Claim**: On Elliptic, the failure is not merely that `margin` or `rejection` should be globally negated. The anomaly direction is regime-dependent: low-degree / low-rejection regions can be more anomalous, while medium/high-degree regimes can have different orientation.

**Evidence**:
- Orientation diagnostic showed `margin_cos_u_d` AUC `0.3404`, but `-margin` AUC `0.6596`.
- `normal_rejection` is also inverted: AUC `0.3489`, `-rejection` AUC `0.6511`.
- Lowest-degree bin has much higher anomaly rate and strong `-margin` signal, while higher-degree bins behave differently.
- Naive label-free orientation rules based on train-normal vs unlabeled shift predicted the wrong sign.

**Implication**: A scientifically defensible route should not simply flip the score. It should model structural regimes and infer orientation/role locally, or design a bidirectional score that does not require committing to one global anomaly-high convention.

---

## H9: R_a selection must be repaired before relying on DualRef margin on Elliptic

**Claim**: Current `R_a` selection (`similarity + rejection_score`) is fragile because it assumes high rejection means anomaly-like. On Elliptic this assumption is inverted, so `R_a` becomes mostly normal/high-degree references.

**Evidence**:
- Global anomaly-reference anomaly ratio is only `0.0235`.
- Anom-ref ratio AUC is `0.4448`.
- `ga/rejection` AUC is `0.3489`, so the bootstrap signal selects the wrong side.
- `R_a` references have higher mean log-degree than `R_n`, aligning with degree rather than true anomaly semantics.

**Candidate directions**:
1. **Bidirectional reference sets**: construct both high-rejection and low-rejection reference candidates, then let downstream scoring decide which side is suspicious per regime.
2. **Degree/regime-conditioned R_a**: choose anomaly-side references within structural strata rather than globally.
3. **Reference purity diagnostic without labels**: use stability/diversity/consistency criteria to avoid degenerate high-degree reference pools.

**Implication**: Improving `R_a` purity/role assignment may be more fundamental than adding a larger head on top of the current references.

---

## H10: Route2.5 / RRDM-style response distribution modeling may be a bridge between matrix signal and regime orientation

**Claim**: The response matrix `M_ij(v)=cos(h_v-r_{n,i}, r_{a,j}-r_{n,i})` contains richer information than scalar margin, but simple summaries such as mean/median are unstable. A response-distribution model could learn normal patterns of the full matrix under labeled normals, then score anomalies as reconstruction/likelihood/role inconsistency.

**Working name**: Route2.5 / RRDM-style response distribution modeling.

**Candidate variants**:
1. **Normal-only Matrix Autoencoder**: train an autoencoder on labeled-normal response matrices; anomaly score is reconstruction error or latent inconsistency.
2. **Denoising Matrix Autoencoder**: corrupt/drop entries of normal response matrices and reconstruct; score nodes whose response pattern is not reconstructable from normal structure.
3. **Regime-conditioned Matrix Autoencoder**: condition decoder or normalization on structural regime features such as degree bin, rejection quantile, or reference-set statistics.
4. **Bidirectional matrix channels**: include both high-rejection and low-rejection reference response matrices so the model can detect orientation flips without a hard global sign.

**Why it is promising**:
- It preserves the full `K_n × K_a` response pattern rather than collapsing it to a scalar mean.
- It can absorb heterogeneous normal regimes better than center-distance scoring.
- It can be trained with labeled normals only, avoiding pseudo-anomaly generation.

**Main risk**:
- A pure reconstruction-error objective may again learn typicality rather than anomaly ranking. It must be tested as a diagnostic first and compared against margin, `-margin`, and simple regime baselines.

---

## Hypothesis Status Table

| Hypothesis | Status | Verification Method |
|---|---|---|
| H1 context mismatch misaligned | confirmed | Previous experiment |
| H2 pseudo anomaly strong but risky | to verify | Literature survey |
| H3 RHO contrastive alignment | partially rejected on Elliptic | Stage4 + embedding structure diagnostics |
| H4 relation-level negative | candidate | Offline diagnosis |
| H5 score semantics constraint | active; global sign insufficient | Orientation/sign diagnostic |
| H6 offline proxy metrics diagnostic | actionable | Run offline analysis |
| H7 minimal probe before sweep | actionable | After H6 validation |
| H8 regime-conditioned orientation | active candidate | Degree/regime orientation diagnostic |
| H9 R_a selection repair | active candidate | Reference autopsy + new selection probes |
| H10 Route2.5/RRDM response distribution modeling | active candidate | Normal-only matrix AE diagnostic |

---

_Updated by Nexus, 2026-05-16._