# Insights — Constraint-Calibrated Reference Relation

## Current scientific position

The response matrix should be treated as a microscope for reference-relation quality, not as a default excuse to add capacity. The important result from the previous investigation is that C-LEG3 / old-exact reference construction restores positive response-matrix orientation, whereas current refs invert the signal.

## Main interpretation rule

A training objective that only makes known-normal nodes lower is not enough. It must also change anomaly ranking in a way that is:

1. stable across seeds;
2. not almost perfectly monotonic with margin/mat_mean;
3. not explainable by `D_psi` residual norm alone;
4. distinguishable from random-direction or GGAD-like pseudo anomalies.

## Why R0/R4 come before R1–R3

R0 tells us how much signal is already present without training. R4 tells us whether PCA normal-subspace residual alone explains the apparent benefit. If either already accounts for the effect, then a learned pseudo-anomaly scorer is scientifically weak.

## Current no-go statements

- Do not present `D_psi` as a new module in the first version.
- Do not add `W` / reliability gate in the first score.
- Do not claim pseudo anomaly as contribution until it beats random-direction and GGAD-like controls under the same normal-only protocol.
- Do not repeat broad k scan under the current reference generator; that was already answered.

## First-step expected output

A JSON table and report section answering:

- Which family wins under fixed C-LEG3: scalar mean, row anchor, column reference, entry residual, PCA residual, or proxy?
- Is the winner complementary to margin/mat_mean by Spearman and top-k overlap?
- Does performance concentrate in degree/rejection regimes?
- Does R4 residual explain the same ranking?
- Should R1–R3 pseudo-anomaly training be launched next?


## Step-1 result insight — 2026-05-22

The first C-LEG3 decomposition gate finished with a clear result: simple full response-matrix mean (`mat_mean`) won all 5 seeds, with AUC 0.8168 ± 0.0054. This improves over centroid margin (0.7928 ± 0.0054) and beats row-anchor, column-reference, entry-normal-manifold, and degree proxy families.

Interpretation:

1. The useful signal is already present in the unweighted C-LEG3 response matrix.
2. The signal is not just degree: degree proxy is ~0.33 AUC and has very low correlation with the winning score.
3. The winning score is not identical to margin: Spearman ≈ 0.707 and top-5% overlap ≈ 0.579, so it changes ranking meaningfully.
4. PCA/entry normal-manifold residuals are not currently stronger than `mat_mean`; this weakens the case for immediately building pseudo-anomaly training around `D_psi`.
5. Next training experiments must treat `mat_mean` as the baseline to beat. A learned scorer that only matches margin or residual_norm is not scientifically interesting.


## Step-2 insight — mat_mean helps by boundary reordering, not by magic separation

The failure autopsy confirms that switching from centroid `margin` to full-matrix `mat_mean` changes the top-K boundary in a scientifically meaningful way.

Aggregate result:

- `mat_mean` AUC: 0.8134 ± 0.0105
- `margin` AUC: 0.7980 ± 0.0060
- Decision: `MAT_MEAN_REORDERS_BOUNDARY_BY_RESCUING_ANOMALIES_AND_REMOVING_MARGIN_FALSE_POSITIVES`

Mechanism:

1. `mat_mean` rescues many true anomalies missed by margin: 360.2 ± 21.9 per seed on average.
2. It removes even more margin-only normal false positives: 1219.0 ± 91.7 per seed on average.
3. Removed false positives have low anomaly-reference purity and large row/column/matrix dispersion; these are likely centroid-margin artifacts.
4. Lost anomalies are a real cost: they often have high margin and reasonably high anomaly-reference purity but heterogeneous matrix support, so a pure average can over-penalize them.

Implication: the next method should not simply add pseudo-anomaly training. A better next gate is a robust matrix readout that keeps the "remove centroid false positives" effect while reducing the "lost heterogeneous true anomalies" cost. Candidate directions: distribution-aware readout, row/column gating, quantile/mixture summaries, or reliability-weighted average; all must beat `mat_mean`, not merely beat `margin`.

## 2026-05-25 Learning Signal Discovery pivot note

The C-LEG3 decomposition result should now be treated as a **positive control / teacher** for Learning Signal Discovery. Its key role is to show that a strong ordering exists in the fixed old-exact reference relation (`mat_mean` AUC `0.8168 ± 0.0054`, `margin` AUC `0.7928 ± 0.0054`), not to certify that current learnable reference losses are adequate.

Cross-investigation evidence from `2026-05-24-dualrefgad-normal-low-dropout-reliability` sharpened this interpretation: A/C/D reference-loss ablation was negative, with best report-only `A_normal_low` test AUC/AP only `0.5544 ± 0.0508 / 0.1007 ± 0.0118`. Therefore the current bottleneck is not primarily response-matrix readout capacity; it is finding a reliable, stable learning signal that can recover C-LEG3-like ordering under label-free/normal-only constraints.

Label boundary: anomaly labels remain allowed only for diagnostic/autopsy use during exploration, including oracle analyses of why `mat_mean` works and why losses fail. They must not be used for final method training, early stopping, checkpoint selection, or non-oracle contribution claims.

## 2026-05-25 LSD oracle-autopsy result insight

`lsd_oracle_autopsy_probe` finished as a runner-registered bundled pure probe (`5/5` seeds, HCCS-25, no training). The label boundary was preserved: anomaly labels were used only for report-only metrics and oracle-autopsy categories.

The probe changes the reading of the C-LEG3 teacher slightly. In this exact probe implementation, `mat_mean` only barely exceeds `margin` on average (`0.7845 ± 0.0308` vs `0.7826 ± 0.0101` AUC), with two seeds where `mat_mean` is worse. This is weaker than the earlier Step-1/Step-2 positive-control numbers and must be treated as a protocol/regime comparison warning. Do not cite this run as a clean `mat_mean` dominance result without explaining the setting difference.

The useful discovery signal is in the discordant boundary categories, not in the aggregate AUC gap alone:

1. `mat_mean` removes many margin-only normal false positives (`1382.2 ± 83.1` per seed), and these removed false positives have high matrix dispersion plus very low anomaly-reference diagnostic ratio.
2. `mat_mean` also loses true anomalies (`173.8 ± 112.6` per seed) whose anomaly-reference diagnostic ratio is still high but matrix support is heterogeneous.
3. Score delta is not a degree shortcut: Spearman(delta, degree) is about `-0.024`.
4. `mat_mean` and `margin` are correlated but not identical: Spearman is about `0.700`.

Scientific implication: the next Learning Signal Discovery step should not directly learn a scalar `mat_mean` teacher. It should isolate label-free proxies for row/column reliability and heterogeneous-support handling: keep the false-positive cleanup effect while preventing reliable but heterogeneous true anomalies from being over-penalized. Future trainable losses should be framed as normal-only stability/reliability proxies and must beat both `margin` and raw `mat_mean` while reporting degree/rejection shortcut checks.

## 2026-05-25 Strict reproduction insight

The C-LEG3 strict reproduction audit clarifies the apparent conflict between Step-1/Step-2 and the LSD oracle-autopsy probe.

Strict facts:

- `mat_mean` was recomputed exactly as the direct mean of the response matrix entries; the formula check is exact (`max_abs_diff=0.0`) for all five seeds.
- The strict audit uses explicit `data_split_seed=seed` and sequential execution, avoiding global RNG interference from concurrent `load_mat()` calls.
- Aggregate result is between the earlier strong positive controls and the weaker LSD run: `mat_mean` AUC `0.8009 ± 0.0182`, `margin` AUC `0.7952 ± 0.0064`.

Interpretation update:

1. The LSD drop is not evidence that `mat_mean` was misdefined; it is most likely a split/RNG protocol drift artifact.
2. Historical Step-1/Step-2 remain useful as positive-control evidence, but they lack split fingerprints, so per-seed equality cannot be audited after the fact.
3. The strict audit should be the new citation baseline when discussing reproducible C-LEG3 `margin`/`mat_mean` behavior.
4. Future matrix-readout or learning-signal probes must save split fingerprints and avoid threaded global RNG around data splitting; otherwise small AUC deltas are not scientifically interpretable.

