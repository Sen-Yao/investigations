# Insights

## Seed0 result — inconclusive / weak positive diagnostic, not a method promotion

The seed0 normal-only Matrix AE probe completed successfully.

| item | value |
|---|---:|
| decision | `INCONCLUSIVE` |
| strongest scalar baseline | `neg_mat_mean` |
| scalar AUC / AP | 0.6377 / 0.1405 |
| best AE | `ae_mse_latent8` |
| AE AUC / AP | 0.6028 / 0.1140 |
| AE ΔAUC vs scalar best | -0.0349 |
| AE Spearman vs margin | -0.4560 |
| AE top5 Jaccard vs margin | 0.0036 |

### Interpretation

- Matrix AE has some signal (AUC≈0.603), but it does **not** beat simple scalar matrix orientation on seed0.
- The strongest scalar here is negative matrix mean, not margin, suggesting orientation/regime effects remain important.
- AE has very low top-k overlap with margin, so it may be complementary, but the effect size is currently too weak to promote.
- Next best step is not a full method sweep yet. Recommended follow-up: compare seed1/2 cheaply or inspect degree/rejection/regime stratification before adding model capacity.

### Conclusion boundary

This is a runner-registered seed0 diagnostic. It is valid as exploratory evidence, not final scientific validation.

## 5-seed result — do not promote Matrix AE as-is

The seed1-4 parallel probes completed successfully, giving a 5-seed view of the normal-only Matrix AE diagnostic.

Per-seed outcome:

- seed0: INCONCLUSIVE, best AE AUC/AP 0.6028 / 0.1140, scalar AUC/AP 0.6377 / 0.1405, ΔAUC -0.0349
- seed1: PROMOTE, best AE AUC/AP 0.6659 / 0.2689, scalar AUC/AP 0.6410 / 0.1785, ΔAUC +0.0249
- seed2: DROP, best AE AUC/AP 0.5463 / 0.1006, scalar AUC/AP 0.6641 / 0.2374, ΔAUC -0.1178
- seed3: PROMOTE, best AE AUC/AP 0.6731 / 0.1597, scalar AUC/AP 0.6491 / 0.1537, ΔAUC +0.0240
- seed4: DROP, best AE AUC/AP 0.5692 / 0.1121, scalar AUC/AP 0.6581 / 0.2386, ΔAUC -0.0889

Aggregate:

- best AE AUC: 0.6115 ± 0.0567
- best AE AP: 0.1511 ± 0.0696
- best scalar AUC: 0.6500 ± 0.0111
- best scalar AP: 0.1897 ± 0.0461
- AE - scalar ΔAUC: -0.0385 ± 0.0647
- AE - scalar ΔAP: -0.0387 ± 0.0951

Interpretation:

- The core finding changed from seed0-only `INCONCLUSIVE` to multi-seed **negative for method promotion**.
- Matrix AE is not uniformly useless: two seeds produce small AUC wins over scalar summaries. But this is not stable and the aggregate effect is negative.
- Scalar response-matrix summaries are much more stable: scalar AUC std is ~0.0111, while AE AUC std is ~0.0567. This suggests the AE is adding optimization/representation instability rather than extracting a robust hidden abnormality signal.
- Top-k overlap with margin remains tiny on average (0.0075), so AE is not merely duplicating margin; however, complementarity without stable metric gain is not enough for promotion.

Practical decision:

- Do not add this normal-only Matrix AE head to DualRefGAD as a main method component.
- If Route2.5 continues, prioritize regime/degree/reference-distribution diagnostics or a simpler scalar/rule-based use of the response matrix before adding learnable capacity.
- A possible repair direction is not “bigger AE”; it is to first identify why seeds 1/3 promote while 2/4 drop, likely via split/regime/reference selection sensitivity.

Conclusion boundary:

This is a 5-seed probe on Elliptic under the Route2.5 Matrix AE protocol. It is strong enough to reject immediate promotion of this exact AE design, but not broad enough to reject all response-matrix-based anomaly scoring.

## Phase 1 instability audit — AE initialization is not the main culprit

Phase 1 reran AE training multiple times per fixed split/reference construction to test whether the seed instability above was simply caused by AE initialization noise.

Per-split best repeat outcome:

- seed0: best-repeat AE AUC/AP 0.6055 / 0.1147; scalar 0.6377 / 0.1405; ΔAUC -0.0322
- seed1: best-repeat AE AUC/AP 0.6696 / 0.2855; scalar 0.6410 / 0.1785; ΔAUC +0.0286
- seed2: best-repeat AE AUC/AP 0.5489 / 0.1010; scalar 0.6641 / 0.2374; ΔAUC -0.1151
- seed3: best-repeat AE AUC/AP 0.6730 / 0.1706; scalar 0.6491 / 0.1537; ΔAUC +0.0239
- seed4: best-repeat AE AUC/AP 0.5744 / 0.1149; scalar 0.6581 / 0.2386; ΔAUC -0.0837

Aggregate:

- best-repeat AE AUC: 0.6143 ± 0.0499
- best-repeat AE ΔAUC vs scalar: -0.0357 ± 0.0571
- scalar AUC: 0.6500 ± 0.0100
- mean within-split AE AUC std: 0.0033 ± 0.0019
- decision: `SPLIT_REFERENCE_INSTABILITY_DOMINATES__DO_NOT_PROMOTE`

Interpretation:

- Within a fixed split, repeated AE initializations are highly stable; the typical AE AUC fluctuation is only ~0.003.
- Across splits, the success/failure pattern remains: seeds 1/3 can promote slightly, while seeds 2/4 fail badly.
- Therefore the instability is not primarily “AE optimizer randomness”. It is more likely caused by split/reference/representation regime differences that change whether the response matrix contains usable normal-only structure.
- This reinforces the practical decision: do **not** promote the Matrix AE head as-is; if Route2.5 continues, repair reference distribution/orientation regime before adding learnable capacity.

Conclusion boundary:

This Phase 1 audit is still a registered diagnostic, not a SOTA sweep. It is sufficient to reject “just rerun/tune the AE” as the next step, but not sufficient to reject all response-matrix scoring families.

## Phase 2 posthoc learning-strategy audit — cheap selection does not repair AE

Phase 2 asked a narrower, lower-cost question after Phase 1: before spending compute on a new method sweep, can the Matrix AE failure be explained by a cheap learning-strategy issue such as latent choice or validation-loss selection?

Key results:

- Scalar baseline AUC: 0.6500 ± 0.0100
- Oracle best-AUC AE selector: 0.6143 ± 0.0499; ΔAUC -0.0357 ± 0.0571
- Validation-loss AE selector: 0.6099 ± 0.0488; ΔAUC -0.0401 ± 0.0556
- Validation-loss selector promotes only 1/5 splits and drops 2/5 splits.
- Fixed latent=8 and latent=16 both keep negative mean ΔAUC; neither gives a deployable repair.
- Validation loss has near-zero rank correlation with AUC across repeated AE runs (`ρ≈-0.062`), so it is not a reliable unsupervised selector for this signal.

Interpretation:

- The apparent seed1/seed3 wins are not recoverable by a clean deployable selection rule.
- Even a label-oracle selector remains below scalar summaries on average, so “try a better checkpoint selector” is not a promising route.
- This strengthens the previous conclusion: do **not** promote the normal-only Matrix AE head as-is.
- The next scientific move, if Route2.5 continues, should not be more AE tuning. It should diagnose or repair reference distribution, orientation regime, or representation construction.

Conclusion boundary:

This is a runner-registered posthoc diagnostic, not a formal SOTA sweep. It is strong enough to reject cheap AE learning-strategy repair for the tested Route2.5 design, but it does not reject all response-matrix methods.

