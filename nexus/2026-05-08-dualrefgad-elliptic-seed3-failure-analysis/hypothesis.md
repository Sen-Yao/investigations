# Hypotheses

## H1: Seed 3 split is structurally harder

Seed 3 may select labeled normal nodes that are less representative of the normal test population.

Tests:

- Compare labeled-normal feature distributions across seeds.
- Compare degree / component / time-step distribution across seeds.
- Compare distance from train normal nodes to test anomalies.

## H2: Seed 3 normal reference quality is worse

`R_n` may contain weaker or less representative normal anchors for seed 3.

Tests:

- Compare `normal_ref_normal_ratio` across seeds.
- Compare normal reference feature/distance distribution.
- Compare target-to-normal-reference similarity distribution.

## H3: Seed 3 deviation reference quality is worse

`R_a` may fail to capture useful deviation support in seed 3.

Tests:

- Compare `anom_ref_anom_ratio` and `anom_ref_anom_ratio_on_anom_nodes` across seeds.
- Compare anomaly-reference coverage over test anomalies.
- Inspect whether seed 3 has fewer useful deviation-side matches.

## H4: Score or margin collapses in seed 3

The head may produce less separable final scores for seed 3.

Tests:

- Compare `final_score_std`, `final_margin_std`, `final_sn_std`, `final_sa_std`.
- Plot normal vs anomaly score distributions per seed.
- Compare ROC ranking errors.

## H5: Seed 3 suffers stronger final epoch degradation

The model may pass through a better checkpoint and degrade more by epoch 50.

Tests:

- Pull WandB history per run.
- Compare `test_auc/test_ap` curves.
- Compare final-vs-peak degradation without using peak as formal metric.

## H6: The low seed is expected variance, not a bug

If all diagnostics look similar, seed 3 may simply reveal high variance in the current head.

Implication:

- Report variance honestly.
- Improve objective robustness rather than treating seed 3 as a data bug.
