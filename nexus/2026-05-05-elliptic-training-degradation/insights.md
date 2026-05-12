# Insights

## 2026-05-05 Initial framing

Elliptic `target_ref_guided` formal sweep shows a consistent gap between early/best-test geometry and final performance. The core scientific question is no longer whether `pseudo_beta` should be tuned, but why training degrades the initial reference geometry.

The working hypothesis is that BCE pseudo anomaly training optimizes synthetic separability, while Elliptic anomaly ranking depends on preserving heterogeneous reference geometry. The first diagnostic priority is to determine whether degradation comes from representation drift or from the classifier/objective itself.

## 2026-05-06 D0/D1 mini diagnosis result

Mini diagnosis sweep finished successfully:

- Job: `exp_20260505_193110_elliptic_training_degradation_diagnosis_`
- Sweep: `6orojp4m`
- WandB: https://wandb.ai/HCCS/VoxG/sweeps/6orojp4m
- Runs: 3/3 finished, failed=0
- Dataset: Elliptic
- Seed: 0
- pseudo_beta: 0.2

| diagnosis_mode | final AUC | final AP | best_test_auc | best_test_ap | pseudo_auc | pseudo_ap |
|---|---:|---:|---:|---:|---:|---:|
| train_all | 0.5657 | 0.1019 | 0.6969 | 0.1589 | 0.5291 | 0.5227 |
| freeze_encoder | 0.7328 | 0.2237 | 0.7785 | 0.3427 | 0.5149 | 0.5123 |
| train_head_only | 0.7328 | 0.2237 | 0.7785 | 0.3427 | 0.5149 | 0.5123 |

Initial interpretation:

1. `train_all` reproduces the degradation pattern: final AUC/AP is much lower than early/best-test performance.
2. Freezing the encoder/reference path strongly improves both final and best-test metrics.
3. This supports H2: the degradation is primarily caused by representation/reference geometry drift under full training, rather than the head alone.
4. `pseudo_auc` remains close to random in all variants, so the current BCE pseudo anomaly objective provides weak synthetic separability; full training still perturbs the representation enough to hurt real anomaly ranking.

Caveat:

- `freeze_encoder` and `train_head_only` are currently equivalent in implementation: both freeze all non-head parameters and train `fc1/fc2/fc3`.
- `emb_drift_mean` is not yet reliable because dropout/train-eval mode can affect embedding snapshots even when encoder parameters are frozen. The next patch should compute drift under `model.eval()` with deterministic no-grad snapshots.

Next diagnostic priority:

- Fix deterministic geometry drift logging.
- Separate `freeze_encoder` vs stricter `head_only` only if needed.
- Add epoch-wise curve extraction for `train_all` vs frozen head to verify when the gap opens.

## 2026-05-06 Head-only 5-seed validation

Head-only 5-seed mini sweep finished successfully:

- Job: `exp_20260506_091315_elliptic_head_only_5seed_validation`
- Sweep: `h21wc31z`
- WandB: https://wandb.ai/HCCS/VoxG/sweeps/h21wc31z
- Runs: 5/5 finished, failed=0
- Dataset: Elliptic
- Objective: `target_ref_guided`
- diagnosis_mode: `train_head_only`
- pseudo_beta: 0.2
- seeds: [0,1,2,3,4]

5-seed aggregate:

| metric | mean±std |
|---|---:|
| final AUC | 0.6649±0.0826 |
| final AP | 0.1536±0.0459 |
| best_test_auc | 0.7514±0.0243 |
| best_test_ap | 0.2566±0.0590 |
| emb_drift_mean | 0.0000±0.0000 |
| pseudo_auc | 0.5154±0.0002 |
| pseudo_ap | 0.5130±0.0005 |

Per-seed final / best-test:

| seed | final AUC | final AP | best_test_auc | best_test_ap |
|---:|---:|---:|---:|---:|
| 0 | 0.5437 | 0.0981 | 0.7656 | 0.2836 |
| 1 | 0.5945 | 0.1096 | 0.7341 | 0.1890 |
| 2 | 0.7530 | 0.2160 | 0.7530 | 0.2160 |
| 3 | 0.6900 | 0.1510 | 0.7172 | 0.2373 |
| 4 | 0.7430 | 0.1936 | 0.7870 | 0.3571 |

Interpretation:

1. 5-seed head-only validation confirms that freezing the reference encoder substantially improves over full end-to-end training on Elliptic.
2. Compared with the previous full train_all formal sweep (`best_test_auc=0.6836±0.0229`, `best_test_ap=0.1530±0.0149`), head-only reaches `best_test_auc=0.7514±0.0243`, `best_test_ap=0.2566±0.0590`.
3. Deterministic drift logging now reports `emb_drift_mean=0`, confirming that the frozen geometry is actually fixed in this run.
4. `pseudo_auc≈0.515` remains near random, suggesting that the BCE pseudo anomaly task itself is weak; the gain mainly comes from preserving useful initial reference geometry and training only the scoring head.
5. Final epoch performance is still lower than best-test for several seeds, so early stopping / validation selection remains important.

Next recommended experiments:

- Repeat head-only with larger `encode_batch_size` (e.g. 256, then 512 if memory permits) for speed.
- Compare against a deterministic cached-embedding head-only implementation only if speed remains problematic.
- Investigate why final epoch still degrades relative to best-test even when geometry is frozen; likely head overfitting / BCE target mismatch rather than encoder drift.
