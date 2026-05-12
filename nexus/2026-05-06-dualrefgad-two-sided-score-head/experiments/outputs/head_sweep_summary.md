# Head Sweep Summary

## Completed sweeps

| purpose | job_id | sweep | status |
|---|---|---|---|
| full head comparison | `exp_20260506_190921_dualrefgad_structured_head_5seed` | `v6rq0rop` | 33/40 finished, 7 failed due CUBLAS alloc failure |
| decomposition rerun | `exp_20260506_194824_dualrefgad_decomposition_head_rerun_5see` | `x1af56l1` | 20/20 finished, 0 failed |

## Reliable aggregated results

| head_mode | AUC | AP | note |
|---|---:|---:|---|
| `scalar_mlp_baseline` | 0.6599±0.0379 | 0.1363±0.0167 | generic pooled scalar head |
| `structured_readout` | 0.6874±0.0268 | 0.1508±0.0172 | explicit relation features |
| `decomposition_head` | **0.7008±0.0309** | **0.1616±0.0222** | best stable head so far |
| `decomposition_split_mismatch` | 0.6952±0.0262 | 0.1558±0.0183 | useful but not better than plain decomposition |

## About the 0.75+ AUC observations

Some individual runs reached `best_test_auc > 0.75`, for example in the decomposition rerun. These values are not directly comparable to the stable mean because they are single-run best-checkpoint values selected by validation behavior.

Interpretation:

```text
0.75+ = possible per-seed/best-checkpoint signal
0.7008±0.0309 = stable grouped method-level result for decomposition_head
```

Because real deployment has no labeled validation set, validation-selected metrics should be treated as diagnostic/oracle evidence only.

## No-validation implication

Future deployable DualRefGAD should avoid relying on validation labels. Recommended next experiments:

- fixed training budget comparison
- unsupervised stopping proxies
- final-score stability diagnostics
- relation-margin saturation diagnostics
