# Next Stage Plan: Elliptic Head Design under 5/95 No-Validation Protocol

## Decision

Continue on Elliptic as the main dataset.

Photo may be used later for auxiliary sanity checks, but the next stage stays on Elliptic because Elliptic is the stress-test dataset where VecGAD previously degraded.

## Hard constraint

```text
No validation set.
Only 5/95 split.
No validation early stopping.
No validation-selected checkpoint reporting as main result.
```

## Current reliable head ranking

| head | AUC | AP |
|---|---:|---:|
| `decomposition_head` | **0.7008±0.0309** | **0.1616±0.0222** |
| `decomposition_split_mismatch` | 0.6952±0.0262 | 0.1558±0.0183 |
| `structured_readout` | 0.6874±0.0268 | 0.1508±0.0172 |
| `scalar_mlp_baseline` | 0.6599±0.0379 | 0.1363±0.0167 |

## Next experimental target

Find the minimal deployable relation-aware/decomposition head.

## Proposed ablation matrix

| variant | question |
|---|---|
| `decomposition_head` | reference best head |
| `decomposition_no_sn_aux` | is normal-side auxiliary signal necessary? |
| `decomposition_no_sa_aux` | is deviation-side auxiliary signal necessary? |
| `decomposition_no_sublogit_to_final` | should `s_n/s_a` feed final, or only regularize? |
| `structured_readout` | non-decomposed relation baseline |

## Protocol

```text
dataset: elliptic
split: 5/95
seed: [0,1,2,3,4]
num_epoch: 50 fixed
validation: none
```

Total if run as one grid:

```text
5 variants × 5 seeds = 25 runs
```

## Main reporting rule

Report only fixed-budget final metrics as primary:

```text
final_test_auc mean±std
final_test_ap mean±std
```

`best_val_auc` and validation-selected metrics should not be used in the main table.
