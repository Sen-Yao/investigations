# Implementation Notes: Structured Head Script

Script:

```text
experiments/scripts/fixed_dual_reference_gt_structured_head_train.py
```

Syntax check:

```text
python3 -m py_compile .../fixed_dual_reference_gt_structured_head_train.py
```

passed on HCCS86.

## Implemented head modes

| head_mode | description |
|---|---|
| `scalar_mlp_baseline` | VecGAD-like pooled target/reference embedding -> scalar normality logit |
| `structured_readout` | explicit target / normal / deviation relation features -> scalar normality logit |
| `decomposition_head` | normal-compatibility sub-logit + deviation-context support sub-logit + final normality logit |
| `decomposition_split_mismatch` | decomposition head trained with separate R_n / R_a mismatch scenes and auxiliary side-specific losses |

## Score semantics

The model outputs a normality/context-validity logit. The anomaly score is:

```text
anomaly_score = -normality_logit
```

`R_a` is treated as deviation-side evidence, not as an anomaly pseudo-label.

## Data leakage guard

The script keeps the semi-supervised red line:

```python
assert np.sum(labels_np[normal_idx]) == 0
```

Only labeled-normal nodes are used to construct training valid/mismatch contexts.

## Prepared sweep config

```text
experiments/configs/structured_head_mini_5seed.yaml
```

Grid:

```text
head_mode = [scalar_mlp_baseline, structured_readout, decomposition_head, decomposition_split_mismatch]
seed = [0,1,2,3,4]
num_epoch = [50,100]
```

Total: 40 runs.
