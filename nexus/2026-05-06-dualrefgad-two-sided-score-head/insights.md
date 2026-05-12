# Insights: DualRefGAD Two-Sided Score Head

## Scope

This investigation studies how fixed dual-reference evidence should be read out into an anomaly score after frozen GT/VecGAD encoding.

Fixed evidence semantics:

- `R_n(v)`: normal-side evidence; answers "if v is normal, what normal references should it agree with?"
- `R_a(v)`: deviation-side evidence; answers "if v deviates from normality, what deviation-side evidence is structurally relevant?"
- `R_a` is **not** an anomaly pseudo-label.
- The GT encoder remains frozen in this investigation; only the score/judgment head is trained.

## Sweeps

### 1. Full head comparison sweep

- Job: `exp_20260506_190921_dualrefgad_structured_head_5seed`
- Sweep: `v6rq0rop`
- WandB: https://wandb.ai/HCCS/VoxG/sweeps/v6rq0rop
- Intended grid: 4 head modes × 5 seeds × 2 epoch settings = 40 runs
- Outcome: 33 finished / 7 failed
- Failure reason: GPU resource/CUBLAS allocation failure, not an identified method-level bug.

Error observed:

```text
RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`
```

Partial aggregation from valid/completed metrics:

| head_mode | n | AUC | AP |
|---|---:|---:|---:|
| `scalar_mlp_baseline` | 10 | 0.6599±0.0379 | 0.1363±0.0167 |
| `structured_readout` | 10 | 0.6874±0.0268 | 0.1508±0.0172 |
| `decomposition_head` | 10 | 0.6119±0.1679 | 0.1372±0.0413 |
| `decomposition_split_mismatch` | 10 | 0.5945±0.1672 | 0.1305±0.0414 |

The decomposition numbers above were judged unreliable because failed/early-stopped runs polluted the statistics.

### 2. Decomposition-only rerun

- Job: `exp_20260506_194824_dualrefgad_decomposition_head_rerun_5see`
- Sweep: `x1af56l1`
- WandB: https://wandb.ai/HCCS/VoxG/sweeps/x1af56l1
- Grid: 2 head modes × 5 seeds × 2 epoch settings = 20 runs
- Concurrency limit: `--max-agents 3`
- Outcome: 20 finished / 0 failed

Final aggregation:

| head_mode | n | AUC | AP |
|---|---:|---:|---:|
| `decomposition_head` | 10 | 0.7008±0.0309 | 0.1616±0.0222 |
| `decomposition_split_mismatch` | 10 | 0.6952±0.0262 | 0.1558±0.0183 |

## Main conclusions

### 1. Generic pooled scalar MLP is insufficient

`scalar_mlp_baseline` is substantially weaker than relation-aware heads:

```text
scalar_mlp_baseline: 0.6599±0.0379 AUC
structured_readout:  0.6874±0.0268 AUC
decomposition_head:  0.7008±0.0309 AUC
```

This supports the design principle that DualRefGAD should preserve target / normal-reference / deviation-reference relation structure in the score head.

### 2. Decomposition is useful, but not yet a decisive breakthrough

The stable rerun shows:

```text
decomposition_head:           0.7008±0.0309 AUC
decomposition_split_mismatch: 0.6952±0.0262 AUC
structured_readout:           0.6874±0.0268 AUC
```

Thus decomposition improves over structured readout, but the margin is modest. Split mismatch did not outperform plain decomposition after full 10-run completion.

### 3. The previous apparent 0.75+ signal is not the same as a stable method-level mean

Several individual runs have `best_test_auc > 0.75`, especially in the decomposition rerun, e.g. individual seeds around:

```text
0.7519, 0.7543, 0.7571, 0.7621
```

However, these are **individual best-checkpoint / best-val-aligned test values**, not the 5-seed method-level mean. They cannot be reported as the method result.

The stable comparison should use grouped 5-seed/10-run aggregation:

```text
decomposition_head: 0.7008±0.0309
```

or, if comparing best-checkpoint behavior separately, it must be explicitly labeled as an oracle/validation-selected analysis.

### 4. Real deployment has no validation labels

The user pointed out an important constraint: real anomaly detection deployment does not have validation labels.

Therefore, future training strategy should not depend on labeled validation early stopping as a core method. Validation can be used only for offline diagnosis, not as an algorithmic requirement.

Implication:

- Do not build DualRefGAD around supervised validation early stopping.
- Prefer unsupervised or label-free stopping criteria, e.g. score stability, consistency loss plateau, reference-relation margin saturation, or fixed training budget selected from validation-free diagnostics.
- Report validation-selected results separately from deployable fixed-budget results.

## Current recommendation

Next step should not be full trainable GT yet. First improve/diagnose the deployable head training protocol under no-validation assumptions:

1. Compare fixed-budget epochs rather than best-val selection.
2. Diagnose final-vs-best gap without using validation as a method dependency.
3. Study unsupervised stopping proxies such as score distribution stability and relation-margin saturation.
4. Only after head training becomes stable, move to projection-only or last-layer GT training.

## 2026-05-06 update: keep Elliptic as the main head-design battlefield

User decision:

- Continue head design on **Elliptic** rather than switching the main line to Photo.
- Photo may remain useful as a future auxiliary sanity-check dataset, but it should not replace Elliptic.

Rationale:

- Elliptic exposes the hard regime where VecGAD previously degraded.
- The current DualRefGAD question is specifically whether fixed dual-reference evidence plus a relation-aware head can handle this difficult setting.
- Moving too early to Photo may reduce iteration cost, but risks optimizing for an easier dataset and losing the specific stress-test signal that motivated DualRefGAD.

Therefore, the next stage remains Elliptic-first.

## Hard protocol constraint: no validation set

User clarified the deployable protocol:



Consequences:

-  / validation-selected  may be used only as offline diagnostic evidence from past runs.
- Future method comparisons should emphasize fixed-budget final metrics under 5/95.
- Head design should be evaluated by deployable behavior, not by oracle checkpoint selection.

## Next head-design questions on Elliptic

The current stable ranking is:



This suggests:

1. Generic pooled scalar MLP is insufficient.
2. Relation-aware readout is necessary.
3. A compact two-sided decomposition head is the current best candidate.
4. More complex split-mismatch supervision has not justified its extra complexity.

Next stage should therefore study the **minimal deployable decomposition head**.

## Proposed next-stage ablations

Under Elliptic-only, 5/95-only, no-validation protocol:

| variant | purpose |
|---|---|
|  | current best stable reference |
|  | test whether normal-side auxiliary compatibility is necessary |
|  | test whether deviation-side auxiliary support is necessary |
|  | test whether  should feed final or only regularize |
|  | relation-aware non-decomposed baseline |

Primary report metrics:

- final test AUC/AP under fixed epoch budget
- mean±std over seeds 
- no validation-selected metrics as main result
- train consistency AUC/AP as auxiliary diagnostic only
- score distribution stability (, normal/test score means)
- sub-logit stability (, ) for decomposition variants

Recommended first fixed budget:



Reason: previous 100-epoch settings often showed final-score drift; without validation labels, a shorter fixed budget is more deployable and less exposed to over-training.


## 2026-05-06 update: keep Elliptic as the main head-design battlefield

User decision:

- Continue head design on **Elliptic** rather than switching the main line to Photo.
- Photo may remain useful as a future auxiliary sanity-check dataset, but it should not replace Elliptic.

Rationale:

- Elliptic exposes the hard regime where VecGAD previously degraded.
- The current DualRefGAD question is specifically whether fixed dual-reference evidence plus a relation-aware head can handle this difficult setting.
- Moving too early to Photo may reduce iteration cost, but risks optimizing for an easier dataset and losing the specific stress-test signal that motivated DualRefGAD.

Therefore, the next stage remains Elliptic-first.

## Hard protocol constraint: no validation set

User clarified the deployable protocol:

```text
Only 5/95 split is allowed.
No validation set is allowed.
No validation-based early stopping or checkpoint selection.
```

Consequences:

- `best_val_auc` / validation-selected `best_test_auc` may be used only as offline diagnostic evidence from past runs.
- Future method comparisons should emphasize fixed-budget final metrics under 5/95.
- Head design should be evaluated by deployable behavior, not by oracle checkpoint selection.

## Next head-design questions on Elliptic

The current stable ranking is:

```text
decomposition_head           0.7008±0.0309 AUC
decomposition_split_mismatch 0.6952±0.0262 AUC
structured_readout           0.6874±0.0268 AUC
scalar_mlp_baseline          0.6599±0.0379 AUC
```

This suggests:

1. Generic pooled scalar MLP is insufficient.
2. Relation-aware readout is necessary.
3. A compact two-sided decomposition head is the current best candidate.
4. More complex split-mismatch supervision has not justified its extra complexity.

Next stage should therefore study the **minimal deployable decomposition head**.

## Proposed next-stage ablations

Under Elliptic-only, 5/95-only, no-validation protocol:

| variant | purpose |
|---|---|
| `decomposition_head` | current best stable reference |
| `decomposition_no_sn_aux` | test whether normal-side auxiliary compatibility is necessary |
| `decomposition_no_sa_aux` | test whether deviation-side auxiliary support is necessary |
| `decomposition_no_sublogit_to_final` | test whether `s_n/s_a` should feed final or only regularize |
| `structured_readout` | relation-aware non-decomposed baseline |

Primary report metrics:

- final test AUC/AP under fixed epoch budget
- mean±std over seeds `[0,1,2,3,4]`
- no validation-selected metrics as main result
- train consistency AUC/AP as auxiliary diagnostic only
- score distribution stability (`score_std`, normal/test score means)
- sub-logit stability (`sn_std`, `sa_std`) for decomposition variants

Recommended first fixed budget:

```text
num_epoch = 50
```

Reason: previous 100-epoch settings often showed final-score drift; without validation labels, a shorter fixed budget is more deployable and less exposed to over-training.

---

## 2026-05-07 Update: Dual Margin Breakthrough

### Key Finding

**lambda-free normalized margin two-score head** 在 Elliptic 5/95 no-validation protocol 下显著优于之前所有 head：

| head | final_test_auc | final_test_ap |
|---|---:|---:|
| dual_margin_two_score | **0.7429±0.0164** | **0.1983±0.0210** |
| contrastive_two_score | 0.7148±0.0210 | 0.1722±0.0224 |

**提升**: AUC +0.0281, AP +0.0261, std 更小

---

### Head Form



其中：



无 lambda、无 aux loss、无 validation selection。

---

### 为什么有效

1. VecGAD 已验证 deviation direction 有信号
2. Normalized margin ∈ [-1,1] 自动匹配 s_n/s_a 尺度
3. Dual-reference geometry 被 head 显式读出

---

### 意义

- R_a - R_n direction 不是 noise
- Margin term 不应该被删
- DualRefGAD 理论方向正确

---

### Next

- 更强 margin 形式
- 跨数据集验证
- 不走 aux ablation 路线

---

详细记录见：
