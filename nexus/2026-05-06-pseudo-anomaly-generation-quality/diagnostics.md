# Diagnostic Plan: Pseudo Quality Audit

## D0: Reuse fixed geometry setup

初始使用已验证的 Elliptic setup：

```text
dataset = elliptic
objective_mode = target_ref_guided
diagnosis_mode = train_head_only / frozen geometry
pseudo_beta = 0.2
seed = 0 initially; then selected seeds if needed
```

优先复用 `2026-05-05-elliptic-training-degradation` 的诊断 runner，但新增 pseudo quality metrics。

## D1: Pseudo separability audit

回答：pseudo positives 是否形成有效二分类任务？

记录：

```text
pseudo_auc
pseudo_ap
pseudo_margin = mean(score_pseudo) - mean(score_normal)
normal-pseudo cosine
normal-pseudo L2 distance
logit distribution overlap
```

判定：

- 若 pseudo_auc 接近 0.5 且 margin 接近 0，说明 BCE positive class 弱；
- 若 pseudo 可分但 real AUC 不升，说明 pseudo task 与真实异常不对齐。

## D2: Pseudo-real manifold proximity

诊断用真实 label，不进入训练。

对每个 pseudo sample 计算：

```text
dist(pseudo, nearest real anomaly)
dist(pseudo, nearest normal)
ratio = dist_to_normal / dist_to_anomaly
kNN anomaly ratio around pseudo
```

判定：

- pseudo 更靠近 normal → generation 没有进入 anomaly manifold；
- pseudo 靠近 anomaly 但 real AUC 不升 → scoring/head 问题；
- pseudo 分散远离两者 → artifact positives。

## D3: Pseudo direction alignment

定义：

```python
direction_pseudo_i = normalize(mean(R_a(i)) - mean(R_n(i)))
direction_real_i = normalize(emb[nearest_real_anomaly(i)] - emb[i])
cos_alignment_i = cosine(direction_pseudo_i, direction_real_i)
```

诊断：

```text
mean/std cos_alignment
alignment vs real anomaly score
alignment vs pseudo score
alignment vs beta
```

注意：真实异常标签只用于诊断，不用于训练。

## D4: R_a multi-modality / mean collapse audit

对每个 target 的 `R_a` refs：

```text
pairwise cosine among R_a embeddings
variance of R_a embeddings
norm(mean(R_a)-mean(R_n))
cos(mean_direction, individual R_a_j - mean(R_n))
```

判定：

- R_a diversity 高 + mean direction weak → mean-direction pseudo 可能 collapse；
- individual directions 与 mean direction 分歧大 → 应考虑 multi-direction pseudo。

## D5: Beta-quality curve

不是 performance sweep，而是 quality sweep：

```text
beta ∈ {0.05, 0.1, 0.2, 0.4, 0.8}
```

记录：

```text
pseudo separability
pseudo-real proximity
pseudo-real alignment
real AUC/AP after head-only learning
```

关键判断：

```text
pseudo separability ↑ 是否伴随 real alignment ↑？
```

若不伴随，则 beta 只是增强 artifact。

## D6: Reference score baseline

不生成 pseudo samples，直接构造 reference anomaly scores：

```text
S1 = normal rejection
S2 = mean cos(target, R_a) - mean cos(target, R_n)
S3 = ||target - mean(R_n)||
S4 = ||target - mean(R_a)|| - ||target - mean(R_n)||
S5 = R_a anomaly-likeness / density score
```

比较：

```text
reference score AUC/AP
pseudo-BCE head AUC/AP
head-only best-test AUC/AP
```

若 reference score ≥ pseudo-BCE，则说明 sample generation 不是必要路径。

## Minimal execution order

1. D1 pseudo separability audit；
2. D2 pseudo-real proximity；
3. D4 mean collapse audit；
4. D6 reference score baseline；
5. D5 beta-quality curve；
6. D3 direction alignment refined analysis。

优先 seed=0 快速诊断；发现清晰信号后再扩展到 5 seed。

## 2026-05-06 Constraint: prefer zero/few-hyperparameter strategies

User preference: subsequent exploration should introduce as few new hyperparameters as possible, ideally none.

Priority order:

1. **Direct reference scoring**: no generated pseudo samples, no beta/alpha.
2. **Use retrieved R_a representations as positives**: no displacement magnitude.
3. **Use mean(R_a) as positive prototype**: no displacement magnitude.
4. **Only if necessary**, test fixed deterministic constructions such as midpoint/mixup, but avoid sweeping alpha/beta at this stage.

This shifts the next audit from beta tuning to strategy comparison:

| strategy | extra hyperparameter? | description |
|---|---|---|
| `reference_score` | no | directly score target by relation to R_n/R_a |
| `ra_mean_positive` | no | positive = mean embedding of R_a refs |
| `ra_individual_positive` | no | positives = individual R_a ref embeddings |
| `local_displacement` | yes (`beta`) | current baseline only, not preferred |

Key criterion:

```text
Does the candidate positive set move closer to real anomaly manifold without adding tunable generation strength?
```
