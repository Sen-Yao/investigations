# Diagnostic Plan

## D0: Reproduce baseline curve

**目的**：确认退化曲线稳定存在。

配置：

```text
dataset = elliptic
objective_mode = target_ref_guided
pseudo_beta = 0.2
seed = 0
num_epoch = 200
```

记录：

- epoch-wise AUC/AP；
- best_val / best_test；
- loss；
- score distribution statistics。

## D1: Freeze encoder

**问题**：退化是否来自 encoder/reference geometry 被更新破坏？

Variants：

| variant | encoder | reference/token module | classifier/head |
|---|---|---|---|
| `train_all` | train | train | train |
| `freeze_encoder` | frozen | frozen or fixed-output | train |
| `train_head_only` | frozen | frozen | train only head |

判定：

- 若 frozen variants 不退化：退化主要来自 representation drift；
- 若 frozen variants 也退化：退化主要来自 BCE/head target mismatch。

## D2: Geometry drift tracking

每 10 epoch 记录：

- embedding drift norm: `||emb_t - emb_0||`；
- normal/anomaly centroid drift；
- reference score separation: `score_ref(anomaly) - score_ref(normal)`；
- model score separation；
- top-k anomaly density。

目标：判断训练是否破坏原始 reference geometry。

## D3: Rank correlation tracking

每 10 epoch 计算：

```text
Spearman(model_score_t, label)
Spearman(reference_score_0, label)
Spearman(model_score_t, reference_score_0)
Spearman(model_score_t, model_score_0)
```

关键判据：

```text
reference_score_0 vs label 高，
但 model_score_t vs reference_score_0 逐渐下降
```

则说明训练偏离原始有效 geometry。

## D4: Pseudo-real alignment

分析 pseudo direction 是否与真实异常方向一致：

```python
direction_pseudo_i = normalize(mean(R_a(i)) - mean(R_n(i)))
direction_real_i = normalize(emb[nearest_anomaly] - emb[i])  # diagnostic only, not training
cos_alignment = cosine(direction_pseudo_i, direction_real_i)
```

注意：真实异常标签只用于诊断，不能用于训练。

记录：

- alignment mean/std；
- alignment vs model score change；
- alignment vs anomaly hit rate。

## D5: Pseudo classification vs real anomaly ranking

同时记录：

- pseudo BCE accuracy/AUC；
- real test AUC/AP；
- pseudo score margin；
- real score margin。

若 pseudo separability 上升但 real ranking 下降，则支持 H1。

## D6: Seed sensitivity

先用 seed=0 快速诊断，再挑 seed=1/3 做复核。

不做 5-seed formal sweep，除非诊断发现明确机制并需要正式验证。

## Minimal execution order

建议优先顺序：

1. D0 baseline curve；
2. D1 freeze/head-only variants；
3. D3 rank correlation tracking；
4. D2 geometry drift；
5. D5 pseudo-vs-real separability；
6. D4 pseudo-real alignment。

D4 涉及真实异常方向，只作为诊断解释，不进入训练目标。
