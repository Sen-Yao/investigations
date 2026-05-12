# Hypotheses

## H1: BCE objective 优化 synthetic separability，而不是 real anomaly ranking

当前训练标签来自：

```text
normal training nodes -> label 0
pseudo anomaly nodes  -> label 1
```

pseudo anomaly 来自：

```python
pseudo_i = emb[i] + beta * normalize(mean(R_a(i)) - mean(R_n(i)))
```

模型可能学到的是区分 synthetic perturbation artifact，而不是保留或放大真实异常排序。

**可观测证据**：

- training loss 下降或保持稳定；
- pseudo classification 变好；
- 但 real test AUC/AP 下降；
- model score 与 original reference score 的 Spearman correlation 随 epoch 下降。

## H2: Encoder / reference geometry 被训练更新破坏

epoch 0 的 embedding/reference geometry 已有较好异常排序。训练过程中 encoder 或 token/reference module 的参数更新可能扭曲该 geometry。

**可观测证据**：

- `freeze_encoder` 后不退化或退化显著减轻；
- embedding drift norm 随 epoch 增大；
- reference score separation 随 epoch 下降。

## H3: Classifier/head 本身不适配 reference geometry

即使冻结 encoder，只训练 classifier/head，也可能把有效的 reference score 映射成不稳定 decision boundary。

**可观测证据**：

- `freeze_encoder` 仍退化；
- `train_head_only` 仍退化；
- model score 与 reference score correlation 下降，但 embedding drift 小。

## H4: pseudo anomaly direction 与真实异常方向覆盖不一致

`R_a(i)-R_n(i)` 对部分异常形态有效，但 Elliptic 的真实异常可能具有 temporal/transactional heterogeneity。训练后模型偏向 synthetic direction，降低对真实异常整体分布的 ranking。

**可观测证据**：

- pseudo-real anomaly cosine alignment 不稳定或下降；
- 不同 anomaly subgroup / degree / temporal slice 的 score drift 不一致；
- AP 比 AUC 更敏感地下滑。

## H5: 训练信号过弱但足以扰乱 ranking

日志中 loss 长期接近 `0.693`，说明 BCE 信号弱；但小幅参数扰动足以让 AUC 大幅波动。

**可观测证据**：

- loss 变化极小；
- score distribution shift 很小但 rank order 大幅变化；
- Spearman/Kendall rank correlation 比均值差异更敏感。
