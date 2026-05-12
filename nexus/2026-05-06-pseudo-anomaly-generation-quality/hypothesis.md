# Hypotheses

## H1: Current pseudo generation creates weak positives

当前 `normal + beta * normalize(mean(R_a)-mean(R_n))` 生成的 pseudo anomalies 与 normal embeddings 几乎不可分，因此 BCE 没有有效学习信号。

**已有证据**：

```text
head-only 5-seed pseudo_auc = 0.5154±0.0002
head-only 5-seed pseudo_ap  = 0.5130±0.0005
```

**预测**：

- pseudo-normal margin 接近 0；
- pseudo samples 的 nearest neighbors 中 normal 占比高；
- 增大 beta 可能提高 pseudo separability，但不一定提高 real anomaly alignment。

## H2: R_a/R_n relation is a scoring cue, not necessarily a generative direction

`R_a` 和 `R_n` 适合作为 reference/scoring relation，但不一定能定义真实异常的生成方向。

**核心区别**：

```text
reference cue: tells how target relates to normal/anomaly-like refs
pseudo generation: assumes normal can be moved along this direction into anomaly manifold
```

**预测**：

- reference score 本身可能比 pseudo BCE 更有效；
- pseudo direction 与 nearest real anomaly direction alignment 低；
- pseudo samples 不靠近真实异常流形。

## H3: Mean direction collapses multi-modal R_a information

当前使用：

```python
mean(R_a) - mean(R_n)
```

但 Elliptic 异常可能是多模态的。对 `R_a` 求均值可能导致方向抵消或 mode collapse。

**预测**：

- `R_a` pairwise diversity 高；
- individual directions `R_a_j - R_n` 方差大；
- mean direction norm 小或与 individual directions cosine 低；
- multi-direction pseudo 比 mean-direction pseudo 更合理。

## H4: Local displacement from normal is a poor model of Elliptic anomaly

Elliptic 异常可能是 transaction/temporal/local behavior anomaly，不是 normal embedding 的局部平移。

**预测**：

- pseudo samples 沿 local displacement 后仍更接近 normal manifold；
- real anomalies 与 pseudo samples 的 neighborhood overlap 低；
- anomaly subgroups 对同一 beta/direction 的响应差异大。

## H5: Stronger pseudo separability may not imply better real alignment

如果增大 beta 后 pseudo_auc 上升，但 real AUC/AP 不上升，说明生成策略只是制造了更容易区分的 artifact，而不是更真实的 anomaly positives。

**预测**：

| beta ↑ | pseudo_auc | real alignment | real AUC |
|---|---|---|---|
| artifact case | ↑ | - / ↓ | - / ↓ |
| useful case | ↑ | ↑ | ↑ |
