# Hypotheses

## H1: Reference geometry 可以驱动 pseudo anomaly synthesis

如果 pseudo anomaly 沿 reference-conditioned residual 方向生成，而非随机噪声或 self-residual 方向生成，则训练过程能放大 dual-reference tokenization 的初始可分性。

**已有证据**: Photo 上 global ref-guided 将 Test AUC 从 0.5799 提升到 0.7646，best epoch 从 0 移到 200。

## H2: Direction 必须是 target-specific，而非 global

全局 `mean(R_a)-mean(R_n)` 对 Photo 有效，但对 Elliptic 失败，说明不同目标节点的异常偏离方向可能异质。per-target direction 应更稳定：

```python
direction_i = normalize(mean(R_a(i)) - mean(R_n(i)))
```

## H3: VecGAD residual synthesis principle 可被吸收，但不应原样照搬多损失系统

VecGAD 的核心价值是 “normal → reconstruction residual direction → pseudo anomaly”，与 normal explanation failure 理论一致。但完整 VecGAD objective 包含 BCE + reconstruction + ring loss，会引入额外启发式和超参数。第一阶段只吸收 residual synthesis principle，保持单一 BCE。

## H4: 半监督安全性是硬约束

所有训练 pseudo anomaly 只能由 normal-only training nodes 生成。训练前必须断言：

```python
assert np.sum(labels[train_idx]) == 0
```
