# Delta 归一化问题 - 2026-03-26

## 问题发现

在分析 Photo 数据集时发现，Delta 向量的值在不同层之间差异巨大：

| 层 | 平均绝对值 | 与 Delta_1 比值 |
|----|-----------|----------------|
| Delta_1 | 6.9e-2 | 1.0 |
| Delta_2 | 3.1e-3 | 0.045 |
| Delta_3 | 2.7e-4 | 0.004 |
| Delta_4 | 3.1e-5 | 0.0004 |
| Delta_5 | 4.1e-6 | 0.00006 |
| Delta_6 | 6.1e-7 | 0.000009 |

**问题**：Delta_6 比 Delta_1 小了 **10^5 倍**！

## 影响

1. **Transformer 无法学习**：尺度差异太大，后面的 Delta 层几乎被忽略
2. **concat 模式效果差**：原始 Token (~1.0) 与 Delta (~0.0000006) 拼接，Delta 信息丢失

## 解决方案

### 方案 1：层归一化

```python
# 对每个 Delta 层单独归一化
delta_tokens = nodes_features[:, 1:] - nodes_features[:, :-1]  # [N, K, D]
# LayerNorm on last dimension
delta_normalized = F.layer_norm(delta_tokens, (D,))
```

### 方案 2：全局归一化

```python
# 对所有 Delta 整体归一化
delta_tokens = nodes_features[:, 1:] - nodes_features[:, :-1]
delta_mean = delta_tokens.mean()
delta_std = delta_tokens.std()
delta_normalized = (delta_tokens - delta_mean) / delta_std
```

### 方案 3：加权 Delta

```python
# 根据层的重要性学习权重
delta_weights = nn.Parameter(torch.ones(K))
delta_weighted = delta_tokens * delta_weights.view(1, K, 1)
```

## 验证方法

1. 在 Photo 数据集上测试归一化后的效果
2. 对比 concat 原始 vs concat 归一化
3. 分析不同归一化策略的效果

## 预期效果

归一化后，concat 模式应该能够：
1. 更好地利用 Delta 信息
2. 在 Photo 数据集上获得更好的效果
3. 减少对模型容量的依赖

---

_发现时间: 2026-03-26_
