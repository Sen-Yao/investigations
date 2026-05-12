# 自我审计报告

## 审计对象
- 双视图 Token 实验
- 分支: feature/dual-view-token
- 时间: 2026-03-27

---

## 发现的问题

### 🔴 严重问题

#### 1. ChannelWiseView 实现错误

**问题**：ChannelWiseView 的 forward 方法与 CrossChannelView 完全相同。

```python
# ChannelWiseView.forward() 中的代码
H = features
for k in range(self.num_hops):
    H = torch.matmul(adj, H)  # 整体传播
    tokens[:, k + 1, :] = H
```

**数学分析**：
$$A \cdot X = A \cdot [x_1, x_2, ..., x_D] = [A \cdot x_1, A \cdot x_2, ..., A \cdot x_D]$$

整体传播后按通道分离，**数学上等价于**每个通道单独传播。

**影响**：
- CrossChannelView 和 ChannelWiseView 输出相同
- 这解释了为什么"双视图融合"没有增益
- 实际上只有一个视图！

**修复建议**：
- 要实现真正的 Channel-wise view，需要每个通道用不同的传播矩阵
- 或者每个通道用不同的传播步数

---

### 🟡 潜在问题

#### 2. 马氏距离在高维时不稳定

**问题**：
- 当特征维度 D > 训练样本数 N 时，协方差矩阵不可逆
- 使用伪逆可能导致数值不稳定

**影响**：
- Delta concat (维度 745×6=4470) 时，AUC 反而下降到 0.716
- 可能是马氏距离在高维空间失效

**修复建议**：
- 使用降维（PCA）后再计算马氏距离
- 或使用其他异常检测方法（如 One-Class SVM）

#### 3. Delta last 的物理意义不明确

**问题**：
- Delta last = H^K - H^{K-1}（最后一跳的变化）
- 为什么这比 Delta mean 更好？

**分析**：
- Delta last 捕捉的是"最后阶段的传播效果"
- 可能与"收敛状态"有关

---

## 实验结论的可信度

| 结论 | 可信度 | 说明 |
|------|--------|------|
| Delta > Cross-channel | 🟡 中 | 代码正确，但需要更多验证 |
| Channel-wise 无增益 | 🔴 低 | 实现有误，结论不可靠 |
| 数据集特异性 | 🟢 高 | 多数据集验证，结论可靠 |
| Amazon 特殊性 | 🟢 高 | 数据集特性分析合理 |

---

## 下一步行动

1. **修复 ChannelWiseView 实现**
   - 实现真正的通道独立传播
   - 例如：每个通道用不同的传播步数

2. **改进评估方法**
   - 使用 PCA 降维后再计算马氏距离
   - 或使用 Transformer 分类器

3. **重新验证实验**
   - 修复后重新运行所有实验
   - 确保双视图真正不同

---

_审计时间: 2026-03-27 09:40_
_审计人: Nexus (自我审计)_