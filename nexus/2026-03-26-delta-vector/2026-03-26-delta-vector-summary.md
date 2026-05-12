# Delta Vector 实验总结 - 2026-03-26

## 核心发现

### 1. Delta 向量 > Delta 范数（理论验证）

| 检测方法 | Photo AUC | Elliptic AUC |
|----------|-----------|--------------|
| Delta 范数 | 0.31~0.44 | - |
| **Delta 向量 (LR)** | **0.9990** | - |

**结论**：Delta 向量包含丰富的异常信息，但 Delta 范数几乎无用。

### 2. Delta 模式不能单独使用

| 模式 | 输入 | Elliptic AUC | 问题 |
|------|------|--------------|------|
| original | 7 tokens | 0.7191 | baseline |
| **delta** | 6 delta | **0.5346** | ❌ 丢失节点特征 |
| concat | 13 tokens | 0.6689 | 容量不足 |
| concat_enhanced | 13 tokens, dim=512 | **0.7552** | ✅ 最佳 |

**结论**：Delta 向量必须与原始 Token 结合使用。

### 3. 模型容量是关键

| embedding_dim | Elliptic AUC | 问题 |
|---------------|--------------|------|
| 256 | 0.6689 | 容量不足 |
| **512** | **0.7552** | ✅ 最佳 |
| 768 | 0.6945 | 过拟合 |

**结论**：concat 模式需要增加模型容量。

### 4. Delta 尺度差异问题（新发现）

| 层 | 平均绝对值 | 问题 |
|----|-----------|------|
| Delta_1 | 6.9e-2 | - |
| Delta_6 | 6.1e-7 | **小 10^5 倍** |

**影响**：Transformer 无法有效学习尺度差异如此大的输入。

**解决方案**：对 Delta 进行归一化。

### 5. 归一化验证

| 方法 | Photo AUC (LR) |
|------|----------------|
| Original Token | 0.9502 |
| **Concat + Delta 归一化** | **0.9560** ✅ |

**结论**：归一化后的 concat 有轻微提升。

---

## 改进建议

### 短期（代码修改）

1. **添加 Delta 归一化选项**
   ```python
   delta_tokens = nodes_features[:, 1:] - nodes_features[:, :-1]
   delta_normalized = F.layer_norm(delta_tokens, (D,))
   ```

2. **使用 concat_enhanced 配置**
   - embedding_dim: 512
   - GT_num_heads: 4
   - epoch: 200

### 中期（架构改进）

1. **加权 Delta**：学习每层 Delta 的重要性
2. **正交化 Delta**：移除与原始 Token 的冗余信息

### 长期（研究方向）

1. **双分支架构**：原始和 Delta 分别处理
2. **自适应融合**：根据数据集特性自动调整融合策略

---

## 数据集建议

| 数据集特点 | 推荐配置 |
|-----------|---------|
| 密集图 + 高维特征 (Photo) | 大 batch, 小 pp_k, 增大 alpha |
| 稀疏图 + 低维特征 (Elliptic) | 标准 concat_enhanced |
| 中等图 (Amazon, Reddit) | 标准 concat_enhanced |

---

_总结时间: 2026-03-26_
