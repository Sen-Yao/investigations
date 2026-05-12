# 探索结晶

> **用途**: 记录探索流程和关键结果，供后续探索复用

---

## 📋 探究标识

| 项目 | 内容 |
|------|------|
| **探究目录** | `2026-04-01-offset-anomaly-relation/` |
| **关键词** | Offset, KS检验, 异常检测, alpha=0 |
| **数据集** | Photo (N=7535, D=745) |
| **方法** | Hop聚合 + Offset计算 + KS检验 |

---

## 🔍 探究问题

**核心问题**: Offset Token 的信息内容是否有助于异常检测？

**子问题**:
1. Offset 范数能否区分正常/异常节点？
2. Offset 方向能否区分正常/异常节点？

---

## 🧪 探究方法

### 方法步骤

1. 加载数据集 Photo.mat
2. 计算 Hop 特征 (alpha=0, 对称归一化)
3. 计算 Offset = hop_k - hop_0
4. KS 检验对比正/异常节点

### 关键代码片段

```python
# 对称归一化 D^{-0.5}AD^{-0.5}
degree = adj.sum(axis=1)
d_inv_sqrt = np.power(degree, -0.5)
adj_norm = np.diag(d_inv_sqrt) @ adj @ np.diag(d_inv_sqrt)

# alpha=0 的 Hop 聚合
for k in range(K):
    X = adj_norm @ X  # 纯邻居聚合
```

### 参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| alpha | 0 | 纯邻居聚合 |
| pp_k | 6 | Hop数量 |
| 归一化 | 对称归一化 | D^{-0.5}AD^{-0.5} |

---

## 🎯 关键结果

| 发现 | 数值 | 说明 |
|------|------|------|
| Offset 范数 KS | 0.08-0.12 | 正常>异常，区分有限 |
| Offset 方向 KS | 0.07-0.08 | 正常有正一致性 |

### 结论

**Offset 单独使用对异常检测帮助有限**

---

## 💡 可复用的洞见

| 启示 | 适用场景 |
|------|---------|
| Offset 单独效果有限 | 设计token策略时 |
| 正常节点Offset更大 | 反映活跃度差异 |

### 可复用脚本

| 脚本 | 用途 |
|------|------|
| analyze_offset_auc.py | Offset 范数分析 |
| analyze_offset_direction.py | Offset 方向分析 |

---

## 🔗 相关探究

| 探究 | 关系 |
|------|------|
| 2026-03-31-offset-information-theory | 前置探究 |

---

## ❓ 遗留问题

| 问题 | 建议 |
|------|------|
| Offset+Delta组合 | 探索组合策略 |

---

## 📊 时间投入

预估60分钟，实际5分钟，节省92%

---

_创建时间: 2026-04-01_