# 最小验证路线设计

## 一、目标

本文件的目标，是把当前已经形成的 token design 草图进一步转成一条低成本、可执行、可逐步收敛的验证路线。

核心原则：

> 在方法框架未完全收敛前，不直接启动大规模训练；先用统计分析、轻量 probe 和小组合对比，验证新 token 是否真正携带互补信息。

---

## 二、当前最值得验证的两个新 token

结合前述分析，当前优先级最高的两个新 token 是：

1. **Delta Prototype Token**
2. **Local Consistency Token**

原因如下：

- Delta Prototype Token 最直接承接 NDC 现象
- Local Consistency Token 最直接承接 ANR 背后“局部一致异常环境”的无监督解释
- 二者都可以在不改大模型的前提下先做轻量统计验证

---

## 三、验证总路线

建议把验证拆成三层，由浅入深推进：

### Level 1：统计可分性验证
不训练复杂模型，只检查新 token 对应的统计量是否能区分 normal / anomaly。

### Level 2：轻量 probe 验证
使用简单分类器（如 logistic regression）验证这些 token 是否携带互补信息。

### Level 3：小规模组合验证
只在小规模 setting 下比较 token family 组合，而不是直接上完整 sweep。

---

## 四、Level 1：统计可分性验证

### 4.1 验证 A：Delta Prototype 是否有区分力

**目标**：检验邻域平均 Delta 模式本身是否对异常有区分信息。

**候选统计量**：
- node delta 与 delta prototype 的相关性
- node delta 与 delta prototype 的欧氏距离
- node delta 与 delta prototype 的 cosine similarity
- delta prototype 的范数 / 方差

**重点问题**：
- anomaly 与 normal 在这些统计量上的分布是否显著不同？
- 这些统计量是否和现有 NDC 一致，还是提供新增信息？

**预期意义**：
如果这些量可分，说明 Delta Prototype Token 不是冗余构件，而是可独立承载异常相关信号。

---

### 4.2 验证 B：Local Consistency 是否有区分力

**目标**：检验局部一致性 proxy 是否能区分异常节点与正常节点。

**候选统计量**：
- 邻域 delta 的平均两两相似度
- 邻域 delta 的方差 / dispersion
- node delta 到 neighborhood prototype 的偏差
- neighborhood 内部一致性分数

**重点问题**：
- anomaly 是否更处于高一致性邻域？
- 这个现象是否跨数据集稳定？

**预期意义**：
如果一致性 proxy 可分，则说明 ANR 所揭示的社区型异常现象可以部分转译为无监督可计算的表示信号。

---

## 五、Level 2：轻量 probe 验证

### 5.1 Probe 目标

统计量可分，并不自动意味着 token 真能帮助模型。因此第二层验证应检查：

> 把这些 token 或其对应特征送入轻量分类器时，是否能比现有 token 提供增益？

### 5.2 推荐 probe 设置

| Probe 输入 | 目的 |
|-----------|------|
| Hop Token flatten | 作为基础对照 |
| Delta Token flatten | 当前最强基线之一 |
| Delta Prototype features | 检查 prototype 单独效果 |
| Local Consistency features | 检查一致性 proxy 单独效果 |
| Delta + Prototype | 检查 prototype 是否补足 delta |
| Delta + Consistency | 检查 consistency 是否补足 delta |
| Delta + Prototype + Consistency | 当前最小 multi-token proxy 组合 |

**模型建议**：
- Logistic Regression
- Linear SVM（可选）

**原因**：
- 轻量、可解释
- 便于看 token 本身的信息含量
- 避免把结果混到大模型能力里

---

## 六、Level 3：小规模组合验证

当前不建议直接做 full-scale VoxG 训练，而建议先设计几个小组合：

### 组合 1：Hop vs Delta
**目的**：继续作为已有基线对照

### 组合 2：Delta vs Delta + Prototype
**目的**：验证 prototype 是否提供增益

### 组合 3：Delta vs Delta + Consistency
**目的**：验证 consistency proxy 是否提供增益

### 组合 4：Delta vs Delta + Prototype + Consistency
**目的**：验证 anomaly-aware multi-token 最小框架是否优于单一 delta token

### 组合 5：Hop + Delta vs Hop + Delta + Prototype + Consistency
**目的**：看多 token family 是否形成系统性增益

---

## 七、数据集选择建议

### 第一阶段建议优先用：Photo
原因：
- 当前已有较完整 NDC / ANR 结果
- 现象最强烈
- 便于快速验证 token 设计是否有效

### 第二阶段建议扩展到：Amazon / Tolokers / Elliptic
原因：
- 可以检查跨数据集稳定性
- 避免只在 Photo 上成立

### 当前不建议优先：Reddit
原因：
- 现象相对弱
- 目前 ANR 验证不完整
- 容易增加不必要噪声

---

## 八、执行顺序建议

### Step 1
先写统计分析脚本，计算：
- delta prototype
- consistency proxy
- 相关统计量

### Step 2
在 Photo 上做分布比较与显著性检验

### Step 3
若结果支持，再做轻量 probe

### Step 4
若 probe 有增益，再决定是否进入小规模组合训练

---

## 九、当前最小可执行任务（MVP）

如果只做一个最小闭环，我建议是：

### MVP
在 Photo 数据集上，围绕 Delta Token 做两件事：

1. 构造 **Delta Prototype Token 对应特征**
2. 构造 **Local Consistency Token 对应特征**

然后完成：
- 分布统计
- KS 检验
- 与现有 Delta flatten 做轻量 logistic probe 对比

这个 MVP 的价值在于：
- 不需要重训练大模型
- 成本低
- 直接检验 NDC / ANR 是否能转成 token-level feature gain

---

## 十、当前结论

最合理的推进方式不是“立刻上大实验”，而是：

> **先把 NDC / ANR 现象转成可计算的 prototype / consistency 特征，再验证这些特征是否具有额外区分力。**

如果这一步成立，我们再进入 anomaly-aware multi-token 的更完整模型设计，会更稳，也更容易解释。
