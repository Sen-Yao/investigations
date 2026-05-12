# Relation Feature Spec（第一版）

## 一、目标

本文件定义 Relation Token 第一轮轻量验证所需的具体 relation-level 特征。目标是低成本验证：

> 节点与上下文之间的关系特征，是否比 delta 更自然地承接 NDC，并提供 node-wise baseline 之外的新增量。

---

## 二、默认上下文定义

第一轮优先使用两种上下文：

1. **Neighborhood context**：一阶邻居均值表示
2. **Patch context**：ego-patch 均值表示

---

## 三、建议提取的 Relation 特征

### R1. node_to_neigh_cos
目标节点与邻域均值表示余弦相似度。

### R2. node_to_neigh_l2
目标节点与邻域均值表示 L2 距离。

### R3. node_to_neigh_corr
目标节点与邻域均值表示相关系数。

### R4. node_to_patch_cos
目标节点与 patch 均值表示余弦相似度。

### R5. node_to_patch_l2
目标节点与 patch 均值表示 L2 距离。

### R6. neigh_internal_mean_pairwise_cos
邻域内部表示一致性。

### R7. patch_internal_mean_pairwise_cos
patch 内部表示一致性。

### R8. node_context_contrast_gap
节点与局部上下文一致性和局部内部一致性之间的差值。

---

## 四、第一轮重点

第一轮最重要的是覆盖三种 relation 语义：

1. **节点-邻域协调性**
2. **节点-patch 归属感**
3. **节点相对局部一致性的偏离**

最低可行子集建议是：
- node_to_neigh_cos
- node_to_neigh_l2
- node_to_patch_cos
- node_to_patch_l2
- neigh_internal_mean_pairwise_cos
- node_context_contrast_gap

---

## 五、第一轮验证方式

### 统计分析
比较 normal / anomaly 在 relation 特征上的分布差异。

### 残差分析
先用 node baseline 或 delta baseline，再检查 relation 特征是否解释 residual。

### 轻量 Probe
比较：
- node-only baseline
- relation-only features
- node + relation features

---

## 六、当前结论

这一版 relation feature spec 已足够支撑第一轮轻量验证脚本实现，并且与 NDC 的证据链保持直接衔接。
