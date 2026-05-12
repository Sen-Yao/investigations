# Patch Feature Spec（第一版）

## 一、目标

本文件定义 Patch Token 第一轮轻量验证所需的具体 patch-level 特征。目标不是立即训练完整 patch token 模型，而是先以低成本方式验证：

> Patch 级结构与一致性特征，是否真的携带超出 node-wise 表示的异常信号。

---

## 二、Patch 定义（第一版）

### 默认 Patch
采用 **Local Ego-Patch**：
- 以目标节点为中心
- 取 1-hop 邻域
- patch = {目标节点 + 一阶邻居}

这是最低复杂度的起点，适合作为第一轮 baseline。

---

## 三、建议提取的 Patch 特征

### P1. patch_size
patch 内节点数。

### P2. patch_density
patch 内部边密度。

### P3. patch_feature_mean_norm
patch 节点特征均值的范数。

### P4. patch_feature_var
patch 节点特征的平均方差。

### P5. patch_internal_mean_pairwise_cos
patch 内节点表示两两余弦相似度均值。

### P6. target_to_patch_center_cos
目标节点与 patch 均值表示的余弦相似度。

### P7. target_to_patch_center_l2
目标节点与 patch 均值表示的 L2 距离。

### P8. patch_boundary_contrast
patch 内均值表示与 patch 外一阶边界邻域均值表示之间的差异。

### P9. patch_internal_external_gap
patch 内部一致性与外部对比之间的 gap。

---

## 四、第一轮重点

第一轮最重要的不是特征数量，而是覆盖三种 patch 语义：

1. **局部规模 / 稠密性**
2. **内部一致性**
3. **内外对比**

所以最低可行子集建议是：
- patch_size
- patch_density
- patch_internal_mean_pairwise_cos
- target_to_patch_center_cos
- target_to_patch_center_l2
- patch_boundary_contrast

---

## 五、第一轮验证方式

### 统计分析
比较 normal / anomaly 在上述 patch 特征上的分布差异。

### 轻量 Probe
比较：
- node-only baseline
- patch-only features
- node + patch features

---

## 六、当前结论

如果 patch 路线要开始做实验，这一版 feature spec 已足够支撑第一轮轻量验证脚本实现。
