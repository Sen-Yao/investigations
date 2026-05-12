# Post-Delta 三类 Token 的最小验证路线

## 一、目标

本文件定义新探究的最小验证路线。目标不是立即做完整模型，而是低成本判断三类新 token 是否真的值得进入 VoxG 的下一阶段方法设计。

三类 token：
- Community / Patch Token
- Prototype Assignment / Contrast Token
- Relation Token

---

## 二、验证原则

1. 先验证“是否真有新增视角”，再考虑完整训练
2. 优先做低成本统计 / probe / 结构分析
3. 每类 token 都要回答：它是否超出了 hop/delta 的重参数化范围

---

## 三、Community / Patch Token

### 最小验证 A
在已有数据集上构造局部 ego-patch，检查：
- anomaly patch 内部的一致性是否显著高于 normal patch
- anomaly patch 与其外部邻域的差异是否更大

### 最小验证 B
对 patch 级摘要特征做轻量 probe，比较：
- node-only feature
- patch-only feature
- node + patch feature

### 当前优先数据集
- Photo
- Amazon

原因：
- Photo 可检查“为何有小幅新增量”
- Amazon 可检查“社区型结构是否存在但未被 node-level token 利用”

---

## 四、Prototype Assignment / Contrast Token

### 最小验证 A
先在局部表示空间中构造 prototype，检查：
- anomaly 节点的 assignment 分布是否更集中或更偏离 normal prototype
- assignment margin 是否具有区分力

### 最小验证 B
比较以下轻量特征：
- mean prototype baseline
- assignment token feature
- contrast-to-normal feature

### 当前重点
重点不在“求均值 prototype”，而在“模式归属是否比均值统计更有新增量”。

---

## 五、Relation Token

### 最小验证 A
把 NDC 思路扩展成多种 relation feature，比较：
- node-to-neighborhood relation
- node-to-patch relation
- node-to-community relation

### 最小验证 B
检查 relation feature 是否比纯 node state feature 更能解释 anomaly residual。

### 当前重点
Relation token 应优先承接 NDC，而不是再回到 delta token 设计。

---

## 六、当前优先顺序

### 第一优先级
- Community / Patch Token
- Relation Token

### 第二优先级
- Prototype Assignment / Contrast Token

原因：
- Patch token 最直接承接 ANR
- Relation token 最直接承接 NDC
- Prototype 路线仍需防止退化为弱均值特征

---

## 七、下一步建议

如果继续推进，我建议优先写两份更细的设计文档：

1. **patch token 设计草案**
2. **relation token 设计草案**

因为这两条线最直接接住当前已有现象，也最有可能形成真正区别于 hop/delta 的新坐标系。
