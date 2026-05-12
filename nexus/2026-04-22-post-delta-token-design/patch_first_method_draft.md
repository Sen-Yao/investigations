# Patch-First Method Draft

## 一、当前目标

在第一轮轻量验证之后，当前最值得推进的不是完整 unified multi-token Transformer，而是先把 Patch 路线收敛为一版真正可训练、可解释、可对接现有 VoxG 主干的方法草案。

本草案的目标是回答：

1. 第一版 Patch Token 应该选哪种定义？
2. Patch Token 应该如何构造？
3. Patch Token 如何与现有 Node Token 共同进入下游模型？
4. 第一版实验应该追求什么，不应该追求什么？

---

## 二、当前推荐路线：以 `P-C1: Local Ego-Patch Token` 为第一版主实现

虽然 Patch 方向当前已有两个优先候选：

- `P-C1: Local Ego-Patch Token`
- `P-C2: Representation Patch Token`

但从当前阶段目标来看，第一版应优先选择：

> **`P-C1: Local Ego-Patch Token`**

### 原因

#### 1. 更贴近 ANR 的原始现象
当前 Patch 路线最重要的现象锚点是 ANR，而 ANR 首先描述的是局部结构单元的异常聚集。因此第一版 Patch 最自然的对象应是：
- 目标节点
- 其 k-hop 或 ego-neighborhood
- 局部连接关系与局部表示分布

#### 2. 可解释性更强
`Local Ego-Patch` 更容易解释：
- 这个 patch 包含哪些节点
- patch 内部密度如何
- patch 内部一致性如何
- 节点与 patch 中心是否协调

相比之下，`Representation Patch Token` 更抽象，更像第二阶段优化对象。

#### 3. 更适合作为第一版模型输入
如果第一版目标是快速判断 Patch 是否能稳定帮助现有主干，那么 `Local Ego-Patch Token` 更适合直接接入。

---

## 三、Patch Token 的第一版定义

### 定义原则
第一版 Patch Token 不追求复杂 patch discovery，而采用：

> **以目标节点为中心的局部 ego-patch 摘要向量。**

### 候选 patch 范围
当前推荐的第一版 patch 范围为：
- `1-hop ego patch`
- 包含目标节点自身
- 包含其直接邻居

理由：
- 与当前轻量验证脚本一致
- 计算简单
- 可解释性强
- 足以承接 ANR 的局部聚集现象

### Patch Token 的特征组成
第一版 patch token 不一定是原始 patch 内所有节点逐个作为 token，而更适合作为：

> **一个 patch summary token**

当前建议包含以下摘要成分：

1. `patch_size`
2. `patch_density`
3. `patch_feature_mean`
4. `patch_feature_var`
5. `patch_internal_consistency`
6. `target_to_patch_center_alignment`
7. `patch_boundary_contrast`

也就是说，第一版更像：
- 局部结构统计
- 局部表示统计
- 目标节点与局部上下文的对齐关系

三者组合的 summary token。

---

## 四、第一版下游接法：`Node Token + Patch Token`

### 当前推荐方案
第一版不做复杂并行体系，而直接采用：

> **Node Token + Patch Token**

### 为什么不直接上多 Patch / 多尺度 / 多 token patch family
因为当前更重要的是先回答：
- Patch 是否真的有新增量？
- 它能否稳定帮助 node baseline？

如果一开始就引入：
- 多 patch token
- patch hierarchy
- patch-to-patch attention

会让结果过早耦合，失去解释性。

### 输入组织建议
第一版可以考虑两种实现：

#### 方案 A：并行双 token
输入序列包括：
- `Node Token`
- `Patch Token`

这是最符合 post-delta 叙事的方案，因为它明确把 Patch 当成独立 token family。

#### 方案 B：Patch 作为 node 增强向量
把 patch summary 直接拼接或融合到 node 表示上。

这更像工程上更稳的过渡方案，但语义上略弱于真正双 token family。

### 当前偏好
我当前推荐：

> **先做方案 A，必要时再保留方案 B 作为对照。**

因为如果 Patch 是 post-delta 的主方向，就应该尽量让它以独立 token family 的身份进入模型，而不是只是做 side feature。

---

## 五、第一版训练目标

第一版 Patch-first 方法不应该承担过多任务。当前更合理的目标是：

### 目标 1：验证 Patch 是否稳定优于 node-only
不是追 SOTA，而是先看它能否在关键数据集上稳定提供增益。

### 目标 2：验证 Patch 是否与 ANR 对齐
也就是结果的解释是否能回到：
- 局部异常聚集
- 局部结构单元异常
- patch 内部不协调

### 目标 3：验证 Patch 是否值得作为主 token family 保留
如果第一版 Patch-first 站住，后续 Relation / Prototype 才有更清晰的位置。

---

## 六、当前明确不做的事情

为了避免第一版目标发散，当前建议明确暂不做：

1. 不直接上完整 unified framework
2. 不引入 Prototype 作为主分支
3. 不先做复杂 multi-scale patch hierarchy
4. 不把 Relation 与 Patch 过早深耦合
5. 不以 SOTA 为第一轮唯一目标

---

## 七、Relation 在当前阶段的角色

虽然本草案是 Patch-first，但 Relation 不应被忽略。

当前更合理的做法是：
- 先把 Patch 作为主方法推进；
- Relation 作为第二阶段角色收紧对象；
- 后续决定它是并行 token、辅助支路，还是 residual explainer。

换句话说：

> **Patch 先站住，Relation 再定位。**

---

## 八、当前结论

如果要把 Patch-first 方法草案压缩成一句话，可以写成：

> **post-delta 路线的第一版正式方法，应以 `P-C1: Local Ego-Patch Token` 为核心，先用 `Node Token + Patch Token` 的最小双 token 结构验证 Patch 是否能作为真正的主 token family 站稳，再决定 Relation 与 Prototype 的系统角色。**
