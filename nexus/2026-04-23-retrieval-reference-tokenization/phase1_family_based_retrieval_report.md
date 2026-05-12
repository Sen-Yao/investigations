# Family-Based Retrieval Tokenization Report (Phase 1)

## 一、背景

当前探究目标不是直接设计一个新的端到端 GAD 模型，而是先回答：

> 在半监督 normal-only 设定下，能否从整图中构造出高质量的 reference token family，供后续 Graph Transformer 使用？

为此，我们围绕三个 family 展开：
- `center_family`
- `delta_ndc_family`
- `anomaly_affinity_family`

其中最关键的挑战是：
- 如何在不依赖异常标签的前提下，构造 anomaly-like reference；
- 如何保证 anomaly-reference 的 top-k purity 足够高，真正对 tokenization 有意义。

---

## 二、旧探究的审计结论

### 1. `2026-04-16-isp-hop2token` 的原始脚本当前不可直接复现
原 investigation 中的脚本存在：
- 数据路径失效；
- 无有效数据时仍输出肯定性结论；
- 因此不能作为当前可信 reproduction 入口。

### 2. 但原始现象层结论大体可信
通过严格按旧报告定义重做后，得到：

#### NDC
在大多数数据集上支持：
\[
\text{Anomaly NDC} > \text{Normal NDC}
\]
而且在旧报告中的 6 个数据集上，严格版 reproduction 也复现了这一方向。

#### ANR
在所有重跑数据集上支持：
\[
\text{Anomaly ANR} > \text{Normal ANR}
\]
但 ANR 原始定义依赖真实异常标签，因此：
> **ANR 只能作为现象锚点，不能直接进入半监督方法。**

---

## 三、三个 family 的当前角色

### 1. `center_family`
定义为节点与正常中心的相似度：
\[
c(v)=\cos(z_v, c_{normal})
\]

当前定位：
> **normal-reference selector**

实验表现：
- 在 `photo` 上 normal top-k purity 很高；
- 在 `elliptic` 上不稳定，不能作为 anomaly selector。

### 2. `delta_ndc_family`
直接使用原始 delta-based NDC：
\[
\text{NDC}(v)=\mathrm{corr}(\Delta(v), \bar{\Delta}_{\mathcal N(v)})
\]

当前定位：
> **relation-pattern selector**

实验表现：
- 能稳定提供统计 signal；
- 但单独用于 anomaly top-k retrieval 时 purity 不高；
- 在 `elliptic` 上比 center 更有价值。

### 3. `anr_proxy_family`
目标是无标签近似“局部异常聚集”。

当前结论：
- 第一版粗糙 patch-style proxy 不够 faithful；
- 还不能稳定承担 anomaly-reference 的主角色。

---

## 四、为什么原始 NDC / ANR 现象很强，而新 score 一开始失效

关键原因在于：
1. 原始 NDC / ANR 测量的是**现象层信号**；
2. 新 retrieval score 需要的是**排序器 / selector**；
3. 现象显著不等于 top-k purity 足够高；
4. 后来构造的 proxy 没有 faithful 保留原始语义。

因此问题不在原始现象，而在：
> **后续构造的 proxy 还不够 faithful。**

---

## 五、从双阶段 anomaly 提纯到单阶段 anomaly affinity

### 1. 双阶段 anomaly 提纯
最初尝试：
- Stage 1: `NDC-cluster` 粗筛 anomaly-like 候选；
- Stage 2: 在候选池内用 local concentration proxy 精排。

其优点是：
- 在 `photo` 和 `elliptic` 上都能提升 anomaly purity。

但缺点也很明显：
- 不够优雅；
- 引入 `topM` 超参数；
- 更像 pipeline trick。

### 2. 更优雅的单阶段方案：Anomaly Affinity Score
定义：

#### anomaly prior
\[
q(v)=\sigma(\mathrm{zscore}(\text{NDC}(v)))
\]

#### local concentration
\[
c(v)=\frac{1}{|\mathcal N(v)|}\sum_{u \in \mathcal N(v)} q(u)
\]

#### anomaly affinity
\[
s_{anom}(v)=q(v)\cdot c(v)
\]

含义：
> 一个节点之所以值得进入 anomaly-reference sequence，不只是因为它自己 anomaly-like，而且因为它处在一个 anomaly-like 节点彼此支持的局部环境中。

---

## 六、单阶段 anomaly affinity 的结果

### `elliptic`
- 原始 `base_ndc`：
  - AUC ≈ 0.5701
  - AP ≈ 0.1448
  - `topk=32` anomaly purity = **0.0**
- `anomaly_affinity`：
  - AUC ≈ 0.5721
  - AP ≈ 0.1459
  - `topk=32` anomaly purity = **0.34375**

### `photo`
- 原始 `base_ndc`：
  - `topk=32` anomaly purity = **0.03125**
- `anomaly_affinity`：
  - `topk=32` anomaly purity = **0.0625**
  - AUC / AP 也略有提升

### 结论
> **单阶段 anomaly affinity 比双阶段更优雅，同时已经能在 `elliptic` 上显著提高 anomaly-reference 的 purity。**

---

## 七、为什么在 tokenization 场景里 purity 比 AUC/AP 更重要

当前方法的目标不是训练一个 standalone detector，而是：
> **构造 reference token selector**

因此：
- AUC / AP 只能说明这个 family 不是纯噪声；
- `top-k purity` 才真正决定：
  - 这些 token 是否像我们想让它们像的那类 reference。

因此当前阶段的指标优先级应为：
1. `normal top-k purity`
2. `anomaly top-k purity`
3. AUC / AP（辅助 sanity check）

---

## 八、当前最可靠的 family 设计

### `center_family`
- 作用：构造 `normal-reference sequence`
- 当前状态：可信，可保留

### `delta_ndc_family`
- 作用：构造 `relation-pattern reference sequence`
- 当前状态：可信，但不适合作为 anomaly selector 单独使用

### `anomaly_affinity_family`
- 作用：构造 `anomaly-reference sequence`
- 当前状态：当前最值得继续推进的 anomaly family 主候选

---

## 九、当前推荐的统一框架

统一的不是单一 score，而是：
> **family-based retrieval tokenization framework**

即：
1. self token
2. `normal-reference tokens` ← `center_family`
3. `relation-reference tokens` ← `delta_ndc_family`
4. `anomaly-reference tokens` ← `anomaly_affinity_family`

这比强行把所有 family 融成一个单分数更符合当前证据，也更有方法论意义。

---

## 十、当前建议的下一步

1. 继续验证 `anomaly_affinity_family` 的稳定性；
2. 以 purity 而不是 AUC 作为 anomaly-reference family 的主优化目标；
3. 开始设计真正的 tokenization v1：
   - 每个 family 取多少 token；
   - summary token 还是 raw selected nodes；
   - 如何避免 token 数爆炸。

---

## 十一、一句话总结

> 当前最可靠、可审计、可复现的结论是：在半监督 normal-only 设定下，原始 NDC/ANR 的现象层结论大体可信；基于此，`center_family`、`delta_ndc_family` 与单阶段 `anomaly_affinity_family` 已经形成一个统一的 family-based retrieval tokenization 框架，其中 `anomaly_affinity_family` 是目前最值得继续推进的 anomaly-reference 选择机制，因为它在 `elliptic` 上显著提升了 top-k anomaly purity，同时在 `photo` 上保持正向。