# Mapping Audited NDC / ANR to Retrieval Score Design

## 一、审计后的映射原则

在完成对 `2026-04-16-isp-hop2token` 的审计之后，当前最合理的映射方式是：

- **NDC**：作为可直接进入半监督方法设计的 `relation anomaly source`；
- **ANR**：作为不能直接使用、但必须转化为无标签 proxy 的 `local aggregation anomaly source`。

---

## 二、NDC 的当前角色

### 审计后结论
- NDC 多数数据集支持 `anomaly > normal`；
- 无标签泄漏；
- 适合作为 retrieval score 中的关系异常项。

### 当前方法映射
NDC 应被映射为：
> **节点与其上下文是否不协调**

因此，在当前 retrieval score 中，NDC 不应再被压缩成抽象 residual，而应成为一个明确的 relation anomaly term。

### 当前最小 proxy
目前较可信的最小 proxy 是：
- `node_to_neigh_mismatch`

它虽然不强，但方向清晰、语义明确，且与旧 NDC 思想一致。

---

## 三、ANR 的当前角色

### 审计后结论
- ANR 现象跨数据集稳定；
- 但原定义依赖异常标签；
- 因此不能直接进入半监督 score。

### 当前方法映射
ANR 应被映射为：
> **局部异常聚集 / 局部异常结构单元的无标签 proxy**

也就是说，ANR 的作用是告诉我们：
- 需要设计一个 local aggregation anomaly term；
- 但该 term 必须完全不依赖真实异常标签。

---

## 四、为什么旧探究在 elliptic 上显著，而新 score 失效

这是当前最关键的问题。审计后的解释是：

### 原因 1：原始 NDC / ANR 是“现象测量”，不是“方法 score”
旧探究测量的是：
- 节点与邻域 Delta 模式的相关性；
- 异常邻居比例。

这些量可以显著区分现象，但并不意味着：
- 它们可以直接作为 retrieval score；
- 或者与 normal-center anchor 线性耦合后仍保持有效。

### 原因 2：原始 ANR 有标签泄漏
ANR 在旧探究里强，不代表我们当前无标签 proxy 已经 faithful 地近似了它。

### 原因 3：原始 NDC 依赖的是 Delta 语义，而当前新 score 里的 proxy 依赖的是轻量表征空间
也就是说：
- 旧 NDC 作用在 Delta 序列上；
- 当前新 proxy 作用在新的轻量 embedding / patch 表征上；
- 二者不是同一个对象。

### 原因 4：elliptic 数据集机制本来就特殊
过去多次探索都显示：
- `elliptic` 经常与 `photo`、`tolokers` 方向不同；
- center-based intuition 往往在 `elliptic` 上失效；
- 关系项和局部结构项更重要。

因此，不能期望旧现象在新的 score family 中自动保持强信号。

---

## 五、当前最合理的设计调整

### 对于 photo
目前可以保留：
- `center + ANR-like proxy`

### 对于 elliptic
当前更合理的方向是：
- 弱化或移除 center 项；
- 保留并加强 NDC 项；
- 重做更 faithful 的 ANR proxy。

---

## 六、当前最小 score 重构建议

### 针对 photo
\[
s_{photo}(v) = \alpha \cdot center(v) - \beta \cdot NDC(v) - \gamma \cdot ANRproxy(v)
\]
其中 ANRproxy 可作为主要增强项。

### 针对 elliptic
\[
s_{elliptic}(v) = - \beta \cdot NDC(v) - \gamma \cdot ANRproxy(v)
\]
或者将 `center(v)` 权重显著降低。

---

## 七、当前结论

> 旧探究中的 NDC / ANR 在 elliptic 上之所以“看起来很稳定很显著”，是因为它们测量的是现象层信号；而我们当前的新 score 设计之所以失效，是因为我们尝试将这些现象直接迁移到新的、无标签、耦合的 retrieval score 中，但代理量并未 faithful 保留原始语义，且 center-based prior 在 elliptic 上本身就不稳定。
