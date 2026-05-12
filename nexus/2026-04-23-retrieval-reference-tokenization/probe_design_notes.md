# Probe Design Notes

## 过去一个月探究中可直接借鉴的 probe 思路

### 1. 信息量 / 可分性 probe
来自 2026-03-31 offset-information-theory 与 2026-04-03 hop-learning-capability：
- 熵 / 信息量比较
- 互信息估计
- Fisher 可分性
- 线性 probe / R²

对当前方向可改写为：
- score(normal) / score(anomaly) 的 KS / AUC / rank 分离度
- retrieval score 与 label / anomaly-relevant residual 的相关性

### 2. residual explanation probe
来自 2026-04-22 post-delta token design：
- 检查某类 feature 是否能解释 baseline 未解释的 residual

对当前方向可改写为：
- reference score 是否能解释 node-only / local-only baseline 的 residual
- reference-based feature 是否比 local feature 更能解释误差样本

### 3. token utilization / deep token research
来自 2026-04-01 token-utilization-research：
- token 是否真的被模型用到
- 深层 token 的信息保留 / 方差 / 相关性如何

对当前方向可改写为：
- reference token 在注意力中的权重分布
- self token 是否真实引用 remote reference token
- normal/anomaly sequence 是否存在明显 usage pattern

### 4. dataset-dependent sanity check
来自 2026-03-30 / 2026-04-01 / 2026-04-22：
- 很多现象都具有强数据集依赖性
- 不能只看最终总体性能，要看 per-dataset 机制是否一致

因此当前 probe 也应优先：
- Photo
- Elliptic
并视情况再扩展。

## 当前建议的最低成本 probe

### Probe A: score distribution sanity check
- normal score / anomaly score 在 normal / anomaly 上的分布差异
- KS, AUC, rank correlation

### Probe B: top-k reference purity
- top-k normal references 的 normal purity
- top-k anomaly-like references 的 anomaly purity
- 与随机参考、局部近邻参考对比

### Probe C: non-locality check
- 被检索 reference 与目标节点的最短路径距离分布
- 检查 reference 是否真的经常超出局部邻域

### Probe D: residual explanation
- 在 node/local baseline 后，reference score 是否更能解释剩余错误

### Probe E: interaction necessity
- 只用 retrieval score 排名 vs retrieval + transformer interaction
- 检查收益是否主要来自 interaction 而非 score 本身
