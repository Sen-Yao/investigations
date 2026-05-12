# Insights: Retrieval Reference Tokenization

## 1. 最小实现与接口验证

Dual-Sequence Reference Tokenization 的最小实现已经验证可行。Photo 上 dry run 结果显示，token 序列形状为 `[7535, 21, 745]`，送入现有 `VecGAD.TransformerEncoder` 后输出 embedding 形状为 `[1, 7535, 256]`。这说明当前方法可以在不侵入 VecGAD 主干的前提下，与现有 Transformer token encoder 对接。

其中 21 个 tokens 对应：1 个 self token、4 个 normal-reference tokens、16 个 anomaly-reference tokens。

## 2. normal-reference 与 anomaly-reference 的质量差异

Photo 上 reference purity 审计显示：

| Reference | 平均纯度 | 解释 |
|---|---:|---|
| normal-reference | 0.8058 | 已较可靠 |
| anomaly-reference | 0.0544 | 明显不足 |

进一步四分数审计显示：

| 分数 | 目标 | Photo 纯度 |
|---|---|---:|
| `G_n` | 正常 | 1.0000 |
| `L_n` | 正常 | 0.8931 |
| `G_a` | 异常 | 0.0625 |
| `L_a` | 异常 | 0.0927 |

因此当前瓶颈不在 normal side，而在 anomaly side。

## 3. richer `h_v` 不是主要瓶颈

曾尝试将 anomaly-local representation 从 `[q,c]` 扩展为 `[q,c,NDC,a]`，以及加入 `patch_mismatch` / `patch_inconsistency`。结果显示，richer `h_v` 与 small `h_v` 在 Photo / Elliptic 上几乎无显著差异。

该结果说明：当前 anomaly-reference 质量不高，主要不是因为 `L_a` 表示维度太低，而更可能是 `G_a` 候选异常本身不够纯，或 `G_a` 与 `L_a` 的组合方式不符合 reference selection 目标。

## 4. 从 VecGAD 学到的关键启发

VecGAD 的异常机制不是直接学习真实异常，而是从 normal-only training 出发，通过 token reconstruction error 生成 pseudo-outlier。它的核心启发是：异常可以被理解为 normal explanation failure。

因此，候选 `G_a` 不应只依赖 NDC / anomaly affinity，也应考虑节点是否无法被 normal center / normal bank / normal density 解释。

## 5. 候选 `G_a` 测试发现

### 5.1 VecGAD-inspired 候选

在 Photo 上，`normal residual × anomaly affinity` 明显提升 AUC/AP：

| 指标 | 旧 `q*c` | `residual_times_affinity` |
|---|---:|---:|
| AUC | 0.6098 | 0.6920 |
| AP | 0.1232 | 0.1708 |

在 Elliptic 上，`center_distance` / `one_minus_Gn` 整体 AUC 很高，约 0.7718，但 top-k purity 不如旧 `q*c`。这说明整体异常排序与 anomaly-reference 的 top-k purity 不是同一个目标。

### 5.2 无监督 GAD 启发指标

测试了 Normal Density Failure、Structure Context Density Failure、Structure-Attribute Disagreement、LOF-style Relative Density 等候选。

Photo 上，`attr_normal_density_failure` 的 top-k purity 很强：

| top-k | `q*c` | `attr_normal_density_failure` |
|---|---:|---:|
| top32 | 0.0625 | 0.1875 |
| top64 | 0.1094 | 0.2500 |
| top128 | 0.1172 | 0.2266 |

Elliptic 上，旧 `q*c` 仍保持最强尖端 top-k：

| top-k | `q*c` |
|---|---:|
| top16 | 0.3125 |
| top32 | 0.3438 |
| top64 | 0.2656 |

但 `struct_context_density_failure` 也有较强补充信号：top16=0.25, top32=0.1875, top64=0.2344, AUC=0.6651。

## 6. 简洁双因子分数的结果

为避免复杂加权，测试了无权重的 `Normal-Deviation × Anomaly-Support`：

- `NormalDeviation`：center / density / attribute deviation
- `AnomalySupport`：旧 `q*c`
- 组合方式：rank percentile product 或 raw product

结果显示：

- 在 Photo 上，该类分数显著改善中等 top-k 与 AUC/AP。
- 在 Elliptic 上，直接全图乘法会伤害旧 `q*c` 的尖端 top-k purity。

因此简单对称乘法仍不足以表达 reference selection 所需的非对称逻辑。

## 7. Support-first rerank 的诊断价值

显式 support-first rerank 不是最终方案，但作为诊断实验非常重要。

### Photo

旧 `q*c` baseline：top16=0.0625, top32=0.0625, top64=0.1094。

当先取 `q*c` top512 候选池，再用 center/density deviation 重排时：

| top-k | purity |
|---|---:|
| top16 | 0.7500 |
| top32 | 0.7813 |
| top64 | 约 0.50-0.53 |

### Elliptic

旧 `q*c` baseline：top16=0.3125, top32=0.3438, top64=0.2656。

当先取 `q*c` top128 候选池，再用 center deviation 重排时：

| top-k | purity |
|---|---:|
| top16 | 0.4375 |
| top32 | 0.5625 |
| top64 | 0.5625 |

这说明：`q*c` 负责召回的候选池里确实存在高纯异常节点，而 normal-deviation 能有效精排。

## 8. 当前核心原则

当前最重要的原则是：

> **Support as eligibility, deviation as ranking.**

也就是：异常支持决定候选资格，正常偏离决定参考排序。

这条原则解释了为什么：

1. 旧 `q*c` 在 Elliptic top-k 上很强，因为它擅长候选资格判断；
2. normal-deviation 直接全图使用不稳定，因为它不擅长 anomaly support 召回；
3. support-first rerank 有效，因为它先保留支持，再用 deviation 排序。

## 9. 后续研究方向

不要将显式 top-M 双阶段 rerank 作为最终方法。下一步应设计一个单阶段、无显式 candidate-size 超参数的分数，使其自然近似 support-first rerank 的非对称行为。

候选方向：

1. support-conditioned deviation percentile；
2. soft eligibility mask without hard top-M；
3. asymmetric rank score；
4. deviation measured within support strata。

目标是得到一个更优雅的 `G_a`：既保留 `q*c` 的候选资格能力，又让 normal-deviation 在高 support 区域内发挥排序作用。
