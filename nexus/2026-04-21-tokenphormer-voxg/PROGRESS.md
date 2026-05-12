## Photo 轻量验证结果（2026-04-22）

### 当前状态
- 已完成 Photo 数据集上的 prototype / consistency 第一轮轻量验证
- 结果支持 anomaly-aware multi-token 方向，但当前新特征更像互补特征，而非主特征

### 统计层结果
多个 prototype / consistency 相关统计量可以显著区分 normal / anomaly：

| 特征 | Normal Mean | Anomaly Mean | Difference | KS p-value |
|------|-------------|--------------|------------|-----------|
| delta_corr_to_proto | -0.353 | -0.282 | +0.070 | 3.35e-14 |
| delta_cos_to_proto | -0.315 | -0.252 | +0.063 | 2.49e-10 |
| delta_l2_to_proto | 14.790 | 13.309 | -1.481 | 8.59e-12 |
| proto_delta_norm | 5.092 | 4.204 | -0.888 | 7.24e-22 |
| neigh_delta_var | 0.03464 | 0.03401 | -0.00064 | 1.88e-16 |

### Probe 层结果

| 输入 | AUC | AP |
|------|-----|----|
| delta_flatten | 0.9602 | 0.8225 |
| prototype_only | 0.6272 | 0.1421 |
| consistency_only | 0.6036 | 0.1352 |
| delta_plus_prototype | 0.9624 | 0.8274 |
| delta_plus_consistency | 0.9618 | 0.8267 |
| delta_plus_prototype_plus_consistency | 0.9633 | 0.8300 |

### 当前解释
1. Delta token 仍然是主特征。
2. Prototype / Consistency 单独效果一般，但与 Delta 组合后提供稳定小幅增益。
3. 这说明 NDC / ANR 现象可以开始转化为表示层的互补特征。
4. 当前 community-aware proxy 还只是初级版本，后续仍需优化。

## Amazon 轻量验证结果（2026-04-22）

### 当前状态
- 已完成 Amazon 数据集上的同构验证
- 统计层信号更强，但与 Delta 主特征组合后没有带来稳定增益

### 统计层结果
多个 prototype / consistency 相关统计量依然可以显著区分 normal / anomaly：

| 特征 | Normal Mean | Anomaly Mean | Difference | KS p-value |
|------|-------------|--------------|------------|-----------|
| delta_corr_to_proto | -0.190 | 0.189 | +0.379 | 3.97e-40 |
| delta_cos_to_proto | -0.188 | 0.190 | +0.379 | 3.82e-40 |
| delta_l2_to_proto | 488.390 | 265.567 | -222.823 | 5.28e-105 |
| proto_delta_norm | 135.546 | 122.432 | -13.114 | 5.18e-03 |
| neigh_delta_var | 4373.280 | 3390.713 | -982.567 | 1.43e-94 |
| neigh_delta_mean_pairwise_cos | 0.116 | 0.262 | +0.146 | 4.10e-66 |

### Probe 层结果

| 输入 | AUC | AP |
|------|-----|----|
| delta_flatten | 0.9807 | 0.8652 |
| prototype_only | 0.7402 | 0.1542 |
| consistency_only | 0.7274 | 0.2301 |
| delta_plus_prototype | 0.9806 | 0.8645 |
| delta_plus_consistency | 0.9807 | 0.8640 |
| delta_plus_prototype_plus_consistency | 0.9807 | 0.8637 |

### 当前解释
1. Amazon 上 prototype / consistency 特征单独具有更强统计可分性。
2. 但 Delta flatten 已经非常强，新增特征没有提供稳定增益，甚至略有回落。
3. 这说明 anomaly-aware multi-token 特征在 Amazon 上更像“可解释辅助视角”，但暂未形成超越 Delta 主干的实际收益。
4. 当前结论是：该方向跨数据集在“统计显著性”上成立，但在“与 Delta 组合的增益”上未表现出一致稳定性。

### 下一步
- 可扩展到 Tolokers / Elliptic，检查这种“统计显著但不一定提升 probe”的模式是否普遍存在
- 若该模式重复出现，需要重新审视 consistency proxy 的定义，避免与 Delta 主特征高度冗余