# Global G_a Score for Anomaly Reference

## 背景

本探究聚焦 Dual-Sequence Reference Tokenization 中的全局 anomaly-reference eligibility score：`G_a`。

核心约束：

```text
G_a ∈ R^N
```

`G_a` 必须是全图一次性计算的节点级数组，只表示候选节点 `u` 是否适合作为 anomaly-reference candidate，不依赖当前 query 节点 `v`。query-specific 相关性由 `L_a(u|v)` 负责。

## 当前收敛结论

当前方法框架已从开放探索收敛为：

```text
Global normal-deviation eligibility + local anomaly relevance
```

即：

```text
S_a(u|v) = G_a(u) + L_a(u|v)
```

其中 `G_a` 使用 5% normal labels 做 normal-calibrated deviation，`L_a` 使用 anomaly context similarity。

## 推荐 G_a family

主方法候选为 Normal-Calibrated Multi-View Deviation：

```text
D_center(u)  = || normalize(z_u) - μ_n ||_2
D_density(u) = 1 - mean_topk_normal cos(z_u, z_r)
D_attr(u)    = 1 - mean_topk_normal cos(x_u, x_r)
```

使用正常训练节点校准：

```text
R_m(u) = |{r ∈ V_train^n : D_m(r) ≤ D_m(u)}| / |V_train^n|
```

跨视角 soft-OR：

```text
G_a(u) = 1 - (1 - R_center(u))(1 - R_density(u))(1 - R_attr(u))
```

## 组合方式决策

已测试：

```text
S = G × L
S = G + L
S = rank(G) + rank(L)
```

当前决策：主方法采用加法。

原因：

1. 加法更符合节点分类 / 异常检测中的 additive evidence 表达；
2. Photo 上 add 与 multiply 几乎持平；
3. Elliptic 上 add 的 anomaly-node reference purity 更高且方差更小；
4. rank-add 在强 G_a 上明显弱于 add/multiply；
5. 加法避免审稿时质疑两个非概率分数为何直接相乘。

## 关键实验文件

```text
experiments/scripts/probe_normal_calibrated_ga.py
experiments/scripts/probe_ga_with_la_reference.py
experiments/scripts/probe_gl_combine_modes.py
experiments/scripts/probe_gl_combine_gpu.py
```

关键输出：

```text
experiments/outputs/photo_normal_calibrated_ga.json
experiments/outputs/elliptic_normal_calibrated_ga.json
experiments/outputs/photo_ga_with_la_refk16.json
experiments/outputs/elliptic_ga_with_la_refk16.json
experiments/outputs/photo_gl_combine_modes.json
experiments/outputs/elliptic_gl_combine_modes_gpu.json
```

## 当前下一步

进入完整模型最小闭环验证，而不是继续无限调 probe：

1. 接入 `run_dual_sequence_reference.py`；
2. 增加 `--ga-mode baseline_qc|normal_attr|normal_max_dev|normal_soft_or`；
3. 增加 `--gl-combine add|multiply`；
4. 主配置使用 `normal_soft_or + add`；
5. Photo/Elliptic seed0 做最小训练验证；
6. 若正向，再扩展到 5 seeds 与更多数据集。
