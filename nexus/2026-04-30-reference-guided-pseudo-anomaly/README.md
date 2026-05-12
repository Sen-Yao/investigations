# Reference-Guided Pseudo Anomaly

**创建日期**: 2026-04-30  
**状态**: Active  
**主题**: Dual-reference tokenization 如何驱动半监督 GAD 中的 pseudo anomaly synthesis objective。

## 背景

前序探究 `2026-04-27-global-ga-score` 已经证明：

1. Dual-reference tokenization 的 `R_a(v)` 具有明确异常排序信号；
2. `full > no_ra / shuffled_ra`，说明 target-reference 对应关系重要；
3. 原始 self-residual pseudo objective 在 Photo/Elliptic 上 best epoch 均为 0，说明训练未有效放大 reference geometry；
4. global reference-guided pseudo anomaly 在 Photo 上显著提升，但在 Elliptic 上退化，提示全局方向过粗。

因此，本探究从“reference 如何选择”转向“reference 如何驱动训练”。

## 核心问题

在 5% normal-only 半监督 GAD 设置下，如何用 dual-reference tokenization 产生稳定、可解释、target-specific 的 pseudo anomaly direction？

## 初始结论基线

| Dataset | Objective | Test AUC | Test AP | Best Epoch | 说明 |
|---|---:|---:|---:|---:|---|
| Photo | self_residual | 0.5799 | 0.1209 | 0 | 原始 objective 未放大 geometry |
| Photo | global_ref_guided | **0.7646** | **0.3967** | 200 | 强正向证据 |
| Elliptic | self_residual | **0.6900** | **0.1518** | 0 | 初始 geometry 强 |
| Elliptic | global_ref_guided | 0.5360 | 0.0984 | 160 | 全局方向误导 |

## 研究目标

设计并验证 **target-specific reference residual pseudo anomaly**：

```python
rn_i = mean(emb[normal_refs[i]])
ra_i = mean(emb[anom_refs[i]])
direction_i = normalize(ra_i - rn_i)
pseudo_i = emb[i] + beta * direction_i
```

约束：

- 保持单一 BCE objective；
- 不引入额外复杂 loss；
- 不新增关键超参数；
- train_rate=0.05；
- 训练集必须只包含正常节点；
- mini 阶段先 seed=0，正式结论必须 5-seed。

## 文件结构

```text
experiments/scripts/   # 实验脚本
experiments/configs/   # 配置文件
experiments/outputs/   # 输出 JSON / log
experiments/plots/     # 可视化
```
