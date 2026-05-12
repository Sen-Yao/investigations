# Delta Vector消融实验设计

**日期**: 2026-03-23
**主题**: Token模式消融实验设计

## 背景

根据之前对delta vector的分析（见 `2026-03-23-delta-vector-findings.md`），我们发现：

1. **Delta向量本身具有异常检测能力**：使用简单的阈值方法，delta向量在多个数据集上已经能达到不错的检测效果
2. **Delta向量捕捉了"偏离正常"的信息**：它编码了节点与其邻居的差异，这与异常的本质（偏离正常模式）高度一致
3. **当前模型使用的是NagPhormer的token方式**：将传播后的特征作为token

## 核心问题

**当前的token是否充分利用了异常检测的关键信息？**

当前token结构：`[N, K+1, D]`
- Token 0: 原始特征
- Token 1~K: 第1到K跳传播后的特征

Delta向量结构：`[N, K, D]`
- Token 1~K: 第i跳特征 - 第i-1跳特征 = 传播过程中的变化量

## 实验设计

### 三种Token模式

| 变体 | 名称 | Token结构 | 说明 |
|------|------|-----------|------|
| V1 | original | `[N, K+1, D]` | 保持现状（baseline） |
| V2 | delta | `[N, K, D]` | 替换为delta向量 |
| V3 | concat | `[N, 2K+1, D]` | 拼接两种信息 |

### 实验矩阵

- **数据集**: photo, t_finance, tolokers, elliptic (4个)
- **Token模式**: original, delta, concat (3种)
- **Seeds**: 0, 1, 2, 3, 4 (5个)
- **总实验数**: 4 × 3 × 5 = 60

### 超参数

使用与之前复现实验相同的超参数：
- train_rate=0.05
- num_epoch=300
- batch_size=8192
- embedding_dim=256
- 其他默认参数

## 预期结果与假设

### 假设1：Delta模式可能优于Original
如果delta向量确实更好地捕捉了异常信息，那么V2应该能够达到或超过V1的性能。

### 假设2：Concat模式可能最优
拼接两种信息理论上应该拥有最多的信息量，如果模型能够有效利用这些信息，V3应该表现最好。

### 假设3：不同数据集可能有不同的最优模式
不同数据集的异常类型不同，可能需要不同的token方式。

## 风险与考虑

1. **信息冗余**：Concat模式可能导致信息冗余，增加模型学习难度
2. **过拟合风险**：更多的token可能增加过拟合风险
3. **计算开销**：Concat模式的序列长度增加，计算开销略增

## 实现细节

### 代码修改

1. `run.py`: 添加 `--token_mode` 参数
2. `utils.py`: 实现三种tokenization方式
   - `original`: 保持原有逻辑
   - `delta`: 计算delta向量
   - `concat`: 拼接两种token

### Delta计算

```python
# Delta tokenization
delta_tokens = []
for k in range(K):
    delta = propagated_features[k+1] - propagated_features[k]
    delta_tokens.append(delta)
tokens = torch.stack(delta_tokens, dim=1)  # [N, K, D]
```

### Concat计算

```python
# Concat tokenization
original_tokens = ...  # [N, K+1, D]
delta_tokens = ...     # [N, K, D]
tokens = torch.cat([original_tokens, delta_tokens], dim=1)  # [N, 2K+1, D]
```

## 后续分析

实验完成后需要分析：

1. **性能对比**：三种模式在AUROC和AUPRC上的差异
2. **统计显著性**：使用t检验验证差异是否显著
3. **收敛速度**：不同模式的训练曲线对比
4. **聚类质量**：Ring Loss效果是否受token模式影响
5. **可解释性**：不同token模式下模型关注的特征是否不同

## 文件位置

- 实验脚本: `run_delta_ablation.sh`
- 汇总脚本: `summarize_ablation_results.py`
- 日志目录: `logs/tmp/`
- 结果报告: `exp/delta_ablation_results.md`

---

*实验设计完成，等待结果...*