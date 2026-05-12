# Hop-Aware Attention 集成到 VoxGFormer

> 在 VoxGFormer 的 Transformer 中集成 Hop-Aware Attention，验证效果

---

## 背景

### 核心发现

| 发现 | 说明 |
|------|------|
| **GT 学习能力孱弱** | 标准注意力 QK^T 计算相似度，没有减法 |
| **hop_bias 探测成功** | 真实数据验证：距离衰减模式有意义 |
| **VoxGFormer 有伪异常生成** | 可以解决半监督探测问题 |

### VoxGFormer 架构

| 组件 | 说明 |
|------|------|
| **输入** | hop tokens: (N, pp_k+1, d) |
| **TransformerEncoder** | 使用 MultiHeadAttention |
| **attn_bias 参数** | 已支持，可以传入偏置 |

---

## 探索方案

### 时间规划
- **开始时间**: 2026-04-03 15:20
- **预计结束**: 2026-04-03 18:00
- **总时长**: ~2.5 小时

### 探索计划

| 活动 | 预估时间 | 说明 |
|------|---------|------|
| 集成设计 | 30 分钟 | hop_bias 注入方案 |
| 代码修改 | 30 分钟 | 修改 MultiHeadAttention |
| Sweep 配置 | 15 分钟 | 创建 WandB sweep |
| Sweep 运行 | 1 小时 | 5-seed 实验 |
| 结果分析 | 15 分钟 | 对比 VecGAD baseline |

### 集成方案



### 预期产出

- [ ] HopAwareMultiHeadAttention 模块
- [ ] 修改后的 VoxGFormer
- [ ] Sweep 配置文件
- [ ] 5-seed AUC/AP 结果
- [ ] 与 VecGAD 对比分析

---

## 风险评估

| 风险 | 应对措施 |
|------|---------|
| 集成失败 | 最小化修改，保持兼容性 |
| 效果不佳 | 多种 hop_bias 设计对比 |
| 时间超支 | 先验证 Photo，再扩展其他数据集 |

---

## 用户审批

- **审批状态**: ⏳ 待审批
- **审批时间**: YYYY-MM-DD HH:MM
- **用户反馈**: [如有]

---

_创建时间: 2026-04-03_
