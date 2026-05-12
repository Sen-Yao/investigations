# 假设列表

## 核心假设

| ID | 假设 | 验证方法 | 状态 |
|----|------|---------|------|
| H1 | hop_bias 能提升 VoxGFormer 性能 | 5-seed AUC/AP | ⏳ 待验证 |
| H2 | hop_bias 与伪异常生成机制互补 | 对比实验 | ⏳ 待验证 |

## 假设详情

### H1: hop_bias 能提升 VoxGFormer 性能

**内容**: 在 VoxGFormer 的 Transformer 中注入 hop_bias，能让模型更好地利用 Hop 因果结构，提升异常检测性能。

**验证方法**: 
- 数据集: Photo (目标 AUC > 0.8960)
- 5-seed 实验
- 对比: VecGAD baseline

**预期**: hop_bias 让模型关注相邻层差异，捕捉异常信号。

---

### H2: hop_bias 与伪异常生成机制互补

**内容**: hop_bias 注入结构信息，伪异常生成提供监督信号，两者结合能更好学习。

**验证方法**: 
- ablation study: 只用 hop_bias vs 只用伪异常 vs 结合

**预期**: 结合效果 > 单独效果。

---

_创建时间: 2026-04-03_
