# HBAFormer 改进探究

## 探究目标
改进 HBAFormer 模型，使其在 Photo 数据集上超越 VecGAD baseline (AUC 0.8960)

## 探究问题
1. 为什么 HBAFormer v1/v2 性能低于 VecGAD？
2. Hop 结构信息是否有助于异常检测？
3. Hop 信息的最优引入方式是什么？

## 时间线
- **2026-04-03 16:54**: 开始探究，分析 v1/v2 失败原因
- **2026-04-03 18:25**: GLM-5.1 辩论，提出方向 C (Loss 正则化)
- **2026-04-03 18:50**: v3 消融实验代码修复完成
- **2026-04-03 19:16**: v3 消融实验完成

## 当前状态
🔄 进行中 - 等待 GLM-5.1 辩论下一步建议

## 关键文件
- `hypothesis.md`: 假设列表
- `insights.md`: 发现和洞见
- `PROGRESS.md`: 详细进度记录
- `experiments/`: 实验相关内容

## 实验结果汇总
| 版本 | AUC | 说明 |
|------|-----|------|
| VecGAD (baseline) | 0.8960 | 目标 |
| HBAFormer v1 | 0.7943±0.0190 | hop_bias 直接注入 |
| HBAFormer v2 | 0.7749±0.0171 | 门控 Value 调制 |
| HBAFormer v3-b1 | 0.8273±0.0172 | 乘法缩放 (最佳) |