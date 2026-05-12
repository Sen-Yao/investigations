# CAA (Convergence-Aware Attention) 探究

**探究时间**: 2026-04-01  
**状态**: ✅ 完成  
**关键发现**: CAA在Photo数据集超越VecGAD SOTA **+5.0%**

---

## 探究目标

设计基于信息论分析的GT注意力机制，用于增强图神经网络异常检测性能。

**核心假设**: Delta深层Token包含最高互信息(MI)，应设计机制放大深层信息。

---

## 背景

本探究源于阶段1-3的信息论分析发现：

| 阶段 | 发现 | 影响 |
|------|------|------|
| Phase 1 | Delta在高维数据集MI最高 | Photo MI=66.4 |
| Phase 2 | Delta与图结构强相关 | PageRank corr=0.89 |
| Phase 3 | Delta深层Token注意力最高 | Deep attn=11.6% |

**结论**: 需要设计机制专门增强Delta深层Token的信息传递。

---

## 目录结构

```
2026-04-01-caa/
├── README.md          # 本文档
├── MOTIVATION.md      # 动机与背景（阶段1-3发现）
├── ARCHITECTURE.md    # CAA架构详细说明
├── EXPERIMENTS.md     # 实验结果汇总
├── experiments/
│   ├── scripts/       # CAA实现代码
│   ├── configs/       # Sweep配置
│   ├── outputs/       # 实验结果数据
│   └── plots/         # 可视化图表
└── insights.md        # 核心发现总结
```

---

## 关键结果

| 方法 | Photo AUC | vs VecGAD SOTA |
|------|-----------|----------------|
| **CAA** | **0.9708** | **+5.0%** |
| VecGAD | 0.8960 | baseline |

---

## 相关探究

- [2026-03-31-offset-information-theory](../2026-03-31-offset-information-theory/) - 信息论分析基础

---

_CAA验证了信息论分析的实用性，Delta策略在高维数据集最优。_
