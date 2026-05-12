# Hop-Offset 分析探究

## 状态
✅ 探索阶段完成

---

## 背景

师兄提出一种新的差分方案：

**当前 Delta（相邻差分）**：
```
Delta_1 = Hop_1 - Hop_0
Delta_2 = Hop_2 - Hop_1
...
Delta_k = Hop_k - Hop_{k-1}
```

**新方案（累计偏移，Hop-Offset）**：
```
Offset_0 = Hop_0（自身）
Offset_1 = Hop_1 - Hop_0
Offset_2 = Hop_2 - Hop_0
...
Offset_k = Hop_k - Hop_0
```

---

## 核心问题

**Hop-Offset 序列是否包含更多信息？能否提升异常检测效果？**

---

## 📊 探索结论（2026-03-30）

| 维度 | Hop-Offset | Delta | 结论 |
|------|------------|-------|------|
| **信息量** | ✅ 更大 | 较小 | **Hop-Offset 优** |
| **可分性** | ≈ | ≈ | 相当 |
| **新颖性** | ✅ 新方法 | 现有方法 | **有创新价值** |

### 关键发现

1. **Hop-Offset 是新颖方法**：文献未发现相同方案
2. **Alpha=0.5 方向反转**：异常节点更高，数值稳定
3. **信息量优势**：Elliptic 26%，3/3 数据集 Hop-Offset 更优

---

## 命名约定

| 名称 | 定义 | 说明 |
|------|------|------|
| **Delta** | `Hop_k - Hop_{k-1}` | 相邻差分，当前使用 |
| **Hop-Offset** | `Hop_k - Hop_0` | 累计偏移，新方案 |

---

## 📁 文件结构

```
investigations/2026-03-30-hop-offset-analysis/
├── README.md           # 探究概述
├── hypothesis.md       # 假设列表（已验证 4/6）
├── insights.md         # 探索发现汇总
└── experiments/
    ├── scripts/
    │   ├── analyze_hop_offset.py
    │   ├── info_entropy_analysis.py
    │   └── fisher_discriminant_ratio.py
    ├── outputs/
    │   ├── entropy_results.txt
    │   └── fisher_discriminant_results.json
    └── notes.md         # 探索记录
```

---

## 🎯 下一步

| 优先级 | 任务 | 状态 |
|--------|------|------|
| 1 | α=0.5 Hop-Offset 完整测试 | ⏳ 待执行 |
| 2 | 结合 Hop-Offset + Delta | ⏳ 待设计 |
| 3 | 实际异常检测 AUC 对比 | ⏳ 待验证 |

---

## 时间线

- 2026-03-30 18:32: 探究启动，提出假设
- 2026-03-30 20:14-20:25: 自由探索完成（四项任务）
- 2026-03-31: 探索结果记录到 insights.md

---

_更新时间: 2026-03-31_