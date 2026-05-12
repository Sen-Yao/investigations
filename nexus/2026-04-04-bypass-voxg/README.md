# BypassVoxG: 高维 Bypass 异常评分架构

## 动机

### 问题背景

VoxGFormer 在 Photo 数据集上的 concat 方法达到 AUC 0.8777，距离 VecGAD 的 0.8960 还有 1.8% 的差距。

### 已验证的失败方案

| 方法 | AUC | 失败原因 |
|------|-----|---------|
| delta-only | 0.53 | Delta 独立流信息不足 |
| DualStreamVoxG | 0.52 | 退化为 delta-only |
| hop_bias | 0.79 | 注意力偏置干扰异常感知 |
| FiLM-Delta | 0.48~0.63 | 半监督陷阱 + 调制干扰 |

### 核心洞察

1. **Transformer = 低通滤波器**：点积注意力抹平高频差异信号
2. **投影层损失 22%**：745→128 维压缩丢失 Delta 信息
3. **concat 效果最佳**：但仍有信息瓶颈

---

## 核心创新

### Bypass 高维评分

**核心思想**：不让关键判别信息经过 Transformer

```python
# 在 745 维空间直接计算统计量（零信息损失）
hop_mean = hop_features.mean(dim=-1)
hop_std = hop_features.std(dim=-1)
delta_norm = deltas.norm(dim=-1)
```

### 架构设计

```
输入: Hop features [N, 7, 745]
      │
      ├──────────────────────┐
      │                      │
      ▼                      ▼
┌─────────────┐      ┌─────────────────┐
│ Transformer │      │ Bypass Channel  │
│  (低频)     │      │  (高频保留)      │
│  745→128    │      │  直接在745维计算  │
└─────┬───────┘      └────────┬────────┘
      │                       │
      └───────────┬───────────┘
                  │
                  ▼
           ┌──────────┐
           │  Fusion  │
           │ + Decoder│
           └──────────┘
```

---

## 预期效果

| 方法 | AUC | 说明 |
|------|-----|------|
| concat (baseline) | 0.8777 | 当前最佳 |
| **BypassVoxG** | **>0.89** | 目标 |

---

## 文献依据

- **VecGAD RDV**: 硬编码减法保留差异信息
- **High-Frequency Signal**: 异常检测需要高频信号
- **Dual-Path Architecture**: CNN 中的双路径设计

---

## 时间线

- 2026-04-04: 启动调研
- 待定: 原型实现
- 待定: 5-seed 验证

---

_创建时间: 2026-04-04_
_作者: Nexus_