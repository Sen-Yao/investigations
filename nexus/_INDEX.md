# 探究目录索引

> **用途**: 快速浏览所有探究，发现相关成果
> **位置**: `investigations/_INDEX.md`
> **更新**: 每次探索完成后更新

---

## 📋 探究列表

| 探究目录 | 关键词 | 数据集 | 状态 | 关键发现 |
|---------|--------|--------|------|---------|
| `2026-04-01-deep-token-information` | CAA, 深层Token, 假设验证 | Photo, Tolokers, Elliptic | ✅ 完成 | **CAA理论基础不普适** |
| `2026-04-01-delta-offset-cross-dataset` | Delta, Offset, SNR, 跨数据集 | Photo, Tolokers, Elliptic | ✅ 完成 | 信息稀释效应非普适，Elliptic反转 |
| `2026-04-01-delta-offset-equivalence` | Delta, Offset, SNR | Photo | ✅ 完成 | Delta SNR是Offset的18.88倍 |
| `2026-04-01-caa-diagnosis` | CAA, VecGAD, 方向信息 | Photo | ✅ 完成 | 方向信息丢失是关键问题 |
| `2026-04-01-offset-anomaly-relation` | Offset, KS检验 | Photo | ✅ 完成 | Offset单独使用效果有限 |
| `2026-04-01-caa` | CAA, 数据泄漏 | Photo | ✅ 完成 | 发现训练集泄漏bug |
| `2026-05-13-dualrefgad-normal-only-residual-probe` | DualRefGAD, normal-only, residual probe, margin | Photo / 待定 | 🟡 活跃 | 诊断 margin 之外是否存在协议干净的稳定残差信号 |

---

## 🔍 按主题分类

### CAA 机制分析（核心发现链）
| 探究 | 状态 | 关键洞见 |
|------|------|---------|
| `2026-04-01-deep-token-information` | ✅ | **CAA理论基础不普适**（深层不一定更优）|
| `2026-04-01-caa-diagnosis` | ✅ | 方向信息丢失是关键问题 |
| `2026-04-01-caa` | ✅ | 数据泄漏修复后性能差 |

### Delta vs Offset 对比
| 探究 | 状态 | 关键洞见 |
|------|------|---------|
| `2026-04-01-delta-offset-cross-dataset` | ✅ | 非普适！Elliptic上Offset更好 |
| `2026-04-01-delta-offset-equivalence` | ✅ | Photo: Delta SNR是Offset的18.88倍 |

---

## 🔗 探究关联图

```
2026-04-01-caa (CAA机制，性能差)
         ↓
2026-04-01-caa-diagnosis (方向信息丢失)
         ↓
2026-04-01-delta-offset-equivalence (Delta更好?)
         ↓
2026-04-01-delta-offset-cross-dataset (跨数据集验证)
         ↓
         发现：Delta不普适！
         ↓
2026-04-01-deep-token-information (CAA理论基础验证)
         ↓
         **关键发现：CAA理论基础不普适！**
```

---

## 🎯 待解决问题汇总

| 问题 | 重要性 | 来源探究 |
|------|--------|---------|
| 如何设计自适应Token策略？ | ⭐⭐⭐ | 2026-04-01-deep-token-information |
| CAA缺少哪些组件？ | ⭐⭐ | 2026-04-01-deep-token-information |
| Elliptic为什么反转？ | ⭐⭐ | 2026-04-01-delta-offset-cross-dataset |

---

## 📊 关键发现汇总

| 发现 | 数值/结论 | 来源 |
|------|----------|------|
| **CAA理论不普适** | 深层不一定更优 | 2026-04-01-deep-token-information |
| **Photo SNR比值** | Delta = Offset × 18.88 | 2026-04-01-delta-offset-equivalence |
| **Elliptic SNR比值** | Offset = Delta × 1.27 | 2026-04-01-delta-offset-cross-dataset |

---

_创建时间: 2026-04-01_
_最后更新: 2026-05-13 (添加 DualRefGAD normal-only residual probe)_