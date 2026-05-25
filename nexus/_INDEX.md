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
| `2026-05-13-dualrefgad-normal-only-residual-probe` | DualRefGAD, normal-only, residual probe, margin | Elliptic | ✅ 完成 | additive residual route closed；correction 主要是 margin compression |
| `2026-05-13-dualrefgad-reference-geometry-anatomy` | DualRefGAD, reference geometry, margin anatomy, distributional inconsistency | Elliptic / 待扩展 | 🟡 活跃 | 解剖 margin-only 信号来源与 top-k failure case，判断是否存在 response distribution 信号 |
| `2026-05-21-dualrefgad-constraint-calibrated-reference-relation` | DualRefGAD, C-LEG3, reference relation, pseudo anomaly gate | Elliptic | 🟡 活跃 | 固定 C-LEG3，先做 R0/R4 decomposition gate，再决定是否启动 R1–R3 伪异常训练 |

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


### DualRefGAD / reference geometry
| 探究 | 状态 | 关键洞见 |
|------|------|---------|
| `2026-05-13-dualrefgad-normal-only-residual-probe` | ✅ | additive residual route closed；learned correction 主要是 margin compression / calibration |
| `2026-05-13-dualrefgad-reference-geometry-anatomy` | 🟡 | 解剖 margin-only 信号来源、reference response distribution 与 top-k failure case |
| `2026-05-21-dualrefgad-constraint-calibrated-reference-relation` | 🟡 | 新探究：约束校准 reference relation；第一步 C-LEG3 decomposition gate 已准备，未启动 |

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
| DualRefGAD margin-only 的信号来源与失败机制是什么？ | ⭐⭐⭐ | 2026-05-13-dualrefgad-reference-geometry-anatomy |
| C-LEG3 固定后，两条 normal-only 约束能否校准无权重 reference relation score？ | ⭐⭐⭐ | 2026-05-21-dualrefgad-constraint-calibrated-reference-relation |

---

## 📊 关键发现汇总

| 发现 | 数值/结论 | 来源 |
|------|----------|------|
| **CAA理论不普适** | 深层不一定更优 | 2026-04-01-deep-token-information |
| **Photo SNR比值** | Delta = Offset × 18.88 | 2026-04-01-delta-offset-equivalence |
| **Elliptic SNR比值** | Offset = Delta × 1.27 | 2026-04-01-delta-offset-cross-dataset |
| **DualRefGAD residual head 关闭** | `margin+correction` 无稳定收益，correction≈margin compression | 2026-05-13-dualrefgad-normal-only-residual-probe |

---

_创建时间: 2026-04-01_
_最后更新: 2026-05-21 (添加 DualRefGAD constraint-calibrated reference relation)_