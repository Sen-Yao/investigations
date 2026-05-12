# Investigations - Nexus Research Archive

> 科研调查记录的独立 Git 仓库，用于长期保存和版本追溯。

## 目录结构

```
investigations/
├── nexus/                    # Nexus 代理的科研调查
│   ├── 2026-03-23-xxx/       # 按 YYYY-MM-DD-topic 命名
│   ├── 2026-04-01-xxx/
│   └── 2026-05-09-semisupervised-negative-signal-for-dualrefgad/  # 最新
├── README.md                 # 本文件
```

## Investigation 目录规范

每个 investigation 目录包含：
- `README.md` - 概述和目标
- `hypothesis.md` - 待验证假设
- `insights.md` - 关键结论和发现
- `PROGRESS.md` - 活动时间线（可选）
- `experiments/` - 实验脚本和输出
  - `configs/` - 配置文件
  - `scripts/` - Python 脚本
  - `outputs/` - 结果和报告

## Git 管理

本仓库从 `.openclaw/workspace/agents/nexus/investigations/` 迁移而来（2026-05-12）。

原位置保留作为备份，建议在确认迁移成功后手动清理。

## 活跃 Investigation

### 2026-05-09-semisupervised-negative-signal-for-dualrefgad

**核心问题**：DualRefGAD 如何利用无标签节点的 negative signal？

**关键发现**：
- Margin-only baseline AUC 0.7952 > learned head 0.7455
- `d` 不能被 latent-space density 消解，必须以 conditioning/explicit geometry 形式进入 scoring function
- Density Probe 失败：仅用 `r` density 信息无法达到 margin baseline 性能

**下一步**：Likelihood-Ratio with `||d||` conditioning，目标 AUC ≥ 0.79

---
*Created: 2026-05-12 | Nexus Agent*