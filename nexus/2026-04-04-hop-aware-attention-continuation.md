# PROGRESS: Hop-aware Attention 集成（续）

## 当前状态
🟡 **运行中** - Hop-bias sweep 已启动

## 背景

### 已有资源
| 文件 | 状态 |
|------|------|
| `HBAFormer.py` | ✅ 已实现 HopBiasMultiHeadAttention |
| `run_hba.py` | ✅ 使用 HBAFormer（带 hop_bias）|
| 探测实验 | ✅ hop_bias 学习成功（距离衰减模式）|

### 代码验证
```python
# HBAFormer.py 第 45-71 行
class HopBiasMultiHeadAttention(nn.Module):
    def __init__(self, ...):
        self.hop_bias = nn.Parameter(torch.zeros(2 * max_hop - 1))

    def forward(self, q, k, v, attn_bias=None, hop_indices=None):
        if hop_indices is not None:
            hop_bias_matrix = self._build_hop_bias_matrix(hop_indices, q.size(1))
            x = x + hop_bias_matrix.unsqueeze(0).unsqueeze(0)
```

## 活动历史

| 时间 | 活动 | 状态 |
|------|------|------|
| 09:56 | 搜索已有探究 | ✅ |
| 09:57 | 确认代码已实现 | ✅ |
| 10:00 | 创建 sweep 配置 | ✅ |
| 10:01 | 启动 sweep | 🔄 |

## Sweep 信息

| 项目 | 内容 |
|------|------|
| **Sweep ID** | `jcfkg39v` |
| **URL** | https://wandb.ai/HCCS/VoxG/sweeps/jcfkg39v |
| **数据集** | Photo |
| **配置** | 5-seed, 200 epochs |
| **目标** | 验证 hop_bias 效果 |

## 当前运行的进程

| 进程 | 数据集 | seed | GPU |
|------|--------|------|-----|
| run_hba.py | photo | 0 | - |

## 待验证

- [ ] 5-seed AUC/AP 结果
- [ ] hop_bias 学习模式
- [ ] vs VecGAD baseline (0.8960)

## 关联探究

- `investigations/2026-04-03-hop-aware-attention/` - 探测实验成功
- `investigations/2026-04-03-hop-aware-voxg/` - 集成方案（未完成）
- `docs/ideas/2026-04-01-hop-semantics-attention.md` - 设计文档

---
_最后更新: 2026-04-04 10:02_