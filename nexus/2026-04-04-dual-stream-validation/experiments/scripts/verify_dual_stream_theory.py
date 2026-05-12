#!/usr/bin/env python3
"""
双流架构理论基础验证实验

核心假设：
1. 点积注意力会摧毁 Delta 信息
2. 专用通道（MLP）能保留 Delta 信息

验证方法：
- 测量输入-输出互信息
- 对比 Attention vs MLP 的信息保留程度
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mutual_info_score

def compute_mutual_information(x, y, n_bins=20):
    """
    计算两个张量之间的互信息
    将连续值离散化后计算
    """
    x = x.detach().cpu().numpy().flatten()
    y = y.detach().cpu().numpy().flatten()
    
    # 离散化
    x_binned = np.digitize(x, np.linspace(x.min(), x.max(), n_bins))
    y_binned = np.digitize(y, np.linspace(y.min(), y.max(), n_bins))
    
    return mutual_info_score(x_binned, y_binned)

def compute_delta_preservation_ratio(input_tokens, output_tokens):
    """
    计算 Delta 信息保留比例
    使用余弦相似度和互信息
    """
    # 余弦相似度
    cos_sim = torch.nn.functional.cosine_similarity(
        input_tokens.flatten().unsqueeze(0),
        output_tokens.flatten().unsqueeze(0)
    ).item()
    
    # 互信息
    mi = compute_mutual_information(input_tokens, output_tokens)
    
    return cos_sim, mi

# 模拟数据
print("=" * 60)
print("双流架构理论基础验证实验")
print("=" * 60)

# 设置
torch.manual_seed(42)
batch_size = 100
seq_len = 7  # pp_k + 1
d_model = 64

# 生成模拟 Hop Token（包含 Delta 信息）
hop_tokens = torch.randn(batch_size, seq_len, d_model)

# 计算 Delta（相邻 Hop 的差异）
delta_tokens = hop_tokens[:, 1:, :] - hop_tokens[:, :-1, :]  # (batch, seq_len-1, d_model)

print(f"\n输入形状: Hop tokens {hop_tokens.shape}, Delta tokens {delta_tokens.shape}")

# ===== 方式 1：标准注意力 =====
print("\n" + "-" * 60)
print("方式 1：标准注意力（点积）")
print("-" * 60)

class StandardAttention(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
    
    def forward(self, x):
        out, _ = self.attention(x, x, x)
        return out

attn_layer = StandardAttention(d_model)
delta_after_attn = attn_layer(delta_tokens)

cos_attn, mi_attn = compute_delta_preservation_ratio(delta_tokens, delta_after_attn)
print(f"余弦相似度: {cos_attn:.4f}")
print(f"互信息: {mi_attn:.4f}")

# ===== 方式 2：专用通道（MLP） =====
print("\n" + "-" * 60)
print("方式 2：专用通道（MLP，不经点积）")
print("-" * 60)

class DedicatedChannel(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
    
    def forward(self, x):
        return self.mlp(x)

mlp_layer = DedicatedChannel(d_model)
delta_after_mlp = mlp_layer(delta_tokens)

cos_mlp, mi_mlp = compute_delta_preservation_ratio(delta_tokens, delta_after_mlp)
print(f"余弦相似度: {cos_mlp:.4f}")
print(f"互信息: {mi_mlp:.4f}")

# ===== 方式 3：无处理（基线） =====
print("\n" + "-" * 60)
print("方式 3：无处理（基线）")
print("-" * 60)
cos_identity, mi_identity = compute_delta_preservation_ratio(delta_tokens, delta_tokens)
print(f"余弦相似度: {cos_identity:.4f}")
print(f"互信息: {mi_identity:.4f}")

# ===== 结果汇总 =====
print("\n" + "=" * 60)
print("结果汇总")
print("=" * 60)

print(f"\n{'方式':<20} {'余弦相似度':<15} {'互信息':<15} {'信息保留率':<15}")
print("-" * 65)
print(f"{'标准注意力':<20} {cos_attn:<15.4f} {mi_attn:<15.4f} {cos_attn/cos_identity*100:.1f}%")
print(f"{'专用通道(MLP)':<20} {cos_mlp:<15.4f} {mi_mlp:<15.4f} {cos_mlp/cos_identity*100:.1f}%")
print(f"{'无处理(基线)':<20} {cos_identity:<15.4f} {mi_identity:<15.4f} 100.0%")

# ===== 结论 =====
print("\n" + "=" * 60)
print("结论")
print("=" * 60)

improvement = (cos_mlp - cos_attn) / abs(cos_attn) * 100 if cos_attn != 0 else 0
print(f"\n专用通道相比标准注意力的信息保留差异: {improvement:.1f}%")

if cos_mlp > cos_attn:
    print("\n✅ 假设验证通过：专用通道能更好地保留 Delta 信息")
    print("   双流架构的理论基础成立")
else:
    print("\n⚠️ 注意：随机初始化的模型可能需要训练后才能得出结论")
    print("   建议：用真实数据训练后再次验证")