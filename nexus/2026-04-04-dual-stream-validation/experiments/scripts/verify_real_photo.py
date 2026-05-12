#!/usr/bin/env python3
"""
真实 Photo 数据 + 真实图结构验证

目标：
1. 加载真实 Photo 数据集（特征 + 邻接矩阵）
2. 用真实图结构生成 Hop features
3. 测量多层 Attention 的 Delta 信息损失

关键改进：
- 使用真实邻接矩阵进行 Hop 特征聚合
- Delta 结构反映真实的图拓扑关系
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.linear_model import Ridge
import os
import sys

print('=' * 70)
print('真实 Photo 数据 + 图结构验证')
print('=' * 70)

# ===== 1. 加载真实 Photo 数据 =====
print('\n[1] 加载真实 Photo 数据集...')

data_path = '/root/gpufree-data/linziyao/MatrixGAD/dataset/photo.mat'
data = sio.loadmat(data_path)

# 提取特征和标签
features = data['Attributes']
if hasattr(features, 'todense'):
    features = features.todense()
features = torch.tensor(np.array(features), dtype=torch.float32)

labels = data['Label'].flatten()
labels = torch.tensor(labels, dtype=torch.long)

# 提取邻接矩阵
adj = data['Network']
if hasattr(adj, 'todense'):
    adj_sparse = adj
else:
    adj_sparse = sp.csr_matrix(adj)

print(f'    节点数: {features.shape[0]}')
print(f'    特征维度: {features.shape[1]}')
print(f'    异常节点: {labels.sum().item()} / {len(labels)} ({labels.sum().item()/len(labels)*100:.1f}%)')
print(f'    边数: {adj_sparse.nnz}')

# ===== 2. 用真实图结构生成 Hop features =====
print('\n[2] 用真实图结构生成 Hop features...')

def normalize_adj(adj):
    """标准化邻接矩阵"""
    adj = adj + sp.eye(adj.shape[0])  # 加自环
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def get_hop_features_real(features, adj, k=6):
    """用真实邻接矩阵生成 k 跳特征"""
    n_nodes = features.shape[0]
    n_features = features.shape[1]
    
    # 标准化邻接矩阵
    adj_norm = normalize_adj(adj)
    adj_tensor = torch.tensor(adj_norm.todense(), dtype=torch.float32)
    
    hop_features = torch.zeros(n_nodes, k+1, n_features)
    hop_features[:, 0, :] = features  # Hop_0 = 自身
    
    for hop in range(1, k+1):
        # 真实邻居聚合
        hop_features[:, hop, :] = torch.matmul(adj_tensor, hop_features[:, hop-1, :])
    
    return hop_features

n_hops = 7  # pp_k + 1 (VoxGFormer 默认)
hop_features = get_hop_features_real(features, adj_sparse, k=6)

print(f'    Hop features shape: {hop_features.shape}')

# 计算 Delta (相邻 Hop 的差异)
delta_features = hop_features[:, 1:, :] - hop_features[:, :-1, :]
print(f'    Delta features shape: {delta_features.shape}')

# ===== 3. 分析 Delta 结构 =====
print('\n[3] 分析真实 Delta 结构...')

delta_norm = delta_features.norm(dim=-1)  # (n_nodes, n_hops-1)
print(f'    Delta norm range: [{delta_norm.min().item():.4f}, {delta_norm.max().item():.4f}]')
print(f'    Delta norm mean: {delta_norm.mean().item():.4f}')

# 异常 vs 正常的 Delta 差异
normal_mask = labels == 0
anomaly_mask = labels == 1

delta_norm_normal = delta_norm[normal_mask].mean().item()
delta_norm_anomaly = delta_norm[anomaly_mask].mean().item()

print(f'    正常节点 Delta norm: {delta_norm_normal:.4f}')
print(f'    异常节点 Delta norm: {delta_norm_anomaly:.4f}')
print(f'    差异: {(delta_norm_anomaly - delta_norm_normal) / delta_norm_normal * 100:.1f}%')

# ===== 4. Linear Probe 测试 =====
print('\n[4] Linear Probe: 从 Hop 预测 Delta...')

def linear_probe_r2(hop_data, delta_data, sample_size=None):
    """测量 R²"""
    if sample_size:
        idx = torch.randperm(hop_data.shape[0])[:sample_size]
        hop_data = hop_data[idx]
        delta_data = delta_data[idx]
    
    hop_flat = hop_data.reshape(hop_data.shape[0], -1).numpy()
    delta_flat = delta_data.reshape(delta_data.shape[0], -1).numpy()
    
    probe = Ridge(alpha=1.0)
    probe.fit(hop_flat, delta_flat)
    r2 = probe.score(hop_flat, delta_flat)
    return r2

# 输入层 R²
r2_input = linear_probe_r2(hop_features, delta_features)
print(f'    输入层 R²: {r2_input:.4f}')

# ===== 5. 多层 Attention 模拟 =====
print('\n[5] 多层 Attention 模拟...')

class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads=1):  # 使用 n_heads=1 以适应任意 d_model
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)

n_layers_to_test = [1, 2, 3, 4, 6]
d_model = features.shape[1]

results = []

for n_layers in n_layers_to_test:
    print(f'\n    测试 {n_layers} 层...')
    
    # 构建模型
    layers = nn.ModuleList([AttentionLayer(d_model) for _ in range(n_layers)])
    
    # 处理 Hop features
    with torch.no_grad():
        x = hop_features
        for layer in layers:
            x = layer(x)
        hop_processed = x
    
    # 测量 R²
    r2_output = linear_probe_r2(hop_processed, delta_features)
    
    # 损失
    loss_pct = (r2_input - r2_output) / abs(r2_input) * 100 if r2_input != 0 else 0
    
    results.append({
        'n_layers': n_layers,
        'r2_input': r2_input,
        'r2_output': r2_output,
        'loss_pct': loss_pct
    })
    
    print(f'      输出 R²: {r2_output:.4f}')
    print(f'      损失: {loss_pct:.1f}%')

# ===== 6. MLP 对照 =====
print('\n[6] MLP 对照实验...')

class MLPChannel(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        return self.norm(x + self.mlp(x))

mlp_layers = nn.ModuleList([MLPChannel(d_model) for _ in range(6)])

with torch.no_grad():
    x = hop_features
    for layer in mlp_layers:
        x = layer(x)
    hop_mlp = x

r2_mlp = linear_probe_r2(hop_mlp, delta_features)
print(f'    6层 MLP R²: {r2_mlp:.4f}')
print(f'    vs Attention: {r2_mlp - results[-1]["r2_output"]:.4f}')

# ===== 7. 结果汇总 =====
print('\n' + '=' * 70)
print('结果汇总')
print('=' * 70)

print(f'\n{"层数":<10} {"输入 R²":<15} {"输出 R²":<15} {"损失":<15}')
print('-' * 55)
for r in results:
    print(f'{r["n_layers"]:<10} {r["r2_input"]:<15.4f} {r["r2_output"]:<15.4f} {r["loss_pct"]:<15.1f}%')
print(f'{"MLP-6":<10} {r2_input:<15.4f} {r2_mlp:<15.4f} {(r2_input-r2_mlp)/abs(r2_input)*100:.1f}%')

# ===== 8. 结论 =====
print('\n' + '=' * 70)
print('结论')
print('=' * 70)

loss_1layer = results[0]['loss_pct']
loss_6layer = results[-1]['loss_pct']
mlp_vs_attn = r2_mlp - results[-1]['r2_output']

print(f'\n关键发现:')
print(f'  输入层 R²: {r2_input:.4f} (Delta 可从 Hop 提取)')
print(f'  单层 Attention 损失: {loss_1layer:.1f}%')
print(f'  6层 Attention 损失: {loss_6layer:.1f}%')
print(f'  6层 MLP 损失: {(r2_input-r2_mlp)/abs(r2_input)*100:.1f}%')
print(f'  MLP vs Attention: {mlp_vs_attn:.4f}')

if loss_6layer > 40:
    print('\n[VALIDATED] 多层累积效应显著')
elif loss_6layer > 20:
    print('\n[PARTIAL] 累积效应存在')
else:
    print('\n[NOTE] 累积效应不如预期')

if mlp_vs_attn > 0:
    print('\n[VALIDATED] MLP 保留 Delta > Attention')
else:
    print('\n[WARNING] MLP 未优于 Attention')

# ===== 9. 保存结果 =====
output_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(output_dir, '../outputs')
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, 'real_photo_results.txt')

with open(output_file, 'w') as f:
    f.write('真实 Photo 数据验证结果\n')
    f.write('=' * 50 + '\n\n')
    f.write(f'数据集: Photo\n')
    f.write(f'节点数: {features.shape[0]}\n')
    f.write(f'特征维度: {features.shape[1]}\n')
    f.write(f'异常比例: {labels.sum().item()/len(labels)*100:.1f}%\n\n')
    f.write('Delta 结构:\n')
    f.write(f'  正常节点 Delta norm: {delta_norm_normal:.4f}\n')
    f.write(f'  异常节点 Delta norm: {delta_norm_anomaly:.4f}\n\n')
    f.write('多层累积:\n')
    for r in results:
        f.write(f'  {r["n_layers"]}层: R²={r["r2_output"]:.4f}, 损失={r["loss_pct"]:.1f}%\n')
    f.write(f'  MLP-6: R²={r2_mlp:.4f}\n')

print(f'\n结果已保存: {output_file}')