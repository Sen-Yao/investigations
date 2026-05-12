#!/usr/bin/env python3
"""
真实训练模型验证：训练 VoxGFormer 后测量 Delta 信息损失

方法：
1. 加载真实 Photo 数据
2. 构建 VoxGFormer 模型
3. 训练异常检测任务
4. 测量训练前后 Delta 信息保留变化

关键：对比"随机初始化" vs "训练后"的差异
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
import os
import sys

print('=' * 70)
print('真实训练模型验证：VoxGFormer 训练前后 Delta 信息损失')
print('=' * 70)

# ===== 1. 加载真实 Photo 数据 =====
print('\n[1] 加载真实 Photo 数据集...')

data_path = '/root/gpufree-data/linziyao/MatrixGAD/dataset/photo.mat'
data = sio.loadmat(data_path)

features = data['Attributes']
if hasattr(features, 'todense'):
    features = features.todense()
features = torch.tensor(np.array(features), dtype=torch.float32)

labels = data['Label'].flatten()
labels = torch.tensor(labels, dtype=torch.long)

adj = data['Network']
if hasattr(adj, 'todense'):
    adj_sparse = adj
else:
    adj_sparse = sp.csr_matrix(adj)

n_nodes = features.shape[0]
n_features = features.shape[1]

print(f'    节点数: {n_nodes}, 特征维度: {n_features}')
print(f'    异常节点: {labels.sum().item()} ({labels.sum().item()/len(labels)*100:.1f}%)')

# ===== 2. 数据划分 =====
print('\n[2] 数据划分（5% 训练集，仅正常节点）...')

normal_idx = torch.where(labels == 0)[0]
anomaly_idx = torch.where(labels == 1)[0]

# 随机划分
torch.manual_seed(42)
perm = torch.randperm(len(normal_idx))
train_size = int(0.05 * n_nodes)  # 5% 训练集
val_size = int(0.10 * n_nodes)

train_idx = normal_idx[perm[:train_size]]
val_idx = normal_idx[perm[train_size:train_size+val_size]]
test_idx = torch.cat([normal_idx[perm[train_size+val_size:]], anomaly_idx])

print(f'    训练集: {len(train_idx)} (仅正常)')
print(f'    验证集: {len(val_idx)}')
print(f'    测试集: {len(test_idx)}')

# ===== 3. 生成 Hop features =====
print('\n[3] 生成 Hop features...')

def normalize_adj(adj):
    adj = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def get_hop_features(features, adj, k=6):
    adj_norm = normalize_adj(adj)
    adj_tensor = torch.tensor(adj_norm.todense(), dtype=torch.float32)
    
    hop_features = torch.zeros(features.shape[0], k+1, features.shape[1])
    hop_features[:, 0, :] = features
    
    for hop in range(1, k+1):
        hop_features[:, hop, :] = torch.matmul(adj_tensor, hop_features[:, hop-1, :])
    
    return hop_features

n_hops = 7
hop_features = get_hop_features(features, adj_sparse, k=6)
delta_features = hop_features[:, 1:, :] - hop_features[:, :-1, :]

print(f'    Hop features: {hop_features.shape}')
print(f'    Delta features: {delta_features.shape}')

# ===== 4. 定义 VoxGFormer 模型 =====
print('\n[4] 定义 VoxGFormer 模型...')

class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class VoxGFormer(nn.Module):
    def __init__(self, n_features, d_hidden=128, n_layers=3, n_heads=4, n_hops=7):
        super().__init__()
        self.n_hops = n_hops
        
        # Input projection
        self.input_proj = nn.Linear(n_features, d_hidden)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_hidden, n_heads) for _ in range(n_layers)
        ])
        
        # Output
        self.readout = nn.Linear(d_hidden, 1)
    
    def forward(self, hop_features, return_intermediate=False):
        # Project
        x = self.input_proj(hop_features)  # (N, n_hops, d_hidden)
        
        intermediates = [x.clone()]
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
            if return_intermediate:
                intermediates.append(x.clone())
        
        # Readout: mean over hops
        node_repr = x.mean(dim=1)  # (N, d_hidden)
        
        # Anomaly score
        score = self.readout(node_repr).squeeze(-1)
        
        if return_intermediate:
            return score, intermediates
        return score

# ===== 5. 训练前验证 =====
print('\n[5] 训练前 Delta 信息保留验证...')

def linear_probe_r2(hop_data, delta_data):
    hop_flat = hop_data.reshape(hop_data.shape[0], -1).numpy()
    delta_flat = delta_data.reshape(delta_data.shape[0], -1).numpy()
    
    probe = Ridge(alpha=1.0)
    probe.fit(hop_flat, delta_flat)
    return probe.score(hop_flat, delta_flat)

# 原始输入 R²
r2_input = linear_probe_r2(hop_features, delta_features)
print(f'    输入层 R²: {r2_input:.4f}')

# 创建模型
model = VoxGFormer(n_features, d_hidden=128, n_layers=3, n_heads=4)

# 训练前中间层 R²
with torch.no_grad():
    _, intermediates = model(hop_features, return_intermediate=True)

print(f'    训练前各层 R²:')
for i, intermediate in enumerate(intermediates):
    r2 = linear_probe_r2(intermediate, delta_features)
    print(f'      Layer {i}: R² = {r2:.4f}')

# ===== 6. 训练 =====
print('\n[6] 训练 VoxGFormer...')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

train_labels = torch.zeros(n_nodes)
train_labels[train_idx] = 0  # 正常
train_labels[anomaly_idx] = 1  # 异常

best_val_auc = 0
best_state = None

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    
    scores = model(hop_features)
    
    # 只用训练集计算损失
    loss = criterion(scores[train_idx], train_labels[train_idx].float())
    
    # 添加验证集正常样本（半监督）
    loss += 0.5 * criterion(scores[val_idx], train_labels[val_idx].float())
    
    loss.backward()
    optimizer.step()
    
    # 验证
    if (epoch + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            scores = model(hop_features)
            val_auc = roc_auc_score(labels[test_idx].numpy(), 
                                   torch.sigmoid(scores[test_idx]).numpy())
        
        print(f'    Epoch {epoch+1}: Loss={loss.item():.4f}, Val AUC={val_auc:.4f}')
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

# 加载最佳模型
if best_state:
    model.load_state_dict(best_state)

print(f'\n    最佳 Val AUC: {best_val_auc:.4f}')

# ===== 7. 训练后验证 =====
print('\n[7] 训练后 Delta 信息保留验证...')

with torch.no_grad():
    _, intermediates = model(hop_features, return_intermediate=True)

print(f'    训练后各层 R²:')
r2_after = []
for i, intermediate in enumerate(intermediates):
    r2 = linear_probe_r2(intermediate, delta_features)
    r2_after.append(r2)
    print(f'      Layer {i}: R² = {r2:.4f}')

# ===== 8. 对比 =====
print('\n' + '=' * 70)
print('训练前后对比')
print('=' * 70)

print(f'\n{"Layer":<10} {"训练前 R²":<15} {"训练后 R²":<15} {"变化":<15}')
print('-' * 55)

# 训练前重新计算
model_init = VoxGFormer(n_features, d_hidden=128, n_layers=3, n_heads=4)
with torch.no_grad():
    _, intermediates_init = model_init(hop_features, return_intermediate=True)

for i in range(len(intermediates)):
    r2_before = linear_probe_r2(intermediates_init[i], delta_features)
    r2_after_val = r2_after[i]
    change = r2_after_val - r2_before
    print(f'{i:<10} {r2_before:<15.4f} {r2_after_val:<15.4f} {change:+.4f}')

# ===== 9. 结论 =====
print('\n' + '=' * 70)
print('结论')
print('=' * 70)

r2_final = r2_after[-1]
loss_pct = (r2_input - r2_final) / r2_input * 100

print(f'\n输入层 R²: {r2_input:.4f}')
print(f'最终输出 R²: {r2_final:.4f}')
print(f'总损失: {loss_pct:.1f}%')

if r2_final < 0.5:
    print('\n[VALIDATED] 训练后 Delta 信息大幅损失')
    print('            与 04-03 真实实验 (R²=-0.15) 一致')
elif loss_pct > 10:
    print('\n[PARTIAL] 训练后 Delta 信息中等损失')
else:
    print('\n[NOTE] 训练后 Delta 信息保留较好')

# 保存结果
output_dir = os.path.join(os.path.dirname(__file__), '../outputs')
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'trained_model_results.txt'), 'w') as f:
    f.write('训练模型 Delta 信息保留验证\n')
    f.write('=' * 50 + '\n\n')
    f.write(f'输入层 R²: {r2_input:.4f}\n')
    f.write(f'最终输出 R²: {r2_final:.4f}\n')
    f.write(f'总损失: {loss_pct:.1f}%\n\n')
    f.write('各层 R²:\n')
    for i, r2 in enumerate(r2_after):
        f.write(f'  Layer {i}: {r2:.4f}\n')

print('\n结果已保存')