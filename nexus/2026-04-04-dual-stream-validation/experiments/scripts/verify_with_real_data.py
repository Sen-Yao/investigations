#!/usr/bin/env python3
"""
Real Data Validation: Delta Information Preservation
Training-based comparison between Attention and MLP
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio

print('=' * 60)
print('Real Data Validation: Delta Information Preservation')
print('=' * 60)

# Load Photo data
data = sio.loadmat('/root/gpufree-data/linziyao/VoxG/data/Photo.mat')
features = torch.tensor(data['X'].todense(), dtype=torch.float32)
labels = torch.tensor(data['y'].flatten(), dtype=torch.long)
adj = data['A']

print(f'Dataset: Photo, Nodes={features.shape[0]}, Features={features.shape[1]}')

# Simulate Hop features (simplified neighbor aggregation)
def get_hop_features(features, adj, k=6):
    hop_features = [features]
    adj_tensor = torch.tensor(adj.todense(), dtype=torch.float32)
    
    for _ in range(k):
        aggregated = torch.matmul(adj_tensor, hop_features[-1])
        aggregated = aggregated / (aggregated.norm(dim=1, keepdim=True) + 1e-8)
        hop_features.append(aggregated)
    
    return torch.stack(hop_features, dim=1)

hop_features = get_hop_features(features, adj, k=6)
print(f'Hop features shape: {hop_features.shape}')

# Compute Delta
delta_features = hop_features[:, 1:, :] - hop_features[:, :-1, :]
print(f'Delta features shape: {delta_features.shape}')

# Method 1: Standard Attention
class SimpleAttention(nn.Module):
    def __init__(self, d_model, n_heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
    
    def forward(self, x):
        return self.attn(x, x, x)[0]

# Method 2: Dedicated Channel (MLP)
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

d_model = features.shape[1]
attn = SimpleAttention(d_model)
mlp = DedicatedChannel(d_model)

# Sample for testing
sample_idx = torch.randint(0, hop_features.shape[0], (50,))
delta_sample = delta_features[sample_idx]

def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0),
        b.flatten().unsqueeze(0)
    ).item()

# Before training
with torch.no_grad():
    delta_after_attn = attn(delta_sample)
    delta_after_mlp = mlp(delta_sample)

cos_attn_before = cosine_sim(delta_sample, delta_after_attn)
cos_mlp_before = cosine_sim(delta_sample, delta_after_mlp)

print(f'\nBefore Training:')
print(f'  Attention: {cos_attn_before:.4f}')
print(f'  MLP:       {cos_mlp_before:.4f}')

# Training: information preservation objective
print(f'\nTraining to preserve Delta information...')

optimizer_attn = torch.optim.Adam(attn.parameters(), lr=0.001)
optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=0.001)

for step in range(200):
    # Attention training
    optimizer_attn.zero_grad()
    out_attn = attn(delta_sample)
    loss_attn = ((out_attn - delta_sample) ** 2).mean()
    loss_attn.backward()
    optimizer_attn.step()
    
    # MLP training
    optimizer_mlp.zero_grad()
    out_mlp = mlp(delta_sample)
    loss_mlp = ((out_mlp - delta_sample) ** 2).mean()
    loss_mlp.backward()
    optimizer_mlp.step()
    
    if (step + 1) % 50 == 0:
        with torch.no_grad():
            cos_a = cosine_sim(delta_sample, attn(delta_sample))
            cos_m = cosine_sim(delta_sample, mlp(delta_sample))
        print(f'  Step {step+1}: Attention={cos_a:.4f}, MLP={cos_m:.4f}')

# After training
with torch.no_grad():
    delta_after_attn_trained = attn(delta_sample)
    delta_after_mlp_trained = mlp(delta_sample)

cos_attn_after = cosine_sim(delta_sample, delta_after_attn_trained)
cos_mlp_after = cosine_sim(delta_sample, delta_after_mlp_trained)

print(f'\n' + '=' * 60)
print('Results')
print('=' * 60)
print(f'\nAfter Training:')
print(f'  Attention: {cos_attn_after:.4f}')
print(f'  MLP:       {cos_mlp_after:.4f}')

improvement = cos_mlp_after - cos_attn_after
print(f'\nMLP vs Attention improvement: {improvement:.4f}')

if cos_mlp_after > cos_attn_after:
    print('\n[PASS] Dual-stream theory validated:')
    print('       Dedicated channel (MLP) preserves Delta better than Attention')
else:
    print('\n[FAIL] Need to reconsider dual-stream design')

print(f'\n' + '=' * 60)
print('Conclusion')
print('=' * 60)
print('Theoretical basis for dual-stream architecture:')
if cos_mlp_after > cos_attn_after:
    print('  [VALID] Delta information requires dedicated processing channel')
    print('  [VALID] Point-product attention is suboptimal for difference signals')
else:
    print('  [UNCERTAIN] Further investigation needed')