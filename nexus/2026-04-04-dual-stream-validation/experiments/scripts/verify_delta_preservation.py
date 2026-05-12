#!/usr/bin/env python3
"""
Simplified Delta Information Preservation Validation
Focus: Does attention destroy delta information?
"""

import torch
import torch.nn as nn
import numpy as np

print('=' * 60)
print('Delta Information Preservation Validation')
print('=' * 60)

# Setup
torch.manual_seed(42)
np.random.seed(42)

batch_size = 100
seq_len = 6  # Delta tokens
d_model = 128

# Generate structured Delta (not random noise)
# Delta = difference between consecutive hop features
# Simulate: normal nodes have small delta, anomalous have large delta

# Create base features
base = torch.randn(batch_size, 1, d_model).repeat(1, seq_len, 1)

# Add structured delta: normal = small variation, anomaly = large variation
normal_mask = torch.rand(batch_size) > 0.1  # 90% normal
delta_scale = torch.where(normal_mask.unsqueeze(-1).unsqueeze(-1), 
                          torch.ones(1) * 0.1,  # normal: small delta
                          torch.ones(1) * 0.8)  # anomaly: large delta

delta_features = base + delta_scale * torch.randn(batch_size, seq_len, d_model)
delta_features = delta_features / delta_features.norm(dim=-1, keepdim=True)

print(f'Delta features: {delta_features.shape}')
print(f'Normal samples: {normal_mask.sum().item()}, Anomaly: {(~normal_mask).sum().item()}')

# Models
class StandardAttention(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        return self.norm(x + self.attn(x, x, x)[0])

class DedicatedChannel(nn.Module):
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

attn = StandardAttention(d_model)
mlp = DedicatedChannel(d_model)

def compute_metrics(original, processed):
    """Compute information preservation metrics"""
    # Cosine similarity
    cos = torch.nn.functional.cosine_similarity(
        original.flatten(), processed.flatten(), dim=0
    ).item()
    
    # MSE
    mse = ((original - processed) ** 2).mean().item()
    
    # Correlation
    corr = torch.corrcoef(torch.stack([original.flatten(), processed.flatten()]))[0, 1].item()
    
    return cos, mse, corr

# Before training
with torch.no_grad():
    out_attn = attn(delta_features)
    out_mlp = mlp(delta_features)

cos_a, mse_a, corr_a = compute_metrics(delta_features, out_attn)
cos_m, mse_m, corr_m = compute_metrics(delta_features, out_mlp)

print(f'\n--- Before Training ---')
print(f'Attention: cos={cos_a:.4f}, mse={mse_a:.4f}, corr={corr_a:.4f}')
print(f'MLP:       cos={cos_m:.4f}, mse={mse_m:.4f}, corr={corr_m:.4f}')

# Training
print(f'\nTraining (reconstruction objective)...')
optimizer_attn = torch.optim.Adam(attn.parameters(), lr=0.001)
optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=0.001)

for step in range(500):
    # Attention
    optimizer_attn.zero_grad()
    out_a = attn(delta_features)
    loss_a = ((out_a - delta_features) ** 2).mean()
    loss_a.backward()
    optimizer_attn.step()
    
    # MLP
    optimizer_mlp.zero_grad()
    out_m = mlp(delta_features)
    loss_m = ((out_m - delta_features) ** 2).mean()
    loss_m.backward()
    optimizer_mlp.step()
    
    if (step + 1) % 100 == 0:
        with torch.no_grad():
            out_a = attn(delta_features)
            out_m = mlp(delta_features)
            cos_a = torch.nn.functional.cosine_similarity(
                delta_features.flatten(), out_a.flatten(), dim=0).item()
            cos_m = torch.nn.functional.cosine_similarity(
                delta_features.flatten(), out_m.flatten(), dim=0).item()
        print(f'  Step {step+1}: Attention={cos_a:.4f}, MLP={cos_m:.4f}')

# Final evaluation
with torch.no_grad():
    out_attn_final = attn(delta_features)
    out_mlp_final = mlp(delta_features)

cos_a_final, mse_a_final, corr_a_final = compute_metrics(delta_features, out_attn_final)
cos_m_final, mse_m_final, corr_m_final = compute_metrics(delta_features, out_mlp_final)

print(f'\n' + '=' * 60)
print('Final Results (After Training)')
print('=' * 60)
print(f'\n{"Metric":<12} {"Attention":<15} {"MLP":<15} {"Winner":<10}')
print('-' * 55)
print(f'{"Cosine":<12} {cos_a_final:<15.4f} {cos_m_final:<15.4f} {"MLP" if cos_m_final > cos_a_final else "Attention":<10}')
print(f'{"MSE":<12} {mse_a_final:<15.4f} {mse_m_final:<15.4f} {"MLP" if mse_m_final < mse_a_final else "Attention":<10}')
print(f'{"Correlation":<12} {corr_a_final:<15.4f} {corr_m_final:<15.4f} {"MLP" if abs(corr_m_final) > abs(corr_a_final) else "Attention":<10}')

print(f'\n' + '=' * 60)
print('Conclusion')
print('=' * 60)

if cos_m_final > cos_a_final and mse_m_final < mse_a_final:
    print('\n[PASS] Dual-stream theory validated!')
    print('       MLP preserves Delta better than Attention')
    print('       Reason: No point-product in dedicated channel')
elif cos_m_final > cos_a_final:
    print('\n[PARTIAL] MLP has better cosine similarity')
    print('          But MSE comparison is inconclusive')
else:
    print('\n[FAIL] Attention performs similarly or better')
    print('       May need different training strategy')

print(f'\nKey insight:')
print(f'  Attention uses point-product (similarity measure)')
print(f'  Delta represents difference (subtraction operation)')
print(f'  MLP can learn any transformation, including preserving differences')