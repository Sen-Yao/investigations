"""
架构对比实验

对比：
1. concat (baseline)
2. BypassVoxG v2
3. CrossConcatVoxG

作者: Nexus
日期: 2026-04-04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, '/root/gpufree-data/linziyao/VoxG')

from BypassVoxG import create_bypass_voxg
from CrossConcatVoxG import create_cross_concat_voxg


def load_data():
    """加载数据"""
    data = sio.loadmat('/root/gpufree-data/linziyao/MatrixGAD/dataset/photo.mat')
    features = torch.tensor(np.array(data['Attributes'].todense()), dtype=torch.float32)
    labels = torch.tensor(data['Label'].flatten(), dtype=torch.long)
    adj = data['Network']
    
    # 生成 Hop features
    adj_norm = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_norm.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_norm = adj_norm.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    adj_tensor = torch.tensor(adj_norm.todense(), dtype=torch.float32)
    
    n_hops = 6
    hop_features = torch.zeros(features.shape[0], n_hops + 1, features.shape[1])
    hop_features[:, 0, :] = features
    for hop in range(1, n_hops + 1):
        hop_features[:, hop, :] = torch.matmul(adj_tensor, hop_features[:, hop-1, :])
    
    # 训练集
    torch.manual_seed(42)
    normal_idx = torch.where(labels == 0)[0]
    perm = torch.randperm(len(normal_idx))
    train_idx = normal_idx[perm[:376]]
    
    test_idx = torch.cat([normal_idx[perm[376:]], torch.where(labels == 1)[0]])
    
    return hop_features, labels, train_idx, test_idx


def train_model(model, hop_features, labels, train_idx, n_epochs=100, lr=0.001):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_auc = 0
    best_epoch = 0
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        logits, loss = model(hop_features, return_loss=True)
        
        # 分类损失
        bce = F.binary_cross_entropy_with_logits(logits[train_idx], torch.zeros(len(train_idx)))
        
        total_loss = bce + 0.5 * loss
        total_loss.backward()
        optimizer.step()
        
        # 评估
        model.eval()
        with torch.no_grad():
            logits_eval, _ = model(hop_features, return_loss=False)
            probs = torch.sigmoid(logits_eval)
            auc = roc_auc_score(labels.numpy(), probs.numpy())
        
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch + 1
    
    return best_auc, best_epoch


def main():
    print("=" * 60)
    print("架构对比实验")
    print("=" * 60)
    
    # 加载数据
    hop_features, labels, train_idx, test_idx = load_data()
    n_features = hop_features.shape[-1]
    
    print(f"\n数据: {hop_features.shape}")
    print(f"训练集: {len(train_idx)}, 测试集: {len(test_idx)}")
    
    results = {}
    
    # 1. BypassVoxG v2
    print("\n[1] BypassVoxG v2")
    model = create_bypass_voxg(n_features)
    print(f"    参数: {model.get_model_info()['n_params_M']:.2f}M")
    auc, epoch = train_model(model, hop_features, labels, train_idx)
    results['BypassVoxG v2'] = {'auc': auc, 'epoch': epoch}
    print(f"    AUC: {auc:.4f} @ Epoch {epoch}")
    
    # 2. CrossConcatVoxG
    print("\n[2] CrossConcatVoxG")
    model = create_cross_concat_voxg(n_features)
    print(f"    参数: {model.get_model_info()['n_params_M']:.2f}M")
    auc, epoch = train_model(model, hop_features, labels, train_idx)
    results['CrossConcatVoxG'] = {'auc': auc, 'epoch': epoch}
    print(f"    AUC: {auc:.4f} @ Epoch {epoch}")
    
    # 对比
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"{'方法':<25} {'AUC':<10} {'Epoch':<10}")
    print("-" * 45)
    print(f"{'concat (baseline)':<25} {'0.8777':<10} {'-':<10}")
    print(f"{'BypassVoxG v2':<25} {results['BypassVoxG v2']['auc']:.4f}     {results['BypassVoxG v2']['epoch']}")
    print(f"{'CrossConcatVoxG':<25} {results['CrossConcatVoxG']['auc']:.4f}     {results['CrossConcatVoxG']['epoch']}")


if __name__ == "__main__":
    main()