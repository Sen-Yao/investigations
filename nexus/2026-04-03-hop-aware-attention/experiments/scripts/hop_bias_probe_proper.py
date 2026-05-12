#!/usr/bin/env python3
"""
Hop-Aware Attention 探测实验 - 符合科研规范版本

遵循：
1. 半监督设置：train 只包含正常节点
2. train_rate 与 VecGAD 一致（5%）
3. 5-seed 验证
4. AUC/AP 作为评估指标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.io as sio
import sys
import os

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# ============================================================
# 模型定义
# ============================================================

class RelativeHopBiasAttention(nn.Module):
    """
    方案 A：相对距离偏置
    
    hop_bias 基于 Hop 相对距离 (hop[j] - hop[i])
    """
    
    def __init__(self, d_model, d_proj, n_head, max_hop, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_proj = d_proj
        self.n_head = n_head
        self.max_hop = max_hop
        self.head_dim = d_proj // n_head
        
        self.W_q = nn.Linear(d_model, d_proj)
        self.W_k = nn.Linear(d_model, d_proj)
        self.W_v = nn.Linear(d_model, d_proj)
        
        # 相对距离偏置（可学习）
        self.hop_bias = nn.Parameter(torch.zeros(2 * max_hop - 1))
        
        self.out_proj = nn.Linear(d_proj, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hop_indices):
        batch_size, seq_len, _ = x.shape
        
        Q = self.W_q(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 构建相对距离偏置矩阵
        hop_i = hop_indices.unsqueeze(1)
        hop_j = hop_indices.unsqueeze(0)
        hop_diff = hop_j - hop_i
        bias_idx = hop_diff + (self.max_hop - 1)
        bias_idx = bias_idx.clamp(0, 2 * self.max_hop - 2)
        bias_matrix = self.hop_bias[bias_idx]
        
        scores = scores + bias_matrix.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_proj)
        
        return self.out_proj(out), attn, self.hop_bias.data


class HopBiasDetector(nn.Module):
    """
    探测模型：使用 HopBiasAttention + 分类头
    """
    
    def __init__(self, d_model, d_proj=256, n_head=4, max_hop=7, dropout=0.3):
        super().__init__()
        self.attention = RelativeHopBiasAttention(d_model, d_proj, n_head, max_hop, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
    def forward(self, hop_features, hop_indices):
        out, attn, hop_bias = self.attention(hop_features, hop_indices)
        # 取 hop_0 的输出作为节点表示
        node_repr = out[:, 0, :]  # [batch, d_model]
        logits = self.classifier(node_repr).squeeze(-1)
        return logits, attn, hop_bias


# ============================================================
# 数据加载和划分
# ============================================================

def load_data(dataset_path):
    """加载图数据并计算 hop 特征"""
    data = sio.loadmat(dataset_path)
    
    # 特征和标签
    features = data['Attributes']
    if hasattr(features, 'toarray'):
        features = features.toarray()
    labels = data['Label'].flatten()
    
    # 邻接矩阵
    adj = data['Network']
    if hasattr(adj, 'toarray'):
        adj = adj.toarray()
    
    # 对称归一化
    degree = adj.sum(axis=1)
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0)
    adj_norm = np.diag(d_inv_sqrt) @ adj @ np.diag(d_inv_sqrt)
    
    # 计算 6-hop 特征
    K = 6
    n_nodes, d_feat = features.shape
    hop_features = np.zeros((n_nodes, K+1, d_feat))
    hop_features[:, 0, :] = features
    
    agg = features.copy()
    for k in range(1, K+1):
        agg = adj_norm @ agg
        hop_features[:, k, :] = agg
    
    return hop_features, labels, adj


def split_data(labels, train_rate=0.05, val_rate=0.15, seed=42):
    """
    半监督数据划分
    
    关键：train 只包含正常节点！
    """
    set_seed(seed)
    
    n_nodes = len(labels)
    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]
    
    # 训练集：只从正常节点中采样
    n_train = int(len(normal_idx) * train_rate)
    train_idx = np.random.choice(normal_idx, n_train, replace=False)
    
    # 验证集：从剩余节点中采样（包含正常和异常）
    remaining_normal = np.setdiff1d(normal_idx, train_idx)
    n_val = int(n_nodes * val_rate)
    
    # 验证集保持异常比例
    val_normal = np.random.choice(remaining_normal, n_val // 2, replace=False)
    val_anomaly = np.random.choice(anomaly_idx, n_val // 2, replace=False)
    val_idx = np.concatenate([val_normal, val_anomaly])
    
    # 测试集：剩余所有节点
    test_idx = np.setdiff1d(np.arange(n_nodes), np.concatenate([train_idx, val_idx]))
    
    # 验证数据划分
    assert np.sum(labels[train_idx]) == 0, "数据泄露！训练集包含异常节点！"
    
    return train_idx, val_idx, test_idx


# ============================================================
# 训练和评估
# ============================================================

def train_epoch(model, hop_features, labels, train_idx, hop_indices, optimizer, criterion):
    """训练一个 epoch"""
    model.train()
    optimizer.zero_grad()
    
    hop_features_tensor = torch.FloatTensor(hop_features)
    labels_tensor = torch.FloatTensor(labels)
    
    logits, attn, hop_bias = model(hop_features_tensor, hop_indices)
    
    # 只计算训练集的 loss
    loss = criterion(logits[train_idx], labels_tensor[train_idx])
    
    loss.backward()
    optimizer.step()
    
    return loss.item(), hop_bias


def evaluate(model, hop_features, labels, eval_idx, hop_indices):
    """评估 AUC 和 AP"""
    model.eval()
    
    with torch.no_grad():
        hop_features_tensor = torch.FloatTensor(hop_features)
        logits, _, _ = model(hop_features_tensor, hop_indices)
        
        probs = torch.sigmoid(logits[eval_idx]).numpy()
        true_labels = labels[eval_idx]
        
        auc = roc_auc_score(true_labels, probs)
        ap = average_precision_score(true_labels, probs)
    
    return auc, ap


# ============================================================
# 主实验
# ============================================================

def run_experiment(dataset_path, seeds=[42, 123, 456, 789, 1024], train_rate=0.05, n_epochs=100):
    """
    运行 5-seed 探测实验
    """
    print("="*60)
    print("Hop-Aware Attention 探测实验（符合科研规范）")
    print("="*60)
    
    # 加载数据
    hop_features, labels, adj = load_data(dataset_path)
    n_nodes, n_hops, d_model = hop_features.shape
    
    print(f"\n数据集信息:")
    print(f"  节点数: {n_nodes}")
    print(f"  特征维度: {d_model}")
    print(f"  异常节点: {labels.sum()} ({labels.mean()*100:.2f}%)")
    print(f"  train_rate: {train_rate}")
    
    hop_indices = torch.arange(n_hops)
    
    results = []
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)
        
        # 数据划分
        train_idx, val_idx, test_idx = split_data(labels, train_rate=train_rate, seed=seed)
        
        print(f"  训练集: {len(train_idx)} (正常节点)")
        print(f"  验证集: {len(val_idx)} (正常: {np.sum(labels[val_idx]==0)}, 异常: {np.sum(labels[val_idx]==1)})")
        print(f"  测试集: {len(test_idx)} (正常: {np.sum(labels[test_idx]==0)}, 异常: {np.sum(labels[test_idx]==1)})")
        
        # 模型
        model = HopBiasDetector(d_model, d_proj=256, n_head=4, max_hop=n_hops, dropout=0.3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
        
        # 使用 BCE loss（半监督设置）
        criterion = nn.BCEWithLogitsLoss()
        
        # 训练
        best_val_auc = 0
        best_test_auc = 0
        best_test_ap = 0
        best_hop_bias = None
        
        for epoch in range(n_epochs):
            loss, hop_bias = train_epoch(model, hop_features, labels, train_idx, hop_indices, optimizer, criterion)
            
            # 验证集评估
            val_auc, val_ap = evaluate(model, hop_features, labels, val_idx, hop_indices)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                test_auc, test_ap = evaluate(model, hop_features, labels, test_idx, hop_indices)
                best_test_auc = test_auc
                best_test_ap = test_ap
                best_hop_bias = hop_bias.clone()
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: Loss={loss:.4f}, Val_AUC={val_auc:.4f}, Test_AUC={test_auc:.4f}")
        
        results.append({
            'seed': seed,
            'test_auc': best_test_auc,
            'test_ap': best_test_ap,
            'hop_bias': best_hop_bias
        })
        
        print(f"  最终: Test_AUC={best_test_auc:.4f}, Test_AP={best_test_ap:.4f}")
    
    # 汇总结果
    print("\n" + "="*60)
    print("5-seed 结果汇总")
    print("="*60)
    
    aucs = [r['test_auc'] for r in results]
    aps = [r['test_ap'] for r in results]
    
    print(f"\nTest AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"Test AP:  {np.mean(aps):.4f} ± {np.std(aps):.4f}")
    
    # 分析 hop_bias
    print("\n--- hop_bias 分析 ---")
    all_hop_biases = torch.stack([r['hop_bias'] for r in results])
    mean_bias = all_hop_biases.mean(dim=0)
    std_bias = all_hop_biases.std(dim=0)
    
    print(f"hop_bias (mean): {mean_bias}")
    print(f"hop_bias (std):  {std_bias}")
    
    # 距离衰减模式
    print("\n距离衰减模式:")
    for k in range(n_hops):
        idx = n_hops - 1 + k
        if idx < len(mean_bias):
            print(f"  B[{k}] = {mean_bias[idx]:.4f} ± {std_bias[idx]:.4f}")
    
    return results


def main():
    # 数据集路径
    dataset_path = '/root/gpufree-data/linziyao/VoxG/dataset/Photo.mat'
    
    # 运行实验
    results = run_experiment(
        dataset_path,
        seeds=[42, 123, 456, 789, 1024],
        train_rate=0.05,
        n_epochs=100
    )
    
    # 保存结果
    import json
    output = {
        'dataset': 'Photo',
        'train_rate': 0.05,
        'n_seeds': 5,
        'test_auc_mean': np.mean([r['test_auc'] for r in results]),
        'test_auc_std': np.std([r['test_auc'] for r in results]),
        'test_ap_mean': np.mean([r['test_ap'] for r in results]),
        'test_ap_std': np.std([r['test_ap'] for r in results]),
    }
    
    with open('probe_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ 结果已保存到 probe_results.json")


if __name__ == "__main__":
    main()