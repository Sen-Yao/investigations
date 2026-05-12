#!/usr/bin/env python3
"""
Hop-Offset vs Delta 信息熵分析
任务：比较 Hop-Offset 和 Delta 两种特征表示的信息量

方法：
- Hop-Offset: alpha=0, pp_k=10 的多跳传播特征
- Delta: 相邻跳特征的差值 (X^{(k+1)} - X^{(k)})
- 信息熵: Shannon Entropy 衡量特征分布的不确定性/信息量

数据集: Photo, Elliptic, Tolokers
"""

import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import torch
from collections import Counter
import sys
import os

# 添加 VoxG 路径
sys.path.insert(0, '/root/gpufree-data/linziyao/VoxG')

def load_dataset(dataset_name):
    """加载数据集"""
    data_path = f'/root/gpufree-data/linziyao/VoxG/dataset/{dataset_name}.mat'
    data = sio.loadmat(data_path)
    
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']
    
    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)
    
    ano_labels = np.squeeze(np.array(label))
    
    return adj, feat, ano_labels

def preprocess_features(features):
    """Row-normalize feature matrix"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def compute_hop_offset_features(adj, features, pp_k=10, alpha=0):
    """
    计算 Hop-Offset 特征 (alpha=0 时的多跳传播)
    
    Args:
        adj: 邻接矩阵 (normalized + self-loop)
        features: 节点特征
        pp_k: 传播跳数
        alpha: 残差系数 (alpha=0 表示纯传播，无残差)
    
    Returns:
        nodes_features: [N, pp_k+1, D] 包含 0-hop 到 pp_k-hop 的特征
    """
    # 预处理邻接矩阵
    adj_normalized = normalize_adj(adj)
    adj_normalized = (adj_normalized + sp.eye(adj_normalized.shape[0])).todense()
    adj_tensor = torch.FloatTensor(np.array(adj_normalized))
    
    # 预处理特征
    if sp.issparse(features):
        features_dense = np.array(features.todense())
    else:
        features_dense = np.array(features)
    features_tensor = torch.FloatTensor(features_dense)
    
    # 收集每跳特征
    nodes_features = features_tensor.unsqueeze(1)  # [N, 1, D]
    x_prev = features_tensor
    
    for hop in range(pp_k):
        # alpha=0: 纯传播 X^{(k+1)} = A * X^{(k)}
        x_k = torch.mm(adj_tensor, x_prev)
        nodes_features = torch.concat((nodes_features, x_k.unsqueeze(1)), dim=1)
        x_prev = x_k
    
    return nodes_features  # [N, pp_k+1, D]

def compute_delta_features(hop_offset_features):
    """
    计算 Delta 特征 (相邻跳特征的差值)
    
    Args:
        hop_offset_features: [N, pp_k+1, D]
    
    Returns:
        delta_features: [N, pp_k, D] (Delta = X^{(k+1)} - X^{(k)})
    """
    delta_features = hop_offset_features[:, 1:, :] - hop_offset_features[:, :-1, :]
    return delta_features

def compute_shannon_entropy_1d(values):
    """
    计算 1D 数组的 Shannon 熵
    将值视为概率分布的一部分
    """
    values = np.abs(values).flatten()
    total = values.sum()
    if total == 0 or len(values) == 0:
        return 0.0
    
    probs = values / total
    probs = probs[probs > 1e-10]  # 过滤接近 0 的值
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

def compute_entropy_per_node(features):
    """计算每个节点的熵 (跨所有维度)"""
    features_np = features.numpy() if isinstance(features, torch.Tensor) else np.array(features)
    # features_np shape: [N, num_hops, D]
    
    # 对每个节点，将其所有特征值视为一个分布
    node_entropies = []
    for i in range(features_np.shape[0]):
        node_values = features_np[i].flatten()
        entropy = compute_shannon_entropy_1d(node_values)
        node_entropies.append(entropy)
    
    return np.mean(node_entropies), np.std(node_entropies)

def compute_entropy_per_hop(features):
    """计算每个 hop 层的平均熵"""
    features_np = features.numpy() if isinstance(features, torch.Tensor) else np.array(features)
    # features_np shape: [N, num_hops, D]
    
    hop_entropies = []
    for hop in range(features_np.shape[1]):
        hop_values = features_np[:, hop, :]  # [N, D]
        entropy = compute_shannon_entropy_1d(hop_values)
        hop_entropies.append(entropy)
    
    return hop_entropies

def compute_entropy_statistics(features, name):
    """计算并打印熵统计信息"""
    features_np = features.numpy() if isinstance(features, torch.Tensor) else np.array(features)
    
    # 1. 每节点的熵
    node_entropy_mean, node_entropy_std = compute_entropy_per_node(features)
    
    # 2. 每 hop 的熵
    hop_entropies = compute_entropy_per_hop(features)
    
    # 3. 整体熵
    total_entropy = compute_shannon_entropy_1d(features_np)
    
    # 4. 特征范数统计 (辅助信息)
    norms = np.linalg.norm(features_np, axis=-1)  # [N, num_hops]
    
    return {
        'name': name,
        'shape': list(features_np.shape),
        'node_entropy_mean': node_entropy_mean,
        'node_entropy_std': node_entropy_std,
        'hop_entropies': hop_entropies,
        'total_entropy': total_entropy,
        'norm_mean': float(np.mean(norms)),
        'norm_std': float(np.std(norms)),
        'norm_min': float(np.min(norms)),
        'norm_max': float(np.max(norms))
    }

def analyze_dataset(dataset_name, pp_k=10, alpha=0):
    """分析单个数据集"""
    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*60}")
    
    # 加载数据
    adj, feat, labels = load_dataset(dataset_name)
    
    # 预处理特征
    if dataset_name in ['Amazon', 'tf_finace', 'reddit', 'elliptic']:
        features = preprocess_features(feat)
    else:
        features = np.array(feat.todense())
    
    num_nodes = features.shape[0]
    feat_dim = features.shape[1]
    print(f"节点数: {num_nodes}, 特征维度: {feat_dim}")
    print(f"异常节点数: {np.sum(labels == 1)}, 正常节点数: {np.sum(labels == 0)}")
    
    # 计算 Hop-Offset 特征
    print(f"\n计算 Hop-Offset 特征 (alpha={alpha}, pp_k={pp_k})...")
    hop_offset_features = compute_hop_offset_features(adj, features, pp_k=pp_k, alpha=alpha)
    print(f"Hop-Offset shape: {hop_offset_features.shape}")
    
    # 计算 Delta 特征
    print(f"\n计算 Delta 特征...")
    delta_features = compute_delta_features(hop_offset_features)
    print(f"Delta shape: {delta_features.shape}")
    
    # 计算熵统计
    hop_offset_stats = compute_entropy_statistics(hop_offset_features, "Hop-Offset")
    delta_stats = compute_entropy_statistics(delta_features, "Delta")
    
    # 打印统计信息
    print(f"\n{'='*40}")
    print(f"Hop-Offset 统计:")
    print(f"{'='*40}")
    print(f"  Shape: {hop_offset_stats['shape']}")
    print(f"  每节点平均熵: {hop_offset_stats['node_entropy_mean']:.4f} ± {hop_offset_stats['node_entropy_std']:.4f} bits")
    print(f"  整体熵: {hop_offset_stats['total_entropy']:.4f} bits")
    print(f"  特征范数: mean={hop_offset_stats['norm_mean']:.4f}, std={hop_offset_stats['norm_std']:.4f}")
    print(f"  各 hop 熵值:")
    for i, h_entropy in enumerate(hop_offset_stats['hop_entropies']):
        print(f"    hop {i}: {h_entropy:.4f} bits")
    
    print(f"\n{'='*40}")
    print(f"Delta 统计:")
    print(f"{'='*40}")
    print(f"  Shape: {delta_stats['shape']}")
    print(f"  每节点平均熵: {delta_stats['node_entropy_mean']:.4f} ± {delta_stats['node_entropy_std']:.4f} bits")
    print(f"  整体熵: {delta_stats['total_entropy']:.4f} bits")
    print(f"  特征范数: mean={delta_stats['norm_mean']:.4f}, std={delta_stats['norm_std']:.4f}")
    print(f"  各 delta 熵值:")
    for i, d_entropy in enumerate(delta_stats['hop_entropies']):
        print(f"    delta {i}: {d_entropy:.4f} bits")
    
    # 对比分析
    print(f"\n{'='*40}")
    print(f"对比分析:")
    print(f"{'='*40}")
    
    entropy_ratio_node = delta_stats['node_entropy_mean'] / (hop_offset_stats['node_entropy_mean'] + 1e-10)
    entropy_ratio_total = delta_stats['total_entropy'] / (hop_offset_stats['total_entropy'] + 1e-10)
    
    print(f"  每节点熵比值 (Delta/Hop-Offset): {entropy_ratio_node:.4f}")
    print(f"  整体熵比值 (Delta/Hop-Offset): {entropy_ratio_total:.4f}")
    
    winner = "Hop-Offset" if hop_offset_stats['node_entropy_mean'] > delta_stats['node_entropy_mean'] else "Delta"
    print(f"  📊 信息量更大的方案: {winner}")
    
    return {
        'dataset': dataset_name,
        'num_nodes': num_nodes,
        'feat_dim': feat_dim,
        'hop_offset_stats': hop_offset_stats,
        'delta_stats': delta_stats,
        'entropy_ratio_node': entropy_ratio_node,
        'entropy_ratio_total': entropy_ratio_total,
        'winner': winner
    }

def main():
    """主函数"""
    print("="*60)
    print("Hop-Offset vs Delta 信息熵分析")
    print("="*60)
    print("配置: alpha=0, pp_k=10")
    print("数据集: Photo, Elliptic, Tolokers")
    
    datasets = ['Photo', 'elliptic', 'tolokers']
    results = []
    
    for dataset in datasets:
        try:
            result = analyze_dataset(dataset, pp_k=10, alpha=0)
            results.append(result)
        except Exception as e:
            print(f"\n❌ 数据集 {dataset} 分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总报告
    print("\n" + "="*60)
    print("📊 汇总报告")
    print("="*60)
    
    print(f"\n| 数据集 | 节点数 | 特征维度 | Hop-Offset熵 | Delta熵 | 比值 | 信息量赢家 |")
    print(f"|--------|--------|----------|--------------|---------|------|------------|")
    
    for r in results:
        print(f"| {r['dataset']} | {r['num_nodes']} | {r['feat_dim']} | "
              f"{r['hop_offset_stats']['node_entropy_mean']:.4f} | "
              f"{r['delta_stats']['node_entropy_mean']:.4f} | "
              f"{r['entropy_ratio_node']:.4f} | {r['winner']} |")
    
    # 最终结论
    print("\n" + "="*60)
    print("🏆 最终结论")
    print("="*60)
    
    hop_offset_wins = sum(1 for r in results if r['winner'] == 'Hop-Offset')
    delta_wins = sum(1 for r in results if r['winner'] == 'Delta')
    
    if hop_offset_wins > delta_wins:
        print(f"✅ Hop-Offset 在 {hop_offset_wins}/{len(results)} 个数据集上信息量更大")
        print("   建议: 使用原始 Hop-Offset 特征表示")
    elif delta_wins > hop_offset_wins:
        print(f"✅ Delta 在 {delta_wins}/{len(results)} 个数据集上信息量更大")
        print("   建议: 使用 Delta 特征表示")
    else:
        print(f"⚖️ Hop-Offset 和 Delta 各在 {hop_offset_wins} 个数据集上信息量更大")
        print("   建议: 根据具体数据集选择表示方式")
    
    # 保存结果到文件
    output_path = '/root/gpufree-data/linziyao/VoxG/nexus/investigations/2026-03-30-hop-offset-analysis/experiments/outputs/entropy_results.txt'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("Hop-Offset vs Delta 信息熵分析\n")
        f.write("="*60 + "\n\n")
        
        for r in results:
            f.write(f"数据集: {r['dataset']}\n")
            f.write(f"  节点数: {r['num_nodes']}, 特征维度: {r['feat_dim']}\n")
            f.write(f"  Hop-Offset 每节点熵: {r['hop_offset_stats']['node_entropy_mean']:.4f} bits\n")
            f.write(f"  Delta 每节点熵: {r['delta_stats']['node_entropy_mean']:.4f} bits\n")
            f.write(f"  熵比值 (Delta/Hop-Offset): {r['entropy_ratio_node']:.4f}\n")
            f.write(f"  信息量赢家: {r['winner']}\n\n")
        
        f.write(f"总结: Hop-Offset 赢 {hop_offset_wins} 次, Delta 赢 {delta_wins} 次\n")
    
    print(f"\n结果已保存到: {output_path}")
    
    return results

if __name__ == "__main__":
    results = main()