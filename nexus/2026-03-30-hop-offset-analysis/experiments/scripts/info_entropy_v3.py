#!/usr/bin/env python3
"""
【v3 实验2】Hop-Offset vs Delta 信息熵分析

关键配置：
- 归一化：对称归一化 D^(-0.5)AD^(-0.5)
- alpha=0.1（残差系数）

任务：
1. 计算 Hop-Offset 和 Delta 的信息熵
2. 对比两者信息量
3. 汇报结果
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

def normalize_adj_symmetric(adj):
    """对称归一化 D^(-0.5)AD^(-0.5)"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return adj_normalized

def compute_hop_features_with_alpha(adj, features, pp_k=10, alpha=0.1):
    """
    计算带残差的多跳传播特征
    
    传播公式: X^{(k+1)} = (1-alpha) * A_norm * X^{(k)} + alpha * X^{(0)}
    
    注意：这里 A_norm 是添加了自环的对称归一化矩阵
    
    Args:
        adj: 原始邻接矩阵
        features: 节点特征 (已预处理)
        pp_k: 传播跳数
        alpha: 残差系数 (alpha=0.1 表示 10% 来自原始特征)
    
    Returns:
        hop_features: [N, pp_k+1, D] 从 0-hop 到 pp_k-hop 的特征
    """
    # 对称归一化 D^(-0.5)AD^(-0.5)
    adj_normalized = normalize_adj_symmetric(adj)
    
    # 添加自环
    adj_with_selfloop = adj_normalized + sp.eye(adj_normalized.shape[0])
    adj_tensor = torch.FloatTensor(np.array(adj_with_selfloop.todense()))
    
    # 预处理特征
    if sp.issparse(features):
        features_dense = np.array(features.todense())
    else:
        features_dense = np.array(features)
    features_tensor = torch.FloatTensor(features_dense)
    
    # 收集每跳特征
    hop_features = features_tensor.unsqueeze(1)  # [N, 1, D]
    x_prev = features_tensor.clone()
    x_0 = features_tensor.clone()  # 原始特征，用于残差
    
    for hop in range(pp_k):
        # 带残差的传播: X^{(k+1)} = (1-alpha) * A * X^{(k)} + alpha * X^{(0)}
        x_k = (1 - alpha) * torch.mm(adj_tensor, x_prev) + alpha * x_0
        hop_features = torch.concat((hop_features, x_k.unsqueeze(1)), dim=1)
        x_prev = x_k
    
    return hop_features  # [N, pp_k+1, D]

def compute_hop_offset_features(hop_features):
    """
    计算 Hop-Offset 特征
    
    Offset_k = Hop_k - Hop_0 (相对于第0跳的偏移)
    
    Args:
        hop_features: [N, pp_k+1, D] 从 0-hop 到 pp_k-hop 的特征
    
    Returns:
        hop_offset_features: [N, pp_k+1, D] 包含 Offset_0 到 Offset_k
    """
    x_0 = hop_features[:, 0:1, :]  # [N, 1, D]
    hop_offset_features = hop_features - x_0  # 广播减法
    return hop_offset_features

def compute_delta_features(hop_features):
    """
    计算 Delta 特征 (相邻跳特征的差值)
    
    Delta_k = Hop_{k+1} - Hop_k
    
    Args:
        hop_features: [N, pp_k+1, D]
    
    Returns:
        delta_features: [N, pp_k, D]
    """
    delta_features = hop_features[:, 1:, :] - hop_features[:, :-1, :]
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
    
    node_entropies = []
    for i in range(features_np.shape[0]):
        node_values = features_np[i].flatten()
        entropy = compute_shannon_entropy_1d(node_values)
        node_entropies.append(entropy)
    
    return np.mean(node_entropies), np.std(node_entropies)

def compute_entropy_per_hop(features, name_prefix="hop"):
    """计算每个 hop 层的平均熵"""
    features_np = features.numpy() if isinstance(features, torch.Tensor) else np.array(features)
    
    hop_entropies = []
    for hop in range(features_np.shape[1]):
        hop_values = features_np[:, hop, :]  # [N, D]
        entropy = compute_shannon_entropy_1d(hop_values)
        hop_entropies.append(entropy)
    
    return hop_entropies

def compute_entropy_statistics(features, name):
    """计算并打印熵统计信息"""
    features_np = features.numpy() if isinstance(features, torch.Tensor) else np.array(features)
    
    node_entropy_mean, node_entropy_std = compute_entropy_per_node(features)
    hop_entropies = compute_entropy_per_hop(features)
    total_entropy = compute_shannon_entropy_1d(features_np)
    
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

def analyze_dataset(dataset_name, pp_k=10, alpha=0.1):
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
    
    # 计算带残差的多跳特征
    print(f"\n计算多跳特征 (alpha={alpha}, pp_k={pp_k})...")
    hop_features = compute_hop_features_with_alpha(adj, features, pp_k=pp_k, alpha=alpha)
    print(f"Hop features shape: {hop_features.shape}")
    
    # 计算 Hop-Offset 特征
    print(f"\n计算 Hop-Offset 特征...")
    hop_offset_features = compute_hop_offset_features(hop_features)
    print(f"Hop-Offset shape: {hop_offset_features.shape}")
    
    # 计算 Delta 特征
    print(f"\n计算 Delta 特征...")
    delta_features = compute_delta_features(hop_features)
    print(f"Delta shape: {delta_features.shape}")
    
    # 计算熵统计
    hop_offset_stats = compute_entropy_statistics(hop_offset_features, "Hop-Offset")
    delta_stats = compute_entropy_statistics(delta_features, "Delta")
    
    # 打印统计信息
    print(f"\n{'='*40}")
    print(f"Hop-Offset 统计:")
    print(f"{'='*40}")
    print(f"  Shape: {hop_offset_stats['shape']}")
    print(f"  每节点平均熵: {hop_offset_stats['node_entropy_mean']:.4f} +/- {hop_offset_stats['node_entropy_std']:.4f} bits")
    print(f"  整体熵: {hop_offset_stats['total_entropy']:.4f} bits")
    print(f"  特征范数: mean={hop_offset_stats['norm_mean']:.4f}, std={hop_offset_stats['norm_std']:.4f}")
    print(f"  各 offset 熵值:")
    for i, h_entropy in enumerate(hop_offset_stats['hop_entropies']):
        print(f"    offset {i}: {h_entropy:.4f} bits")
    
    print(f"\n{'='*40}")
    print(f"Delta 统计:")
    print(f"{'='*40}")
    print(f"  Shape: {delta_stats['shape']}")
    print(f"  每节点平均熵: {delta_stats['node_entropy_mean']:.4f} +/- {delta_stats['node_entropy_std']:.4f} bits")
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
    entropy_diff_pct = abs(hop_offset_stats['node_entropy_mean'] - delta_stats['node_entropy_mean']) / max(hop_offset_stats['node_entropy_mean'], delta_stats['node_entropy_mean']) * 100
    print(f"  [INFO] 信息量更大的方案: {winner} (+{entropy_diff_pct:.1f}%)")
    
    return {
        'dataset': dataset_name,
        'num_nodes': num_nodes,
        'feat_dim': feat_dim,
        'hop_offset_stats': hop_offset_stats,
        'delta_stats': delta_stats,
        'entropy_ratio_node': entropy_ratio_node,
        'entropy_ratio_total': entropy_ratio_total,
        'winner': winner,
        'entropy_diff_pct': entropy_diff_pct
    }

def main():
    """主函数"""
    print("="*60)
    print("【v3 实验2】Hop-Offset vs Delta 信息熵分析")
    print("="*60)
    print("配置: alpha=0.1, 对称归一化 D^(-0.5)AD^(-0.5)")
    print("数据集: Photo, Elliptic, Tolokers")
    
    datasets = ['Photo', 'elliptic', 'tolokers']
    results = []
    
    for dataset in datasets:
        try:
            result = analyze_dataset(dataset, pp_k=10, alpha=0.1)
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] 数据集 {dataset} 分析失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总报告
    print("\n" + "="*60)
    print("[SUMMARY] 汇总报告")
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
    print("[CONCLUSION] 最终结论")
    print("="*60)
    
    hop_offset_wins = sum(1 for r in results if r['winner'] == 'Hop-Offset')
    delta_wins = sum(1 for r in results if r['winner'] == 'Delta')
    
    if hop_offset_wins > delta_wins:
        print(f"[OK] Hop-Offset 在 {hop_offset_wins}/{len(results)} 个数据集上信息量更大")
        print("   建议: 使用 Hop-Offset 特征表示")
    elif delta_wins > hop_offset_wins:
        print(f"[OK] Delta 在 {delta_wins}/{len(results)} 个数据集上信息量更大")
        print("   建议: 使用 Delta 特征表示")
    else:
        print(f"[EQUAL] Hop-Offset 和 Delta 各在 {hop_offset_wins} 个数据集上信息量更大")
        print("   建议: 根据具体数据集选择表示方式")
    
    # 详细对比
    print("\n" + "="*60)
    print("[DETAIL] 详细对比")
    print("="*60)
    
    avg_hop_offset_entropy = np.mean([r['hop_offset_stats']['node_entropy_mean'] for r in results])
    avg_delta_entropy = np.mean([r['delta_stats']['node_entropy_mean'] for r in results])
    avg_ratio = np.mean([r['entropy_ratio_node'] for r in results])
    
    print(f"\n平均每节点熵:")
    print(f"  Hop-Offset: {avg_hop_offset_entropy:.4f} bits")
    print(f"  Delta: {avg_delta_entropy:.4f} bits")
    print(f"  比值 (Delta/Hop-Offset): {avg_ratio:.4f}")
    
    # 保存结果
    output_path = '/root/gpufree-data/linziyao/VoxG/nexus/investigations/2026-03-30-hop-offset-analysis/experiments/outputs/entropy_v3_results.txt'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("【v3 实验2】Hop-Offset vs Delta 信息熵分析\n")
        f.write("配置: alpha=0.1, 对称归一化 D^(-0.5)AD^(-0.5)\n")
        f.write("="*60 + "\n\n")
        
        for r in results:
            f.write(f"数据集: {r['dataset']}\n")
            f.write(f"  节点数: {r['num_nodes']}, 特征维度: {r['feat_dim']}\n")
            f.write(f"  Hop-Offset 每节点熵: {r['hop_offset_stats']['node_entropy_mean']:.4f} +/- {r['hop_offset_stats']['node_entropy_std']:.4f} bits\n")
            f.write(f"  Delta 每节点熵: {r['delta_stats']['node_entropy_mean']:.4f} +/- {r['delta_stats']['node_entropy_std']:.4f} bits\n")
            f.write(f"  熵比值 (Delta/Hop-Offset): {r['entropy_ratio_node']:.4f}\n")
            f.write(f"  信息量赢家: {r['winner']}\n\n")
        
        f.write(f"\n总结:\n")
        f.write(f"  Hop-Offset 赢 {hop_offset_wins} 次\n")
        f.write(f"  Delta 赢 {delta_wins} 次\n")
        f.write(f"  平均 Hop-Offset 熵: {avg_hop_offset_entropy:.4f} bits\n")
        f.write(f"  平均 Delta 熵: {avg_delta_entropy:.4f} bits\n")
        f.write(f"  平均比值: {avg_ratio:.4f}\n")
    
    print(f"\n结果已保存到: {output_path}")
    
    return results

if __name__ == "__main__":
    results = main()