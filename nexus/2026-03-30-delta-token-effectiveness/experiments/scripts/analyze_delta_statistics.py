#!/usr/bin/env python3
"""
Delta 向量统计特性分析

目标：
1. 计算 Delta 向量的均值、方差、稀疏性
2. 对比正常/异常节点的 Delta 分布
3. 分析不同数据集的 Delta 特性差异
"""

import numpy as np
import torch
import scipy.io as sio
import scipy.sparse as sp
from pathlib import Path
import sys
import argparse

# 添加 VoxG 路径
sys.path.insert(0, '/root/gpufree-data/linziyao/VoxG')


def load_dataset(dataset_name, data_dir='/root/gpufree-data/linziyao/VoxG/dataset'):
    """加载数据集 - 使用与 run.py 相同的方式"""
    data_path = Path(data_dir) / f"{dataset_name}.mat"
    
    if not data_path.exists():
        raise FileNotFoundError(f"数据集 {data_path} 不存在")
    
    data = sio.loadmat(str(data_path))
    
    # 兼容不同字段名
    label = data['Label'] if ('Label' in data) else data['gnd'] if ('gnd' in data) else data['y']
    attr = data['Attributes'] if ('Attributes' in data) else data['X'] if ('X' in data) else data['x']
    network = data['Network'] if ('Network' in data) else data['A'] if ('A' in data) else data['adj']
    
    # 转换稀疏矩阵
    if sp.issparse(attr):
        features = attr.toarray()
    else:
        features = np.array(attr)
    
    if sp.issparse(network):
        adj = network.toarray()
    else:
        adj = np.array(network)
    
    labels = np.squeeze(np.array(label))
    
    return features, adj, labels


def node_neighborhood_feature(adj, features, hop, alpha):
    """计算 hop-hop 特征"""
    n = adj.shape[0]
    
    # 归一化邻接矩阵
    rowsum = np.array(adj.sum(1))
    d_inv = np.power(rowsum, -alpha).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = np.diag(d_inv)
    
    normalized_adj = d_mat @ adj
    
    # 多跳聚合
    result = features.copy()
    power_adj = normalized_adj
    for _ in range(hop - 1):
        power_adj = power_adj @ normalized_adj
    
    result = power_adj @ features
    return result


def compute_tokens(features, adj, pp_k=6, alpha=0.1):
    """计算原始 tokens 和 delta tokens"""
    n_nodes = features.shape[0]
    n_features = features.shape[1]
    
    # Original tokens: [N, K+1, D]
    tokens = np.zeros((n_nodes, pp_k + 1, n_features))
    tokens[:, 0, :] = features  # hop 0 = 原始特征
    
    for hop in range(1, pp_k + 1):
        tokens[:, hop, :] = node_neighborhood_feature(adj, features, hop, alpha)
    
    # Delta tokens: [N, K, D]
    delta = tokens[:, 1:, :] - tokens[:, :-1, :]
    
    return tokens, delta


def analyze_dataset(dataset_name, features, adj, labels, pp_k=6, alpha=0.1):
    """分析单个数据集"""
    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*60}")
    
    # 基本信息
    n_nodes = features.shape[0]
    n_features = features.shape[1]
    n_normal = (labels == 0).sum()
    n_anomaly = (labels == 1).sum()
    avg_degree = adj.sum() / n_nodes
    
    print(f"\n数据集基本信息:")
    print(f"  节点数: {n_nodes}")
    print(f"  边数: {int(adj.sum()/2)}")
    print(f"  平均度: {avg_degree:.1f}")
    print(f"  特征维度: {n_features}")
    print(f"  正常节点: {n_normal} ({n_normal/n_nodes*100:.1f}%)")
    print(f"  异常节点: {n_anomaly} ({n_anomaly/n_nodes*100:.1f}%)")
    
    # 计算 tokens
    print(f"\n正在计算 tokens...")
    original_tokens, delta_tokens = compute_tokens(features, adj, pp_k, alpha)
    
    print(f"\nToken 形状:")
    print(f"  Original: {original_tokens.shape}")
    print(f"  Delta: {delta_tokens.shape}")
    
    # 分析 Original 特性
    print(f"\n--- Original Token 统计 ---")
    orig_mean = np.mean(original_tokens)
    orig_std = np.std(original_tokens)
    orig_sparsity = np.mean(original_tokens == 0) * 100
    print(f"  均值: {orig_mean:.6f}")
    print(f"  标准差: {orig_std:.6f}")
    print(f"  稀疏性: {orig_sparsity:.2f}%")
    
    # 分析 Delta 特性
    print(f"\n--- Delta Token 统计 ---")
    delta_mean = np.mean(delta_tokens)
    delta_std = np.std(delta_tokens)
    delta_sparsity = np.mean(np.abs(delta_tokens) < 1e-6) * 100
    print(f"  均值: {delta_mean:.6f}")
    print(f"  标准差: {delta_std:.6f}")
    print(f"  近零稀疏性: {delta_sparsity:.2f}%")
    
    # 信息保留比例
    info_ratio = delta_std / (orig_std + 1e-8)
    print(f"  信息保留比 (std delta / std orig): {info_ratio:.4f}")
    
    # 按节点类型分析
    print(f"\n--- 按节点类型分析 Delta ---")
    
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    delta_normal = delta_tokens[normal_mask]
    delta_anomaly = delta_tokens[anomaly_mask]
    
    print(f"\n正常节点 Delta:")
    print(f"  均值: {np.mean(delta_normal):.6f}")
    print(f"  标准差: {np.std(delta_normal):.6f}")
    normal_l2 = np.mean(np.linalg.norm(delta_normal, axis=(1,2)))
    print(f"  L2 范数均值: {normal_l2:.4f}")
    
    print(f"\n异常节点 Delta:")
    print(f"  均值: {np.mean(delta_anomaly):.6f}")
    print(f"  标准差: {np.std(delta_anomaly):.6f}")
    anomaly_l2 = np.mean(np.linalg.norm(delta_anomaly, axis=(1,2)))
    print(f"  L2 范数均值: {anomaly_l2:.4f}")
    
    # 可分性分析
    normal_norms = np.linalg.norm(delta_normal, axis=(1, 2))
    anomaly_norms = np.linalg.norm(delta_anomaly, axis=(1, 2))
    
    print(f"\n--- Delta L2 范数可分性 ---")
    print(f"  正常节点 L2: {np.mean(normal_norms):.4f} ± {np.std(normal_norms):.4f}")
    print(f"  异常节点 L2: {np.mean(anomaly_norms):.4f} ± {np.std(anomaly_norms):.4f}")
    
    # 可分性分数 (Fisher's discriminant ratio)
    separation = abs(np.mean(anomaly_norms) - np.mean(normal_norms)) / (np.std(normal_norms) + np.std(anomaly_norms) + 1e-8)
    print(f"  可分性分数: {separation:.4f}")
    
    # 按 hop 分析
    print(f"\n--- 按 Hop 分析 Delta ---")
    for hop in range(delta_tokens.shape[1]):
        hop_delta = delta_tokens[:, hop, :]
        print(f"  Hop {hop+1}: mean={np.mean(hop_delta):.6f}, std={np.std(hop_delta):.6f}")
    
    return {
        'dataset': dataset_name,
        'n_nodes': n_nodes,
        'n_features': n_features,
        'avg_degree': avg_degree,
        'anomaly_ratio': n_anomaly / n_nodes,
        'orig_mean': orig_mean,
        'orig_std': orig_std,
        'delta_mean': delta_mean,
        'delta_std': delta_std,
        'info_ratio': info_ratio,
        'delta_normal_mean': np.mean(delta_normal),
        'delta_anomaly_mean': np.mean(delta_anomaly),
        'separation_score': separation
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['photo', 'Amazon', 'elliptic'],
                       help='要分析的数据集')
    parser.add_argument('--data_dir', type=str, default='/root/gpufree-data/linziyao/VoxG/dataset',
                       help='数据目录')
    args = parser.parse_args()
    
    results = []
    
    for dataset in args.datasets:
        try:
            print(f"\n加载 {dataset}...")
            features, adj, labels = load_dataset(dataset, args.data_dir)
            result = analyze_dataset(dataset, features, adj, labels)
            results.append(result)
        except Exception as e:
            print(f"加载 {dataset} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 汇总对比
    if results:
        print(f"\n{'='*100}")
        print("汇总对比")
        print(f"{'='*100}")
        print(f"{'数据集':<10} {'节点数':>8} {'平均度':>10} {'特征维':>8} {'异常比':>8} {'Delta/Orig信息比':>14} {'可分性':>10}")
        print(f"{'-'*100}")
        for r in results:
            print(f"{r['dataset']:<10} {r['n_nodes']:>8} {r['avg_degree']:>10.1f} {r['n_features']:>8} {r['anomaly_ratio']*100:>7.1f}% {r['info_ratio']:>14.4f} {r['separation_score']:>10.4f}")
    
    print(f"\n分析完成！")


if __name__ == '__main__':
    main()
