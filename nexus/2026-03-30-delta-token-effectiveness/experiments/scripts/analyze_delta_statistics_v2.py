#!/usr/bin/env python3
"""
Delta 向量统计特性分析 V2 - 优化版本

优化点：
1. 使用稀疏矩阵计算，避免 dense 矩阵
2. 只计算前 2 hop，避免多层聚合
3. 采样部分节点进行统计，减少计算量
"""

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from pathlib import Path
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

def load_dataset(dataset_name, data_dir='/root/gpufree-data/linziyao/VoxG/dataset'):
    """加载数据集"""
    data_path = Path(data_dir) / f"{dataset_name}.mat"
    
    if not data_path.exists():
        raise FileNotFoundError(f"数据集 {data_path} 不存在")
    
    data = sio.loadmat(str(data_path))
    
    # 兼容不同字段名
    label = data.get('Label', data.get('gnd', data.get('y')))
    attr = data.get('Attributes', data.get('X', data.get('x')))
    network = data.get('Network', data.get('A', data.get('adj')))
    
    # 保持稀疏格式
    if sp.issparse(attr):
        features = attr
    else:
        features = sp.csr_matrix(np.array(attr))
    
    if sp.issparse(network):
        adj = network
    else:
        adj = sp.csr_matrix(np.array(network))
    
    labels = np.squeeze(np.array(label))
    
    return features, adj, labels


def compute_hop_features_sparse(features, adj, hop, alpha=0.1, sample_nodes=None):
    """计算 hop-hop 特征（稀疏版本）"""
    # 归一化邻接矩阵
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -alpha, where=rowsum>0)
    d_inv = np.nan_to_num(d_inv, nan=0.0, posinf=0.0, neginf=0.0)
    
    # D^{-alpha} A
    d_mat = sp.diags(d_inv)
    normalized_adj = d_mat @ adj
    
    # A^hop @ features
    power_adj = normalized_adj
    for _ in range(hop - 1):
        power_adj = power_adj @ normalized_adj
    
    result = power_adj @ features
    
    # 如果指定采样节点，只计算这些节点
    if sample_nodes is not None:
        result = result[sample_nodes]
    
    return result


def analyze_dataset(dataset_name, features, adj, labels, pp_k=6, alpha=0.1, sample_ratio=0.1):
    """分析单个数据集"""
    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*60}")
    
    n_nodes = features.shape[0]
    n_features = features.shape[1]
    
    # 基本信息
    avg_degree = adj.sum() / n_nodes
    n_normal = (labels == 0).sum()
    n_anomaly = (labels == 1).sum()
    
    print(f"\n数据集基本信息:")
    print(f"  节点数: {n_nodes}")
    print(f"  平均度: {avg_degree:.1f}")
    print(f"  特征维度: {n_features}")
    print(f"  正常节点: {n_normal} ({n_normal/n_nodes*100:.1f}%)")
    print(f"  异常节点: {n_anomaly} ({n_anomaly/n_nodes*100:.1f}%)")
    
    # 采样节点（用于统计分析）
    sample_size = max(100, int(n_nodes * sample_ratio))
    np.random.seed(42)
    sample_indices = np.random.choice(n_nodes, sample_size, replace=False)
    
    print(f"\n采样节点数: {sample_size} (用于统计分析)")
    
    # 计算 tokens（只计算前 pp_k hop）
    print(f"\n正在计算 tokens (pp_k={pp_k})...")
    
    # Hop 0: 原始特征
    hop0_features = features[sample_indices]
    if sp.issparse(hop0_features):
        hop0_features = hop0_features.toarray()
    
    # 计算 hop 1-pp_k
    hop_features_list = [hop0_features]
    for hop in range(1, pp_k + 1):
        print(f"  计算 hop {hop}...")
        hop_feat = compute_hop_features_sparse(features, adj, hop, alpha, sample_indices)
        if sp.issparse(hop_feat):
            hop_feat = hop_feat.toarray()
        hop_features_list.append(hop_feat)
    
    # Stack: [sample_size, pp_k+1, n_features]
    original_tokens = np.stack(hop_features_list, axis=1)
    
    # Delta: [sample_size, pp_k, n_features]
    delta_tokens = original_tokens[:, 1:, :] - original_tokens[:, :-1, :]
    
    print(f"\nToken 形状:")
    print(f"  Original: {original_tokens.shape}")
    print(f"  Delta: {delta_tokens.shape}")
    
    # 统计分析
    print(f"\n--- Original Token 统计 ---")
    orig_mean = np.mean(original_tokens)
    orig_std = np.std(original_tokens)
    orig_sparsity = np.mean(original_tokens == 0) * 100
    print(f"  均值: {orig_mean:.6f}")
    print(f"  标准差: {orig_std:.6f}")
    print(f"  稀疏性: {orig_sparsity:.2f}%")
    
    print(f"\n--- Delta Token 统计 ---")
    delta_mean = np.mean(delta_tokens)
    delta_std = np.std(delta_tokens)
    delta_sparsity = np.mean(np.abs(delta_tokens) < 1e-6) * 100
    print(f"  均值: {delta_mean:.6f}")
    print(f"  标准差: {delta_std:.6f}")
    print(f"  近零稀疏性: {delta_sparsity:.2f}%")
    
    # 信息保留比
    info_ratio = delta_std / (orig_std + 1e-8)
    print(f"  信息保留比: {info_ratio:.4f}")
    
    # 按节点类型分析（使用采样节点）
    sample_labels = labels[sample_indices]
    normal_mask = sample_labels == 0
    anomaly_mask = sample_labels == 1
    
    delta_normal = delta_tokens[normal_mask]
    delta_anomaly = delta_tokens[anomaly_mask]
    
    print(f"\n--- 按节点类型分析 Delta (采样) ---")
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
    
    # Fisher's discriminant ratio
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
        'separation_score': separation,
        'sample_size': sample_size
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['photo', 'Amazon', 'elliptic'],
                       help='要分析的数据集')
    parser.add_argument('--data_dir', type=str, default='/root/gpufree-data/linziyao/VoxG/dataset',
                       help='数据目录')
    parser.add_argument('--sample_ratio', type=float, default=0.1,
                       help='采样节点比例（用于统计分析）')
    args = parser.parse_args()
    
    results = []
    
    for dataset in args.datasets:
        try:
            print(f"\n加载 {dataset}...")
            features, adj, labels = load_dataset(dataset, args.data_dir)
            result = analyze_dataset(dataset, features, adj, labels, 
                                     sample_ratio=args.sample_ratio)
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
        print(f"{'数据集':<10} {'节点数':>8} {'平均度':>10} {'特征维':>8} {'异常比':>8} {'Delta/Orig信息比':>14} {'可分性':>10} {'采样数':>8}")
        print(f"{'-'*100}")
        for r in results:
            print(f"{r['dataset']:<10} {r['n_nodes']:>8} {r['avg_degree']:>10.1f} {r['n_features']:>8} {r['anomaly_ratio']*100:>7.1f}% {r['info_ratio']:>14.4f} {r['separation_score']:>10.4f} {r['sample_size']:>8}")
    
    print(f"\n分析完成！")


if __name__ == '__main__':
    main()
