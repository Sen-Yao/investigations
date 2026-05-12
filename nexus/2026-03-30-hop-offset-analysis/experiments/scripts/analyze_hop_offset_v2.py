#!/usr/bin/env python3
"""
Hop-Offset 分析脚本（修正版）

使用完整的对称归一化：norm_adj = D^(-0.5) @ A @ D^(-0.5)

Hop-Offset 定义：Offset_k = Hop_k - Hop_0
- Offset_0 = Hop_0（自身）
- Offset_1 = Hop_1 - Hop_0
- Offset_2 = Hop_2 - Hop_0
- ...
- Offset_k = Hop_k - Hop_0

参数：
- alpha: 归一化参数
  - alpha=0.5: 完整对称归一化 D^(-0.5) @ A @ D^(-0.5)
  - alpha=0.0: 不归一化（仅用于对比，会数值爆炸）
  - alpha=1.0: Random Walk 归一化 D^(-1) @ A
- pp_k = 10（10 跳）
- sample_size = 1000

用法：
    python analyze_hop_offset.py --dataset photo --alpha 0.5
    python analyze_hop_offset.py --dataset elliptic --alpha 0.5
    python analyze_hop_offset.py --dataset tolokers --alpha 0.5
"""

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import argparse
import warnings
import json
warnings.filterwarnings('ignore')

np.random.seed(42)


def load_dataset(name):
    """加载数据集"""
    path = f'/root/gpufree-data/linziyao/VoxG/dataset/{name}.mat'
    data = sio.loadmat(path)
    
    label = data.get('Label', data.get('gnd', data.get('y')))
    attr = data.get('Attributes', data.get('X', data.get('x')))
    network = data.get('Network', data.get('A', data.get('adj')))
    
    labels = np.squeeze(np.array(label))
    
    if sp.issparse(attr):
        attr = sp.csr_matrix(attr)
    if sp.issparse(network):
        network = sp.csr_matrix(network)
    
    return attr, network, labels


def normalize_adjacency(network, alpha=0.5):
    """
    归一化邻接矩阵
    
    alpha=0.5: 完整对称归一化 D^(-0.5) @ A @ D^(-0.5)
    alpha=0.0: 不归一化（返回原始矩阵）
    alpha=1.0: Random Walk 归一化 D^(-1) @ A
    """
    if alpha == 0.0:
        return network
    
    rowsum = np.array(network.sum(1)).flatten()
    
    if alpha == 0.5:
        # 对称归一化: D^(-0.5) @ A @ D^(-0.5)
        d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum > 0)
        d_inv_sqrt = np.nan_to_num(d_inv_sqrt)
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj = D_inv_sqrt @ network @ D_inv_sqrt
    elif alpha == 1.0:
        # Random Walk: D^(-1) @ A
        d_inv = np.power(rowsum, -1.0, where=rowsum > 0)
        d_inv = np.nan_to_num(d_inv)
        D_inv = sp.diags(d_inv)
        norm_adj = D_inv @ network
    else:
        # 通用形式（只左乘）
        d_inv = np.power(rowsum, -alpha, where=rowsum > 0)
        d_inv = np.nan_to_num(d_inv)
        D = sp.diags(d_inv)
        norm_adj = D @ network
    
    return norm_adj


def compute_hop_offset(attr, network, sample_idx, pp_k=10, alpha=0.5):
    """
    计算 Hop-Offset
    
    使用完整对称归一化（默认 alpha=0.5）
    """
    # 归一化邻接矩阵
    norm_adj = normalize_adjacency(network, alpha)
    
    # Hop 0
    hop0 = attr[sample_idx]
    if sp.issparse(hop0):
        hop0 = hop0.toarray()
    
    # 计算 Hop 1 到 pp_k
    hops = [hop0]
    power_adj = norm_adj.copy()
    
    for k in range(1, pp_k + 1):
        sample_row = power_adj[sample_idx]
        hop_k = sample_row @ attr
        if sp.issparse(hop_k):
            hop_k = hop_k.toarray()
        hops.append(hop_k)
        power_adj = power_adj @ norm_adj
    
    # 计算 Hop-Offset: Offset_k = Hop_k - Hop_0
    offsets = [hop0]  # Offset_0 = Hop_0
    for k in range(1, pp_k + 1):
        offset_k = hops[k] - hop0  # Offset_k = Hop_k - Hop_0
        offsets.append(offset_k)
    
    return offsets


def analyze_hop_offset(dataset, sample_size=1000, pp_k=10, alpha=0.5):
    """分析 Hop-Offset"""
    print(f"\n{'='*70}")
    print(f"数据集: {dataset}")
    print(f"{'='*70}")
    
    # 加载数据
    attr, network, labels = load_dataset(dataset)
    
    n_nodes = attr.shape[0]
    n_features = attr.shape[1]
    avg_degree = network.sum() / n_nodes
    
    print(f"节点数: {n_nodes}, 特征维度: {n_features}, 平均度: {avg_degree:.1f}")
    
    # 采样
    sample_idx = np.random.choice(n_nodes, min(sample_size, n_nodes), replace=False)
    sample_labels = labels[sample_idx]
    normal_mask = sample_labels == 0
    anomaly_mask = sample_labels == 1
    
    print(f"采样: {len(sample_idx)} 节点 (正常 {normal_mask.sum()}, 异常 {anomaly_mask.sum()})")
    print(f"参数: alpha={alpha} (对称归一化 D^(-0.5)AD^(-0.5)), pp_k={pp_k}")
    
    # 计算 Hop-Offset
    print(f"\n计算 Hop-Offset...")
    offsets = compute_hop_offset(attr, network, sample_idx, pp_k=pp_k, alpha=alpha)
    
    # 分析每跳的 L2 范数
    print(f"\n{'Hop':>4} {'类型':>10} {'正常节点':>18} {'异常节点':>18} {'差异':>10} {'结论':>12}")
    print(f"{'-'*80}")
    
    results = []
    
    for k, offset in enumerate(offsets):
        # L2 范数
        l2 = np.linalg.norm(offset, axis=1)
        
        l2_normal = l2[normal_mask]
        l2_anomaly = l2[anomaly_mask]
        
        mean_n = np.mean(l2_normal)
        mean_a = np.mean(l2_anomaly)
        std_n = np.std(l2_normal)
        std_a = np.std(l2_anomaly)
        
        if k == 0:
            offset_type = "自身"
            diff = 0.0
        else:
            offset_type = f"Offset_{k}"
            diff = abs(mean_a - mean_n) / (mean_n + 1e-8) * 100
        
        normal_higher = mean_n > mean_a
        conclusion = 'N' if normal_higher else 'A'
        
        print(f"{k:>4} {offset_type:>10} {mean_n:>12.4e}±{std_n:<5.2e} {mean_a:>12.4e}±{std_a:<5.2e} {diff:>9.2f}% {conclusion:>12}")
        
        results.append({
            'hop': k,
            'type': offset_type,
            'normal_mean': float(mean_n),
            'normal_std': float(std_n),
            'anomaly_mean': float(mean_a),
            'anomaly_std': float(std_a),
            'diff_ratio': float(diff),
            'normal_higher': bool(normal_higher)
        })
    
    # 汇总
    print(f"\n{'='*70}")
    print("汇总")
    print(f"{'='*70}")
    
    n_normal_higher = sum(1 for r in results[1:] if r['normal_higher'])
    n_anomaly_higher = len(results) - 1 - n_normal_higher
    
    print(f"正常更高 (N): {n_normal_higher}/{len(results)-1} hops")
    print(f"异常更高 (A): {n_anomaly_higher}/{len(results)-1} hops")
    
    avg_diff = np.mean([r['diff_ratio'] for r in results[1:]])
    print(f"平均差异: {avg_diff:.2f}%")
    
    # 返回结果
    return {
        'dataset': dataset,
        'avg_degree': float(avg_degree),
        'sample_size': len(sample_idx),
        'alpha': alpha,
        'pp_k': pp_k,
        'results': results,
        'n_normal_higher': n_normal_higher,
        'n_anomaly_higher': n_anomaly_higher,
        'avg_diff': float(avg_diff)
    }


def main():
    parser = argparse.ArgumentParser(description='Hop-Offset 分析（修正版）')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['photo', 'elliptic', 'tolokers'],
                       help='数据集名称')
    parser.add_argument('--sample_size', type=int, default=1000,
                       help='采样节点数')
    parser.add_argument('--pp_k', type=int, default=10,
                       help='跳数')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='归一化参数 (0.5=对称归一化)')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径 (JSON)')
    args = parser.parse_args()
    
    result = analyze_hop_offset(
        args.dataset,
        sample_size=args.sample_size,
        pp_k=args.pp_k,
        alpha=args.alpha
    )
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n结果已保存到: {args.output}")
    
    print(f"\n分析完成！")


if __name__ == '__main__':
    main()
