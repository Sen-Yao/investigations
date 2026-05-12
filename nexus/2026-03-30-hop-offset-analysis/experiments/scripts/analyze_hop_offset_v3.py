#!/usr/bin/env python3
"""
Hop-Offset 分析脚本（修正版 v3）

归一化方式：始终使用对称归一化 D^(-0.5) @ A @ D^(-0.5)
alpha 参数：残差系数（类似 PPR），控制自环权重

传播公式：H^{(k)} = (1-alpha) * norm_adj @ H^{(k-1)} + alpha * H^{(0)}

alpha=0.0: 纯邻居聚合（无自环）
alpha=0.1: PPR 风格（少量自环）
alpha=0.5: 平衡自环和邻居

Hop-Offset 定义：Offset_k = Hop_k - Hop_0
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


def symmetric_normalize(network):
    """
    对称归一化：D^(-0.5) @ A @ D^(-0.5)
    始终应用，确保数值稳定
    """
    rowsum = np.array(network.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum > 0)
    d_inv_sqrt = np.nan_to_num(d_inv_sqrt)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return D_inv_sqrt @ network @ D_inv_sqrt


def compute_ppr_propagation(attr, network, sample_idx, pp_k=10, alpha=0.1):
    """
    PPR 风格传播
    
    H^{(k)} = (1-alpha) * norm_adj @ H^{(k-1)} + alpha * H^{(0)}
    
    alpha: 残差系数（自环权重）
    - alpha=0.0: 纯邻居聚合
    - alpha=0.1: PPR 风格（推荐）
    - alpha=0.5: 平衡
    """
    # 始终使用对称归一化
    norm_adj = symmetric_normalize(network)
    
    # Hop 0
    hop0 = attr[sample_idx]
    if sp.issparse(hop0):
        hop0 = hop0.toarray()
    
    # PPR 风格传播
    hops = [hop0]
    prev_hop = hop0.copy()
    
    for k in range(1, pp_k + 1):
        # 传播：H_k = (1-alpha) * A_norm @ H_{k-1} + alpha * H_0
        propagated = norm_adj[sample_idx] @ attr
        if sp.issparse(propagated):
            propagated = propagated.toarray()
        
        if alpha > 0:
            hop_k = (1 - alpha) * propagated + alpha * hop0
        else:
            hop_k = propagated
        
        hops.append(hop_k)
        prev_hop = hop_k
    
    # 计算 Hop-Offset: Offset_k = Hop_k - Hop_0
    offsets = [hop0]
    for k in range(1, pp_k + 1):
        offset_k = hops[k] - hop0
        offsets.append(offset_k)
    
    # 计算 Delta: Delta_k = Hop_k - Hop_{k-1}
    deltas = []
    for k in range(pp_k + 1):
        if k == 0:
            deltas.append(hop0)
        else:
            delta_k = hops[k] - hops[k-1]
            deltas.append(delta_k)
    
    return hops, offsets, deltas


def analyze_hop_offset(dataset, sample_size=1000, pp_k=10, alpha=0.1):
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
    print(f"参数: alpha={alpha} (残差系数), pp_k={pp_k}")
    print(f"归一化: 对称归一化 D^(-0.5)AD^(-0.5)")
    
    # 计算
    print(f"\n计算 Hop-Offset 和 Delta...")
    hops, offsets, deltas = compute_ppr_propagation(attr, network, sample_idx, pp_k=pp_k, alpha=alpha)
    
    # 分析
    print(f"\n{'Hop':>4} {'类型':>10} {'正常节点':>18} {'异常节点':>18} {'差异':>10} {'结论':>12}")
    print(f"{'-'*80}")
    
    results = []
    
    for k, offset in enumerate(offsets):
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
    parser = argparse.ArgumentParser(description='Hop-Offset 分析 v3')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['photo', 'elliptic', 'tolokers'])
    parser.add_argument('--sample_size', type=int, default=1000)
    parser.add_argument('--pp_k', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='残差系数 (0=纯邻居聚合, 0.1=PPR推荐)')
    parser.add_argument('--output', type=str, default=None)
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