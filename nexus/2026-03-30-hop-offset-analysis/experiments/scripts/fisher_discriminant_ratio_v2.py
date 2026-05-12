#!/usr/bin/env python3
"""
Fisher 判别比分析 v2 - 对称归一化版本

Fisher 判别比 = (mu_normal - mu_anomaly)^2 / (sigma_normal^2 + sigma_anomaly^2)

衡量特征对正常/异常节点的可分性，值越大表示可分性越好。

关键修改（v2）：
- 归一化方式：D^(-0.5) @ A @ D^(-0.5)（对称归一化）
- alpha = 0.5（默认）

分析内容：
1. 各 hop 的 Fisher 判别比（Hop_0, Hop_1, ..., Hop_k）
2. Hop-Offset 的 Fisher 判别比（Offset_k = Hop_k - Hop_0）
3. Delta 的 Fisher 判别比（Delta_k = Hop_{k+1} - Hop_k）
"""

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import warnings
import json
from datetime import datetime

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


def symmetric_normalize_adj(network):
    """
    对称归一化：D^(-0.5) @ A @ D^(-0.5)
    这是在 GCN 中使用的标准归一化方式。
    """
    # 计算度矩阵 D
    rowsum = np.array(network.sum(1)).flatten()
    
    # D^(-0.5)，处理度为0的节点
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum > 0)
    d_inv_sqrt = np.nan_to_num(d_inv_sqrt)
    
    # 构建稀疏对角矩阵
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    
    # 对称归一化：D^(-0.5) @ A @ D^(-0.5)
    norm_adj = D_inv_sqrt @ network @ D_inv_sqrt
    
    return norm_adj


def compute_hops(attr, network, sample_idx, pp_k=10):
    """
    计算各 hop 的特征（使用对称归一化）
    
    对称归一化：A_sym = D^(-0.5) @ A @ D^(-0.5)
    Hop_k = A_sym^k @ X
    
    注意：Hop_0 = X（原始特征，不受归一化影响）
    """
    # 对称归一化邻接矩阵
    norm_adj = symmetric_normalize_adj(network)
    
    # Hop 0（自身特征）
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
    
    return hops


def compute_fisher_ratio(normal_values, anomaly_values):
    """
    计算 Fisher 判别比
    F = (mu_normal - mu_anomaly)^2 / (sigma_normal^2 + sigma_anomaly^2)
    """
    mu_n = np.mean(normal_values)
    mu_a = np.mean(anomaly_values)
    var_n = np.var(normal_values, ddof=1)  # 无偏估计
    var_a = np.var(anomaly_values, ddof=1)
    
    # 避免除零
    denominator = var_n + var_a
    if denominator < 1e-10:
        return 0.0
    
    fisher_ratio = (mu_n - mu_a) ** 2 / denominator
    return fisher_ratio


def analyze_dataset(dataset_name, sample_size=2000, pp_k=10):
    """分析单个数据集"""
    print(f"\n{'='*80}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*80}")
    
    # 加载数据
    attr, network, labels = load_dataset(dataset_name)
    
    n_nodes = attr.shape[0]
    n_features = attr.shape[1]
    avg_degree = network.sum() / n_nodes
    
    print(f"节点数: {n_nodes}, 特征维度: {n_features}, 平均度: {avg_degree:.1f}")
    
    # 采样
    sample_idx = np.random.choice(n_nodes, min(sample_size, n_nodes), replace=False)
    sample_labels = labels[sample_idx]
    normal_mask = sample_labels == 0
    anomaly_mask = sample_labels == 1
    
    n_normal = normal_mask.sum()
    n_anomaly = anomaly_mask.sum()
    print(f"采样: {len(sample_idx)} 节点 (正常 {n_normal}, 异常 {n_anomaly})")
    
    if n_normal < 5 or n_anomaly < 5:
        print(f"样本不足，跳过该数据集")
        return None
    
    print(f"参数: 对称归一化 (D^-0.5 @ A @ D^-0.5), pp_k={pp_k}")
    
    # 计算各 hop 特征
    print(f"\n计算各 Hop 特征...")
    hops = compute_hops(attr, network, sample_idx, pp_k=pp_k)
    
    # 结果存储
    results = {
        'dataset': dataset_name,
        'n_nodes': int(n_nodes),
        'n_features': int(n_features),
        'avg_degree': float(avg_degree),
        'sample_size': len(sample_idx),
        'n_normal': int(n_normal),
        'n_anomaly': int(n_anomaly),
        'normalization': 'symmetric (D^-0.5 @ A @ D^-0.5)',
        'pp_k': pp_k,
        'hop_fisher': [],
        'offset_fisher': [],
        'delta_fisher': []
    }
    
    # 1. 计算各 Hop 的 Fisher 判别比
    print(f"\n--- Hop Fisher 判别比 ---")
    print(f"{'Hop':>6} {'mu_normal':>12} {'mu_anomaly':>12} {'sigma_normal':>10} {'sigma_anomaly':>10} {'Fisher':>10}")
    print("-" * 70)
    
    best_hop_fisher = 0
    best_hop_k = 0
    
    for k, hop in enumerate(hops):
        # 使用 L2 范数作为标量特征
        l2_norms = np.linalg.norm(hop, axis=1)
        
        l2_normal = l2_norms[normal_mask]
        l2_anomaly = l2_norms[anomaly_mask]
        
        mu_n = np.mean(l2_normal)
        mu_a = np.mean(l2_anomaly)
        sigma_n = np.std(l2_normal)
        sigma_a = np.std(l2_anomaly)
        
        fisher = compute_fisher_ratio(l2_normal, l2_anomaly)
        
        print(f"Hop_{k:>2} {mu_n:>12.4f} {mu_a:>12.4f} {sigma_n:>10.4f} {sigma_a:>10.4f} {fisher:>10.4f}")
        
        results['hop_fisher'].append({
            'hop': k,
            'mu_normal': float(mu_n),
            'mu_anomaly': float(mu_a),
            'sigma_normal': float(sigma_n),
            'sigma_anomaly': float(sigma_a),
            'fisher_ratio': float(fisher)
        })
        
        if fisher > best_hop_fisher:
            best_hop_fisher = fisher
            best_hop_k = k
    
    print(f"\n[OK] 最佳 Hop: Hop_{best_hop_k} (Fisher={best_hop_fisher:.4f})")
    
    # 2. 计算 Hop-Offset 的 Fisher 判别比
    print(f"\n--- Hop-Offset Fisher 判别比 ---")
    print(f"{'Offset':>8} {'mu_normal':>12} {'mu_anomaly':>12} {'sigma_normal':>10} {'sigma_anomaly':>10} {'Fisher':>10}")
    print("-" * 70)
    
    hop0 = hops[0]
    best_offset_fisher = 0
    best_offset_k = 0
    
    for k in range(1, len(hops)):
        offset = hops[k] - hop0  # Offset_k = Hop_k - Hop_0
        l2_norms = np.linalg.norm(offset, axis=1)
        
        l2_normal = l2_norms[normal_mask]
        l2_anomaly = l2_norms[anomaly_mask]
        
        mu_n = np.mean(l2_normal)
        mu_a = np.mean(l2_anomaly)
        sigma_n = np.std(l2_normal)
        sigma_a = np.std(l2_anomaly)
        
        fisher = compute_fisher_ratio(l2_normal, l2_anomaly)
        
        print(f"Offset_{k:>2} {mu_n:>12.4f} {mu_a:>12.4f} {sigma_n:>10.4f} {sigma_a:>10.4f} {fisher:>10.4f}")
        
        results['offset_fisher'].append({
            'offset': k,
            'mu_normal': float(mu_n),
            'mu_anomaly': float(mu_a),
            'sigma_normal': float(sigma_n),
            'sigma_anomaly': float(sigma_a),
            'fisher_ratio': float(fisher)
        })
        
        if fisher > best_offset_fisher:
            best_offset_fisher = fisher
            best_offset_k = k
    
    print(f"\n[OK] 最佳 Offset: Offset_{best_offset_k} (Fisher={best_offset_fisher:.4f})")
    
    # 3. 计算 Delta 的 Fisher 判别比
    print(f"\n--- Delta Fisher 判别比 ---")
    print(f"{'Delta':>8} {'mu_normal':>12} {'mu_anomaly':>12} {'sigma_normal':>10} {'sigma_anomaly':>10} {'Fisher':>10}")
    print("-" * 70)
    
    best_delta_fisher = 0
    best_delta_k = 0
    
    for k in range(len(hops) - 1):
        delta = hops[k + 1] - hops[k]  # Delta_k = Hop_{k+1} - Hop_k
        l2_norms = np.linalg.norm(delta, axis=1)
        
        l2_normal = l2_norms[normal_mask]
        l2_anomaly = l2_norms[anomaly_mask]
        
        mu_n = np.mean(l2_normal)
        mu_a = np.mean(l2_anomaly)
        sigma_n = np.std(l2_normal)
        sigma_a = np.std(l2_anomaly)
        
        fisher = compute_fisher_ratio(l2_normal, l2_anomaly)
        
        print(f"Delta_{k:>2} {mu_n:>12.4f} {mu_a:>12.4f} {sigma_n:>10.4f} {sigma_a:>10.4f} {fisher:>10.4f}")
        
        results['delta_fisher'].append({
            'delta': k,
            'mu_normal': float(mu_n),
            'mu_anomaly': float(mu_a),
            'sigma_normal': float(sigma_n),
            'sigma_anomaly': float(sigma_a),
            'fisher_ratio': float(fisher)
        })
        
        if fisher > best_delta_fisher:
            best_delta_fisher = fisher
            best_delta_k = k
    
    print(f"\n[OK] 最佳 Delta: Delta_{best_delta_k} (Fisher={best_delta_fisher:.4f})")
    
    # 汇总
    results['best_hop'] = {'k': best_hop_k, 'fisher': float(best_hop_fisher)}
    results['best_offset'] = {'k': best_offset_k, 'fisher': float(best_offset_fisher)}
    results['best_delta'] = {'k': best_delta_k, 'fisher': float(best_delta_fisher)}
    
    return results


def main():
    print("=" * 80)
    print("Fisher 判别比分析 v2 - 对称归一化 (D^-0.5 @ A @ D^-0.5)")
    print("=" * 80)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    datasets = ['Photo', 'elliptic', 'tolokers']
    sample_size = 2000
    pp_k = 10
    
    all_results = []
    
    for dataset in datasets:
        result = analyze_dataset(dataset, sample_size=sample_size, pp_k=pp_k)
        if result:
            all_results.append(result)
    
    # 最终汇总
    print("\n" + "=" * 80)
    print("跨数据集汇总")
    print("=" * 80)
    
    print(f"\n{'数据集':>12} {'最佳Hop':>18} {'最佳Offset':>20} {'最佳Delta':>20} {'综合最佳':>20}")
    print("-" * 100)
    
    for r in all_results:
        best_hop = f"Hop_{r['best_hop']['k']}({r['best_hop']['fisher']:.4f})"
        best_offset = f"Offset_{r['best_offset']['k']}({r['best_offset']['fisher']:.4f})"
        best_delta = f"Delta_{r['best_delta']['k']}({r['best_delta']['fisher']:.4f})"
        
        # 综合最佳（三种特征中 Fisher 最大的）
        best_overall = max(
            ('Hop', r['best_hop']['k'], r['best_hop']['fisher']),
            ('Offset', r['best_offset']['k'], r['best_offset']['fisher']),
            ('Delta', r['best_delta']['k'], r['best_delta']['fisher']),
            key=lambda x: x[2]
        )
        overall_str = f"{best_overall[0]}_{best_overall[1]}({best_overall[2]:.4f})"
        
        print(f"{r['dataset']:>12} {best_hop:>18} {best_offset:>20} {best_delta:>20} {overall_str:>20}")
    
    # 保存结果
    output_path = '/root/gpufree-data/linziyao/VoxG/nexus/investigations/2026-03-30-hop-offset-analysis/experiments/outputs/fisher_discriminant_results_v2.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存到: {output_path}")
    
    print("\n分析完成！")


if __name__ == '__main__':
    main()