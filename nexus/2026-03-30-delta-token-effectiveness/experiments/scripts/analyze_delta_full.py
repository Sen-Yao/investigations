#!/usr/bin/env python3
"""
Delta 统计分析 - 完整版本

策略：只计算采样节点的多 hop 特征，不计算全量。
"""

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import sys
import time
import warnings
warnings.filterwarnings('ignore')

def load_data(dataset_name):
    data = sio.loadmat(f"/root/gpufree-data/linziyao/VoxG/dataset/{dataset_name}.mat")
    label = data.get('Label', data.get('gnd', data.get('y')))
    attr = data.get('Attributes', data.get('X', data.get('x')))
    network = data.get('Network', data.get('A', data.get('adj')))
    labels = np.squeeze(np.array(label))
    return attr, network, labels

def compute_sample_hop_features(features, adj, sample_idx, pp_k=6, alpha=0.1):
    """只计算采样节点的 hop 特征"""
    n_samples = len(sample_idx)
    n_features = features.shape[1]
    
    # 结果数组
    tokens = np.zeros((n_samples, pp_k + 1, n_features))
    
    # Hop 0: 原始特征
    if sp.issparse(features):
        tokens[:, 0, :] = features[sample_idx].toarray()
    else:
        tokens[:, 0, :] = features[sample_idx]
    
    # 计算归一化邻接矩阵
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -alpha, where=rowsum > 0)
    d_inv = np.nan_to_num(d_inv)
    D = sp.diags(d_inv)
    norm_adj = D @ adj
    
    # 逐层计算 hop 特征
    for hop in range(1, pp_k + 1):
        start = time.time()
        # (A^hop)[sample_idx, :] @ features
        # 先计算 power_adj 的采样行
        power_adj = norm_adj.copy()
        for _ in range(hop - 1):
            power_adj = power_adj @ norm_adj
        
        # 只提取采样行
        sample_row_adj = power_adj[sample_idx]
        result = sample_row_adj @ features
        
        if sp.issparse(result):
            tokens[:, hop, :] = result.toarray()
        else:
            tokens[:, hop, :] = result
        
        print(f"    Hop {hop}: {time.time()-start:.2f}s")
    
    return tokens

def analyze_dataset(dataset_name, sample_size=500, pp_k=6):
    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*60}")
    
    start_total = time.time()
    
    features, adj, labels = load_data(dataset_name)
    n_nodes = features.shape[0]
    n_features = features.shape[1]
    avg_degree = adj.sum() / n_nodes
    n_anomaly = (labels == 1).sum()
    
    print(f"节点数: {n_nodes}, 平均度: {avg_degree:.1f}, 特征维度: {n_features}")
    
    # 采样
    np.random.seed(42)
    sample_idx = np.random.choice(n_nodes, min(sample_size, n_nodes), replace=False)
    print(f"采样节点数: {len(sample_idx)}")
    
    # 计算 tokens
    print(f"\n计算多 hop 特征...")
    start = time.time()
    tokens = compute_sample_hop_features(features, adj, sample_idx, pp_k)
    print(f"总计算时间: {time.time()-start:.2f}s")
    
    # Delta
    delta = tokens[:, 1:, :] - tokens[:, :-1, :]
    
    # 统计
    print(f"\n--- 统计结果 ---")
    orig_mean = np.mean(tokens)
    orig_std = np.std(tokens)
    delta_mean = np.mean(delta)
    delta_std = np.std(delta)
    info_ratio = delta_std / (orig_std + 1e-8)
    
    print(f"Original: mean={orig_mean:.6f}, std={orig_std:.6f}")
    print(f"Delta: mean={delta_mean:.6f}, std={delta_std:.6f}")
    print(f"信息保留比: {info_ratio:.4f}")
    
    # 可分性
    sample_labels = labels[sample_idx]
    normal_mask = sample_labels == 0
    anomaly_mask = sample_labels == 1
    
    delta_normal = delta[normal_mask]
    delta_anomaly = delta[anomaly_mask]
    
    normal_l2 = np.linalg.norm(delta_normal, axis=(1,2))
    anomaly_l2 = np.linalg.norm(delta_anomaly, axis=(1,2))
    
    separation = abs(np.mean(anomaly_l2) - np.mean(normal_l2)) / (np.std(normal_l2) + np.std(anomaly_l2) + 1e-8)
    print(f"\n可分性分数: {separation:.4f}")
    print(f"  正常节点: n={len(normal_l2)}, L2={np.mean(normal_l2):.4f}±{np.std(normal_l2):.4f}")
    print(f"  异常节点: n={len(anomaly_l2)}, L2={np.mean(anomaly_l2):.4f}±{np.std(anomaly_l2):.4f}")
    
    print(f"\n总分析时间: {time.time()-start_total:.2f}s")
    
    return {
        'dataset': dataset_name,
        'avg_degree': avg_degree,
        'info_ratio': info_ratio,
        'separation': separation,
        'sample_size': len(sample_idx)
    }

def main():
    datasets = ['photo', 'Amazon', 'elliptic']
    results = []
    
    for ds in datasets:
        try:
            r = analyze_dataset(ds, sample_size=500)
            results.append(r)
        except Exception as e:
            print(f"分析 {ds} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总
    print(f"\n{'='*80}")
    print("汇总对比")
    print(f"{'='*80}")
    print(f"{'数据集':<10} {'平均度':>10} {'信息保留比':>12} {'可分性分数':>12} {'采样数':>8}")
    print(f"{'-'*80}")
    for r in results:
        print(f"{r['dataset']:<10} {r['avg_degree']:>10.1f} {r['info_ratio']:>12.4f} {r['separation']:>12.4f} {r['sample_size']:>8}")

if __name__ == '__main__':
    main()
