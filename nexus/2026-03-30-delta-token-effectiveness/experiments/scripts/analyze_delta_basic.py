#!/usr/bin/env python3
"""
Delta 统计分析 - 基础版本

只分析原始特征分布和图结构，不计算多 hop 特征。
多 hop 特征的计算量太大（Amazon 368 平均度）。
"""

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import sys
import warnings
warnings.filterwarnings('ignore')

def analyze_dataset_basic(dataset_name, data_dir='/root/gpufree-data/linziyao/VoxG/dataset'):
    """基础分析：不计算多 hop 特征"""
    data_path = f"{data_dir}/{dataset_name}.mat"
    data = sio.loadmat(data_path)
    
    # 获取数据
    label = data.get('Label', data.get('gnd', data.get('y')))
    attr = data.get('Attributes', data.get('X', data.get('x')))
    network = data.get('Network', data.get('A', data.get('adj')))
    
    # 基本信息
    n_nodes = attr.shape[0]
    n_features = attr.shape[1]
    avg_degree = network.sum() / n_nodes
    
    labels = np.squeeze(np.array(label))
    n_normal = (labels == 0).sum()
    n_anomaly = (labels == 1).sum()
    
    # 转换为 numpy
    if sp.issparse(attr):
        features = attr.toarray()
    else:
        features = np.array(attr)
    
    # 原始特征统计
    orig_mean = np.mean(features)
    orig_std = np.std(features)
    orig_sparsity = np.mean(features == 0) * 100
    
    # 节点类型分析
    normal_features = features[labels == 0]
    anomaly_features = features[labels == 1]
    
    normal_norms = np.linalg.norm(normal_features, axis=1)
    anomaly_norms = np.linalg.norm(anomaly_features, axis=1)
    
    # 可分性分数
    separation = abs(np.mean(anomaly_norms) - np.mean(normal_norms)) / (np.std(normal_norms) + np.std(anomaly_norms) + 1e-8)
    
    # Delta 特性推导
    # 由于 Delta = hop_k - hop_{k-1}
    # 在极稀疏图 (Elliptic): Delta ≈ -original (邻居少)
    # 在超密集图 (Amazon): Delta ≈ original * (avg_degree^{alpha} - 1) (邻居多，聚合后放大)
    # 在中等图 (Photo): Delta ≈ 适当的变换
    
    # 推导信息保留比
    # 假设: Delta_std / orig_std ≈ |avg_degree^alpha - 1|
    # 对于 alpha=0.1
    alpha = 0.1
    
    if avg_degree < 5:  # 极稀疏
        # 邻居少，Delta ≈ -original
        expected_info_ratio = 1.0
        delta_mode = "极稀疏，Delta ≈ -original"
    elif avg_degree > 100:  # 超密集
        # 邻居多，Delta ≈ original * (avg_degree^alpha - 1)
        expected_info_ratio = abs(avg_degree**alpha - 1) + 1
        delta_mode = "超密集，Delta ≈ original * scale"
    else:  # 中等密度
        expected_info_ratio = 1.0 + 0.1 * (avg_degree / 50)
        delta_mode = "中等密度，Delta 有效"
    
    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*60}")
    print(f"节点数: {n_nodes}")
    print(f"平均度: {avg_degree:.1f}")
    print(f"特征维度: {n_features}")
    print(f"异常比例: {n_anomaly/n_nodes*100:.1f}%")
    print(f"\n--- Original 特征统计 ---")
    print(f"均值: {orig_mean:.6f}")
    print(f"标准差: {orig_std:.6f}")
    print(f"稀疏性: {orig_sparsity:.2f}%")
    print(f"\n--- 可分性分析 ---")
    print(f"正常节点 L2: {np.mean(normal_norms):.4f} ± {np.std(normal_norms):.4f}")
    print(f"异常节点 L2: {np.mean(anomaly_norms):.4f} ± {np.std(anomaly_norms):.4f}")
    print(f"可分性分数: {separation:.4f}")
    print(f"\n--- Delta 特性推导 ---")
    print(f"Delta 模式: {delta_mode}")
    print(f"预期信息保留比: {expected_info_ratio:.4f}")
    
    return {
        'dataset': dataset_name,
        'n_nodes': n_nodes,
        'n_features': n_features,
        'avg_degree': avg_degree,
        'anomaly_ratio': n_anomaly / n_nodes,
        'orig_mean': orig_mean,
        'orig_std': orig_std,
        'orig_sparsity': orig_sparsity,
        'separation_score': separation,
        'expected_info_ratio': expected_info_ratio,
        'delta_mode': delta_mode
    }

def main():
    datasets = ['photo', 'Amazon', 'elliptic']
    results = []
    
    for ds in datasets:
        try:
            r = analyze_dataset_basic(ds)
            results.append(r)
        except Exception as e:
            print(f"分析 {ds} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总
    print(f"\n{'='*100}")
    print("汇总对比")
    print(f"{'='*100}")
    print(f"{'数据集':<10} {'节点数':>8} {'平均度':>10} {'特征维':>8} {'异常比':>8} {'Original可分性':>12} {'预期Delta信息比':>14}")
    print(f"{'-'*100}")
    for r in results:
        print(f"{r['dataset']:<10} {r['n_nodes']:>8} {r['avg_degree']:>10.1f} {r['n_features']:>8} {r['anomaly_ratio']*100:>7.1f}% {r['separation_score']:>12.4f} {r['expected_info_ratio']:>14.4f}")
    
    print(f"\n分析完成！")

if __name__ == '__main__':
    main()
