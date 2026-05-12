#!/usr/bin/env python3
"""
验证 Delta 范数在正常/异常节点间的差异

实验设计：
- 每数据集采样 1000 节点
- 计算 hop 1 的 Delta L2 范数
- 比较正常/异常节点的均值

数据集：Photo, Elliptic, T-Finance, Tolokers
"""

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import warnings
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

def compute_delta_norm_sample(attr, network, labels, sample_size=1000):
    """计算采样节点的 Delta L2 范数"""
    n_nodes = attr.shape[0]
    avg_degree = network.sum() / n_nodes
    
    # 采样
    sample_idx = np.random.choice(n_nodes, min(sample_size, n_nodes), replace=False)
    sample_labels = labels[sample_idx]
    
    # 归一化邻接矩阵
    rowsum = np.array(network.sum(1)).flatten()
    d_inv = np.power(rowsum, -0.1, where=rowsum > 0)
    d_inv = np.nan_to_num(d_inv)
    D = sp.diags(d_inv)
    norm_adj = D @ network
    
    # 计算 hop 0 和 hop 1
    hop0 = attr[sample_idx]
    if sp.issparse(hop0):
        hop0 = hop0.toarray()
    
    sample_adj = norm_adj[sample_idx]
    hop1 = sample_adj @ attr
    if sp.issparse(hop1):
        hop1 = hop1.toarray()
    
    # Delta = hop1 - hop0
    delta = hop1 - hop0
    
    # L2 范数
    l2_norm = np.linalg.norm(delta, axis=1)
    
    # 按节点类型分组
    normal_mask = sample_labels == 0
    anomaly_mask = sample_labels == 1
    
    l2_normal = l2_norm[normal_mask]
    l2_anomaly = l2_norm[anomaly_mask]
    
    return {
        'avg_degree': avg_degree,
        'sample_size': len(sample_idx),
        'n_normal': len(l2_normal),
        'n_anomaly': len(l2_anomaly),
        'normal_mean': np.mean(l2_normal),
        'normal_std': np.std(l2_normal),
        'anomaly_mean': np.mean(l2_anomaly),
        'anomaly_std': np.std(l2_anomaly),
        'diff_ratio': abs(np.mean(l2_anomaly) - np.mean(l2_normal)) / (np.mean(l2_normal) + 1e-8) * 100,
        'normal_higher': np.mean(l2_normal) > np.mean(l2_anomaly)
    }

def main():
    datasets = ['photo', 'elliptic', 't_finance', 'tolokers']
    
    print('='*70)
    print('Delta 范数差异验证实验')
    print('采样: 1000 节点/数据集, Hop: 1')
    print('='*70)
    
    results = []
    
    for name in datasets:
        try:
            print(f"\n处理 {name}...")
            attr, network, labels = load_dataset(name)
            result = compute_delta_norm_sample(attr, network, labels, sample_size=1000)
            result['dataset'] = name
            results.append(result)
            
            print(f"  平均度: {result['avg_degree']:.1f}")
            print(f"  采样: {result['sample_size']} (正常 {result['n_normal']}, 异常 {result['n_anomaly']})")
            print(f"  正常节点: {result['normal_mean']:.4f} ± {result['normal_std']:.4f}")
            print(f"  异常节点: {result['anomaly_mean']:.4f} ± {result['anomaly_std']:.4f}")
            print(f"  差异比例: {result['diff_ratio']:.1f}%")
            print(f"  结论: {'正常节点更高' if result['normal_higher'] else '异常节点更高'}")
            
        except Exception as e:
            print(f"  失败: {e}")
    
    # 汇总表格
    print(f"\n{'='*80}")
    print("汇总结果")
    print(f"{'='*80}")
    print(f"{'数据集':<12} {'平均度':>8} {'正常节点':>15} {'异常节点':>15} {'差异':>8} {'结论':>15}")
    print(f"{'-'*80}")
    
    for r in results:
        conclusion = '正常更高' if r['normal_higher'] else '异常更高'
        print(f"{r['dataset']:<12} {r['avg_degree']:>8.1f} {r['normal_mean']:>12.2f}±{r['normal_std']:<5.2f} {r['anomaly_mean']:>12.2f}±{r['anomaly_std']:<5.2f} {r['diff_ratio']:>7.1f}% {conclusion:>15}")
    
    print(f"\n实验完成！")

if __name__ == '__main__':
    main()
