#!/usr/bin/env python3
"""
验证 Delta 范数在正常/异常节点间的差异 - 10 跳版本

实验设计：
- 每数据集采样 1000 节点
- 计算 hop 0-10（共 11 个 hop）
- Delta 1-10（共 10 个 Delta）
- 分析每跳的 L2 范数差异

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

def compute_multi_hop_delta(attr, network, sample_idx, pp_k=10, alpha=0.1):
    """计算多 hop Delta 特征"""
    # 归一化邻接矩阵
    rowsum = np.array(network.sum(1)).flatten()
    d_inv = np.power(rowsum, -alpha, where=rowsum > 0)
    d_inv = np.nan_to_num(d_inv)
    D = sp.diags(d_inv)
    norm_adj = D @ network
    
    # 计算 hop 0
    hop0 = attr[sample_idx]
    if sp.issparse(hop0):
        hop0 = hop0.toarray()
    
    # 存储 hops
    hops = [hop0]
    power_adj = norm_adj.copy()
    
    # 计算 hop 1 到 pp_k
    for k in range(1, pp_k + 1):
        sample_row = power_adj[sample_idx]
        hop_k = sample_row @ attr
        if sp.issparse(hop_k):
            hop_k = hop_k.toarray()
        hops.append(hop_k)
        power_adj = power_adj @ norm_adj
        print(f"    Hop {k} 完成")
    
    # 计算 Delta 1 到 pp_k
    deltas = []
    for k in range(pp_k):
        delta = hops[k+1] - hops[k]
        deltas.append(delta)
    
    return hops, deltas

def analyze_delta_by_hop(deltas, labels, sample_idx):
    """分析每跳 Delta 的 L2 范数差异"""
    sample_labels = labels[sample_idx]
    normal_mask = sample_labels == 0
    anomaly_mask = sample_labels == 1
    
    results = []
    for k, delta in enumerate(deltas):
        l2 = np.linalg.norm(delta, axis=1)
        
        l2_normal = l2[normal_mask]
        l2_anomaly = l2[anomaly_mask]
        
        mean_n = np.mean(l2_normal)
        mean_a = np.mean(l2_anomaly)
        std_n = np.std(l2_normal)
        std_a = np.std(l2_anomaly)
        
        diff = abs(mean_a - mean_n) / (mean_n + 1e-8) * 100
        normal_higher = mean_n > mean_a
        
        results.append({
            'hop': k + 1,
            'normal_mean': mean_n,
            'normal_std': std_n,
            'anomaly_mean': mean_a,
            'anomaly_std': std_a,
            'diff_ratio': diff,
            'normal_higher': normal_higher
        })
    
    return results

def main():
    datasets = ['photo', 'elliptic', 't_finance', 'tolokers']
    pp_k = 10
    sample_size = 1000
    
    print('='*80)
    print(f'Delta 范数差异验证实验 - {pp_k} 跳版本')
    print(f'采样: {sample_size} 节点/数据集')
    print('='*80)
    
    all_results = {}
    
    for name in datasets:
        try:
            print(f"\n{'='*60}")
            print(f"处理 {name}...")
            print(f"{'='*60}")
            
            attr, network, labels = load_dataset(name)
            n_nodes = attr.shape[0]
            avg_degree = network.sum() / n_nodes
            
            print(f"  节点数: {n_nodes}, 平均度: {avg_degree:.1f}")
            
            # 采样
            sample_idx = np.random.choice(n_nodes, min(sample_size, n_nodes), replace=False)
            print(f"  采样: {len(sample_idx)} 节点")
            print(f"  计算 Hop 0-{pp_k}...")
            
            # 计算多 hop
            hops, deltas = compute_multi_hop_delta(attr, network, sample_idx, pp_k=pp_k)
            
            # 分析每跳
            results = analyze_delta_by_hop(deltas, labels, sample_idx)
            all_results[name] = {
                'avg_degree': avg_degree,
                'results': results
            }
            
            # 打印结果
            print(f"\n  {'Hop':>4} {'正常节点':>15} {'异常节点':>15} {'差异':>8} {'结论':>12}")
            print(f"  {'-'*60}")
            for r in results:
                conclusion = '正常更高' if r['normal_higher'] else '异常更高'
                print(f"  {r['hop']:>4} {r['normal_mean']:>10.2f}±{r['normal_std']:<5.2f} {r['anomaly_mean']:>10.2f}±{r['anomaly_std']:<5.2f} {r['diff_ratio']:>7.1f}% {conclusion:>12}")
            
        except Exception as e:
            print(f"  失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总表格
    print(f"\n{'='*100}")
    print("汇总 - 每跳结论")
    print(f"{'='*100}")
    print(f"{'数据集':<12}", end='')
    for k in range(1, pp_k + 1):
        print(f"{'Hop'+str(k):>7}", end='')
    print()
    print(f"{'-'*100}")
    
    for name, data in all_results.items():
        print(f"{name:<12}", end='')
        for r in data['results']:
            symbol = 'N' if r['normal_higher'] else 'A'
            print(f"{symbol:>7}", end='')
        print()
    
    print(f"\n注: N = 正常节点更高, A = 异常节点更高")
    print(f"\n实验完成！")

if __name__ == '__main__':
    main()
