#!/usr/bin/env python3
"""ANR (Anomaly Neighbor Ratio) 分析

验证「社区型异常」假设：异常节点邻居中异常比例是否更高

作者: Nexus
日期: 2026-04-21
"""

import numpy as np
from scipy import stats
import scipy.io as sio
import scipy.sparse as sp

def compute_anr(adj, labels):
    """计算每个节点的 ANR（异常邻居比例）"""
    N = len(labels)
    anr = np.zeros(N)
    
    for i in range(N):
        # 找邻居
        neighbors = np.where(adj[i] > 0)[0]
        
        if len(neighbors) == 0:
            anr[i] = 0
            continue
        
        # 计算邻居中异常比例
        anomaly_neighbors = np.sum(labels[neighbors] == 1)
        anr[i] = anomaly_neighbors / len(neighbors)
    
    return anr

def analyze_dataset(dataset_name):
    """分析单个数据集"""
    # 加载数据
    data = sio.loadmat(f'./dataset/{dataset_name}.mat')
    adj = data.get('Network', data.get('A'))
    if sp.issparse(adj):
        adj = adj.todense()
    labels = data.get('Label', data.get('Y')).flatten()
    labels = labels - labels.min()
    
    adj = np.array(adj)
    labels = np.array(labels)
    
    print(f'{dataset_name}: N={len(labels)}, anomaly={sum(labels)}, ratio={sum(labels)/len(labels)*100:.2f}%')
    
    # 计算 ANR
    anr = compute_anr(adj, labels)
    
    # 分析
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    normal_anr = anr[normal_mask]
    anomaly_anr = anr[anomaly_mask]
    
    # 统计检验
    ks_stat, ks_pval = stats.ks_2samp(normal_anr, anomaly_anr)
    
    print(f'  Normal ANR:  mean={np.mean(normal_anr):.4f}, std={np.std(normal_anr):.4f}')
    print(f'  Anomaly ANR: mean={np.mean(anomaly_anr):.4f}, std={np.std(anomaly_anr):.4f}')
    print(f'  Difference:   {np.mean(anomaly_anr) - np.mean(normal_anr):.4f}')
    print(f'  KS test:      stat={ks_stat:.4f}, p={ks_pval:.4e}')
    
    # 判断方向
    if np.mean(anomaly_anr) > np.mean(normal_anr):
        direction = 'Anomaly > Normal (支持社区假设)'
    else:
        direction = 'Anomaly < Normal (不支持社区假设)'
    print(f'  方向:         {direction}')
    
    return {
        'dataset': dataset_name,
        'normal_mean': np.mean(normal_anr),
        'anomaly_mean': np.mean(anomaly_anr),
        'ks_pval': ks_pval,
        'direction': np.mean(anomaly_anr) > np.mean(normal_anr)
    }

def main():
    datasets = ['photo', 'Amazon', 'tolokers', 'elliptic', 'reddit', 't_finance']
    
    print('='*70)
    print('ANR (Anomaly Neighbor Ratio) 分析')
    print('='*70)
    print()
    
    results = []
    for dataset in datasets:
        try:
            result = analyze_dataset(dataset)
            results.append(result)
            print()
        except Exception as e:
            print(f'Error on {dataset}: {e}')
            print()
    
    # 汇总
    print('='*70)
    print('汇总')
    print('='*70)
    print(f'{"Dataset":<15} {"Normal ANR":<12} {"Anomaly ANR":<12} {"方向":<15} {"p-value":<12}')
    print('-'*70)
    for r in results:
        direction_str = '✅ 支持社区' if r['direction'] else '❌ 不支持'
        print(f'{r["dataset"]:<15} {r["normal_mean"]:<12.4f} {r["anomaly_mean"]:<12.4f} {direction_str:<15} {r["ks_pval"]:<12.4e}')
    
    # 统计
    support_count = sum([r['direction'] for r in results])
    print()
    print(f'支持社区假设的数据集: {support_count}/{len(results)}')
    
    if support_count == len(results):
        print('✅ 所有数据集支持「社区型异常」假设')
    elif support_count >= len(results) * 0.5:
        print('⚠️ 多数数据集支持，部分不支持')
    else:
        print('❌ 多数数据集不支持「社区型异常」假设')

if __name__ == '__main__':
    main()
