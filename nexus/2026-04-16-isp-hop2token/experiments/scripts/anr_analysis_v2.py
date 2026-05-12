#!/usr/bin/env python3
"""ANR 分析 - 处理大小写文件名"""

import numpy as np
from scipy import stats
import scipy.io as sio
import scipy.sparse as sp
import os

def find_dataset_file(dataset_name):
    """查找数据集文件（处理大小写）"""
    possible_names = [
        dataset_name.lower() + '.mat',
        dataset_name.capitalize() + '.mat',
        dataset_name + '.mat'
    ]
    for name in possible_names:
        path = f'./dataset/{name}'
        if os.path.exists(path):
            return path
    return None

def compute_anr(adj, labels):
    """计算每个节点的 ANR"""
    N = len(labels)
    anr = np.zeros(N)
    
    for i in range(N):
        neighbors = np.where(adj[i] > 0)[0]
        if len(neighbors) == 0:
            anr[i] = 0
            continue
        anomaly_neighbors = np.sum(labels[neighbors] == 1)
        anr[i] = anomaly_neighbors / len(neighbors)
    
    return anr

def analyze_dataset(dataset_name):
    """分析单个数据集"""
    # 查找文件
    file_path = find_dataset_file(dataset_name)
    if file_path is None:
        print(f'{dataset_name}: 文件未找到')
        return None
    
    # 加载
    data = sio.loadmat(file_path)
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
    
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    normal_anr = anr[normal_mask]
    anomaly_anr = anr[anomaly_mask]
    
    ks_stat, ks_pval = stats.ks_2samp(normal_anr, anomaly_anr)
    
    print(f'  Normal ANR:  mean={np.mean(normal_anr):.4f}, std={np.std(normal_anr):.4f}')
    print(f'  Anomaly ANR: mean={np.mean(anomaly_anr):.4f}, std={np.std(anomaly_anr):.4f}')
    print(f'  Difference:   {np.mean(anomaly_anr) - np.mean(normal_anr):.4f}')
    print(f'  KS test:      stat={ks_stat:.4f}, p={ks_pval:.4e}')
    
    if np.mean(anomaly_anr) > np.mean(normal_anr):
        direction = '✅ 支持社区假设'
    else:
        direction = '❌ 不支持社区假设'
    print(f'  方向:         {direction}')
    
    return {
        'dataset': dataset_name,
        'normal_mean': np.mean(normal_anr),
        'anomaly_mean': np.mean(anomaly_anr),
        'difference': np.mean(anomaly_anr) - np.mean(normal_anr),
        'ks_pval': ks_pval,
        'support': np.mean(anomaly_anr) > np.mean(normal_anr)
    }

def main():
    datasets = ['photo', 'Amazon', 'tolokers', 'elliptic', 'Reddit', 't_finance']
    
    print('='*70)
    print('ANR (Anomaly Neighbor Ratio) 分析')
    print('验证「社区型异常」假设')
    print('='*70)
    print()
    
    results = []
    for dataset in datasets:
        result = analyze_dataset(dataset)
        if result:
            results.append(result)
        print()
    
    print('='*70)
    print('汇总')
    print('='*70)
    print(f'{"Dataset":<12} {"Normal ANR":<10} {"Anomaly ANR":<10} {"Diff":<10} {"Support":<15} {"p-value":<12}')
    print('-'*70)
    
    for r in results:
        support_str = '✅ 支持社区' if r['support'] else '❌ 不支持'
        print(f'{r["dataset"]:<12} {r["normal_mean"]:<10.4f} {r["anomaly_mean"]:<10.4f} {r["difference"]:<10.4f} {support_str:<15} {r["ks_pval"]:<12.4e}')
    
    support_count = sum([r['support'] for r in results])
    print()
    print(f'支持社区假设的数据集: {support_count}/{len(results)}')
    
    if support_count == len(results):
        print('\n✅✅✅ 所有数据集支持「社区型异常」假设！')
        print('\n结论: 异常节点邻居中异常比例显著高于正常节点')
        print('       这解释了为什么异常节点 NDC 高（与邻居 Delta 相似）')
    elif support_count >= len(results) * 0.5:
        print('⚠️ 多数数据集支持')
    else:
        print('❌ 多数数据集不支持')

if __name__ == '__main__':
    main()
