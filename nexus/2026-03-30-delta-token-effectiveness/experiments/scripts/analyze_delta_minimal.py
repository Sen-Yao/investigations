#!/usr/bin/env python3
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import warnings
warnings.filterwarnings('ignore')

def load_data(dataset):
    data = sio.loadmat(f"/root/gpufree-data/linziyao/VoxG/dataset/{dataset}.mat")
    label = data.get('Label', data.get('gnd', data.get('y')))
    attr = data.get('Attributes', data.get('X', data.get('x')))
    network = data.get('Network', data.get('A', data.get('adj')))
    labels = np.squeeze(np.array(label))
    return attr, network, labels

def analyze_dataset(dataset, sample_size=100, pp_k=2):
    print(f"\n=== {dataset} ===")
    
    features, adj, labels = load_data(dataset)
    n_nodes = features.shape[0]
    avg_degree = adj.sum() / n_nodes
    
    print(f"节点数: {n_nodes}, 平均度: {avg_degree:.1f}")
    
    # 采样
    np.random.seed(42)
    sample_idx = np.random.choice(n_nodes, min(sample_size, n_nodes), replace=False)
    
    # 转换
    if sp.issparse(features):
        features_arr = features.toarray()
    else:
        features_arr = np.array(features)
    
    if sp.issparse(adj):
        adj_arr = adj.toarray()
    else:
        adj_arr = np.array(adj)
    
    # 归一化
    rowsum = adj_arr.sum(1)
    d_inv = np.power(rowsum, -0.1, where=rowsum > 0)
    d_inv = np.nan_to_num(d_inv)
    norm_adj = np.diag(d_inv) @ adj_arr
    
    # 计算 hop 0, 1, 2
    hop0 = features_arr[sample_idx]
    hop1 = norm_adj[sample_idx] @ features_arr
    hop2 = norm_adj[sample_idx] @ norm_adj @ features_arr
    
    # Delta
    delta1 = hop1 - hop0
    delta2 = hop2 - hop1
    
    # 统计
    tokens = np.stack([hop0, hop1, hop2], axis=1)
    delta = np.stack([delta1, delta2], axis=1)
    
    orig_std = np.std(tokens)
    delta_std = np.std(delta)
    info_ratio = delta_std / (orig_std + 1e-8)
    
    print(f"信息保留比: {info_ratio:.4f}")
    
    # 可分性
    sample_labels = labels[sample_idx]
    normal_mask = sample_labels == 0
    anomaly_mask = sample_labels == 1
    
    delta_normal = delta[normal_mask]
    delta_anomaly = delta[anomaly_mask]
    
    normal_l2 = np.linalg.norm(delta_normal.reshape(len(delta_normal), -1), axis=1)
    anomaly_l2 = np.linalg.norm(delta_anomaly.reshape(len(delta_anomaly), -1), axis=1)
    
    separation = abs(np.mean(anomaly_l2) - np.mean(normal_l2)) / (np.std(normal_l2) + np.std(anomaly_l2) + 1e-8)
    print(f"可分性分数: {separation:.4f}")
    print(f"  正常: {len(normal_l2)} nodes")
    print(f"  异常: {len(anomaly_l2)} nodes")
    
    return {
        'dataset': dataset,
        'avg_degree': avg_degree,
        'info_ratio': info_ratio,
        'separation': separation
    }

results = []
for ds in ['photo', 'Amazon', 'elliptic']:
    try:
        r = analyze_dataset(ds, sample_size=200, pp_k=2)
        results.append(r)
    except Exception as e:
        print(f"{ds} 失败: {e}")

print(f"\n{'='*60}")
print("汇总")
print(f"{'='*60}")
print(f"{'数据集':<10} {'平均度':>10} {'信息保留比':>12} {'可分性分数':>12}")
for r in results:
    print(f"{r['dataset']:<10} {r['avg_degree']:>10.1f} {r['info_ratio']:>12.4f} {r['separation']:>12.4f}")
