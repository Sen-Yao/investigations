#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.metrics import pairwise_distances

DATA = Path('/data/linziyao/DualRefGAD/dataset/elliptic.mat')
OUT = Path(__file__).resolve().parent / 'outputs_split_graph'
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [0,1,2,3,4]
TRAIN_RATE=0.05
VAL_RATE=0.0
SAMPLE_RATE=0.15


def split_indices(n, labels, seed):
    # Matches load_mat path used by fixed script: set_seed(seed) then random.shuffle(all_idx).
    import random
    random.seed(seed)
    np.random.seed(seed)
    all_idx=list(range(n))
    random.shuffle(all_idx)
    num_train=int(n*TRAIN_RATE)
    num_val=int(n*VAL_RATE)
    idx_train=np.array(all_idx[:num_train], dtype=int)
    idx_val=np.array(all_idx[num_train:num_train+num_val], dtype=int)
    idx_test=np.array(all_idx[num_train+num_val:], dtype=int)
    normal_train=idx_train[labels[idx_train]==0]
    normal_gen=normal_train[:int(len(normal_train)*SAMPLE_RATE)]
    return idx_train, idx_val, idx_test, normal_train, normal_gen


def stats_for(arr):
    arr=np.asarray(arr, dtype=float)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)) if len(arr)>1 else 0.0,
        'min': float(np.min(arr)),
        'p25': float(np.percentile(arr,25)),
        'median': float(np.median(arr)),
        'p75': float(np.percentile(arr,75)),
        'max': float(np.max(arr)),
    }


def main():
    data=sio.loadmat(DATA)
    labels=np.squeeze(np.array(data['Label'] if 'Label' in data else data['gnd'])).astype(int)
    X=data['Attributes'] if 'Attributes' in data else data['X']
    A=sp.csr_matrix(data['Network'] if 'Network' in data else data['A'])
    X=sp.csr_matrix(X).astype(float)
    n=A.shape[0]
    deg=np.asarray(A.sum(axis=1)).ravel()
    # Use dense feature summary only; Elliptic is 45564x93 manageable.
    Xd=X.toarray()
    feat_norm=np.linalg.norm(Xd, axis=1)
    anomaly_idx=np.where(labels==1)[0]
    normal_idx=np.where(labels==0)[0]

    rows=[]
    for seed in SEEDS:
        idx_train, idx_val, idx_test, normal_train, normal_gen = split_indices(n, labels, seed)
        test_anom=idx_test[labels[idx_test]==1]
        test_norm=idx_test[labels[idx_test]==0]
        # Graph proximity: anomaly nodes adjacent to any labeled normal; 2-hop via sparse multiply indicator.
        train_mask=np.zeros(n, dtype=bool); train_mask[normal_train]=True
        neigh1=(A[:, normal_train].sum(axis=1).A.ravel() > 0)
        # 2-hop approximate: neighbors of train normals then nodes adjacent to those.
        first_neighbors=np.where(neigh1)[0]
        neigh2=(A[:, first_neighbors].sum(axis=1).A.ravel() > 0) if len(first_neighbors) else np.zeros(n, dtype=bool)
        # Feature distance from test anomalies/normals to labeled-normal centroid.
        centroid=Xd[normal_train].mean(axis=0)
        dist=np.linalg.norm(Xd-centroid, axis=1)
        row={
            'seed': seed,
            'n_train': int(len(idx_train)),
            'n_train_anom': int(np.sum(labels[idx_train]==1)),
            'n_train_normal': int(len(normal_train)),
            'n_test': int(len(idx_test)),
            'n_test_anom': int(len(test_anom)),
            'test_anom_rate': float(np.mean(labels[idx_test]==1)),
            'normal_gen_count': int(len(normal_gen)),
            'train_normal_degree': stats_for(deg[normal_train]),
            'test_anom_degree': stats_for(deg[test_anom]),
            'train_normal_feat_norm': stats_for(feat_norm[normal_train]),
            'test_anom_feat_norm': stats_for(feat_norm[test_anom]),
            'test_anom_dist_to_train_normal_centroid': stats_for(dist[test_anom]),
            'test_norm_dist_to_train_normal_centroid': stats_for(dist[test_norm]),
            'test_anom_adjacent_to_train_normal_ratio': float(np.mean(neigh1[test_anom])),
            'test_norm_adjacent_to_train_normal_ratio': float(np.mean(neigh1[test_norm])),
            'test_anom_within_2hop_train_normal_ratio': float(np.mean(neigh2[test_anom])),
            'test_norm_within_2hop_train_normal_ratio': float(np.mean(neigh2[test_norm])),
            'train_normal_indices_head': normal_train[:20].tolist(),
        }
        rows.append(row)

    # deltas seed3 vs non3
    seed3=next(r for r in rows if r['seed']==3)
    flat_keys=['n_train_anom','n_train_normal','n_test_anom','test_anom_rate','test_anom_adjacent_to_train_normal_ratio','test_norm_adjacent_to_train_normal_ratio','test_anom_within_2hop_train_normal_ratio','test_norm_within_2hop_train_normal_ratio']
    stat_keys=[('train_normal_degree','mean'),('test_anom_degree','mean'),('train_normal_feat_norm','mean'),('test_anom_feat_norm','mean'),('test_anom_dist_to_train_normal_centroid','mean'),('test_norm_dist_to_train_normal_centroid','mean')]
    deltas={}
    for k in flat_keys:
        vals=[r[k] for r in rows if r['seed']!=3]
        deltas[k]={'seed3': seed3[k], 'non3_mean': float(np.mean(vals)), 'delta': float(seed3[k]-np.mean(vals))}
    for k, sub in stat_keys:
        vals=[r[k][sub] for r in rows if r['seed']!=3]
        x=seed3[k][sub]
        deltas[f'{k}.{sub}']={'seed3': x, 'non3_mean': float(np.mean(vals)), 'delta': float(x-np.mean(vals))}

    out={'rows': rows, 'seed3_deltas': deltas}
    (OUT/'split_graph_analysis.json').write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')

    lines=['# Split / Graph Structure Analysis', '', '## Seed-level summary', '', '| seed | train normal | train anom in raw train | test anom | test anom rate | anom 1-hop to trainN | anom 2-hop to trainN | anom dist centroid |', '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['seed']} | {r['n_train_normal']} | {r['n_train_anom']} | {r['n_test_anom']} | {r['test_anom_rate']:.4f} | {r['test_anom_adjacent_to_train_normal_ratio']:.4f} | {r['test_anom_within_2hop_train_normal_ratio']:.4f} | {r['test_anom_dist_to_train_normal_centroid']['mean']:.4f} |")
    lines += ['', '## Seed 3 deltas from non-seed3 mean', '', '| metric | seed3 | non3 mean | delta |', '|---|---:|---:|---:|']
    for k,v in deltas.items():
        lines.append(f"| {k} | {v['seed3']:.6f} | {v['non3_mean']:.6f} | {v['delta']:.6f} |")
    (OUT/'split_graph_analysis.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(OUT/'split_graph_analysis.md')

if __name__ == '__main__':
    main()
