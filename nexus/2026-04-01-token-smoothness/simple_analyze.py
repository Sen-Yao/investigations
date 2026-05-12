#!/usr/bin/env python3
"""简化版平滑度分析"""
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from pathlib import Path
from scipy.stats import ks_2samp

def load_dataset(name, path):
    f = {'photo': 'photo.mat', 'tolokers': 'tolokers.mat', 'elliptic': 'elliptic.mat'}
    data = sio.loadmat(Path(path) / f[name])
    X = data.get('Attributes', data.get('X')).toarray().astype(np.float32)
    Y = data.get('Label', data.get('Y')).flatten().astype(np.int64)
    A = data.get('Network', data.get('A')).toarray().astype(np.float32)
    if Y.min() > 0: Y = Y - Y.min()
    return X, Y, A

def compute_hop(X, A, K=6):
    d = A.sum(1); d[d==0]=1
    An = np.diag(d**-0.5) @ A @ np.diag(d**-0.5)
    H = np.zeros((X.shape[0], K+1, X.shape[1]), np.float32)
    H[:,0] = X
    t = X.copy()
    for k in range(K): t = An @ t; H[:,k+1] = t
    return H

results = {}
for ds in ['photo', 'tolokers', 'elliptic']:
    print(f'分析 {ds}...')
    X, Y, A = load_dataset(ds, '/root/gpufree-data/linziyao/VoxG/dataset')
    H = compute_hop(X, A)
    nm, am = Y==0, Y==1
    
    vars_, ks_ = [], []
    for k in range(7):
        v = np.var(H[:,k])
        n = np.linalg.norm(H[:,k], axis=1)
        ks, _ = ks_2samp(n[nm], n[am])
        vars_.append(v); ks_.append(ks)
    
    corr = np.corrcoef(vars_, ks_)[0,1]
    results[ds] = {'variances': vars_, 'ks_stats': ks_, 'corr': corr}
    print(f'  方差范围: {min(vars_):.4f} - {max(vars_):.4f}')
    print(f'  方差-KS相关性: {corr:.4f}')

print('\n--- 假设验证 ---')
print('H1 (深层方差更低):')
for ds, r in results.items():
    deeper = r['variances'][-1] < r['variances'][0]
    print(f'  {ds}: {"是" if deeper else "否"}')

print('\nH2 (方差-KS负相关):')
for ds, r in results.items():
    print(f'  {ds}: r={r["corr"]:.4f} ({"负相关" if r["corr"]<0 else "正相关"})')
