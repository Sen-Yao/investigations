#!/usr/bin/env python3
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from pathlib import Path
from scipy.stats import ks_2samp

def load_dataset(name, path):
    f = {'photo': 'photo.mat', 'tolokers': 'tolokers.mat', 'elliptic': 'elliptic.mat'}
    data = sio.loadmat(Path(path) / f[name])
    X = data.get('Attributes', data.get('X', data.get('features')))
    if sp.issparse(X): X = X.toarray()
    X = X.astype(np.float32)
    Y = data.get('Label', data.get('Y', data.get('labels'))).flatten().astype(np.int64)
    A = data.get('Network', data.get('A', data.get('adj')))
    if sp.issparse(A): A = A.toarray()
    A = A.astype(np.float32)
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
    try:
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
    except Exception as e:
        print(f'  错误: {e}')
        results[ds] = {'error': str(e)}

print('\n=== 假设验证 ===')
print('\nH1 (深层方差更低):')
for ds, r in results.items():
    if 'variances' in r:
        deeper = r['variances'][-1] < r['variances'][0]
        print(f'  {ds}: hop_0={r["variances"][0]:.4f}, hop_6={r["variances"][-1]:.4f}, 深层更低={"是" if deeper else "否"}')

print('\nH2 (方差-KS负相关):')
for ds, r in results.items():
    if 'corr' in r:
        print(f'  {ds}: r={r["corr"]:.4f} ({"支持" if r["corr"]<0 else "不支持"})')

print('\nH3 (高频信号更优):')
for ds, r in results.items():
    if 'variances' in r:
        print(f'  {ds}: 需要结合区分力分析')
