#!/usr/bin/env python3
"""
Probe several candidate G_a scores under full-graph setting.
Candidates are offline scores derived from current MPS/normal space and anomaly-affinity family.
"""

import argparse, json
from pathlib import Path
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import ks_2samp


def load_mat_dataset(path):
    data=sio.loadmat(path)
    label=data['Label'] if 'Label' in data else data['gnd']
    attr=data['Attributes'] if 'Attributes' in data else data['X']
    network=data['Network'] if 'Network' in data else data['A']
    feat=attr.toarray() if sp.issparse(attr) else np.asarray(attr)
    adj=network.toarray() if sp.issparse(network) else np.asarray(network)
    labels=np.asarray(label).reshape(-1)
    labels=labels-labels.min()
    return adj.astype(np.float32), feat.astype(np.float32), labels.astype(int)


def row_norm(x):
    s=x.sum(1, keepdims=True); s[s==0]=1.0
    return x/s


def norm_adj(adj):
    deg=adj.sum(1)
    inv=np.where(deg>0,1/np.sqrt(deg),0.0)
    return inv[:,None]*adj*inv[None,:]


def mps(features, adj, k):
    a=norm_adj(adj); outs=[features]; cur=features.copy()
    for _ in range(k):
        cur=a@cur; outs.append(cur.copy())
    return np.concatenate(outs,1).astype(np.float32)


def cosine_matrix(a,b):
    a=a/(np.linalg.norm(a,axis=1,keepdims=True)+1e-12)
    b=b/(np.linalg.norm(b,axis=1,keepdims=True)+1e-12)
    return a@b.T


def ndc_score(features, adj, k):
    f=row_norm(features.copy()); a=norm_adj(adj); n,d=f.shape
    hops=np.zeros((n,k+1,d), dtype=np.float32); hops[:,0]=f; cur=f.copy()
    for h in range(1,k+1):
        cur=a@cur; hops[:,h]=cur
    delta=hops[:,1:]-hops[:,:-1]
    ndc=np.zeros(n, dtype=np.float32)
    for i in range(n):
        neigh=np.where(adj[i]>0)[0]
        if len(neigh)==0: continue
        x=delta[i].reshape(-1); y=delta[neigh].mean(0).reshape(-1)
        if np.std(x)<1e-8 or np.std(y)<1e-8: ndc[i]=0
        else: ndc[i]=np.corrcoef(x,y)[0,1]
    return ndc


def minmax(x):
    x=np.asarray(x,dtype=np.float64)
    return (x-x.min())/(x.max()-x.min()+1e-12)


def zsig(x):
    return 1/(1+np.exp(-(x-x.mean())/(x.std()+1e-8)))


def eval_score(s, labels, topks):
    out={
        'auc': float(roc_auc_score(labels,s)),
        'ap': float(average_precision_score(labels,s)),
        'normal_mean': float(s[labels==0].mean()),
        'anomaly_mean': float(s[labels==1].mean()),
        'ks': float(ks_2samp(s[labels==0], s[labels==1]).statistic),
    }
    order=np.argsort(s)[::-1]
    for k in topks:
        idx=order[:k]
        out[f'top{k}_anom_ratio']=float(np.mean(labels[idx]==1))
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--train-rate', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--pp-k', type=int, default=6)
    p.add_argument('--normal-bank-k', type=int, default=32)
    p.add_argument('--out', required=True)
    args=p.parse_args()
    np.random.seed(args.seed)
    adj, feat, labels=load_mat_dataset(args.data)
    normal=np.where(labels==0)[0]
    np.random.shuffle(normal)
    n_train=max(1,min(len(normal), int(len(labels)*args.train_rate)))
    train_normal=np.sort(normal[:n_train])

    z=mps(feat, adj, args.pp_k)
    zn=z/(np.linalg.norm(z,axis=1,keepdims=True)+1e-12)
    center=zn[train_normal].mean(0,keepdims=True)
    center=center/(np.linalg.norm(center)+1e-12)
    gn=(zn@center.T).reshape(-1)
    center_dist=np.linalg.norm(zn-center, axis=1)

    # normal explainability by top-k similarity to train normal bank
    sims=zn@zn[train_normal].T
    top=np.sort(sims,axis=1)[:,-min(args.normal_bank_k,sims.shape[1]):]
    normal_explain=top.mean(1)
    residual=1-normal_explain

    ndc=ndc_score(feat, adj, args.pp_k)
    q=zsig(ndc)
    c=np.zeros_like(q)
    for i in range(len(q)):
        neigh=np.where(adj[i]>0)[0]
        c[i]=q[i] if len(neigh)==0 else q[neigh].mean()
    anom_aff=q*c

    # ring score: high if moderately far from center, not extreme. Use Gaussian bump around upper-normal quantile.
    normal_d=center_dist[train_normal]
    mu=np.quantile(normal_d,0.95)
    sigma=np.std(normal_d)+1e-8
    ring=np.exp(-((center_dist-(mu+sigma))**2)/(2*(sigma**2)))

    candidates={
        'ndc_zsig': minmax(q),
        'local_concentration_c': minmax(c),
        'anom_affinity_qc': minmax(anom_aff),
        'one_minus_Gn': minmax(1-gn),
        'center_distance': minmax(center_dist),
        'normal_explain_residual_topk': minmax(residual),
        'ring_distance_score': minmax(ring),
        'residual_plus_affinity': minmax(minmax(residual)+minmax(anom_aff)),
        'centerdist_plus_affinity': minmax(minmax(center_dist)+minmax(anom_aff)),
        'residual_times_affinity': minmax(minmax(residual)*minmax(anom_aff)),
    }
    topks=[16,32,64,128]
    result={'data':args.data,'n':int(len(labels)),'anom_rate':float(labels.mean()),'train_normal':int(len(train_normal)),'scores':{}}
    for name,score in candidates.items():
        result['scores'][name]=eval_score(score, labels, topks)
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
