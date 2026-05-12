#!/usr/bin/env python3
"""
Probe rank-based two-factor G_a:
G_a(v) = percentile(NormalDeviation(v)) * percentile(AnomalySupport(v))
No new weighting hyper-parameters.
"""
import argparse, json
from pathlib import Path
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import ks_2samp, rankdata


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

def l2_rows(x):
    return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12)

def norm_adj(adj):
    deg=adj.sum(1)
    inv=np.zeros_like(deg,dtype=np.float32); m=deg>0; inv[m]=1/np.sqrt(deg[m])
    return inv[:,None]*adj*inv[None,:]

def mps(features, adj, k):
    a=norm_adj(adj); outs=[features]; cur=features.copy()
    for _ in range(k):
        cur=a@cur; outs.append(cur.copy())
    return np.concatenate(outs,1).astype(np.float32)

def ndc_score(features, adj, k):
    f=row_norm(features.copy()); a=norm_adj(adj); n,d=f.shape
    hops=np.zeros((n,k+1,d),dtype=np.float32); hops[:,0]=f; cur=f.copy()
    for h in range(1,k+1):
        cur=a@cur; hops[:,h]=cur
    delta=hops[:,1:]-hops[:,:-1]
    ndc=np.zeros(n,dtype=np.float32)
    for i in range(n):
        neigh=np.where(adj[i]>0)[0]
        if len(neigh)==0: continue
        x=delta[i].reshape(-1); y=delta[neigh].mean(0).reshape(-1)
        if np.std(x)<1e-8 or np.std(y)<1e-8: ndc[i]=0
        else: ndc[i]=np.corrcoef(x,y)[0,1]
    return ndc

def zsig(x):
    return 1/(1+np.exp(-(x-x.mean())/(x.std()+1e-8)))

def minmax(x):
    x=np.asarray(x,dtype=np.float64)
    return (x-x.min())/(x.max()-x.min()+1e-12)

def percentile(x):
    # high score -> percentile close to 1, average ranks for ties
    return (rankdata(x, method='average')-1)/(len(x)-1+1e-12)

def local_mean(values, adj):
    deg=adj.sum(1); out=adj@values; res=np.zeros_like(values,dtype=np.float64)
    m=deg>0; res[m]=out[m]/deg[m]; res[~m]=values[~m]
    return res

def topk_mean_sim_to_bank(x, bank, k=32, block=2048):
    vals=[]; k=min(k, bank.shape[0])
    for st in range(0,x.shape[0],block):
        sims=x[st:st+block]@bank.T
        vals.append(np.partition(sims,-k,axis=1)[:,-k:].mean(1))
    return np.concatenate(vals)

def eval_score(s, labels, topks=(16,32,64,128,256)):
    out={
        'auc': float(roc_auc_score(labels,s)),
        'ap': float(average_precision_score(labels,s)),
        'normal_mean': float(s[labels==0].mean()),
        'anomaly_mean': float(s[labels==1].mean()),
        'ks': float(ks_2samp(s[labels==0],s[labels==1]).statistic),
    }
    order=np.argsort(s)[::-1]
    for k in topks:
        out[f'top{k}_anom_ratio']=float(np.mean(labels[order[:k]]==1))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--train-rate', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pp-k', type=int, default=6)
    ap.add_argument('--bank-k', type=int, default=32)
    ap.add_argument('--out', required=True)
    args=ap.parse_args()
    np.random.seed(args.seed)
    adj,feat,labels=load_mat_dataset(args.data)
    normal=np.where(labels==0)[0]; np.random.shuffle(normal)
    n_train=max(1,min(len(normal),int(len(labels)*args.train_rate)))
    train_normal=np.sort(normal[:n_train])

    # Normal deviation via normal center and normal-bank density
    z=l2_rows(mps(feat,adj,args.pp_k))
    center=z[train_normal].mean(0,keepdims=True); center=center/(np.linalg.norm(center)+1e-12)
    center_dist=np.linalg.norm(z-center,axis=1)
    center_dev=minmax(center_dist)
    mps_density=topk_mean_sim_to_bank(z,z[train_normal],args.bank_k)
    density_dev=minmax(1-mps_density)

    # Attribute density failure (for ablation only)
    xa=l2_rows(row_norm(feat.copy()).astype(np.float32))
    attr_density=topk_mean_sim_to_bank(xa, xa[train_normal], args.bank_k)
    attr_dev=minmax(1-attr_density)

    # Anomaly support via existing q*c
    ndc=ndc_score(feat,adj,args.pp_k)
    q=zsig(ndc); c=local_mean(q,adj); support=minmax(q*c)

    candidates={
        'baseline_support_qc': support,
        'center_deviation': center_dev,
        'mps_density_deviation': density_dev,
        'attr_density_deviation': attr_dev,
        # rank-based two-factor variants
        'rank_center_x_rank_support': percentile(center_dev)*percentile(support),
        'rank_density_x_rank_support': percentile(density_dev)*percentile(support),
        'rank_attr_x_rank_support': percentile(attr_dev)*percentile(support),
        'rank_max_normaldev_x_rank_support': percentile(np.maximum(center_dev,density_dev))*percentile(support),
        'rank_max_allnormaldev_x_rank_support': percentile(np.maximum.reduce([center_dev,density_dev,attr_dev]))*percentile(support),
        # non-rank products for comparison
        'raw_center_x_support': minmax(center_dev*support),
        'raw_density_x_support': minmax(density_dev*support),
        'raw_attr_x_support': minmax(attr_dev*support),
    }
    res={'data':args.data,'n':int(len(labels)),'anom_rate':float(labels.mean()),'train_normal':int(len(train_normal)),'scores':{}}
    for name,s in candidates.items():
        res['scores'][name]=eval_score(s,labels)
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(res,indent=2))
    print(json.dumps(res,indent=2))

if __name__=='__main__':
    main()
