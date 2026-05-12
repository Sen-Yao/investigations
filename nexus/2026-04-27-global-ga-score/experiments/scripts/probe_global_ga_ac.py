#!/usr/bin/env python3
"""
Probe global query-independent G_a scores:
A) support-conditioned deviation percentile
C) Pareto / dominance style scores

All scores are computed once globally as arrays of shape [N].
"""
import argparse, json
from pathlib import Path
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.stats import rankdata, ks_2samp
from sklearn.metrics import roc_auc_score, average_precision_score


def load_mat_dataset(path):
    data=sio.loadmat(path)
    label=data['Label'] if 'Label' in data else data['gnd']
    attr=data['Attributes'] if 'Attributes' in data else data['X']
    network=data['Network'] if 'Network' in data else data['A']
    feat=attr.toarray() if sp.issparse(attr) else np.asarray(attr)
    adj=network.toarray() if sp.issparse(network) else np.asarray(network)
    labels=np.asarray(label).reshape(-1); labels=labels-labels.min()
    return adj.astype(np.float32), feat.astype(np.float32), labels.astype(int)

def row_norm(x):
    s=x.sum(1,keepdims=True); s[s==0]=1.0; return x/s

def l2_rows(x): return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12)

def norm_adj(adj):
    deg=adj.sum(1); inv=np.zeros_like(deg,dtype=np.float32); m=deg>0; inv[m]=1/np.sqrt(deg[m]); return inv[:,None]*adj*inv[None,:]

def mps(feat,adj,k):
    a=norm_adj(adj); outs=[feat]; cur=feat.copy()
    for _ in range(k): cur=a@cur; outs.append(cur.copy())
    return np.concatenate(outs,1).astype(np.float32)

def ndc_score(feat,adj,k):
    f=row_norm(feat.copy()); a=norm_adj(adj); n,d=f.shape
    hops=np.zeros((n,k+1,d),dtype=np.float32); hops[:,0]=f; cur=f.copy()
    for h in range(1,k+1): cur=a@cur; hops[:,h]=cur
    delta=hops[:,1:]-hops[:,:-1]; ndc=np.zeros(n,dtype=np.float32)
    for i in range(n):
        neigh=np.where(adj[i]>0)[0]
        if len(neigh)==0: continue
        x=delta[i].reshape(-1); y=delta[neigh].mean(0).reshape(-1)
        ndc[i]=0 if np.std(x)<1e-8 or np.std(y)<1e-8 else np.corrcoef(x,y)[0,1]
    return ndc

def zsig(x): return 1/(1+np.exp(-(x-x.mean())/(x.std()+1e-8)))
def minmax(x):
    x=np.asarray(x,dtype=np.float64); return (x-x.min())/(x.max()-x.min()+1e-12)
def percentile(x): return (rankdata(x,method='average')-1)/(len(x)-1+1e-12)
def local_mean(v,adj):
    deg=adj.sum(1); out=adj@v; res=np.zeros_like(v,dtype=np.float64); m=deg>0; res[m]=out[m]/deg[m]; res[~m]=v[~m]; return res

def topk_mean(x,bank,k=32,block=2048):
    vals=[]; k=min(k,bank.shape[0])
    for st in range(0,x.shape[0],block):
        sims=x[st:st+block]@bank.T
        vals.append(np.partition(sims,-k,axis=1)[:,-k:].mean(1))
    return np.concatenate(vals)

def eval_score(s, labels, topks=(16,32,64,128,256)):
    order=np.argsort(s)[::-1]
    out={
        'auc':float(roc_auc_score(labels,s)),
        'ap':float(average_precision_score(labels,s)),
        'ks':float(ks_2samp(s[labels==0],s[labels==1]).statistic),
        'normal_mean':float(s[labels==0].mean()),
        'anomaly_mean':float(s[labels==1].mean())
    }
    for k in topks:
        out[f'top{k}_anom_ratio']=float(np.mean(labels[order[:k]]==1))
    return out


def support_conditioned_percentile(support, deviation, bins=20):
    """Global score: deviation percentile within support quantile bin."""
    sr=percentile(support)
    out=np.zeros_like(sr,dtype=np.float64)
    edges=np.linspace(0,1,bins+1)
    for b in range(bins):
        lo,hi=edges[b],edges[b+1]
        mask=(sr>=lo)&(sr<=hi if b==bins-1 else sr<hi)
        if mask.sum()==0: continue
        out[mask]=percentile(deviation[mask])
    # multiply by support rank softly so low-support bins cannot dominate completely
    return minmax(out*sr)


def rolling_support_conditioned_percentile(support, deviation, window_frac=0.1):
    """For each node, percentile of deviation in nearest support-rank window.
    O(N * window) after sorting; acceptable for current datasets.
    """
    n=len(support); sr=percentile(support); order=np.argsort(sr)
    dev_sorted=deviation[order]
    w=max(16,int(n*window_frac))
    out=np.zeros(n,dtype=np.float64)
    for pos,idx in enumerate(order):
        l=max(0,pos-w//2); r=min(n,pos+w//2+1)
        vals=dev_sorted[l:r]
        out[idx]=(rankdata(vals,method='average')[-(r-pos)] if False else np.mean(vals<=deviation[idx]))
    return minmax(out*sr)


def pareto_dominance_count(support, deviation):
    """Approximate dominance strength by ranks.
    Higher is better: product of marginal percentiles approximates fraction dominated if independent.
    """
    rs=percentile(support); rd=percentile(deviation)
    return minmax(rs*rd)


def pareto_min_rank(support, deviation):
    rs=percentile(support); rd=percentile(deviation)
    return np.minimum(rs,rd)


def pareto_ideal_distance(support, deviation):
    rs=percentile(support); rd=percentile(deviation)
    dist=np.sqrt((1-rs)**2+(1-rd)**2)
    return minmax(1-dist/np.sqrt(2))


def pareto_harmonic(support, deviation):
    rs=percentile(support); rd=percentile(deviation)
    return 2*rs*rd/(rs+rd+1e-12)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True)
    ap.add_argument('--train-rate',type=float,default=0.05)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--pp-k',type=int,default=6)
    ap.add_argument('--bank-k',type=int,default=32)
    ap.add_argument('--out',required=True)
    args=ap.parse_args(); np.random.seed(args.seed)
    adj,feat,labels=load_mat_dataset(args.data)
    normal=np.where(labels==0)[0]; np.random.shuffle(normal)
    train=np.sort(normal[:max(1,min(len(normal),int(len(labels)*args.train_rate)))])

    z=l2_rows(mps(feat,adj,args.pp_k)); center=z[train].mean(0,keepdims=True); center=center/(np.linalg.norm(center)+1e-12)
    center_dev=minmax(np.linalg.norm(z-center,axis=1))
    density_dev=minmax(1-topk_mean(z,z[train],args.bank_k))
    xa=l2_rows(row_norm(feat.copy()).astype(np.float32)); attr_dev=minmax(1-topk_mean(xa,xa[train],args.bank_k))
    ndc=ndc_score(feat,adj,args.pp_k); q=zsig(ndc); support=minmax(q*local_mean(q,adj))

    deviations={
        'center':center_dev,
        'density':density_dev,
        'attr':attr_dev,
        'max_center_density':np.maximum(center_dev,density_dev),
    }
    candidates={'baseline_support_qc':support}
    for dname,dev in deviations.items():
        candidates[f'base_{dname}_dev']=dev
        candidates[f'rank_product_{dname}']=percentile(support)*percentile(dev)
        candidates[f'scp_bin20_{dname}']=support_conditioned_percentile(support,dev,bins=20)
        candidates[f'scp_bin50_{dname}']=support_conditioned_percentile(support,dev,bins=50)
        candidates[f'scp_roll10_{dname}']=rolling_support_conditioned_percentile(support,dev,window_frac=0.10)
        candidates[f'pareto_product_{dname}']=pareto_dominance_count(support,dev)
        candidates[f'pareto_min_{dname}']=pareto_min_rank(support,dev)
        candidates[f'pareto_ideal_{dname}']=pareto_ideal_distance(support,dev)
        candidates[f'pareto_harmonic_{dname}']=pareto_harmonic(support,dev)

    res={'data':args.data,'n':int(len(labels)),'anom_rate':float(labels.mean()),'train_normal':int(len(train)),'scores':{}}
    for name,s in candidates.items(): res['scores'][name]=eval_score(minmax(s),labels)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))

if __name__=='__main__': main()
