#!/usr/bin/env python3
"""Diagnostic support-first rerank probe.
Not intended as final method. Tests if normal deviation can improve ranking inside top-M by anomaly support q*c.
"""
import argparse, json
from pathlib import Path
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.stats import rankdata


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

def l2(x): return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12)

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

def purity(order,labels,ks): return {f'top{k}':float(np.mean(labels[order[:k]]==1)) for k in ks}

def rerank_order(support, reranker, M):
    pool=np.argsort(support)[::-1][:M]
    return pool[np.argsort(reranker[pool])[::-1]]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--train-rate',type=float,default=0.05); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--pp-k',type=int,default=6); ap.add_argument('--bank-k',type=int,default=32); ap.add_argument('--out',required=True)
    args=ap.parse_args(); np.random.seed(args.seed)
    adj,feat,labels=load_mat_dataset(args.data)
    normal=np.where(labels==0)[0]; np.random.shuffle(normal); train=normal[:max(1,min(len(normal),int(len(labels)*args.train_rate)))]
    z=l2(mps(feat,adj,args.pp_k)); center=z[train].mean(0,keepdims=True); center=center/(np.linalg.norm(center)+1e-12)
    center_dev=minmax(np.linalg.norm(z-center,axis=1))
    dens=topk_mean(z,z[train],args.bank_k); density_dev=minmax(1-dens)
    xa=l2(row_norm(feat.copy()).astype(np.float32)); attr_dens=topk_mean(xa,xa[train],args.bank_k); attr_dev=minmax(1-attr_dens)
    ndc=ndc_score(feat,adj,args.pp_k); q=zsig(ndc); support=minmax(q*local_mean(q,adj))
    rerankers={
      'center_dev': center_dev,
      'density_dev': density_dev,
      'attr_dev': attr_dev,
      'rank_center_x_support': percentile(center_dev)*percentile(support),
      'rank_density_x_support': percentile(density_dev)*percentile(support),
      'rank_attr_x_support': percentile(attr_dev)*percentile(support),
    }
    ks=[16,32,64,128,256]
    Ms=[64,128,256,512,1024]
    res={'data':args.data,'anom_rate':float(labels.mean()),'baseline_support':purity(np.argsort(support)[::-1],labels,ks),'rerank':{}}
    for M in Ms:
        res['rerank'][str(M)]={}
        for name,r in rerankers.items():
            res['rerank'][str(M)][name]=purity(rerank_order(support,r,M),labels,[k for k in ks if k<=M])
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))
if __name__=='__main__': main()
