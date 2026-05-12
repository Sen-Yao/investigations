#!/usr/bin/env python3
"""Normal-calibrated global G_a probe.
Uses only normal labels in normal_idx to calibrate support/deviation scores.
Tests 5% and 15% normal train rates across seeds.
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
def local_mean(v,adj):
    deg=adj.sum(1); out=adj@v; res=np.zeros_like(v,dtype=np.float64); m=deg>0; res[m]=out[m]/deg[m]; res[~m]=v[~m]; return res

def topk_mean(x,bank,k=32,block=2048):
    vals=[]; k=min(k,bank.shape[0])
    for st in range(0,x.shape[0],block):
        sims=x[st:st+block]@bank.T
        vals.append(np.partition(sims,-k,axis=1)[:,-k:].mean(1))
    return np.concatenate(vals)

def normal_percentile(x, normal_idx):
    base=np.sort(x[normal_idx])
    # fraction of normal values <= x
    return np.searchsorted(base, x, side='right')/(len(base)+1e-12)

def normal_tail_z(x, normal_idx):
    mu=x[normal_idx].mean(); sd=x[normal_idx].std()+1e-8
    return 1/(1+np.exp(-(x-mu)/sd))

def eval_score(s, labels, topks=(16,32,64,128,256)):
    order=np.argsort(s)[::-1]
    out={'auc':float(roc_auc_score(labels,s)),'ap':float(average_precision_score(labels,s)),'ks':float(ks_2samp(s[labels==0],s[labels==1]).statistic)}
    for k in topks: out[f'top{k}']=float(np.mean(labels[order[:k]]==1))
    return out

def summarize(vals):
    keys=vals[0].keys(); return {k:{'mean':float(np.mean([v[k] for v in vals])),'std':float(np.std([v[k] for v in vals],ddof=0))} for k in keys}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--pp-k',type=int,default=6); ap.add_argument('--bank-k',type=int,default=32); ap.add_argument('--rates',default='0.05,0.15'); ap.add_argument('--seeds',default='0,1,2,3,4'); ap.add_argument('--out',required=True)
    args=ap.parse_args(); rates=[float(x) for x in args.rates.split(',')]; seeds=[int(x) for x in args.seeds.split(',')]
    adj,feat,labels=load_mat_dataset(args.data)
    ndc=ndc_score(feat,adj,args.pp_k); q=zsig(ndc); support=minmax(q*local_mean(q,adj))
    z=l2(mps(feat,adj,args.pp_k)); xa=l2(row_norm(feat.copy()).astype(np.float32))
    normal_all=np.where(labels==0)[0]
    result={'data':args.data,'n':int(len(labels)),'anom_rate':float(labels.mean()),'rates':{}}
    for rate in rates:
        seed_results=[]
        for seed in seeds:
            rng=np.random.default_rng(seed); normal=normal_all.copy(); rng.shuffle(normal)
            n_train=max(1,min(len(normal),int(len(labels)*rate)))
            train=np.sort(normal[:n_train])
            center=z[train].mean(0,keepdims=True); center=center/(np.linalg.norm(center)+1e-12)
            center_dev=minmax(np.linalg.norm(z-center,axis=1))
            density_dev=minmax(1-topk_mean(z,z[train],args.bank_k))
            attr_dev=minmax(1-topk_mean(xa,xa[train],args.bank_k))
            max_dev=np.maximum(center_dev,density_dev)
            # normal-calibrated ranks: how far above normal distribution
            ns=normal_percentile(support, train)
            nc=normal_percentile(center_dev, train)
            nd=normal_percentile(density_dev, train)
            na=normal_percentile(attr_dev, train)
            nm=normal_percentile(max_dev, train)
            zc=normal_tail_z(center_dev, train); zd=normal_tail_z(density_dev, train); za=normal_tail_z(attr_dev, train)
            candidates={
                'baseline_support_qc': support,
                'normal_cal_support': ns,
                'normal_cal_center_dev': nc,
                'normal_cal_density_dev': nd,
                'normal_cal_attr_dev': na,
                'normal_cal_max_dev': nm,
                'normal_cal_max_support_center': np.maximum(ns,nc),
                'normal_cal_min_support_center': np.minimum(ns,nc),
                'normal_cal_hmean_support_center': 2*ns*nc/(ns+nc+1e-12),
                'normal_cal_max_support_density': np.maximum(ns,nd),
                'normal_cal_min_support_density': np.minimum(ns,nd),
                'normal_cal_hmean_support_density': 2*ns*nd/(ns+nd+1e-12),
                'normal_cal_max_support_attr': np.maximum(ns,na),
                'normal_cal_min_support_attr': np.minimum(ns,na),
                'normal_cal_hmean_support_attr': 2*ns*na/(ns+na+1e-12),
                'normal_z_hmean_support_center': 2*ns*zc/(ns+zc+1e-12),
                'normal_z_hmean_support_density': 2*ns*zd/(ns+zd+1e-12),
                'normal_z_hmean_support_attr': 2*ns*za/(ns+za+1e-12),
            }
            one={'seed':seed,'train_normal':int(len(train)),'scores':{}}
            for name,s in candidates.items(): one['scores'][name]=eval_score(minmax(s),labels)
            seed_results.append(one)
        # aggregate
        agg={}
        names=seed_results[0]['scores'].keys()
        for name in names: agg[name]=summarize([sr['scores'][name] for sr in seed_results])
        result['rates'][str(rate)]={'seeds':seed_results,'aggregate':agg}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
