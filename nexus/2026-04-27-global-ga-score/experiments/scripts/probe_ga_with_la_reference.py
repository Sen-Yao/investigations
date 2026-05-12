#!/usr/bin/env python3
"""Evaluate final anomaly-reference purity with global G_a and local L_a.
G_a is computed once globally [N]. L_a(u|v) uses current h=[q,c] similarity.
Selection score is multiplicative: S_a(u|v)=G_a(u)*L_a(u|v), no weight hyperparameter.
"""
import argparse, json
from pathlib import Path
import numpy as np
import scipy.io as sio
import scipy.sparse as sp


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
    return np.searchsorted(base,x,side='right')/(len(base)+1e-12)

def eval_global_topk(g, labels, ks=(16,32,64,128,256)):
    order=np.argsort(g)[::-1]
    return {f'top{k}':float(np.mean(labels[order[:k]]==1)) for k in ks}

def eval_reference(g, h, labels, ref_k=16, block=512):
    h=l2(h.astype(np.float32)); n=len(labels)
    pur=[]
    g=np.asarray(g,dtype=np.float32)
    for st in range(0,n,block):
        sim=h[st:st+block]@h.T
        scores=sim*g[None,:]
        rows=scores.shape[0]
        for r in range(rows):
            scores[r, st+r] = -1e9
        idx=np.argpartition(scores, -ref_k, axis=1)[:,-ref_k:]
        pur.extend(np.mean(labels[idx]==1,axis=1).tolist())
    pur=np.array(pur)
    return {
        'avg_ref_purity':float(pur.mean()),
        'normal_node_ref_purity':float(pur[labels==0].mean()),
        'anomaly_node_ref_purity':float(pur[labels==1].mean()),
    }

def summarize(items):
    keys=items[0].keys(); return {k:{'mean':float(np.mean([x[k] for x in items])),'std':float(np.std([x[k] for x in items]))} for k in keys}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--pp-k',type=int,default=6); ap.add_argument('--train-rate',type=float,default=0.05); ap.add_argument('--seeds',default='0,1,2,3,4'); ap.add_argument('--ref-k',type=int,default=16); ap.add_argument('--block',type=int,default=512); ap.add_argument('--out',required=True)
    args=ap.parse_args(); seeds=[int(x) for x in args.seeds.split(',')]
    adj,feat,labels=load_mat_dataset(args.data); normal_all=np.where(labels==0)[0]
    ndc=ndc_score(feat,adj,args.pp_k); q=zsig(ndc); c=local_mean(q,adj); support=minmax(q*c)
    h=np.stack([q,c],axis=1)
    z=l2(mps(feat,adj,args.pp_k)); xa=l2(row_norm(feat.copy()).astype(np.float32))
    result={'data':args.data,'n':int(len(labels)),'anom_rate':float(labels.mean()),'train_rate':args.train_rate,'ref_k':args.ref_k,'seeds':[]}
    for seed in seeds:
        rng=np.random.default_rng(seed); normal=normal_all.copy(); rng.shuffle(normal)
        train=np.sort(normal[:max(1,min(len(normal),int(len(labels)*args.train_rate)))])
        center=z[train].mean(0,keepdims=True); center=center/(np.linalg.norm(center)+1e-12)
        center_dev=minmax(np.linalg.norm(z-center,axis=1))
        density_dev=minmax(1-topk_mean(z,z[train],32))
        attr_dev=minmax(1-topk_mean(xa,xa[train],32))
        max_dev=np.maximum(center_dev,density_dev)
        ns=normal_percentile(support,train)
        nc=normal_percentile(center_dev,train)
        nd=normal_percentile(density_dev,train)
        na=normal_percentile(attr_dev,train)
        nm=normal_percentile(max_dev,train)
        candidates={
            'baseline_support_qc':support,
            'normal_cal_support':ns,
            'normal_cal_max_dev':nm,
            'normal_cal_max_support_density':np.maximum(ns,nd),
            'normal_cal_max_support_attr':np.maximum(ns,na),
            'normal_cal_attr_dev':na,
            'normal_cal_hmean_support_density':2*ns*nd/(ns+nd+1e-12),
            'normal_cal_min_support_center':np.minimum(ns,nc),
        }
        one={'seed':seed,'train_normal':int(len(train)),'scores':{}}
        for name,g in candidates.items():
            g=minmax(g)
            one['scores'][name]={'global_topk':eval_global_topk(g,labels),'reference':eval_reference(g,h,labels,args.ref_k,args.block)}
        result['seeds'].append(one)
    # aggregate
    agg={}
    for name in result['seeds'][0]['scores'].keys():
        agg[name]={
            'global_topk':summarize([s['scores'][name]['global_topk'] for s in result['seeds']]),
            'reference':summarize([s['scores'][name]['reference'] for s in result['seeds']])
        }
    result['aggregate']=agg
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
