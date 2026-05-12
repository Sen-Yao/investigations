#!/usr/bin/env python3
"""Probe G/L combination rules: multiply vs add vs rank_add.
Fixed 5% normal-calibrated G_a candidates; compare final reference purity.
"""
import argparse,json
from pathlib import Path
import numpy as np, scipy.io as sio, scipy.sparse as sp

def load_mat_dataset(path):
    d=sio.loadmat(path); label=d['Label'] if 'Label' in d else d['gnd']; attr=d['Attributes'] if 'Attributes' in d else d['X']; net=d['Network'] if 'Network' in d else d['A']
    feat=attr.toarray() if sp.issparse(attr) else np.asarray(attr); adj=net.toarray() if sp.issparse(net) else np.asarray(net)
    y=np.asarray(label).reshape(-1); y=y-y.min(); return adj.astype(np.float32),feat.astype(np.float32),y.astype(int)
def row_norm(x):
    s=x.sum(1,keepdims=True); s[s==0]=1; return x/s
def l2(x): return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12)
def norm_adj(adj):
    deg=adj.sum(1); inv=np.zeros_like(deg,dtype=np.float32); m=deg>0; inv[m]=1/np.sqrt(deg[m]); return inv[:,None]*adj*inv[None,:]
def mps(feat,adj,k):
    a=norm_adj(adj); outs=[feat]; cur=feat.copy()
    for _ in range(k): cur=a@cur; outs.append(cur.copy())
    return np.concatenate(outs,1).astype(np.float32)
def ndc_score(feat,adj,k):
    f=row_norm(feat.copy()); a=norm_adj(adj); n,d=f.shape; hops=np.zeros((n,k+1,d),dtype=np.float32); hops[:,0]=f; cur=f.copy()
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
    deg=adj.sum(1); out=adj@v; r=np.zeros_like(v,dtype=np.float64); m=deg>0; r[m]=out[m]/deg[m]; r[~m]=v[~m]; return r
def topk_mean(x,bank,k=32,block=2048):
    vals=[]; k=min(k,bank.shape[0])
    for st in range(0,x.shape[0],block):
        sims=x[st:st+block]@bank.T; vals.append(np.partition(sims,-k,axis=1)[:,-k:].mean(1))
    return np.concatenate(vals)
def normal_percentile(x,idx):
    b=np.sort(x[idx]); return np.searchsorted(b,x,side='right')/(len(b)+1e-12)
def rank01_vec(x):
    order=np.argsort(np.argsort(x, kind='mergesort'), kind='mergesort').astype(np.float64)
    return order/(len(x)-1+1e-12)
def eval_reference(g,h,labels,mode,ref_k=16,block=512):
    h=l2(h.astype(np.float32)); n=len(labels); g=minmax(g).astype(np.float32); gr=rank01_vec(g).astype(np.float32); pur=[]
    for st in range(0,n,block):
        sim=h[st:st+block]@h.T; l=minmax(sim) if mode=='add' else sim
        if mode=='multiply': scores=sim*g[None,:]
        elif mode=='add': scores=l+g[None,:]
        elif mode=='rank_add':
            # query-wise rank of local relevance plus global rank
            lr=np.argsort(np.argsort(sim,axis=1),axis=1).astype(np.float32)/(n-1+1e-12)
            scores=lr+gr[None,:]
        else: raise ValueError(mode)
        for r in range(scores.shape[0]): scores[r,st+r]=-1e9
        idx=np.argpartition(scores,-ref_k,axis=1)[:,-ref_k:]
        pur.extend(np.mean(labels[idx]==1,axis=1).tolist())
    pur=np.array(pur)
    return {'avg_ref_purity':float(pur.mean()),'normal_node_ref_purity':float(pur[labels==0].mean()),'anomaly_node_ref_purity':float(pur[labels==1].mean())}
def summarize(xs):
    return {k:{'mean':float(np.mean([x[k] for x in xs])),'std':float(np.std([x[k] for x in xs]))} for k in xs[0]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--pp-k',type=int,default=6); ap.add_argument('--train-rate',type=float,default=0.05); ap.add_argument('--seeds',default='0,1,2,3,4'); ap.add_argument('--ref-k',type=int,default=16); ap.add_argument('--block',type=int,default=512); ap.add_argument('--out',required=True)
    args=ap.parse_args(); seeds=[int(x) for x in args.seeds.split(',')]
    adj,feat,labels=load_mat_dataset(args.data); normal_all=np.where(labels==0)[0]
    ndc=ndc_score(feat,adj,args.pp_k); q=zsig(ndc); c=local_mean(q,adj); support=minmax(q*c); h=np.stack([q,c],1)
    z=l2(mps(feat,adj,args.pp_k)); xa=l2(row_norm(feat.copy()).astype(np.float32))
    res={'data':args.data,'n':int(len(labels)),'anom_rate':float(labels.mean()),'train_rate':args.train_rate,'ref_k':args.ref_k,'seeds':[]}
    for seed in seeds:
        rng=np.random.default_rng(seed); normal=normal_all.copy(); rng.shuffle(normal); train=np.sort(normal[:int(len(labels)*args.train_rate)])
        center=z[train].mean(0,keepdims=True); center=center/(np.linalg.norm(center)+1e-12)
        center_dev=minmax(np.linalg.norm(z-center,axis=1)); density_dev=minmax(1-topk_mean(z,z[train],32)); attr_dev=minmax(1-topk_mean(xa,xa[train],32)); max_dev=np.maximum(center_dev,density_dev)
        ns=normal_percentile(support,train); nc=normal_percentile(center_dev,train); nd=normal_percentile(density_dev,train); na=normal_percentile(attr_dev,train); nm=normal_percentile(max_dev,train)
        soft_or=1-(1-nc)*(1-nd)*(1-na)
        candidates={'baseline_support_qc':support,'normal_cal_attr_dev':na,'normal_cal_max_dev':nm,'normal_cal_soft_or':soft_or,'normal_cal_support':ns,'normal_cal_max_support_attr':np.maximum(ns,na)}
        one={'seed':seed,'scores':{}}
        for name,g in candidates.items():
            one['scores'][name]={m:eval_reference(g,h,labels,m,args.ref_k,args.block) for m in ['multiply','add','rank_add']}
        res['seeds'].append(one)
    agg={}
    for name in res['seeds'][0]['scores']:
        agg[name]={}
        for mode in ['multiply','add','rank_add']:
            agg[name][mode]=summarize([s['scores'][name][mode] for s in res['seeds']])
    res['aggregate']=agg; Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))
if __name__=='__main__': main()
