#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np, scipy.io as sio, scipy.sparse as sp
# Reuse simple functions
def load(path):
 d=sio.loadmat(path); y=d['Label'] if 'Label'in d else d['gnd']; x=d['Attributes'] if 'Attributes'in d else d['X']; a=d['Network'] if 'Network'in d else d['A']
 return (a.toarray() if sp.issparse(a) else np.asarray(a)).astype(np.float32),(x.toarray() if sp.issparse(x) else np.asarray(x)).astype(np.float32),(np.asarray(y).reshape(-1)-np.asarray(y).min()).astype(int)
def row_norm(x): s=x.sum(1,keepdims=True); s[s==0]=1; return x/s
def l2(x): return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12)
def norm_adj(a):
 d=a.sum(1); inv=np.zeros_like(d,dtype=np.float32); m=d>0; inv[m]=1/np.sqrt(d[m]); return inv[:,None]*a*inv[None,:]
def mps(x,a,k):
 na=norm_adj(a); outs=[x]; cur=x.copy()
 for _ in range(k): cur=na@cur; outs.append(cur.copy())
 return np.concatenate(outs,1).astype(np.float32)
def ndc(x,a,k):
 f=row_norm(x.copy()); na=norm_adj(a); n,d=f.shape; hops=np.zeros((n,k+1,d),dtype=np.float32); hops[:,0]=f; cur=f.copy()
 for h in range(1,k+1): cur=na@cur; hops[:,h]=cur
 delta=hops[:,1:]-hops[:,:-1]; out=np.zeros(n,dtype=np.float32)
 for i in range(n):
  nb=np.where(a[i]>0)[0]
  if len(nb)==0: continue
  p=delta[i].reshape(-1); q=delta[nb].mean(0).reshape(-1)
  out[i]=0 if np.std(p)<1e-8 or np.std(q)<1e-8 else np.corrcoef(p,q)[0,1]
 return out
def zsig(x): return 1/(1+np.exp(-(x-x.mean())/(x.std()+1e-8)))
def minmax(x): x=np.asarray(x,dtype=np.float64); return (x-x.min())/(x.max()-x.min()+1e-12)
def local_mean(v,a):
 d=a.sum(1); o=a@v; r=np.zeros_like(v,dtype=np.float64); m=d>0; r[m]=o[m]/d[m]; r[~m]=v[~m]; return r
def topk_mean(x,b,k=32,block=1024):
 vals=[]; k=min(k,b.shape[0])
 for st in range(0,x.shape[0],block):
  sims=x[st:st+block]@b.T; vals.append(np.partition(sims,-k,axis=1)[:,-k:].mean(1))
 return np.concatenate(vals)
def pct(x,idx): b=np.sort(x[idx]); return np.searchsorted(b,x,side='right')/(len(b)+1e-12)
def rank01(x): return np.argsort(np.argsort(x,kind='mergesort'),kind='mergesort').astype(np.float32)/(len(x)-1+1e-12)
def eval_stream(g,h,y,mode,ref_k=16):
 n=len(y); h=l2(h.astype(np.float32)); g=minmax(g).astype(np.float32); gr=rank01(g); pur=[]
 for i in range(n):
  sim=h@h[i]
  if mode=='multiply': score=sim*g
  elif mode=='add': score=minmax(sim)+g
  elif mode=='rank_add': score=rank01(sim)+gr
  else: raise ValueError(mode)
  score[i]=-1e9; idx=np.argpartition(score,-ref_k)[-ref_k:]
  pur.append(float(np.mean(y[idx]==1)))
 pur=np.array(pur); return {'avg_ref_purity':float(pur.mean()),'normal_node_ref_purity':float(pur[y==0].mean()),'anomaly_node_ref_purity':float(pur[y==1].mean())}
def summ(xs): return {k:{'mean':float(np.mean([x[k] for x in xs])),'std':float(np.std([x[k] for x in xs]))} for k in xs[0]}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--out',required=True); ap.add_argument('--pp-k',type=int,default=6); ap.add_argument('--train-rate',type=float,default=.05); ap.add_argument('--seeds',default='0,1,2,3,4'); args=ap.parse_args()
 a,x,y=load(args.data); normals=np.where(y==0)[0]; q=zsig(ndc(x,a,args.pp_k)); c=local_mean(q,a); support=minmax(q*c); h=np.stack([q,c],1); z=l2(mps(x,a,args.pp_k)); xa=l2(row_norm(x.copy()).astype(np.float32))
 res={'data':args.data,'n':int(len(y)),'anom_rate':float(y.mean()),'seeds':[]}
 for seed in [int(s) for s in args.seeds.split(',')]:
  rng=np.random.default_rng(seed); normal=normals.copy(); rng.shuffle(normal); train=np.sort(normal[:int(len(y)*args.train_rate)])
  cen=z[train].mean(0,keepdims=True); cen=cen/(np.linalg.norm(cen)+1e-12); cd=minmax(np.linalg.norm(z-cen,axis=1)); dd=minmax(1-topk_mean(z,z[train],32)); ad=minmax(1-topk_mean(xa,xa[train],32)); md=np.maximum(cd,dd)
  nc,nd,na,nm=pct(cd,train),pct(dd,train),pct(ad,train),pct(md,train); soft=1-(1-nc)*(1-nd)*(1-na)
  cand={'baseline_support_qc':support,'normal_cal_attr_dev':na,'normal_cal_max_dev':nm,'normal_cal_soft_or':soft}
  one={'seed':seed,'scores':{}}
  for name,g in cand.items(): one['scores'][name]={m:eval_stream(g,h,y,m) for m in ['multiply','add','rank_add']}
  res['seeds'].append(one)
 res['aggregate']={}
 for name in res['seeds'][0]['scores']:
  res['aggregate'][name]={m:summ([s['scores'][name][m] for s in res['seeds']]) for m in ['multiply','add','rank_add']}
 Path(args.out).write_text(json.dumps(res,indent=2)); print(json.dumps(res['aggregate'],indent=2))
if __name__=='__main__': main()
