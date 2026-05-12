#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np, scipy.io as sio, scipy.sparse as sp
import torch
# CPU preprocessing functions
def load(path):
 d=sio.loadmat(path); y=d['Label'] if 'Label'in d else d['gnd']; x=d['Attributes'] if 'Attributes'in d else d['X']; a=d['Network'] if 'Network'in d else d['A']
 return (a.toarray() if sp.issparse(a) else np.asarray(a)).astype(np.float32),(x.toarray() if sp.issparse(x) else np.asarray(x)).astype(np.float32),(np.asarray(y).reshape(-1)-np.asarray(y).min()).astype(int)
def row_norm(x): s=x.sum(1,keepdims=True); s[s==0]=1; return x/s
def l2_np(x): return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12)
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
def minmax_np(x): x=np.asarray(x,dtype=np.float64); return ((x-x.min())/(x.max()-x.min()+1e-12)).astype(np.float32)
def local_mean(v,a):
 d=a.sum(1); o=a@v; r=np.zeros_like(v,dtype=np.float64); m=d>0; r[m]=o[m]/d[m]; r[~m]=v[~m]; return r
def topk_mean(x,b,k=32,block=2048):
 vals=[]; k=min(k,b.shape[0])
 for st in range(0,x.shape[0],block):
  sims=x[st:st+block]@b.T; vals.append(np.partition(sims,-k,axis=1)[:,-k:].mean(1))
 return np.concatenate(vals)
def pct(x,idx): b=np.sort(x[idx]); return np.searchsorted(b,x,side='right')/(len(b)+1e-12)
def summ(xs): return {k:{'mean':float(np.mean([x[k] for x in xs])),'std':float(np.std([x[k] for x in xs]))} for k in xs[0]}

def eval_gpu(g_np,h_np,y_np,mode,device,ref_k=16,block=4096):
 n=len(y_np); y=torch.tensor(y_np,device=device); h=torch.tensor(h_np,dtype=torch.float32,device=device); h=torch.nn.functional.normalize(h,dim=1)
 g=torch.tensor(minmax_np(g_np),dtype=torch.float32,device=device); gr=torch.argsort(torch.argsort(g)).float()/(n-1+1e-12)
 pur=[]
 for st in range(0,n,block):
  hb=h[st:st+block]; sim=hb@h.T
  rows=sim.shape[0]
  if mode=='multiply': score=sim*g.unsqueeze(0)
  elif mode=='add':
   mn=sim.min(dim=1,keepdim=True).values; mx=sim.max(dim=1,keepdim=True).values
   score=(sim-mn)/(mx-mn+1e-12)+g.unsqueeze(0)
  elif mode=='rank_add':
   lr=torch.argsort(torch.argsort(sim,dim=1),dim=1).float()/(n-1+1e-12)
   score=lr+gr.unsqueeze(0)
  else: raise ValueError(mode)
  idx_self=torch.arange(rows,device=device); score[idx_self, st+idx_self] = -1e9
  idx=torch.topk(score,ref_k,dim=1).indices
  p=(y[idx]==1).float().mean(dim=1).detach().cpu().numpy(); pur.extend(p.tolist())
  del sim,score
 pur=np.array(pur); return {'avg_ref_purity':float(pur.mean()),'normal_node_ref_purity':float(pur[y_np==0].mean()),'anomaly_node_ref_purity':float(pur[y_np==1].mean())}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--out',required=True); ap.add_argument('--pp-k',type=int,default=6); ap.add_argument('--train-rate',type=float,default=.05); ap.add_argument('--seeds',default='0,1,2,3,4'); ap.add_argument('--block',type=int,default=4096); args=ap.parse_args()
 device='cuda' if torch.cuda.is_available() else 'cpu'; a,x,y=load(args.data); normals=np.where(y==0)[0]
 q=zsig(ndc(x,a,args.pp_k)); c=local_mean(q,a); support=minmax_np(q*c); h=np.stack([q,c],1).astype(np.float32); z=l2_np(mps(x,a,args.pp_k)); xa=l2_np(row_norm(x.copy()).astype(np.float32))
 res={'data':args.data,'device':device,'n':int(len(y)),'anom_rate':float(y.mean()),'seeds':[]}
 for seed in [int(s) for s in args.seeds.split(',')]:
  rng=np.random.default_rng(seed); normal=normals.copy(); rng.shuffle(normal); train=np.sort(normal[:int(len(y)*args.train_rate)])
  cen=z[train].mean(0,keepdims=True); cen=cen/(np.linalg.norm(cen)+1e-12); cd=minmax_np(np.linalg.norm(z-cen,axis=1)); dd=minmax_np(1-topk_mean(z,z[train],32)); ad=minmax_np(1-topk_mean(xa,xa[train],32)); md=np.maximum(cd,dd)
  nc,nd,na,nm=pct(cd,train),pct(dd,train),pct(ad,train),pct(md,train); soft=1-(1-nc)*(1-nd)*(1-na)
  cand={'baseline_support_qc':support,'normal_cal_attr_dev':na,'normal_cal_max_dev':nm,'normal_cal_soft_or':soft}
  one={'seed':seed,'scores':{}}
  for name,g in cand.items(): one['scores'][name]={m:eval_gpu(g,h,y,m,device,16,args.block) for m in ['multiply','add','rank_add']}
  res['seeds'].append(one); Path(args.out).write_text(json.dumps(res,indent=2))
 res['aggregate']={}
 for name in res['seeds'][0]['scores']:
  res['aggregate'][name]={m:summ([s['scores'][name][m] for s in res['seeds']]) for m in ['multiply','add','rank_add']}
 Path(args.out).write_text(json.dumps(res,indent=2)); print(json.dumps(res['aggregate'],indent=2))
if __name__=='__main__': main()
