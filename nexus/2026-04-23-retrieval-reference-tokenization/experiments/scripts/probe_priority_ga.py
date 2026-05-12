#!/usr/bin/env python3
"""
Probe priority G_a candidates inspired by unsupervised GAD:
1) Normal Density Failure (KDE / kNN similarity to normal bank)
2) Structure-Attribute Disagreement
3) LOF-style Relative Normal Density
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

def normalize_rows_l2(x):
    return x/(np.linalg.norm(x, axis=1, keepdims=True)+1e-12)

def norm_adj(adj):
    deg=adj.sum(1)
    inv=np.zeros_like(deg, dtype=np.float32)
    mask=deg>0
    inv[mask]=1/np.sqrt(deg[mask])
    return inv[:,None]*adj*inv[None,:]

def mps(features, adj, k):
    a=norm_adj(adj); outs=[features]; cur=features.copy()
    for _ in range(k):
        cur=a@cur; outs.append(cur.copy())
    return np.concatenate(outs,1).astype(np.float32)

def minmax(x):
    x=np.asarray(x,dtype=np.float64)
    return (x-x.min())/(x.max()-x.min()+1e-12)

def eval_score(s, labels, topks=(16,32,64,128)):
    out={
        'auc': float(roc_auc_score(labels,s)),
        'ap': float(average_precision_score(labels,s)),
        'normal_mean': float(s[labels==0].mean()),
        'anomaly_mean': float(s[labels==1].mean()),
        'ks': float(ks_2samp(s[labels==0], s[labels==1]).statistic),
    }
    order=np.argsort(s)[::-1]
    for k in topks:
        out[f'top{k}_anom_ratio']=float(np.mean(labels[order[:k]]==1))
    return out

def topk_mean_sim_to_bank(x, bank, k=32, block=2048):
    vals=[]
    k=min(k, bank.shape[0])
    for st in range(0, x.shape[0], block):
        sims=x[st:st+block]@bank.T
        part=np.partition(sims, -k, axis=1)[:,-k:]
        vals.append(part.mean(1))
    return np.concatenate(vals)

def kth_density(x, bank, k=32, block=2048):
    vals=[]; k=min(k, bank.shape[0])
    for st in range(0, x.shape[0], block):
        sims=x[st:st+block]@bank.T
        kth=np.partition(sims, -k, axis=1)[:,-k]
        vals.append(kth)
    return np.concatenate(vals)

def local_neighbor_mean(values, adj):
    deg=adj.sum(1)
    out=adj@values
    mask=deg>0
    res=np.zeros_like(values, dtype=np.float64)
    res[mask]=out[mask]/deg[mask]
    res[~mask]=values[~mask]
    return res

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

def zsig(x):
    return 1/(1+np.exp(-(x-x.mean())/(x.std()+1e-8)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--train-rate', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pp-k', type=int, default=6)
    ap.add_argument('--bank-k', type=int, default=32)
    ap.add_argument('--block', type=int, default=2048)
    ap.add_argument('--out', required=True)
    args=ap.parse_args()
    np.random.seed(args.seed)
    adj, feat, labels=load_mat_dataset(args.data)
    normal=np.where(labels==0)[0]
    np.random.shuffle(normal)
    n_train=max(1,min(len(normal), int(len(labels)*args.train_rate)))
    train_normal=np.sort(normal[:n_train])

    # Existing anomaly affinity baseline
    ndc=ndc_score(feat, adj, args.pp_k)
    q=zsig(ndc)
    c=local_neighbor_mean(q, adj)
    qc=minmax(q*c)

    # Attribute normal density
    x_attr=normalize_rows_l2(row_norm(feat.copy()).astype(np.float32))
    attr_bank=x_attr[train_normal]
    attr_density=topk_mean_sim_to_bank(x_attr, attr_bank, args.bank_k, args.block)
    attr_failure=minmax(1-attr_density)

    # MPS / propagation normal density
    z=normalize_rows_l2(mps(feat, adj, args.pp_k))
    z_bank=z[train_normal]
    mps_density=topk_mean_sim_to_bank(z, z_bank, args.bank_k, args.block)
    mps_failure=minmax(1-mps_density)

    # Structure signature: normalized adjacency row projected to train-normal anchors by connectivity pattern.
    # Full adjacency row is too sparse/noisy; use propagated one-hop feature signature as structural context.
    neigh_attr=normalize_rows_l2((norm_adj(adj) @ row_norm(feat.copy())).astype(np.float32))
    neigh_bank=neigh_attr[train_normal]
    struct_density=topk_mean_sim_to_bank(neigh_attr, neigh_bank, args.bank_k, args.block)
    struct_failure=minmax(1-struct_density)

    # Structure-attribute disagreement: attribute normality and structural normality disagree.
    attr_normal=minmax(attr_density)
    struct_normal=minmax(struct_density)
    struct_attr_disagree=minmax(np.abs(attr_normal-struct_normal))
    joint_disagree_failure=minmax(struct_attr_disagree + 0.5*(attr_failure+struct_failure))

    # LOF-style relative normal density: node is anomalous if its normal density is low relative to graph neighbors.
    dens=minmax(mps_density)
    neigh_dens=local_neighbor_mean(dens, adj)
    lof_relative=minmax((neigh_dens+1e-8)/(dens+1e-8))
    lof_failure=minmax(lof_relative * mps_failure)

    # Hybrids with prior qc
    candidates={
        'baseline_anom_affinity_qc': qc,
        'attr_normal_density_failure': attr_failure,
        'mps_normal_density_failure': mps_failure,
        'struct_context_density_failure': struct_failure,
        'struct_attr_disagreement': struct_attr_disagree,
        'joint_disagreement_failure': joint_disagree_failure,
        'lof_relative_density_failure': lof_failure,
        'mps_failure_plus_qc': minmax(mps_failure + qc),
        'mps_failure_times_qc': minmax(mps_failure * qc),
        'joint_disagreement_plus_qc': minmax(joint_disagree_failure + qc),
        'lof_plus_qc': minmax(lof_failure + qc),
    }
    res={'data':args.data,'n':int(len(labels)),'anom_rate':float(labels.mean()),'train_normal':int(len(train_normal)),'scores':{}}
    for name,score in candidates.items():
        res['scores'][name]=eval_score(score, labels)
    out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

if __name__=='__main__':
    main()
