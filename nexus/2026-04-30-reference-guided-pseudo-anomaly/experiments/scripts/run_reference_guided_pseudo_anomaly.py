#!/usr/bin/env python3
"""Clean GT runner for normal-calibrated dual-reference tokenization.

Non-intrusive entry point. Does not modify run.py / VecGAD.py.
Uses 5% labeled-normal nodes to define normal support/rejection descriptors.
"""
import argparse, json, time, random, sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

ROOT = Path.home() / "VoxG"
sys.path.insert(0, str(ROOT))
from utils import load_mat, preprocess_features, normalize_adj
from VecGAD import VecGAD


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_dense_features(dataset, features):
    if dataset in ["Amazon", "tf_finace", "reddit", "elliptic"]:
        features, _ = preprocess_features(features)
        return np.asarray(features, dtype=np.float32)
    return np.asarray(features.todense(), dtype=np.float32)


def l2_rows(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
def rank_percentile(x):
    x=np.asarray(x,dtype=np.float64); order=np.argsort(x); r=np.empty(len(x),dtype=np.float64); r[order]=np.arange(len(x)); return r/max(1,len(x)-1)

def build_hop_attr(features, adj, hops=2):
    adj_norm=normalize_adj(adj); x=features.astype(np.float32); outs=[x]; cur=x
    for _ in range(hops): cur=np.asarray(adj_norm.dot(cur),dtype=np.float32); outs.append(cur)
    return np.concatenate(outs,axis=1).astype(np.float32)

def rwse(adj, steps=8):
    csr=adj.tocsr().astype(np.float64); deg=np.asarray(csr.sum(axis=1)).reshape(-1)
    inv=np.divide(1.0,deg,out=np.zeros_like(deg),where=deg>0); P=sp.diags(inv).dot(csr).tocsr(); cur=P.copy(); feats=[]
    for k in range(1,steps+1):
        feats.append(cur.diagonal().astype(np.float64))
        if k<steps: cur=cur.dot(P).tocsr()
    return np.stack(feats,axis=1).astype(np.float32)

def build_descriptor(mode, features, adj, hops=2, rw_steps=8):
    if mode=="hop_attr": return build_hop_attr(features,adj,hops)
    if mode=="rwse": return rwse(adj,rw_steps)
    if mode=="hop_attr_rwse": return np.concatenate([build_hop_attr(features,adj,hops), rwse(adj,rw_steps)],axis=1).astype(np.float32)
    raise ValueError(mode)

class NormalModel:
    def __init__(self, estimator, z, normal_idx, pca_components=32):
        self.estimator=estimator; self.scaler=StandardScaler(with_mean=True,with_std=True)
        self.zs=self.scaler.fit_transform(z[normal_idx]); self.z_all=self.scaler.transform(z)
        self.mu=self.zs.mean(axis=0,keepdims=True); self.std=self.zs.std(axis=0,keepdims=True)+1e-6; self.pca=None
        if estimator=="pca_residual":
            ncomp=int(min(pca_components,self.zs.shape[0]-1,self.zs.shape[1]))
            if ncomp>0:
                self.pca=PCA(n_components=ncomp,svd_solver="randomized",random_state=0); self.pca.fit(self.zs)
        elif estimator!="diag_gaussian": raise ValueError(estimator)
    def rejection(self):
        if self.estimator=="diag_gaussian" or self.pca is None: return np.mean(((self.z_all-self.mu)/self.std)**2,axis=1)
        rec=self.pca.inverse_transform(self.pca.transform(self.z_all)); return np.mean((self.z_all-rec)**2,axis=1)
    def residual(self):
        if self.estimator=="diag_gaussian" or self.pca is None: return ((self.z_all-self.mu)/self.std).astype(np.float32)
        rec=self.pca.inverse_transform(self.pca.transform(self.z_all)); return (self.z_all-rec).astype(np.float32)
    def density_score(self): return -self.rejection()

def normal_soft_or_score(features, adj, normal_idx):
    hop=build_hop_attr(features,adj,hops=2); hm=NormalModel("diag_gaussian",hop,normal_idx); h=rank_percentile(hm.rejection())
    st=rwse(adj,steps=8); sm=NormalModel("diag_gaussian",st,normal_idx); s=rank_percentile(sm.rejection())
    return (1.0-(1.0-h)*(1.0-s)).astype(np.float32)

def cosine_rows_to_matrix(a,b,block=1024):
    an=l2_rows(a.astype(np.float32)); bn=l2_rows(b.astype(np.float32)); outs=[]
    for st in range(0,an.shape[0],block): outs.append(an[st:st+block]@bn.T)
    return np.vstack(outs)

def select_refs(z,residual,normal_idx,nm,features,adj,args,labels=None):
    n=z.shape[0]; rejection=nm.rejection(); density=nm.density_score(); residual_norm=np.linalg.norm(residual,axis=1)
    if args.gn_mode=="label_gate": normal_pool=np.asarray(normal_idx); gn=np.zeros(n,dtype=np.float32); gn[normal_pool]=1.0
    elif args.gn_mode=="normal_density": normal_pool=np.arange(n); gn=rank_percentile(density).astype(np.float32)
    elif args.gn_mode=="label_gate_density": normal_pool=np.asarray(normal_idx); gn=rank_percentile(density).astype(np.float32); mask=np.ones(n,bool); mask[normal_pool]=False; gn[mask]=-1e9
    else: raise ValueError(args.gn_mode)
    if args.ga_mode=="normal_rejection": ga=rank_percentile(rejection).astype(np.float32)
    elif args.ga_mode=="residual_norm": ga=rank_percentile(residual_norm).astype(np.float32)
    elif args.ga_mode=="normal_soft_or": ga=normal_soft_or_score(features,adj,normal_idx).astype(np.float32)
    else: raise ValueError(args.ga_mode)
    sim_n=cosine_rows_to_matrix(z,z[normal_pool])
    ln_mat=sim_n if args.ln_mode=="descriptor_similarity" else sim_n**2
    n_scores=ln_mat+gn[normal_pool][None,:]
    normal_refs=normal_pool[np.argsort(-n_scores,axis=1)[:,:args.normal_k]]
    if args.la_mode=="residual_cosine": l_a=cosine_rows_to_matrix(residual,residual)
    elif args.la_mode=="descriptor_similarity": l_a=cosine_rows_to_matrix(z,z)
    else: raise ValueError(args.la_mode)
    a_scores=l_a+ga[None,:]; np.fill_diagonal(a_scores,-1e9)
    anom_refs=np.argsort(-a_scores,axis=1)[:,:args.anom_k].astype(np.int64)
    return normal_refs, anom_refs, {"ga":ga,"rejection":rejection,"residual_norm":residual_norm}

def apply_ablation(normal_refs, anom_refs, normal_idx, labels, args):
    rng=np.random.default_rng(args.seed)
    n=normal_refs.shape[0]
    if args.ablation_mode=="full": return normal_refs, anom_refs
    if args.ablation_mode=="no_ra":
        # replace anomaly refs by repeated normal refs to keep token length stable
        rep=np.resize(normal_refs, (n,args.anom_k))
        return normal_refs, rep.astype(np.int64)
    if args.ablation_mode=="shuffled_ra":
        flat=anom_refs.copy(); rng.shuffle(flat, axis=0); return normal_refs, flat
    if args.ablation_mode=="fixed_labeled_normal":
        pool=np.asarray(normal_idx); fixed=np.resize(pool, args.normal_k); normal_fixed=np.tile(fixed[None,:],(n,1))
        return normal_fixed.astype(np.int64), anom_refs
    raise ValueError(args.ablation_mode)

def build_tokens(features, normal_refs, anom_refs):
    toks=[]
    for i in range(features.shape[0]): toks.append(np.concatenate([features[i:i+1], features[normal_refs[i]], features[anom_refs[i]]],axis=0))
    return torch.from_numpy(np.stack(toks).astype(np.float32))

def reference_purity(normal_refs, anom_refs, labels):
    return {"normal_ref_normal_ratio":float(np.mean(labels[normal_refs]==0)),"anom_ref_anom_ratio":float(np.mean(labels[anom_refs]==1)),"anom_ref_anom_ratio_on_anom_nodes":float(np.mean(labels[anom_refs[labels==1]]==1)) if np.any(labels==1) else 0.0}

def encode_tokens_batched(model, token_tensor_cpu, device, batch_size:int):
    n=token_tensor_cpu.shape[0]
    if batch_size<=0 or batch_size>=n: return model.TransformerEncoder(token_tensor_cpu.to(device)).squeeze(0)
    chunks=[]
    for st in range(0,n,batch_size): chunks.append(model.TransformerEncoder(token_tensor_cpu[st:st+batch_size].to(device,non_blocking=True)).squeeze(0))
    return torch.cat(chunks,dim=0)

def eval_logits(logits,labels,idx): return float(roc_auc_score(labels[idx],logits[idx])), float(average_precision_score(labels[idx],logits[idx]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',required=True); ap.add_argument('--device',type=int,default=0); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--train_rate',type=float,default=0.05); ap.add_argument('--num_epoch',type=int,default=200)
    ap.add_argument('--descriptor_mode',choices=['hop_attr','rwse','hop_attr_rwse'],default='hop_attr_rwse'); ap.add_argument('--pn_estimator',choices=['diag_gaussian','pca_residual'],default='diag_gaussian')
    ap.add_argument('--gn_mode',choices=['label_gate','normal_density','label_gate_density'],default='label_gate_density'); ap.add_argument('--ln_mode',choices=['descriptor_similarity','reconstruction_gain'],default='descriptor_similarity')
    ap.add_argument('--ga_mode',choices=['normal_rejection','residual_norm','normal_soft_or'],default='normal_rejection'); ap.add_argument('--la_mode',choices=['residual_cosine','descriptor_similarity'],default='residual_cosine')
    ap.add_argument('--reference_mode',default='dual_reference'); ap.add_argument('--ablation_mode',choices=['full','no_ra','shuffled_ra','fixed_labeled_normal'],default='full')
    ap.add_argument('--normal_k',type=int,default=4); ap.add_argument('--anom_k',type=int,default=16); ap.add_argument('--pp_k',type=int,default=6); ap.add_argument('--hops',type=int,default=2); ap.add_argument('--rw_steps',type=int,default=8); ap.add_argument('--pca_components',type=int,default=32)
    ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--weight_decay',type=float,default=0.0); ap.add_argument('--pseudo_beta',type=float,default=0.3); ap.add_argument('--pseudo_noise',type=float,default=0.01)
    ap.add_argument('--embedding_dim',type=int,default=256); ap.add_argument('--GT_ffn_dim',type=int,default=256); ap.add_argument('--GT_dropout',type=float,default=0.4); ap.add_argument('--GT_attention_dropout',type=float,default=0.4); ap.add_argument('--GT_num_heads',type=int,default=2); ap.add_argument('--GT_num_layers',type=int,default=3)
    ap.add_argument('--sample_rate',type=float,default=0.15); ap.add_argument('--mean',type=float,default=0.02); ap.add_argument('--var',type=float,default=0.01); ap.add_argument('--outlier_beta',type=float,default=0.3); ap.add_argument('--ring_R_max',type=float,default=1.0); ap.add_argument('--ring_R_min',type=float,default=0.3); ap.add_argument('--lambda_rec_tok',type=float,default=1.0); ap.add_argument('--lambda_rec_emb',type=float,default=0.1)
    ap.add_argument('--encode_batch_size',type=int,default=2048); ap.add_argument('--wandb',type=lambda x: str(x).lower() in ['1','true','yes'],default=False); ap.add_argument('--dry_run',action='store_true'); ap.add_argument('--out',default=''); ap.add_argument('--objective_mode',choices=['self_residual','global_ref_guided','target_ref_guided'],default='target_ref_guided')
    args=ap.parse_args(); set_seed(args.seed); device=torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device>=0 else 'cpu')
    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx=load_mat(args.dataset,args.train_rate,0.1,args=args)
    features_np=to_dense_features(args.dataset,features); labels_np=np.asarray(ano_label).reshape(-1).astype(int); normal_idx=np.asarray(normal_for_train_idx,dtype=int)
    assert np.sum(labels_np[normal_idx])==0, 'Data leakage: normal_for_train_idx contains anomalies'
    z=build_descriptor(args.descriptor_mode,features_np,adj,args.hops,args.rw_steps); nm=NormalModel(args.pn_estimator,z,normal_idx,args.pca_components); residual=nm.residual()
    normal_refs, anom_refs, score_meta=select_refs(z,residual,normal_idx,nm,features_np,adj,args,labels_np); normal_refs, anom_refs=apply_ablation(normal_refs,anom_refs,normal_idx,labels_np,args)
    pur=reference_purity(normal_refs,anom_refs,labels_np); token_tensor=build_tokens(features_np,normal_refs,anom_refs)
    if args.dry_run:
        print(json.dumps({'dataset':args.dataset,'token_shape':list(token_tensor.shape),'purity':pur,'config':vars(args)},indent=2,ensure_ascii=False)); return
    wandb_run=None
    if args.wandb:
        import wandb
        wandb_run=wandb.init(project='VoxG', entity='HCCS', config=vars(args), name=f"rgpa_{args.dataset}_{args.objective_mode}_{args.ga_mode}_{args.ablation_mode}")
        wandb.summary.update(pur)
    model=VecGAD(features_np.shape[1],args.embedding_dim,'prelu',args).to(device); opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay); bce=nn.BCEWithLogitsLoss(); normal_t=torch.tensor(normal_idx,dtype=torch.long,device=device)
    best={'val_auc':-1,'val_ap':-1,'test_auc':-1,'test_ap':-1,'epoch':-1}; start=time.time()
    for epoch in range(args.num_epoch+1):
        model.train(); opt.zero_grad(); emb=encode_tokens_batched(model,token_tensor,device,args.encode_batch_size); normal_emb=emb[normal_t]
        # Pseudo anomaly objective variants
        if args.objective_mode == 'self_residual':
            center=normal_emb.mean(dim=0,keepdim=True)
            direction=F.normalize(normal_emb-center,dim=1)
            outlier_emb=normal_emb+args.pseudo_beta*direction+torch.randn_like(normal_emb)*args.pseudo_noise
        elif args.objective_mode == 'global_ref_guided':
            anom_ref_t=torch.tensor(list(set(anom_refs.flatten().tolist())),dtype=torch.long,device=device)
            normal_ref_t=torch.tensor(list(set(normal_refs.flatten().tolist())),dtype=torch.long,device=device)
            pool_ra=emb[anom_ref_t].mean(dim=0)
            pool_rn=emb[normal_ref_t].mean(dim=0)
            direction=F.normalize(pool_ra-pool_rn,dim=0).unsqueeze(0).expand_as(normal_emb)
            outlier_emb=normal_emb+args.pseudo_beta*direction+torch.randn_like(normal_emb)*args.pseudo_noise
        elif args.objective_mode == 'target_ref_guided':
            # For each normal training target i, use its own R_a(i)-R_n(i) direction.
            nr_t=torch.tensor(normal_refs[normal_idx],dtype=torch.long,device=device)
            ar_t=torch.tensor(anom_refs[normal_idx],dtype=torch.long,device=device)
            pool_rn=emb[nr_t].mean(dim=1)
            pool_ra=emb[ar_t].mean(dim=1)
            direction=F.normalize(pool_ra-pool_rn,dim=1)
            outlier_emb=normal_emb+args.pseudo_beta*direction+torch.randn_like(normal_emb)*args.pseudo_noise
        else:
            raise ValueError(args.objective_mode)
        emb_c=torch.cat([normal_emb,outlier_emb],dim=0); logits_train=model.fc3(model.act(model.fc2(model.act(model.fc1(emb_c))))).squeeze(-1); y=torch.cat([torch.zeros(len(normal_emb),device=device),torch.ones(len(outlier_emb),device=device)])
        loss=bce(logits_train,y); loss.backward(); opt.step()
        if epoch%10==0 or epoch==args.num_epoch:
            model.eval()
            with torch.no_grad(): emb_eval=encode_tokens_batched(model,token_tensor,device,args.encode_batch_size); logits=model.fc3(model.act(model.fc2(model.act(model.fc1(emb_eval))))).squeeze(-1).detach().cpu().numpy()
            val_auc,val_ap=eval_logits(logits,labels_np,idx_val); test_auc,test_ap=eval_logits(logits,labels_np,idx_test)
            if val_auc+val_ap>best['val_auc']+best['val_ap']: best.update({'val_auc':val_auc,'val_ap':val_ap,'test_auc':test_auc,'test_ap':test_ap,'epoch':epoch})
            row={'epoch':epoch,'loss':float(loss.item()),'val_auc':val_auc,'val_ap':val_ap,'test_auc':test_auc,'test_ap':test_ap,'AUC':test_auc,'AP':test_ap,'best_val_auc':best['val_auc'],'best_val_ap':best['val_ap'],'best_test_auc':best['test_auc'],'best_test_ap':best['test_ap']}
            print(json.dumps(row,ensure_ascii=False),flush=True)
            if wandb_run: wandb.log(row,step=epoch)
    result={'dataset':args.dataset,'seed':args.seed,'objective_mode':args.objective_mode,'config':vars(args),'best':best,'purity':pur,'time_sec':time.time()-start}
    print('FINAL',json.dumps(result,indent=2,ensure_ascii=False))
    if wandb_run:
        wandb.summary['best_val_auc']=best['val_auc']; wandb.summary['best_val_ap']=best['val_ap']; wandb.summary['test_auc']=best['test_auc']; wandb.summary['test_ap']=best['test_ap']; wandb.summary['AUC']=best['test_auc']; wandb.summary['AP']=best['test_ap']; wandb.finish()
    if args.out:
        Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
