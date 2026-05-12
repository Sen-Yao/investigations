#!/usr/bin/env python3
"""Stage-1 relation density probe.

Encoder/reference frozen. Learn normal-only density on DualRef relation features.
Anomaly labels are diagnostics only and never enter training loss.
"""
import argparse, json, os, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def safe_auc(y, s):
    try: return float(roc_auc_score(y, s)), float(average_precision_score(y, s))
    except Exception: return 0.0, 0.0


def safe_spearman(a,b):
    try:
        v=spearmanr(np.asarray(a), np.asarray(b)).correlation
        return float(0.0 if np.isnan(v) else v)
    except Exception: return 0.0


def top_ratio(labels, scores, frac):
    n=max(1, int(len(scores)*frac)); idx=np.argsort(-scores)[:n]
    return float(np.mean(labels[idx]))


def build_components(emb, normal_refs, anom_refs, node_idx, rn_idx=None, ra_idx=None):
    node_idx=np.asarray(node_idx,dtype=np.int64)
    rn_base=node_idx if rn_idx is None else np.asarray(rn_idx,dtype=np.int64)
    ra_base=node_idx if ra_idx is None else np.asarray(ra_idx,dtype=np.int64)
    h=emb[node_idx]
    rn=emb[normal_refs[rn_base]].mean(dim=1)
    ra=emb[anom_refs[ra_base]].mean(dim=1)
    u=h-rn; d=ra-rn
    margin=torch.sum(F.normalize(u,p=2,dim=1,eps=1e-12)*F.normalize(d,p=2,dim=1,eps=1e-12), dim=1)
    return h,rn,ra,u,d,margin


def build_relation_features(emb, normal_refs, anom_refs, node_idx, input_mode, rn_idx=None, ra_idx=None):
    h,rn,ra,u,d,margin=build_components(emb, normal_refs, anom_refs, node_idx, rn_idx, ra_idx)
    u_norm = F.normalize(u, p=2, dim=1, eps=1e-12)
    d_norm = F.normalize(d, p=2, dim=1, eps=1e-12)
    if input_mode == 'ud':
        x=torch.cat([u,d], dim=1)
    elif input_mode == 'ud_prod_absdiff':
        x=torch.cat([u,d,u*d,torch.abs(u-d)], dim=1)
    elif input_mode == 'h_rn_ra_ud':
        x=torch.cat([h,rn,ra,u,d,u*d,torch.abs(u-d)], dim=1)
    elif input_mode == 'ud_norm':
        x=torch.cat([u_norm,d_norm], dim=1)
    elif input_mode == 'ud_prod_absdiff_norm':
        x=torch.cat([u_norm,d_norm,u_norm*d_norm,torch.abs(u_norm-d_norm)], dim=1)
    elif input_mode == 'h_rn_ra_ud_norm':
        x=torch.cat([F.normalize(h,p=2,dim=1,eps=1e-12),F.normalize(rn,p=2,dim=1,eps=1e-12),F.normalize(ra,p=2,dim=1,eps=1e-12),u_norm,d_norm,u_norm*d_norm,torch.abs(u_norm-d_norm)], dim=1)
    elif input_mode == 'ud_mixed_norm':
        x=torch.cat([u,d,u_norm,d_norm], dim=1)
    else:
        raise ValueError(input_mode)
    return x, margin


class Standardizer(nn.Module):
    def __init__(self, mean, std):
        super().__init__(); self.register_buffer('mean', mean); self.register_buffer('std', std.clamp_min(1e-6))
    def forward(self,x): return (x-self.mean)/self.std


class DiagGaussian(nn.Module):
    def __init__(self, mean, std):
        super().__init__(); self.register_buffer('mean', mean); self.register_buffer('std', std.clamp_min(1e-6))
    def log_prob(self,x):
        z=(x-self.mean)/self.std
        return -0.5*(z*z + 2*torch.log(self.std) + np.log(2*np.pi)).sum(dim=1)


class Coupling(nn.Module):
    def __init__(self, dim, hidden, mask):
        super().__init__(); self.register_buffer('mask', mask)
        self.net=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,dim*2))
    def forward(self,x):
        xm=x*self.mask
        st=self.net(xm); s,t=st.chunk(2,dim=1)
        s=torch.tanh(s)*2.0
        inv=1.0-self.mask
        y=xm + inv*(x*torch.exp(s)+t)
        logdet=(inv*s).sum(dim=1)
        return y,logdet


class RealNVP(nn.Module):
    def __init__(self, dim, hidden=256, layers=4):
        super().__init__(); mods=[]
        for k in range(layers):
            mask=((torch.arange(dim)+k)%2).float()
            mods.append(Coupling(dim,hidden,mask))
        self.mods=nn.ModuleList(mods)
    def forward(self,x):
        logdet=torch.zeros(x.shape[0],device=x.device)
        z=x
        for m in self.mods:
            z,ld=m(z); logdet=logdet+ld
        return z,logdet
    def log_prob(self,x):
        z,ld=self.forward(x)
        base=-0.5*(z*z+np.log(2*np.pi)).sum(dim=1)
        return base+ld


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project-root', default='/data/linziyao/DualRefGAD')
    ap.add_argument('--dataset', default='elliptic'); ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--train_rate', type=float, default=0.05); ap.add_argument('--val_rate', type=float, default=0.0)
    ap.add_argument('--descriptor_mode', choices=['hop_attr','rwse','hop_attr_rwse'], default='hop_attr')
    ap.add_argument('--pn_estimator', choices=['diag_gaussian','pca_residual'], default='pca_residual')
    ap.add_argument('--gn_mode', choices=['label_gate','normal_density','label_gate_density'], default='label_gate')
    ap.add_argument('--ln_mode', choices=['descriptor_similarity','reconstruction_gain'], default='descriptor_similarity')
    ap.add_argument('--ga_mode', choices=['normal_rejection','residual_norm','normal_soft_or'], default='normal_soft_or')
    ap.add_argument('--la_mode', choices=['residual_cosine','descriptor_similarity'], default='descriptor_similarity')
    ap.add_argument('--ablation_mode', choices=['full','no_ra','shuffled_ra','fixed_labeled_normal'], default='full')
    ap.add_argument('--normal_k', type=int, default=4); ap.add_argument('--anom_k', type=int, default=16)
    ap.add_argument('--pp_k', type=int, default=6); ap.add_argument('--hops', type=int, default=2); ap.add_argument('--rw_steps', type=int, default=8); ap.add_argument('--pca_components', type=int, default=32)
    ap.add_argument('--embedding_dim', type=int, default=256); ap.add_argument('--GT_ffn_dim', type=int, default=256); ap.add_argument('--GT_dropout', type=float, default=0.4); ap.add_argument('--GT_attention_dropout', type=float, default=0.4); ap.add_argument('--GT_num_heads', type=int, default=2); ap.add_argument('--GT_num_layers', type=int, default=3)
    ap.add_argument('--sample_rate', type=float, default=0.15); ap.add_argument('--mean', type=float, default=0.02); ap.add_argument('--var', type=float, default=0.01); ap.add_argument('--outlier_beta', type=float, default=0.3); ap.add_argument('--ring_R_max', type=float, default=1.0); ap.add_argument('--ring_R_min', type=float, default=0.3); ap.add_argument('--lambda_rec_tok', type=float, default=1.0); ap.add_argument('--lambda_rec_emb', type=float, default=0.1)
    ap.add_argument('--encode_batch_size', type=int, default=512)
    ap.add_argument('--input_mode', choices=['ud','ud_prod_absdiff','h_rn_ra_ud','ud_norm','ud_prod_absdiff_norm','h_rn_ra_ud_norm','ud_mixed_norm'], default='ud_prod_absdiff_norm')
    ap.add_argument('--density_mode', choices=['diag_gaussian','realnvp'], default='diag_gaussian')
    ap.add_argument('--flow_layers', type=int, default=4); ap.add_argument('--hidden', type=int, default=256); ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--weight_decay', type=float, default=1e-4); ap.add_argument('--num_epoch', type=int, default=80)
    ap.add_argument('--wandb', type=lambda x: str(x).lower() in ['1','true','yes'], default=False)
    ap.add_argument('--out', default='')
    args=ap.parse_args()

    root=Path(args.project_root); sys.path.insert(0,str(root)); sys.path.insert(0,str(root/'scripts')); os.chdir(str(root))
    from utils import load_mat
    from VecGAD import VecGAD
    from run_training_degradation_diagnosis import to_dense_features, build_descriptor, NormalModel, select_refs, apply_ablation, reference_purity, build_tokens, encode_tokens_batched
    t0=time.time(); set_seed(args.seed)
    device=torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device>=0 else 'cpu')
    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx=load_mat(args.dataset,args.train_rate,args.val_rate,args=args)
    features_np=to_dense_features(args.dataset,features); labels_np=np.asarray(ano_label).reshape(-1).astype(int)
    normal_idx=np.asarray(normal_for_train_idx,dtype=int); idx_test=np.asarray(idx_test,dtype=int)
    assert np.sum(labels_np[normal_idx]) == 0, 'Data leakage: train normal contains anomalies'
    z=build_descriptor(args.descriptor_mode,features_np,adj,args.hops,args.rw_steps)
    nm=NormalModel(args.pn_estimator,z,normal_idx,args.pca_components); residual=nm.residual()
    normal_refs, anom_refs, score_meta=select_refs(z,residual,normal_idx,nm,features_np,adj,args,labels_np)
    normal_refs, anom_refs=apply_ablation(normal_refs,anom_refs,normal_idx,labels_np,args)
    pur=reference_purity(normal_refs,anom_refs,labels_np)
    token_tensor=build_tokens(features_np,normal_refs,anom_refs)
    model=VecGAD(features_np.shape[1],args.embedding_dim,'prelu',args).to(device); model.eval()
    with torch.no_grad(): emb=encode_tokens_batched(model,token_tensor,device,args.encode_batch_size).detach()
    normal_refs_t=torch.as_tensor(normal_refs,dtype=torch.long,device=device); anom_refs_t=torch.as_tensor(anom_refs,dtype=torch.long,device=device)
    all_nodes=np.arange(len(labels_np),dtype=np.int64)
    with torch.no_grad():
        x_all, margin0=build_relation_features(emb,normal_refs_t,anom_refs_t,all_nodes,args.input_mode)
    x_train=x_all[torch.as_tensor(normal_idx,dtype=torch.long,device=device)]
    mean=x_train.mean(dim=0); std=x_train.std(dim=0).clamp_min(1e-6)
    scaler=Standardizer(mean,std).to(device)
    xs_train=scaler(x_train).detach(); xs_all=scaler(x_all).detach()
    margin0_np=margin0.detach().cpu().numpy()
    margin_auc, margin_ap=safe_auc(labels_np[idx_test], margin0_np[idx_test])

    run=None
    if args.wandb:
        import wandb
        run=wandb.init(project='DualRefGAD', entity='HCCS', config=vars(args), name=f'stage1_relation_density_{args.density_mode}_{args.input_mode}_h{args.hidden}_l{args.flow_layers}_s{args.seed}')
        wandb.summary.update(pur)
    if args.density_mode=='diag_gaussian':
        density=DiagGaussian(xs_train.mean(dim=0), xs_train.std(dim=0)).to(device)
        with torch.no_grad(): score=(-density.log_prob(xs_all)).detach().cpu().numpy()
        best_epoch=0; best_nll=float((-density.log_prob(xs_train)).mean().item())
    else:
        density=RealNVP(xs_train.shape[1], args.hidden, args.flow_layers).to(device)
        opt=torch.optim.AdamW(density.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_score=None; best_nll=1e9; best_epoch=-1
        for epoch in range(args.num_epoch+1):
            if epoch>0:
                density.train(); opt.zero_grad(); nll=(-density.log_prob(xs_train)).mean(); nll.backward(); opt.step()
            density.eval()
            with torch.no_grad():
                train_nll=float((-density.log_prob(xs_train)).mean().item())
                cur_score=(-density.log_prob(xs_all)).detach().cpu().numpy()
            if train_nll < best_nll:
                best_nll=train_nll; best_score=cur_score; best_epoch=epoch
            if run and (epoch%10==0 or epoch==args.num_epoch):
                auc,apv=safe_auc(labels_np[idx_test], cur_score[idx_test])
                import wandb; wandb.log({'epoch':epoch,'train_nll':train_nll,'test_auc':auc,'test_ap':apv,'top1_ratio':top_ratio(labels_np[idx_test],cur_score[idx_test],0.01),'top5_ratio':top_ratio(labels_np[idx_test],cur_score[idx_test],0.05),'spearman_score_margin':safe_spearman(cur_score[idx_test], margin0_np[idx_test])}, step=epoch)
        score=best_score
    auc, apv=safe_auc(labels_np[idx_test], score[idx_test])
    result={'status':'stage1_relation_density_probe','dataset':args.dataset,'seed':args.seed,'config':vars(args),'purity':pur,'best':{'epoch':best_epoch,'train_nll':best_nll,'test_auc':auc,'test_ap':apv,'top1_ratio':top_ratio(labels_np[idx_test],score[idx_test],0.01),'top5_ratio':top_ratio(labels_np[idx_test],score[idx_test],0.05),'spearman_score_margin':safe_spearman(score[idx_test], margin0_np[idx_test])},'margin_baseline':{'auc':margin_auc,'ap':margin_ap,'top1_ratio':top_ratio(labels_np[idx_test],margin0_np[idx_test],0.01),'top5_ratio':top_ratio(labels_np[idx_test],margin0_np[idx_test],0.05)},'time_sec':time.time()-t0}
    out=Path(args.out) if args.out else root/'outputs/stage1_probe/stage1_relation_density_probe.json'
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'out':str(out),'margin_baseline':result['margin_baseline'],'best':result['best'],'purity':pur,'time_sec':result['time_sec']},indent=2,ensure_ascii=False), flush=True)
    if run:
        import wandb
        wandb.summary.update({'best_epoch':best_epoch,'best_train_nll':best_nll,'best_test_auc':auc,'best_test_ap':apv,'best_top1_ratio':result['best']['top1_ratio'],'best_top5_ratio':result['best']['top5_ratio'],'best_spearman_margin':result['best']['spearman_score_margin'],'margin_auc':margin_auc,'margin_ap':margin_ap,'margin_top1_ratio':result['margin_baseline']['top1_ratio'],'margin_top5_ratio':result['margin_baseline']['top5_ratio']})
        run.finish()

if __name__=='__main__': main()
