#!/usr/bin/env python3
"""Stage-1F margin-regression utility probe.

Encoder/reference frozen. MLP consumes normalized relation inputs and is trained to regress the closed-form DualRef margin using normal-only nodes. In addition to fidelity (MSE/MAE/Spearman), this probe logs anomaly utility (AUC/AP/top-k) of the learned head score on the held-out test set. Anomaly labels are diagnostics only and never enter the loss.
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
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def safe_spearman(a,b):
    try:
        v=spearmanr(np.asarray(a), np.asarray(b)).correlation
        return float(0.0 if np.isnan(v) else v)
    except Exception:
        return 0.0


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


class RelationMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, dropout=0.2):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, 1),
        )
    def forward(self,x): return self.net(x).squeeze(-1)


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
    ap.add_argument('--input_mode', choices=['ud','ud_prod_absdiff','h_rn_ra_ud','ud_norm','ud_prod_absdiff_norm','h_rn_ra_ud_norm','ud_mixed_norm'], default='ud_mixed_norm')
    ap.add_argument('--neg_mode', choices=['none'], default='none')
    ap.add_argument('--hidden', type=int, default=256); ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--num_epoch', type=int, default=50); ap.add_argument('--gamma', type=float, default=0.1)
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
        _, margin0=build_relation_features(emb,normal_refs_t,anom_refs_t,all_nodes,args.input_mode)
        margin0_np=margin0.detach().cpu().numpy()
    margin_auc, margin_ap=safe_auc(labels_np[idx_test], margin0_np[idx_test])
    in_dim={'ud':2,'ud_prod_absdiff':4,'h_rn_ra_ud':7,'ud_norm':2,'ud_prod_absdiff_norm':4,'h_rn_ra_ud_norm':7,'ud_mixed_norm':4}[args.input_mode]*args.embedding_dim
    head=RelationMLP(in_dim,args.hidden,args.dropout).to(device)
    opt=torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng=np.random.default_rng(args.seed); normal_np=normal_idx.copy(); n=len(normal_np)
    train_nodes=normal_np.copy()  # normal-only target regression; no anomaly labels in loss
    run=None
    if args.wandb:
        import wandb
        run=wandb.init(project='DualRefGAD', entity='HCCS', config=vars(args), name=f'stage1_margin_regression_{args.input_mode}_h{args.hidden}_s{args.seed}')
        wandb.summary.update(pur)
    best={'mse':1e9,'mae':1e9,'epoch':-1,'spearman':-1,'top1':0,'top5':0}
    last={}
    for epoch in range(args.num_epoch+1):
        if epoch>0:
            head.train(); opt.zero_grad()
            x_train, m_train=build_relation_features(emb,normal_refs_t,anom_refs_t,train_nodes,args.input_mode)
            pred=head(x_train)
            loss=F.mse_loss(pred, m_train.detach())
            loss.backward(); opt.step()
        else:
            loss=torch.tensor(0.0)
        head.eval()
        with torch.no_grad():
            x_all,_=build_relation_features(emb,normal_refs_t,anom_refs_t,all_nodes,args.input_mode)
            s_all=head(x_all).detach().cpu().numpy()
        mse=float(F.mse_loss(torch.as_tensor(s_all[idx_test],device=device), torch.as_tensor(margin0_np[idx_test],device=device)).item())
        mae=float(torch.mean(torch.abs(torch.as_tensor(s_all[idx_test],device=device)-torch.as_tensor(margin0_np[idx_test],device=device))).item())
        row={'epoch':epoch,'loss':float(loss.detach().cpu().item()),'test_mse':mse,'test_mae':mae,'spearman_score_margin':safe_spearman(s_all[idx_test],margin0_np[idx_test]),'top1_ratio':top_ratio(labels_np[idx_test],s_all[idx_test],0.01),'top5_ratio':top_ratio(labels_np[idx_test],s_all[idx_test],0.05)}
        row['test_auc'], row['test_ap'] = safe_auc(labels_np[idx_test], s_all[idx_test])
        last=row
        if row['test_mse'] < best.get('mse', 1e9):
            best={'mse':mse,'mae':mae,'epoch':epoch,'spearman':row['spearman_score_margin'],'auc':row['test_auc'],'ap':row['test_ap'],'top1':row['top1_ratio'],'top5':row['top5_ratio']}
        if run and (epoch%10==0 or epoch==args.num_epoch):
            import wandb; wandb.log(row, step=epoch)
    result={'status':'stage1_margin_regression_mse_probe','dataset':args.dataset,'seed':args.seed,'config':vars(args),'purity':pur,'margin_baseline':{'mse':float(F.mse_loss(torch.as_tensor(margin0_np[idx_test],device=device), torch.as_tensor(margin0_np[idx_test],device=device)).item()),'mae':0.0,'spearman':1.0,'auc':safe_auc(labels_np[idx_test],margin0_np[idx_test])[0],'ap':safe_auc(labels_np[idx_test],margin0_np[idx_test])[1],'top1_ratio':top_ratio(labels_np[idx_test],margin0_np[idx_test],0.01),'top5_ratio':top_ratio(labels_np[idx_test],margin0_np[idx_test],0.05)},'best':best,'last':last,'time_sec':time.time()-t0}
    out=Path(args.out) if args.out else root/'outputs/stage1_probe/stage1_relation_mlp_margin_regression.json'
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'out':str(out),'margin_baseline':result['margin_baseline'],'best':best,'last':last,'purity':pur,'time_sec':result['time_sec']},indent=2,ensure_ascii=False), flush=True)
    if run:
        import wandb
        wandb.summary.update({'best_test_mse':best['mse'],'best_test_mae':best['mae'],'best_epoch':best['epoch'],'best_spearman':best['spearman'],'best_test_auc':best['auc'],'best_test_ap':best['ap'],'best_top1_ratio':best['top1'],'best_top5_ratio':best['top5'],'margin_auc':result['margin_baseline']['auc'],'margin_ap':result['margin_baseline']['ap'],'margin_top1_ratio':result['margin_baseline']['top1_ratio'],'margin_top5_ratio':result['margin_baseline']['top5_ratio']})
        run.finish()

if __name__=='__main__': main()
