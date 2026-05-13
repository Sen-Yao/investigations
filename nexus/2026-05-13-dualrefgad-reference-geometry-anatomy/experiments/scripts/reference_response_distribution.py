#!/usr/bin/env python3
"""Multi-reference response distribution diagnostic for DualRefGAD.

No-training Phase-2 anatomy. It checks whether reference response distributions
contain signal not captured by scalar mean-pooled margin.
"""
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def safe_auc(y, s):
    try: return float(roc_auc_score(y, s)), float(average_precision_score(y, s))
    except Exception: return 0.0, 0.0


def safe_spearman(a, b):
    try:
        v = spearmanr(np.asarray(a), np.asarray(b)).correlation
        return float(0.0 if np.isnan(v) else v)
    except Exception: return 0.0


def safe_pearson(a, b):
    try:
        v = pearsonr(np.asarray(a), np.asarray(b))[0]
        return float(0.0 if np.isnan(v) else v)
    except Exception: return 0.0


def top_indices(scores, frac):
    return np.argsort(-np.asarray(scores))[:max(1, int(len(scores) * frac))]


def top_ratio(labels, scores, frac):
    idx = top_indices(scores, frac)
    return float(np.mean(np.asarray(labels)[idx]))


def jaccard(a, b):
    sa, sb = set(map(int, a)), set(map(int, b))
    return float(len(sa & sb) / max(1, len(sa | sb)))


def summarize_by_label(labels, values):
    labels = np.asarray(labels).astype(int); values = np.asarray(values, dtype=float)
    out = {}
    for lab, name in [(0, 'normal'), (1, 'anom')]:
        m = labels == lab
        if np.any(m):
            out[f'{name}_mean'] = float(np.mean(values[m])); out[f'{name}_std'] = float(np.std(values[m]))
            out[f'{name}_q05'] = float(np.quantile(values[m], .05)); out[f'{name}_q50'] = float(np.quantile(values[m], .50)); out[f'{name}_q95'] = float(np.quantile(values[m], .95))
        else:
            out[f'{name}_mean'] = out[f'{name}_std'] = out[f'{name}_q05'] = out[f'{name}_q50'] = out[f'{name}_q95'] = 0.0
    out['anom_minus_normal_mean'] = out['anom_mean'] - out['normal_mean']
    return out


def metric_block(labels, idx, arrays, base_name='margin'):
    labels_i = labels[idx]; base = arrays[base_name][idx]
    out = {}
    base_top5 = top_indices(base, .05); base_top1 = top_indices(base, .01)
    for name, vals in arrays.items():
        v = np.asarray(vals)[idx]
        auc, ap = safe_auc(labels_i, v)
        out[name] = {
            'auc': auc, 'ap': ap,
            'top1_ratio': top_ratio(labels_i, v, .01), 'top5_ratio': top_ratio(labels_i, v, .05),
            'spearman_with_margin': safe_spearman(v, base), 'pearson_with_margin': safe_pearson(v, base),
            'top1_jaccard_with_margin': jaccard(top_indices(v, .01), base_top1),
            'top5_jaccard_with_margin': jaccard(top_indices(v, .05), base_top5),
            **summarize_by_label(labels_i, v),
        }
    return out


def entropy_from_scores(x, temp=0.25):
    # x: [N,K], higher response -> higher prob. Returns normalized entropy [0,1].
    z = x / temp
    z = z - np.max(z, axis=1, keepdims=True)
    p = np.exp(z); p = p / (np.sum(p, axis=1, keepdims=True) + 1e-12)
    ent = -np.sum(p * np.log(p + 1e-12), axis=1)
    return ent / np.log(x.shape[1])


def pairwise_cos_stats(x):
    # x [N,K,D]
    xn = x / (np.linalg.norm(x, axis=2, keepdims=True) + 1e-12)
    sim = np.einsum('nkd,nld->nkl', xn, xn)
    k = x.shape[1]
    mask = ~np.eye(k, dtype=bool)
    vals = sim[:, mask].reshape(x.shape[0], k * (k - 1))
    return vals.mean(axis=1), vals.std(axis=1), vals.min(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', default='/data/linziyao/DualRefGAD')
    ap.add_argument('--dataset', default='elliptic'); ap.add_argument('--device', type=int, default=0); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train_rate', type=float, default=0.05); ap.add_argument('--val_rate', type=float, default=0.0)
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
    ap.add_argument('--encode_batch_size', type=int, default=512); ap.add_argument('--out', default='')
    args = ap.parse_args(); t0=time.time(); set_seed(args.seed)
    root = Path(args.project_root); sys.path.insert(0, str(root)); sys.path.insert(0, str(root/'scripts')); os.chdir(str(root))
    from utils import load_mat
    from VecGAD import VecGAD
    from run_training_degradation_diagnosis import to_dense_features, build_descriptor, NormalModel, select_refs, apply_ablation, reference_purity, build_tokens, encode_tokens_batched

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(args.dataset, args.train_rate, args.val_rate, args=args)
    features_np = to_dense_features(args.dataset, features); labels_np=np.asarray(ano_label).reshape(-1).astype(int); idx_test=np.asarray(idx_test,dtype=np.int64); idx_train=np.asarray(idx_train,dtype=np.int64)
    normal_idx=np.asarray(normal_for_train_idx,dtype=np.int64); assert np.sum(labels_np[normal_idx]) == 0
    z=build_descriptor(args.descriptor_mode, features_np, adj, args.hops, args.rw_steps); nm=NormalModel(args.pn_estimator,z,normal_idx,args.pca_components); residual=nm.residual()
    normal_refs, anom_refs, score_meta=select_refs(z,residual,normal_idx,nm,features_np,adj,args,labels_np); normal_refs, anom_refs=apply_ablation(normal_refs,anom_refs,normal_idx,labels_np,args)
    pur=reference_purity(normal_refs, anom_refs, labels_np)
    token_tensor=build_tokens(features_np,normal_refs,anom_refs); model=VecGAD(features_np.shape[1], args.embedding_dim, 'prelu', args).to(device); model.eval()
    with torch.no_grad():
        emb=encode_tokens_batched(model, token_tensor, device, args.encode_batch_size).detach().cpu().numpy()
    h=emb
    rn_set=emb[normal_refs]       # [N,Kn,D]
    ra_set=emb[anom_refs]         # [N,Ka,D]
    rn_mean=rn_set.mean(axis=1); ra_mean=ra_set.mean(axis=1)
    u=h-rn_mean; d=ra_mean-rn_mean
    margin=np.sum((u/(np.linalg.norm(u,axis=1,keepdims=True)+1e-12))*(d/(np.linalg.norm(d,axis=1,keepdims=True)+1e-12)),axis=1)

    # A. anomaly-ref response vector around mean normal anchor: m_a[j] = cos(h-rn_mean, ra_j-rn_mean)
    u_n=u/(np.linalg.norm(u,axis=1,keepdims=True)+1e-12)
    da=ra_set-rn_mean[:,None,:]
    da_n=da/(np.linalg.norm(da,axis=2,keepdims=True)+1e-12)
    ra_resp=np.einsum('nd,nkd->nk', u_n, da_n)
    # B. normal-anchor x anomaly-ref response matrix: M[i,j] = cos(h-rn_i, ra_j-rn_i)
    un=h[:,None,:]-rn_set
    dn=ra_set[:,None,:,:]-rn_set[:,:,None,:]  # [N,Kn,Ka,D]
    un_n=un/(np.linalg.norm(un,axis=2,keepdims=True)+1e-12)
    dn_n=dn/(np.linalg.norm(dn,axis=3,keepdims=True)+1e-12)
    mat=np.einsum('nid,nijd->nij', un_n, dn_n)

    ra_pair_mean, ra_pair_std, ra_pair_min = pairwise_cos_stats(ra_set)
    rn_pair_mean, rn_pair_std, rn_pair_min = pairwise_cos_stats(rn_set)
    arrays={
        'margin': margin,
        'ra_resp_mean': ra_resp.mean(axis=1), 'ra_resp_std': ra_resp.std(axis=1), 'ra_resp_max': ra_resp.max(axis=1), 'ra_resp_min': ra_resp.min(axis=1),
        'ra_resp_q75': np.quantile(ra_resp,.75,axis=1), 'ra_resp_q90': np.quantile(ra_resp,.90,axis=1),
        'ra_resp_top3_mean': np.sort(ra_resp,axis=1)[:,-3:].mean(axis=1),
        'ra_resp_positive_ratio': (ra_resp>0).mean(axis=1), 'ra_resp_high08_ratio': (ra_resp>0.8).mean(axis=1),
        'ra_resp_entropy': entropy_from_scores(ra_resp), 'neg_ra_resp_entropy': -entropy_from_scores(ra_resp),
        'mat_mean': mat.mean(axis=(1,2)), 'mat_std': mat.std(axis=(1,2)), 'mat_max': mat.max(axis=(1,2)), 'mat_min': mat.min(axis=(1,2)),
        'mat_q90': np.quantile(mat,.90,axis=(1,2)), 'mat_top5_mean': np.sort(mat.reshape(mat.shape[0],-1),axis=1)[:,-5:].mean(axis=1),
        'mat_positive_ratio': (mat>0).mean(axis=(1,2)), 'mat_high08_ratio': (mat>0.8).mean(axis=(1,2)), 'mat_entropy': entropy_from_scores(mat.reshape(mat.shape[0],-1)), 'neg_mat_entropy': -entropy_from_scores(mat.reshape(mat.shape[0],-1)),
        'ra_pair_cos_mean': ra_pair_mean, 'ra_pair_cos_std': ra_pair_std, 'ra_pair_cos_min': ra_pair_min,
        'rn_pair_cos_mean': rn_pair_mean, 'rn_pair_cos_std': rn_pair_std, 'rn_pair_cos_min': rn_pair_min,
        'ra_centroid_dist_mean': np.linalg.norm(ra_set-ra_mean[:,None,:],axis=2).mean(axis=1),
        'ra_centroid_dist_std': np.linalg.norm(ra_set-ra_mean[:,None,:],axis=2).std(axis=1),
        'rn_centroid_dist_mean': np.linalg.norm(rn_set-rn_mean[:,None,:],axis=2).mean(axis=1),
        'ra_anom_ratio_diagnostic': (labels_np[anom_refs] == 1).mean(axis=1),
    }
    test_metrics=metric_block(labels_np, idx_test, arrays)
    # candidates satisfying route2 gate approximately
    candidates=[]
    for k,m in test_metrics.items():
        if k == 'margin': continue
        if m['auc'] >= 0.70 and (abs(m['spearman_with_margin']) <= 0.85 or m['top5_jaccard_with_margin'] <= 0.90):
            candidates.append({'signal':k, **{kk:m[kk] for kk in ['auc','ap','spearman_with_margin','top5_jaccard_with_margin','top1_ratio','top5_ratio']}})
    candidates=sorted(candidates,key=lambda x:(x['auc'],x['ap']),reverse=True)
    # failure case table for top margin FPs and low-margin FNs
    test_margin=margin[idx_test]
    top_nodes=idx_test[top_indices(test_margin,.05)]
    fp=top_nodes[labels_np[top_nodes]==0][:50]
    low_nodes=idx_test[np.argsort(test_margin)]
    fn=low_nodes[labels_np[low_nodes]==1][:50]
    failure_rows=[]
    selected=list(map(int,fp[:25]))+list(map(int,fn[:25]))
    for n in selected:
        row={'node':int(n),'label':int(labels_np[n]),'case':'fp_top_margin' if n in set(fp) else 'fn_low_margin'}
        for k in ['margin','ra_resp_mean','ra_resp_std','ra_resp_top3_mean','ra_resp_entropy','mat_mean','mat_std','mat_top5_mean','mat_entropy','ra_pair_cos_mean','ra_centroid_dist_mean','ra_anom_ratio_diagnostic']:
            row[k]=float(arrays[k][n])
        failure_rows.append(row)
    out_base=Path(args.out) if args.out else root/f'outputs/reference_response_distribution/reference_response_distribution_s{args.seed}'
    out_base.parent.mkdir(parents=True, exist_ok=True)
    summary={'status':'reference_response_distribution_no_training','protocol':'No training; labels diagnostic-only; route2 multi-reference distribution anatomy','dataset':args.dataset,'seed':args.seed,'config':vars(args),'counts':{'num_nodes':int(len(labels_np)),'num_test':int(len(idx_test)),'num_labeled_normals':int(len(normal_idx))},'reference_purity':pur,'test_metrics':test_metrics,'route2_candidates':candidates,'topk_failure_cases':{'fp_top_margin_first50':[int(x) for x in fp],'fn_low_margin_first50':[int(x) for x in fn],'failure_rows':failure_rows},'time_sec':time.time()-t0}
    summary_path=out_base.with_suffix('.summary.json'); csv_path=out_base.with_suffix('.per_node.csv'); npz_path=out_base.with_suffix('.arrays.npz')
    summary_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    fields=['node','label','split']+list(arrays.keys())
    split=np.full(len(labels_np),'other',dtype=object); split[idx_train]='train'; split[idx_test]='test'
    lines=[','.join(fields)]
    for i in range(len(labels_np)):
        lines.append(','.join([str(i),str(int(labels_np[i])),str(split[i])]+[str(float(arrays[k][i])) for k in arrays]))
    csv_path.write_text('\n'.join(lines)+'\n')
    np.savez_compressed(npz_path, labels=labels_np, idx_test=idx_test, normal_refs=normal_refs, anom_refs=anom_refs, **arrays)
    print(json.dumps({'summary':str(summary_path),'csv':str(csv_path),'npz':str(npz_path),'margin':test_metrics['margin'],'route2_candidates':candidates[:10],'time_sec':summary['time_sec']},indent=2,ensure_ascii=False),flush=True)


if __name__ == '__main__': main()
