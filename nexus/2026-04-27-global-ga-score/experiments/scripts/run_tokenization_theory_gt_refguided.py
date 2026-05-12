#!/usr/bin/env python3
"""
Tokenization Theory GT Training - Reference-Guided Pseudo Anomaly Version

Objective改进：pseudo anomaly 方向从 self-residual 变为 reference-guided residual。
核心变化：
  - 旧版：outlier_emb = normal_emb + beta * (normal_emb - center) + noise
  - 新版：outlier_emb = normal_emb + beta * (pool_ra - pool_rn) + noise

保持：
  - 半监督设置（训练集只有正常节点）
  - 单一 BCEWithLogitsLoss（不引入多损失）
  - 原有超参数 pseudo_beta, pseudo_noise（不增加新超参数）
"""

import argparse, json, time, sys, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
ROOT = Path.home() / 'VoxG'
sys.path.insert(0, str(ROOT))
from VecGAD import VecGAD
from utils import load_mat, preprocess_features, normalize_adj
from utils.data import load_graph_data
from utils.split import semi_supervised_split
from experiments.scripts.tokenization_refs import build_descriptor, NormalModel, select_refs, apply_ablation, reference_purity, build_tokens

def encode_tokens_batched(model, tokens, device, batch_size):
    """Batched encoding to avoid OOM"""
    N = tokens.shape[0]
    emb = []
    for i in range(0, N, batch_size):
        batch = torch.tensor(tokens[i:i+batch_size], dtype=torch.long, device=device)
        with torch.no_grad():
            emb.append(model.encode(batch).cpu())
    return torch.cat(emb, dim=0).numpy()

def eval_logits(logits, labels, idx):
    from sklearn.metrics import roc_auc_score, average_precision_score
    y_true = labels[idx]
    y_score = logits[idx]
    try:
        auc = roc_auc_score(y_true, y_score)
    except:
        auc = 0.5
    try:
        ap = average_precision_score(y_true, y_score)
    except:
        ap = 0.0
    return auc, ap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train_rate', type=float, default=0.05)
    ap.add_argument('--val_rate', type=float, default=0.15)
    ap.add_argument('--num_epoch', type=int, default=200)
    ap.add_argument('--embedding_dim', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--weight_decay', type=float, default=5e-4)
    ap.add_argument('--pseudo_beta', type=float, default=1.0)
    ap.add_argument('--pseudo_noise', type=float, default=0.1)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--hops', type=int, default=2)
    ap.add_argument('--rw_steps', type=int, default=16)
    ap.add_argument('--pca_components', type=int, default=32)
    ap.add_argument('--normal_k', type=int, default=4)
    ap.add_argument('--anom_k', type=int, default=16)
    ap.add_argument('--pp_k', type=int, default=6)
    ap.add_argument('--descriptor_mode', default='hop_attr')
    ap.add_argument('--pn_estimator', default='pca_residual')
    ap.add_argument('--gn_mode', default='label_gate')
    ap.add_argument('--ln_mode', default='descriptor_similarity')
    ap.add_argument('--ga_mode', default='normal_rejection')
    ap.add_argument('--la_mode', default='residual_cosine')
    ap.add_argument('--ablation_mode', default='full')
    ap.add_argument('--encode_batch_size', type=int, default=2048)
    ap.add_argument('--wandb', type=lambda x: str(x).lower() in ['1','true','yes'], default=False)
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--out', default='')
    ap.add_argument('--objective_mode', default='ref_guided')  # 'self_residual' or 'ref_guided'
    args = ap.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    features_np, adj, labels_np = load_graph_data(args.dataset)
    idx_train, idx_val, idx_test = semi_supervised_split(labels_np, args.train_rate, args.val_rate, args.seed)
    normal_idx = idx_train.tolist()
    assert np.sum(labels_np[normal_idx]) == 0, "数据泄漏！训练集包含异常节点！"
    
    z = build_descriptor(args.descriptor_mode, features_np, adj, args.hops, args.rw_steps)
    nm = NormalModel(args.pn_estimator, z, normal_idx, args.pca_components)
    residual = nm.residual()
    
    normal_refs, anom_refs, score_meta = select_refs(z, residual, normal_idx, nm, features_np, adj, args, labels_np)
    normal_refs, anom_refs = apply_ablation(normal_refs, anom_refs, normal_idx, labels_np, args)
    pur = reference_purity(normal_refs, anom_refs, labels_np)
    token_tensor = build_tokens(features_np, normal_refs, anom_refs)
    
    if args.dry_run:
        print(json.dumps({'dataset':args.dataset,'token_shape':list(token_tensor.shape),'purity':pur,'objective_mode':args.objective_mode},indent=2))
        return
    
    wandb_run=None
    if args.wandb:
        import wandb
        wandb_run=wandb.init(project='VoxG', entity='HCCS', config=vars(args), name=f"tokgt_refguided_{args.dataset}_{args.objective_mode}")
        wandb.summary.update(pur)
    
    model = VecGAD(features_np.shape[1], args.embedding_dim, 'prerelu', args).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    normal_t = torch.tensor(normal_idx, dtype=torch.long, device=device)
    
    best = {'val_auc':-1, 'val_ap':-1, 'test_auc':-1, 'test_ap':-1, 'epoch':-1}
    start_time = time.time()
    
    for epoch in range(args.num_epoch+1):
        model.train()
        opt.zero_grad()
        
        # Encode all tokens
        emb_np = encode_tokens_batched(model, token_tensor, device, args.encode_batch_size)
        emb = torch.tensor(emb_np, dtype=torch.float, device=device)
        normal_emb = emb[normal_t]
        
        # ========== Objective mode selection ==========
        if args.objective_mode == 'self_residual':
            # 旧版：pseudo anomaly 方向由 normal_emb 自己的 residual 决定
            center = normal_emb.mean(dim=0, keepdim=True)
            direction = F.normalize(normal_emb - center, dim=1)
            outlier_emb = normal_emb + args.pseudo_beta * direction + torch.randn_like(normal_emb) * args.pseudo_noise
        else:
            # 新版：pseudo anomaly 方向由 pooled R_a - pooled R_n 决定
            anom_ref_indices = []
            for refs in anom_refs[:min(100, len(anom_refs))]:
                anom_ref_indices.extend(refs)
            if len(anom_ref_indices) == 0:
                anom_ref_indices = normal_idx
            
            normal_ref_indices = []
            for refs in normal_refs[:min(100, len(normal_refs))]:
                normal_ref_indices.extend(refs)
            if len(normal_ref_indices) == 0:
                normal_ref_indices = normal_idx
            
            anom_ref_t = torch.tensor(list(set(anom_ref_indices)), dtype=torch.long, device=device)
            normal_ref_t = torch.tensor(list(set(normal_ref_indices)), dtype=torch.long, device=device)
            pool_ra = emb[anom_ref_t].mean(dim=0)
            pool_rn = emb[normal_ref_t].mean(dim=0)
            ref_direction = F.normalize(pool_ra - pool_rn, dim=0)
            
            outlier_emb = normal_emb + args.pseudo_beta * ref_direction.unsqueeze(0) + torch.randn_like(normal_emb) * args.pseudo_noise
        # ==============================================
        
        emb_c = torch.cat([normal_emb, outlier_emb], dim=0)
        logits_train = model.fc3(model.act(model.fc2(model.act(model.fc1(emb_c))))).squeeze(-1)
        y = torch.cat([torch.zeros(len(normal_emb), device=device), torch.ones(len(outlier_emb), device=device)])
        loss = bce(logits_train, y)
        loss.backward()
        opt.step()
        
        if epoch % 10 == 0 or epoch == args.num_epoch:
            model.eval()
            with torch.no_grad():
                emb_eval = encode_tokens_batched(model, token_tensor, device, args.encode_batch_size)
                logits = model.fc3(model.act(model.fc2(model.act(model.fc1(torch.tensor(emb_eval, dtype=torch.float, device=device)))))).squeeze(-1).detach().cpu().numpy()
            val_auc, val_ap = eval_logits(logits, labels_np, idx_val)
            test_auc, test_ap = eval_logits(logits, labels_np, idx_test)
            if val_auc + val_ap > best['val_auc'] + best['val_ap']:
                best.update({'val_auc':val_auc, 'val_ap':val_ap, 'test_auc':test_auc, 'test_ap':test_ap, 'epoch':epoch})
            row = {'epoch':epoch, 'loss':float(loss.item()), 'val_auc':val_auc, 'val_ap':val_ap, 'test_auc':test_auc, 'test_ap':test_ap, 'objective_mode':args.objective_mode}
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if wandb_run:
                wandb.log(row, step=epoch)
    
    elapsed = time.time() - start_time
    result = {'dataset':args.dataset, 'seed':args.seed, 'objective_mode':args.objective_mode, 'config':vars(args), 'best':best, 'purity':pur, 'elapsed_sec':elapsed}
    print('FINAL', json.dumps(result, indent=2, ensure_ascii=False))
    if wandb_run:
        wandb.summary['best_val_auc'] = best['val_auc']
        wandb.summary['test_auc'] = best['test_auc']
        wandb.finish()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()