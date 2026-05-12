#!/usr/bin/env python3
"""DualRefGAD two-sided score-head training.

Non-intrusive diagnostic for the investigation
`2026-05-06-dualrefgad-two-sided-score-head`.

The script reuses fixed dual references and frozen VecGAD/GT embeddings from the
previous investigation, then compares several score-head designs:

- scalar_mlp_baseline: pooled target/reference embedding -> scalar normality logit
- structured_readout: explicit target / normal / deviation relation features
- decomposition_head: normal-compatibility + deviation-context support + final logit
- decomposition_split_mismatch: decomposition head trained with split R_n/R_a mismatch

Important semantics:
- `R_a` is deviation-side evidence, not anomaly pseudo-label.
- The trained logit is a normality / context-validity logit; anomaly score is `-logit`.
- Train split uses only labeled-normal nodes and asserts no anomaly leakage.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path.home() / "VoxG"
sys.path.insert(0, str(ROOT)); os.chdir(str(ROOT))
SRC = ROOT / "nexus/investigations/2026-05-05-elliptic-training-degradation/experiments/scripts"
sys.path.insert(0, str(SRC))

from run_training_degradation_diagnosis import (  # noqa: E402
    set_seed, to_dense_features, build_descriptor, NormalModel, select_refs,
    apply_ablation, reference_purity, build_tokens, encode_tokens_batched
)
from utils import load_mat  # noqa: E402
from VecGAD import VecGAD  # noqa: E402


def safe_auc(y, s):
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def _idx_to_numpy(idx):
    if isinstance(idx, torch.Tensor):
        return idx.detach().cpu().numpy()
    return idx


def build_pair_batch(emb, normal_refs, anom_refs, node_idx, rn_idx=None, ra_idx=None):
    """Return target, R_n embeddings, R_a embeddings.

    rn_idx/ra_idx choose whose references are used. If None, use node_idx.
    This enables split mismatch without mutating fixed reference arrays.
    """
    device = emb.device
    node_np = _idx_to_numpy(node_idx)
    rn_np = node_np if rn_idx is None else _idx_to_numpy(rn_idx)
    ra_np = node_np if ra_idx is None else _idx_to_numpy(ra_idx)
    target = emb[torch.as_tensor(node_np, dtype=torch.long, device=device)]
    rn_ids = normal_refs[rn_np]
    ra_ids = anom_refs[ra_np]
    rn = emb[torch.as_tensor(rn_ids, dtype=torch.long, device=device)]
    ra = emb[torch.as_tensor(ra_ids, dtype=torch.long, device=device)]
    return target, rn, ra


class ScalarMLPHead(nn.Module):
    """VecGAD-like generic pooled scalar readout."""
    def __init__(self, dim, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, target, rn, ra):
        rn_mean = rn.mean(dim=1)
        ra_mean = ra.mean(dim=1)
        pooled = torch.stack([target, rn_mean, ra_mean], dim=1).mean(dim=1)
        final = self.net(pooled).squeeze(-1)
        return {"final": final}


class StructuredReadoutHead(nn.Module):
    """Explicit target / normal / deviation relation features."""
    def __init__(self, dim, hidden=256, dropout=0.2):
        super().__init__()
        in_dim = dim * 8
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def features(self, target, rn, ra):
        rn_mean = rn.mean(dim=1)
        ra_mean = ra.mean(dim=1)
        return torch.cat([
            target, rn_mean, ra_mean,
            target - rn_mean, target - ra_mean, ra_mean - rn_mean,
            target * rn_mean, target * ra_mean,
        ], dim=1)

    def forward(self, target, rn, ra):
        final = self.net(self.features(target, rn, ra)).squeeze(-1)
        return {"final": final}


class DecompositionHead(nn.Module):
    """Two-sided normality head with inspectable sub-logits.

    `sn`: normal-side compatibility logit.
    `sa`: deviation-side context-support logit (matched deviation evidence), not anomaly label.
    `final`: normality/context-validity logit used for anomaly score = -final.
    """
    def __init__(self, dim, hidden=256, dropout=0.2, use_sublogits_to_final=True):
        super().__init__()
        self.use_sublogits_to_final = use_sublogits_to_final
        rel_dim = dim * 4
        self.n_net = nn.Sequential(
            nn.Linear(rel_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
        )
        self.a_net = nn.Sequential(
            nn.Linear(rel_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
        )
        self.sn = nn.Linear(hidden // 2, 1)
        self.sa = nn.Linear(hidden // 2, 1)
        final_in = hidden + (2 if use_sublogits_to_final else 0)
        self.final = nn.Sequential(
            nn.Linear(final_in, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    @staticmethod
    def rel(target, ref_mean):
        return torch.cat([target, ref_mean, target - ref_mean, target * ref_mean], dim=1)

    def forward(self, target, rn, ra):
        rn_mean = rn.mean(dim=1)
        ra_mean = ra.mean(dim=1)
        hn = self.n_net(self.rel(target, rn_mean))
        ha = self.a_net(self.rel(target, ra_mean))
        sn = self.sn(hn).squeeze(-1)
        sa = self.sa(ha).squeeze(-1)
        if self.use_sublogits_to_final:
            final_in = torch.cat([hn, ha, sn[:, None], sa[:, None]], dim=1)
        else:
            final_in = torch.cat([hn, ha], dim=1)
        final = self.final(final_in).squeeze(-1)
        return {"final": final, "sn": sn, "sa": sa}



class ContrastiveTwoScoreHead(nn.Module):
    """Theory-aligned two-score head.

    sn: normal-side compatibility/support from (v, R_n)
    sa: deviation-side support from (v, R_a)

    Deployable anomaly score is intended as:
        anomaly_score = sa - sn

    The existing training/evaluation pipeline expects a normality logit whose
    negative is the anomaly score, so this head returns:
        final = sn - sa
    """
    def __init__(self, dim, hidden=256, dropout=0.2):
        super().__init__()
        rel_dim = dim * 4
        self.n_net = nn.Sequential(
            nn.Linear(rel_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.a_net = nn.Sequential(
            nn.Linear(rel_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    @staticmethod
    def rel(target, ref_mean):
        return torch.cat([target, ref_mean, target - ref_mean, target * ref_mean], dim=1)

    def forward(self, target, rn, ra):
        rn_mean = rn.mean(dim=1)
        ra_mean = ra.mean(dim=1)
        sn = self.n_net(self.rel(target, rn_mean)).squeeze(-1)
        sa = self.a_net(self.rel(target, ra_mean)).squeeze(-1)
        final = sn - sa
        return {"final": final, "sn": sn, "sa": sa}



class DualMarginTwoScoreHead(nn.Module):
    """Lambda-free direction-aware margin two-score head.

    Same training objective as contrastive_two_score:
        positive: (v, R_n(v), R_a(v)) -> 1
        negative: (v, R_n(c), R_a(c)) -> 0

    Head geometry:
        sn = normal-side consistency
        sa = deviation-side pressure
        m_norm = <normalize(v-rn), normalize(ra-rn)>
        final normality/consistency logit = sn - sa - m_norm
        anomaly score at eval = -final = sa - sn + m_norm
    """
    def __init__(self, dim, hidden=256, dropout=0.2):
        super().__init__()
        rel_dim = dim * 4
        self.n_net = nn.Sequential(
            nn.Linear(rel_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.a_net = nn.Sequential(
            nn.Linear(rel_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    @staticmethod
    def rel(target, ref_mean):
        return torch.cat([target, ref_mean, target - ref_mean, target * ref_mean], dim=1)

    def forward(self, target, rn, ra):
        rn_mean = rn.mean(dim=1)
        ra_mean = ra.mean(dim=1)
        sn = self.n_net(self.rel(target, rn_mean)).squeeze(-1)
        sa = self.a_net(self.rel(target, ra_mean)).squeeze(-1)
        u = F.normalize(target - rn_mean, p=2, dim=1, eps=1e-12)
        d = F.normalize(ra_mean - rn_mean, p=2, dim=1, eps=1e-12)
        margin = torch.sum(u * d, dim=1)
        final = sn - sa - margin
        return {"final": final, "sn": sn, "sa": sa, "margin": margin}


def make_head(args):
    if args.head_mode == "scalar_mlp_baseline":
        return ScalarMLPHead(args.embedding_dim, args.head_hidden, args.head_dropout)
    if args.head_mode == "structured_readout":
        return StructuredReadoutHead(args.embedding_dim, args.head_hidden, args.head_dropout)
    if args.head_mode == "contrastive_two_score":
        return ContrastiveTwoScoreHead(args.embedding_dim, args.head_hidden, args.head_dropout)
    if args.head_mode == "dual_margin_two_score":
        return DualMarginTwoScoreHead(args.embedding_dim, args.head_hidden, args.head_dropout)
    use_sublogits = args.head_mode != "decomposition_no_sublogit_to_final"
    return DecompositionHead(args.embedding_dim, args.head_hidden, args.head_dropout, use_sublogits_to_final=use_sublogits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="elliptic")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_rate", type=float, default=0.05)
    ap.add_argument("--descriptor_mode", choices=["hop_attr","rwse","hop_attr_rwse"], default="hop_attr")
    ap.add_argument("--pn_estimator", choices=["diag_gaussian","pca_residual"], default="pca_residual")
    ap.add_argument("--gn_mode", choices=["label_gate","normal_density","label_gate_density"], default="label_gate")
    ap.add_argument("--ln_mode", choices=["descriptor_similarity","reconstruction_gain"], default="descriptor_similarity")
    ap.add_argument("--ga_mode", choices=["normal_rejection","residual_norm","normal_soft_or"], default="normal_soft_or")
    ap.add_argument("--la_mode", choices=["residual_cosine","descriptor_similarity"], default="descriptor_similarity")
    ap.add_argument("--reference_mode", default="dual_reference")
    ap.add_argument("--ablation_mode", choices=["full","no_ra","shuffled_ra","fixed_labeled_normal"], default="full")
    ap.add_argument("--head_mode", choices=["scalar_mlp_baseline","structured_readout","decomposition_head","decomposition_split_mismatch","decomposition_no_sn_aux","decomposition_no_sa_aux","decomposition_no_sublogit_to_final","contrastive_two_score","dual_margin_two_score"], default="scalar_mlp_baseline")
    ap.add_argument("--normal_k", type=int, default=4)
    ap.add_argument("--anom_k", type=int, default=16)
    ap.add_argument("--pp_k", type=int, default=6)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--rw_steps", type=int, default=8)
    ap.add_argument("--pca_components", type=int, default=32)
    ap.add_argument("--embedding_dim", type=int, default=256)
    ap.add_argument("--GT_ffn_dim", type=int, default=256)
    ap.add_argument("--GT_dropout", type=float, default=0.4)
    ap.add_argument("--GT_attention_dropout", type=float, default=0.4)
    ap.add_argument("--GT_num_heads", type=int, default=2)
    ap.add_argument("--GT_num_layers", type=int, default=3)
    ap.add_argument("--encode_batch_size", type=int, default=256)
    ap.add_argument("--num_epoch", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--head_hidden", type=int, default=256)
    ap.add_argument("--head_dropout", type=float, default=0.2)
    ap.add_argument("--aux_weight", type=float, default=0.25)
    ap.add_argument("--wandb", type=lambda x: str(x).lower() in ["1","true","yes"], default=False)
    ap.add_argument("--out", default="")
    # Compatibility args expected by VecGAD internals.
    ap.add_argument("--sample_rate", type=float, default=0.15)
    ap.add_argument("--mean", type=float, default=0.02)
    ap.add_argument("--var", type=float, default=0.01)
    ap.add_argument("--outlier_beta", type=float, default=0.3)
    ap.add_argument("--ring_R_max", type=float, default=1.0)
    ap.add_argument("--ring_R_min", type=float, default=0.3)
    ap.add_argument("--lambda_rec_tok", type=float, default=1.0)
    ap.add_argument("--lambda_rec_emb", type=float, default=0.1)
    args = ap.parse_args()

    t0 = time.time()
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    adj, features, labels, all_idx, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label, normal_for_train_idx, normal_for_generation_idx = load_mat(args.dataset, args.train_rate, 0.1, args=args)
    features_np = to_dense_features(args.dataset, features)
    labels_np = np.asarray(ano_label).reshape(-1).astype(int)
    normal_idx = np.asarray(normal_for_train_idx, dtype=int)
    idx_val = np.asarray(idx_val, dtype=int)
    idx_test = np.asarray(idx_test, dtype=int)
    assert np.sum(labels_np[normal_idx]) == 0, "Data leakage: normal_for_train_idx contains anomalies"

    z = build_descriptor(args.descriptor_mode, features_np, adj, args.hops, args.rw_steps)
    nm = NormalModel(args.pn_estimator, z, normal_idx, args.pca_components)
    residual = nm.residual()
    normal_refs, anom_refs, score_meta = select_refs(z, residual, normal_idx, nm, features_np, adj, args, labels_np)
    normal_refs, anom_refs = apply_ablation(normal_refs, anom_refs, normal_idx, labels_np, args)
    pur = reference_purity(normal_refs, anom_refs, labels_np)

    token_tensor = build_tokens(features_np, normal_refs, anom_refs)
    model = VecGAD(features_np.shape[1], args.embedding_dim, "prelu", args).to(device)
    model.eval()
    with torch.no_grad():
        emb = encode_tokens_batched(model, token_tensor, device, args.encode_batch_size).detach()

    head = make_head(args).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    normal_t_np = normal_idx.copy()
    n_norm = len(normal_t_np)
    all_nodes_np = np.arange(len(labels_np), dtype=int)
    rng = np.random.default_rng(args.seed)
    best = {"val_auc": -1.0, "val_ap": -1.0, "test_auc": -1.0, "test_ap": -1.0, "epoch": -1}
    last = {}

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="VoxG", entity="HCCS", config=vars(args), name=f"dualref_head_{args.head_mode}_{args.dataset}_s{args.seed}")
        wandb.summary.update(pur)

    for epoch in range(args.num_epoch + 1):
        if epoch > 0:
            head.train(); opt.zero_grad()
            perm = rng.permutation(n_norm)
            corrupt = normal_t_np[perm]
            same = corrupt == normal_t_np
            if np.any(same):
                corrupt[same] = normal_t_np[(np.where(same)[0] + 1) % n_norm]
            v_t = torch.tensor(normal_t_np, dtype=torch.long, device=device)
            c_t = torch.tensor(corrupt, dtype=torch.long, device=device)

            if args.head_mode == "decomposition_split_mismatch":
                pos = head(*build_pair_batch(emb, normal_refs, anom_refs, v_t, None, None))
                neg_n = head(*build_pair_batch(emb, normal_refs, anom_refs, v_t, c_t, None))
                neg_a = head(*build_pair_batch(emb, normal_refs, anom_refs, v_t, None, c_t))
                neg_b = head(*build_pair_batch(emb, normal_refs, anom_refs, v_t, c_t, c_t))
                logits = torch.cat([pos["final"], neg_n["final"], neg_a["final"], neg_b["final"]], dim=0)
                y = torch.cat([
                    torch.ones(n_norm, device=device),
                    torch.zeros(n_norm * 3, device=device),
                ], dim=0)
                loss = bce(logits, y)
                # Auxiliary relation labels: sn tests R_n match; sa tests R_a match.
                aux_sn = torch.cat([pos["sn"], neg_n["sn"], neg_a["sn"], neg_b["sn"]], dim=0)
                aux_sn_y = torch.cat([
                    torch.ones(n_norm, device=device), torch.zeros(n_norm, device=device),
                    torch.ones(n_norm, device=device), torch.zeros(n_norm, device=device),
                ], dim=0)
                aux_sa = torch.cat([pos["sa"], neg_n["sa"], neg_a["sa"], neg_b["sa"]], dim=0)
                aux_sa_y = torch.cat([
                    torch.ones(n_norm, device=device), torch.ones(n_norm, device=device),
                    torch.zeros(n_norm, device=device), torch.zeros(n_norm, device=device),
                ], dim=0)
                loss = loss + args.aux_weight * (bce(aux_sn, aux_sn_y) + bce(aux_sa, aux_sa_y))
            else:
                pos = head(*build_pair_batch(emb, normal_refs, anom_refs, v_t, None, None))
                neg = head(*build_pair_batch(emb, normal_refs, anom_refs, v_t, c_t, c_t))
                logits = torch.cat([pos["final"], neg["final"]], dim=0)
                y = torch.cat([torch.ones(n_norm, device=device), torch.zeros(n_norm, device=device)], dim=0)
                loss = bce(logits, y)
                if args.head_mode in {"decomposition_head", "decomposition_no_sn_aux", "decomposition_no_sa_aux", "decomposition_no_sublogit_to_final"}:
                    # Mild auxiliary labels: valid tuples should be side-matched, corrupted tuples side-mismatched.
                    aux_parts = []
                    aux_targets = []
                    if args.head_mode != "decomposition_no_sn_aux":
                        aux_parts.extend([pos["sn"], neg["sn"]])
                        aux_targets.extend([torch.ones(n_norm, device=device), torch.zeros(n_norm, device=device)])
                    if args.head_mode != "decomposition_no_sa_aux":
                        aux_parts.extend([pos["sa"], neg["sa"]])
                        aux_targets.extend([torch.ones(n_norm, device=device), torch.zeros(n_norm, device=device)])
                    if aux_parts:
                        aux_logits = torch.cat(aux_parts, dim=0)
                        aux_y = torch.cat(aux_targets, dim=0)
                        loss = loss + args.aux_weight * bce(aux_logits, aux_y)
            loss.backward(); opt.step()
            train_auc, train_ap = safe_auc(y.detach().cpu().numpy(), logits.detach().cpu().numpy())
        else:
            loss = torch.tensor(0.0)
            train_auc = 0.5
            train_ap = 0.5

        head.eval()
        with torch.no_grad():
            all_t = torch.tensor(all_nodes_np, dtype=torch.long, device=device)
            out = head(*build_pair_batch(emb, normal_refs, anom_refs, all_t, None, None))
            normality_logit = out["final"].detach().cpu().numpy()
            sn_np = out.get("sn", torch.zeros_like(out["final"])).detach().cpu().numpy()
            margin_np = out.get("margin", torch.zeros_like(out["final"])).detach().cpu().numpy()
            sa_np = out.get("sa", torch.zeros_like(out["final"])).detach().cpu().numpy()
        anomaly_score = -normality_logit
        val_auc, val_ap = safe_auc(labels_np[idx_val], anomaly_score[idx_val])
        test_auc, test_ap = safe_auc(labels_np[idx_test], anomaly_score[idx_test])
        row = {
            "epoch": epoch,
            "loss": float(loss.detach().cpu().item()),
            "valid_corrupt_train_auc": train_auc,
            "valid_corrupt_train_ap": train_ap,
            "val_auc": val_auc,
            "val_ap": val_ap,
            "test_auc": test_auc,
            "test_ap": test_ap,
            "score_std": float(np.std(anomaly_score)),
            "normal_score_mean": float(np.mean(anomaly_score[normal_idx])),
            "test_score_mean": float(np.mean(anomaly_score[idx_test])),
            "sn_std": float(np.std(sn_np)),
            "sa_std": float(np.std(sa_np)),
            "sn_test_mean": float(np.mean(sn_np[idx_test])),
            "sa_test_mean": float(np.mean(sa_np[idx_test])),
            "margin_std": float(np.std(margin_np)),
            "margin_test_mean": float(np.mean(margin_np[idx_test])),
        }
        last = row
        if val_auc > best["val_auc"]:
            best.update({"val_auc": val_auc, "val_ap": val_ap, "test_auc": test_auc, "test_ap": test_ap, "epoch": epoch})
        if run and (epoch % 10 == 0 or epoch == args.num_epoch):
            import wandb
            wandb.log(row, step=epoch)

    final = {
        "seed": args.seed,
        "dataset": args.dataset,
        "train_rate": args.train_rate,
        "head_mode": args.head_mode,
        "ablation_mode": args.ablation_mode,
        "n_labeled_normal": int(n_norm),
        "normal_ref_normal_ratio": pur["normal_ref_normal_ratio"],
        "anom_ref_anom_ratio": pur["anom_ref_anom_ratio"],
        "anom_ref_anom_ratio_on_anom_nodes": pur["anom_ref_anom_ratio_on_anom_nodes"],
        "best_val_auc": best["val_auc"],
        "best_val_ap": best["val_ap"],
        "best_test_auc": best["test_auc"],
        "best_test_ap": best["test_ap"],
        "best_epoch": best["epoch"],
        "final_test_auc": last.get("test_auc"),
        "final_test_ap": last.get("test_ap"),
        "final_train_auc": last.get("valid_corrupt_train_auc"),
        "final_score_std": last.get("score_std"),
        "final_sn_std": last.get("sn_std"),
        "final_sa_std": last.get("sa_std"),
        "final_margin_std": last.get("margin_std"),
        "final_margin_test_mean": last.get("margin_test_mean"),
        "time_sec": time.time() - t0,
    }
    print(json.dumps(final, ensure_ascii=False), flush=True)
    if run:
        import wandb
        wandb.summary.update(final)
        run.finish()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
