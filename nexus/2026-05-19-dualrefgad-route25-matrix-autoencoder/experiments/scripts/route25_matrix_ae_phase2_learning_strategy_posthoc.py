#!/usr/bin/env python3
"""Phase 2 low-cost posthoc audit for Route2.5 Matrix AE.

This probe uses already completed Phase-1 output. It does not train or touch GPUs.
Question: can the Matrix AE failure be repaired by a cheap learning-strategy choice
(latent selection, validation-loss selection, early-stop proxy), or is the remaining
negative decision still dominated by representation/reference regime?
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import math
import statistics


def mean_std(xs):
    xs = [float(x) for x in xs]
    if not xs:
        return {"mean": None, "std": None, "min": None, "max": None}
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / len(xs)
    return {"mean": float(mu), "std": float(math.sqrt(var)), "min": float(min(xs)), "max": float(max(xs))}


def safe_corr(xs, ys):
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    if len(xs) < 3 or len(ys) < 3:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return float(cov / math.sqrt(vx * vy))


def rank(xs):
    xs = [float(x) for x in xs]
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    for pos, idx in enumerate(order):
        r[idx] = float(pos)
    return r


def spearman(xs, ys):
    return safe_corr(rank(xs), rank(ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase1', required=True, help='Phase-1 instability audit JSON')
    ap.add_argument('--source_phase1', default='', help='Alias used by runner config validation; if set and --phase1 is omitted by a wrapper, it names the Phase-1 JSON.')
    ap.add_argument('--dataset', default='elliptic', help='Metadata-only dataset name for runner validation.')
    ap.add_argument('--training_cost', default='zero_new_training', help='Metadata-only cost marker for runner validation.')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    start = time.time()
    phase1_path = Path(args.phase1)
    data = json.loads(phase1_path.read_text(encoding='utf-8'))
    split_results = data['split_results']
    all_runs = []
    split_summaries = []
    for split in split_results:
        sid = int(split['split_seed'])
        scalar = split['scalar_best']
        runs = split['runs']
        for r in runs:
            rr = dict(r)
            rr['split_seed'] = sid
            rr['scalar_auc'] = float(scalar['auc'])
            rr['scalar_ap'] = float(scalar['ap'])
            rr['scalar_name'] = scalar['name']
            rr['anom_ref_anom_ratio'] = float(split['reference_diagnostics']['anom_ref_anom_ratio_diagnostic'])
            all_runs.append(rr)
        best_auc = max(runs, key=lambda r: r['auc'])
        best_val = min(runs, key=lambda r: r['best_val_loss'])
        latent_best = {}
        for ld in sorted({int(r['latent_dim']) for r in runs}):
            rr = [r for r in runs if int(r['latent_dim']) == ld]
            latent_best[str(ld)] = max(rr, key=lambda r: r['auc'])
        split_summaries.append({
            'split_seed': sid,
            'scalar_auc': float(scalar['auc']),
            'scalar_ap': float(scalar['ap']),
            'scalar_name': scalar['name'],
            'best_auc_selector': {
                'auc': float(best_auc['auc']),
                'ap': float(best_auc['ap']),
                'latent_dim': int(best_auc['latent_dim']),
                'ae_seed': int(best_auc['ae_seed']),
                'delta_auc_vs_scalar': float(best_auc['auc'] - scalar['auc']),
            },
            'best_val_loss_selector': {
                'auc': float(best_val['auc']),
                'ap': float(best_val['ap']),
                'latent_dim': int(best_val['latent_dim']),
                'ae_seed': int(best_val['ae_seed']),
                'delta_auc_vs_scalar': float(best_val['auc'] - scalar['auc']),
                'best_val_loss': float(best_val['best_val_loss']),
            },
            'latent_best': {
                k: {
                    'auc': float(v['auc']),
                    'ap': float(v['ap']),
                    'delta_auc_vs_scalar': float(v['auc'] - scalar['auc']),
                    'ae_seed': int(v['ae_seed']),
                } for k, v in latent_best.items()
            },
            'reference_diagnostics': split['reference_diagnostics'],
        })
    def selector_values(selector):
        return [s[selector]['auc'] for s in split_summaries]
    def selector_delta(selector):
        return [s[selector]['delta_auc_vs_scalar'] for s in split_summaries]
    scalar_auc = [s['scalar_auc'] for s in split_summaries]
    latent_dims = sorted({int(r['latent_dim']) for r in all_runs})
    by_latent = {}
    for ld in latent_dims:
        best_per_split = [s['latent_best'][str(ld)] for s in split_summaries]
        by_latent[str(ld)] = {
            'best_per_split_auc': mean_std([x['auc'] for x in best_per_split]),
            'best_per_split_delta_auc_vs_scalar': mean_std([x['delta_auc_vs_scalar'] for x in best_per_split]),
            'promote_splits': int(sum(x['delta_auc_vs_scalar'] > 0.02 for x in best_per_split)),
            'drop_splits': int(sum(x['auc'] < 0.58 for x in best_per_split)),
        }
    val_losses = [r['best_val_loss'] for r in all_runs]
    aucs = [r['auc'] for r in all_runs]
    deltas = [r['delta_auc_vs_scalar'] for r in all_runs]
    degree_corr_abs = [abs(r['spearman_with_degree']) for r in all_runs]
    margin_corr_abs = [abs(r['spearman_with_margin']) for r in all_runs]
    ref_ratio = [r['anom_ref_anom_ratio'] for r in all_runs]
    best_auc_deltas = selector_delta('best_auc_selector')
    best_val_deltas = selector_delta('best_val_loss_selector')
    # Decision rules: cheap repair succeeds only if deployable selection (val loss or fixed latent)
    # recovers positive mean delta and >=3/5 promote splits. Oracle best-AUC is upper bound only.
    fixed_latent_promising = [k for k, v in by_latent.items() if (v['best_per_split_delta_auc_vs_scalar']['mean'] or -9) > 0.02 and v['promote_splits'] >= 3]
    val_selector_promising = (sum(best_val_deltas) / len(best_val_deltas)) > 0.02 and sum(x > 0.02 for x in best_val_deltas) >= 3
    oracle_promising = (sum(best_auc_deltas) / len(best_auc_deltas)) > 0.02 and sum(x > 0.02 for x in best_auc_deltas) >= 3
    if val_selector_promising or fixed_latent_promising:
        decision = 'CHEAP_LEARNING_STRATEGY_REPAIR_PROMISING'
    elif oracle_promising:
        decision = 'ONLY_ORACLE_SELECTION_LOOKS_PROMISING__NOT_DEPLOYABLE'
    else:
        decision = 'NO_CHEAP_LEARNING_STRATEGY_REPAIR__KEEP_DROP_DECISION'
    result = {
        'status': 'finished',
        'probe': 'route25_matrix_ae_phase2_learning_strategy_posthoc',
        'protocol': 'Posthoc low-cost audit using existing Phase-1 repeated-AE output; no new training; labels used for diagnosis only.',
        'source_phase1': str(phase1_path),
        'split_summaries': split_summaries,
        'aggregate': {
            'scalar_auc': mean_std(scalar_auc),
            'oracle_best_auc_selector_auc': mean_std(selector_values('best_auc_selector')),
            'oracle_best_auc_selector_delta_auc_vs_scalar': mean_std(best_auc_deltas),
            'val_loss_selector_auc': mean_std(selector_values('best_val_loss_selector')),
            'val_loss_selector_delta_auc_vs_scalar': mean_std(best_val_deltas),
            'val_loss_selector_promote_splits': int(sum(x > 0.02 for x in best_val_deltas)),
            'val_loss_selector_drop_splits': int(sum(s['best_val_loss_selector']['auc'] < 0.58 for s in split_summaries)),
            'by_latent': by_latent,
            'all_run_corr_val_loss_with_auc': spearman(val_losses, aucs),
            'all_run_corr_val_loss_with_delta_auc': spearman(val_losses, deltas),
            'all_run_corr_abs_degree_with_auc': spearman(degree_corr_abs, aucs),
            'all_run_corr_abs_margin_with_auc': spearman(margin_corr_abs, aucs),
            'all_run_corr_ref_anom_ratio_with_delta_auc': spearman(ref_ratio, deltas),
        },
        'interpretation': {
            'val_loss_is_reliable_selector': bool(val_selector_promising),
            'fixed_latent_repair_candidates': fixed_latent_promising,
            'oracle_best_auc_is_deployable': False,
            'oracle_note': 'Best-AUC selector uses labels and is only an upper bound/autopsy, not a deployable training strategy.',
        },
        'decision': decision,
        'time_sec': float(time.time() - start),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print('FINAL ' + json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
