#!/usr/bin/env python3
from __future__ import annotations
import json, math, statistics as stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)
SRC = Path('/home/openclawvm/.openclaw/workspace/agents/nexus/investigations/2026-05-08-dualrefgad-elliptic-seed3-failure-analysis/experiments/outputs/wandb_histories.json')


def isnum(x):
    return isinstance(x, (int,float)) and not (isinstance(x,float) and math.isnan(x))

def pearson(xs, ys):
    pairs=[(x,y) for x,y in zip(xs,ys) if isnum(x) and isnum(y)]
    if len(pairs)<2: return None
    xs=[p[0] for p in pairs]; ys=[p[1] for p in pairs]
    mx=sum(xs)/len(xs); my=sum(ys)/len(ys)
    vx=sum((x-mx)**2 for x in xs); vy=sum((y-my)**2 for y in ys)
    if vx==0 or vy==0: return None
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/(vx*vy)**0.5

def slope(xs, ys):
    pairs=[(x,y) for x,y in zip(xs,ys) if isnum(x) and isnum(y)]
    if len(pairs)<2: return None
    xs=[p[0] for p in pairs]; ys=[p[1] for p in pairs]
    mx=sum(xs)/len(xs); my=sum(ys)/len(ys)
    den=sum((x-mx)**2 for x in xs)
    if den==0: return None
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/den

def main():
    histories=json.loads(SRC.read_text())
    rows=[]
    for seed,hist in sorted(histories.items(), key=lambda x:int(x[0])):
        # rows may have _step etc. Use epoch sorted.
        hist=[h for h in hist if isnum(h.get('epoch'))]
        hist=sorted(hist, key=lambda h:h['epoch'])
        losses=[h.get('loss') for h in hist]
        aucs=[h.get('test_auc') for h in hist]
        aps=[h.get('test_ap') for h in hist]
        epochs=[h.get('epoch') for h in hist]
        if not hist: continue
        first=hist[0]; last=hist[-1]
        row={
            'seed': int(seed),
            'n_points': len(hist),
            'first_epoch': first.get('epoch'),
            'last_epoch': last.get('epoch'),
            'loss_first': first.get('loss'),
            'loss_final': last.get('loss'),
            'loss_delta': (last.get('loss')-first.get('loss')) if isnum(last.get('loss')) and isnum(first.get('loss')) else None,
            'auc_first': first.get('test_auc'),
            'auc_final': last.get('test_auc'),
            'auc_delta': (last.get('test_auc')-first.get('test_auc')) if isnum(last.get('test_auc')) and isnum(first.get('test_auc')) else None,
            'ap_first': first.get('test_ap'),
            'ap_final': last.get('test_ap'),
            'ap_delta': (last.get('test_ap')-first.get('test_ap')) if isnum(last.get('test_ap')) and isnum(first.get('test_ap')) else None,
            'corr_loss_auc': pearson(losses, aucs),
            'corr_loss_ap': pearson(losses, aps),
            'slope_auc_over_epoch': slope(epochs, aucs),
            'slope_loss_over_epoch': slope(epochs, losses),
        }
        # best diagnostic
        valid_auc=[(h.get('epoch'),h.get('test_auc')) for h in hist if isnum(h.get('test_auc'))]
        valid_loss=[(h.get('epoch'),h.get('loss')) for h in hist if isnum(h.get('loss'))]
        if valid_auc:
            row['auc_peak_epoch'], row['auc_peak'] = max(valid_auc, key=lambda x:x[1])
        if valid_loss:
            row['loss_min_epoch'], row['loss_min'] = min(valid_loss, key=lambda x:x[1])
        rows.append(row)
    summary={
        'rows': rows,
        'mean_loss_delta': stats.mean([r['loss_delta'] for r in rows if isnum(r.get('loss_delta'))]) if rows else None,
        'mean_auc_delta': stats.mean([r['auc_delta'] for r in rows if isnum(r.get('auc_delta'))]) if rows else None,
        'mean_ap_delta': stats.mean([r['ap_delta'] for r in rows if isnum(r.get('ap_delta'))]) if rows else None,
    }
    (OUT/'loss_auc_audit.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    lines=['# Loss vs AUC Audit','', 'Source: WandB history from no-val sweep `0d0py9y1`','', '| seed | loss first | loss final | Δloss | AUC first | AUC final | ΔAUC | AP first | AP final | ΔAP | corr(loss,AUC) |', '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        def f(x): return '-' if x is None else f'{x:.4f}'
        lines.append(f"| {r['seed']} | {f(r['loss_first'])} | {f(r['loss_final'])} | {f(r['loss_delta'])} | {f(r['auc_first'])} | {f(r['auc_final'])} | {f(r['auc_delta'])} | {f(r['ap_first'])} | {f(r['ap_final'])} | {f(r['ap_delta'])} | {f(r['corr_loss_auc'])} |")
    lines += ['', '## Aggregate', '', f"- Mean Δloss: `{summary['mean_loss_delta']}`", f"- Mean ΔAUC: `{summary['mean_auc_delta']}`", f"- Mean ΔAP: `{summary['mean_ap_delta']}`"]
    (OUT/'loss_auc_audit.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__': main()
