#!/usr/bin/env python3
import itertools, json, subprocess, sys, time
from pathlib import Path

ROOT = Path.home() / "VoxG"
SCRIPT = ROOT / "nexus/investigations/2026-04-27-global-ga-score/experiments/scripts/probe_tokenization_theory.py"
OUTDIR = ROOT / "nexus/investigations/2026-04-27-global-ga-score/experiments/outputs/tokenization_theory_probe/grid"
OUTDIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["photo", "elliptic"]
DESCRIPTORS = ["hop_attr", "rwse", "hop_attr_rwse"]
PN = ["diag_gaussian", "pca_residual"]
GN = ["label_gate", "normal_density", "label_gate_density"]
LN = ["descriptor_similarity", "reconstruction_gain"]
GA = ["normal_rejection", "residual_norm", "normal_soft_or"]
LA = ["residual_cosine", "descriptor_similarity"]

summary = []
total = len(DATASETS)*len(DESCRIPTORS)*len(PN)*len(GN)*len(LN)*len(GA)*len(LA)
start = time.time(); done = 0
for ds, desc, pn, gn, ln, ga, la in itertools.product(DATASETS, DESCRIPTORS, PN, GN, LN, GA, LA):
    done += 1
    name = f"{ds}__{desc}__{pn}__{gn}__{ln}__{ga}__{la}.json"
    out = OUTDIR / name
    print(f"[{done}/{total}] {name}", flush=True)
    cmd = [
        sys.executable, str(SCRIPT),
        "--dataset", ds, "--seed", "0", "--train_rate", "0.05",
        "--descriptor_mode", desc, "--pn_estimator", pn,
        "--gn_mode", gn, "--ln_mode", ln,
        "--ga_mode", ga, "--la_mode", la,
        "--normal_k", "4", "--anom_k", "16", "--out", str(out),
    ]
    t0=time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    rec = {"dataset": ds, "descriptor_mode": desc, "pn_estimator": pn, "gn_mode": gn, "ln_mode": ln, "ga_mode": ga, "la_mode": la, "file": str(out), "returncode": proc.returncode, "elapsed_sec": time.time()-t0}
    if proc.returncode != 0:
        rec["error"] = (proc.stderr or proc.stdout)[-4000:]
        print("FAILED", name, rec["error"].splitlines()[-5:], flush=True)
    else:
        try:
            data = json.loads(out.read_text())
            rec["scores"] = data.get("scores", {})
            rec["reference"] = data.get("reference", {})
            rec["meta"] = data.get("meta", {})
        except Exception as e:
            rec["error"] = f"parse output failed: {e}"
    summary.append(rec)
    (OUTDIR / "summary_partial.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

(OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print("ALL_DONE", OUTDIR / "summary.json", "elapsed", time.time()-start)
