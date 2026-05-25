# lsd_oracle_autopsy_probe blocked launch note

Remote sync/launch command for `lsd_oracle_autopsy_probe` was blocked by tool/user policy before any remote write occurred.

Already completed:
- Read experiment-runner runner-registered pure probe SOP and watchdog SOP.
- Created local probe script/config.
- Local `py_compile` passed.
- `experiment.py validate --profile probe` passed.
- HCCS-25 direct SSH read-only preflight passed:
  - host `VM-2-25-ubuntu`
  - project `~/DualRefGAD` present
  - dataset `~/DualRefGAD/dataset/elliptic.mat` present (36M)
  - Python `/home/linziyao/.conda/envs/DualRefGAD/bin/python`
  - torch `2.0.0+cu117`, CUDA available, 8 GPUs
  - GPUs 0-7 all 0 MiB used / 0% util at preflight
- Registered runner job in `created` state: `exp_20260525_171733_lsd_oracle_autopsy_probe`.

Blocked point:
- Minimal `scp`/remote mkdir/remote py_compile was denied with: `BLOCKED: Command denied by user. Do NOT retry this command.`
- Therefore no Hermes-tracked background SSH process and no watchdog cron were started.

Next step if allowed later:
1. Sync only these files to `/home/linziyao/DualRefGAD/investigations/2026-05-21-dualrefgad-constraint-calibrated-reference-relation/experiments/{scripts,configs}/`:
   - `experiments/scripts/lsd_oracle_autopsy_probe.py`
   - `experiments/configs/lsd_oracle_autopsy_probe.yaml`
   - helper scripts imported by the probe if missing/stale: `route25_mat_mean_margin_failure_autopsy.py`, `route25_leg3_response_matrix_decomposition_probe.py`, `route25_matrix_autoencoder_probe.py`
2. Remote py_compile under `/home/linziyao/.conda/envs/DualRefGAD/bin/python`.
3. Update job `exp_20260525_171733_lsd_oracle_autopsy_probe` to running immediately before Hermes-tracked background SSH launch.
4. Launch via terminal background=true, not nohup.
5. Render/schedule pure-probe watchdog if long-running.
